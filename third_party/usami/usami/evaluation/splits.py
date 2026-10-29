"""Frozen public validation and caller-owned private test manifests."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Literal, Optional, Union

from pydantic import Field

from usami.randomness import StableRandom

from usami.evaluation.scenario import (SCENARIO_VERSION, Scenario, generate_scenarios,
                                       validate_scenario)
from usami.world.generator import GeneratorConfig, composition_signature, generate_world
from usami.world.models import GENERATOR_VERSION, Model, WorldState, digest

MANIFEST_VERSION = "usami-manifest-0.1.0"


class ManifestEntry(Model):
    seed: int
    world_id: str
    initial_state_hash: str
    business_fingerprint: str
    scenarios: List[Scenario]


class Manifest(Model):
    version: str = MANIFEST_VERSION
    split: Literal["train", "validation", "test"]
    visibility: Literal["public", "private", "demo"]
    generator_version: str = GENERATOR_VERSION
    scenario_version: str = SCENARIO_VERSION
    config: GeneratorConfig
    config_hash: str
    rules_hash: str
    operation_budget: int = Field(default=20, ge=1)
    entries: List[ManifestEntry]
    manifest_hash: str = ""

    def computed_hash(self) -> str:
        return digest(self.model_dump(mode="json", exclude={"manifest_hash"}))

    def save(self, path: Union[str, Path]) -> None:
        validate_manifest(self)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


def business_fingerprint(world: WorldState) -> str:
    # Exclude provenance, retain all actual relations, permissions and business state.
    return digest(world.model_dump(mode="json", exclude={"world_id", "seed", "config_hash", "generator_version"}))


def create_manifest(split: str, seeds: Iterable[int], config: Optional[GeneratorConfig] = None,
                    scenarios_per_world: int = 12, operation_budget: int = 20,
                    visibility: Optional[str] = None) -> Manifest:
    if scenarios_per_world < 1:
        raise ValueError("at least one scenario per world")
    cfg = config or GeneratorConfig(profile="train_single" if split == "train" else
                                     "test_composed" if split == "test" else "standard")
    entries = []
    for seed in seeds:
        world = generate_world(seed, cfg)
        candidates = generate_scenarios(world)
        # A fixed selection first covers categories, then samples within categories.
        rng = StableRandom(seed)
        rng.shuffle(candidates)
        chosen, categories = [], set()
        for scenario in candidates:
            if scenario.category not in categories and len(chosen) < scenarios_per_world:
                chosen.append(scenario)
                categories.add(scenario.category)
        chosen_ids = {s.scenario_id for s in chosen}
        for scenario in candidates:
            if scenario.scenario_id not in chosen_ids and len(chosen) < scenarios_per_world:
                chosen.append(scenario)
                chosen_ids.add(scenario.scenario_id)
        for scenario in chosen:
            if len(scenario.steps) > operation_budget:
                raise ValueError("operation budget too small for selected scenario")
            scenario.operation_budget = operation_budget
        entries.append(ManifestEntry(seed=seed, world_id=world.world_id, initial_state_hash=world.state_hash,
                                     business_fingerprint=business_fingerprint(world), scenarios=chosen))
    manifest = Manifest(split=split, visibility=visibility or ("private" if split == "test" else "public"),
                        config=cfg.model_copy(deep=True), config_hash=digest(cfg), rules_hash=digest(cfg.rules),
                        operation_budget=operation_budget, entries=entries)
    manifest.manifest_hash = manifest.computed_hash()
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Manifest) -> None:
    if (manifest.version != MANIFEST_VERSION or manifest.generator_version != GENERATOR_VERSION
            or manifest.scenario_version != SCENARIO_VERSION or not manifest.entries):
        raise ValueError("empty manifest or unsupported manifest/generator/scenario version")
    if (manifest.config_hash != digest(manifest.config) or manifest.rules_hash != digest(manifest.config.rules)
            or manifest.manifest_hash != manifest.computed_hash()):
        raise ValueError("manifest hash/config/rules mismatch")
    if manifest.split == "test" and manifest.visibility == "public":
        raise ValueError("test manifest must be private or explicitly labeled demo")
    seen_worlds, seen_scenarios, seen_business = set(), set(), set()
    for entry in manifest.entries:
        world = generate_world(entry.seed, manifest.config)
        if world.world_id in seen_worlds or entry.business_fingerprint in seen_business:
            raise ValueError("duplicate world/business instance in manifest")
        seen_worlds.add(world.world_id)
        seen_business.add(entry.business_fingerprint)
        if (world.world_id != entry.world_id or world.state_hash != entry.initial_state_hash
                or business_fingerprint(world) != entry.business_fingerprint):
            raise ValueError("manifest world cannot be reproduced")
        if manifest.config.profile == "train_single" and composition_signature(world):
            raise ValueError("training contains held-out business combination")
        if manifest.config.profile == "test_composed" and not composition_signature(world):
            raise ValueError("test missing held-out business combination")
        if not entry.scenarios:
            raise ValueError("manifest entry has no scenarios")
        for scenario in entry.scenarios:
            if scenario.scenario_id in seen_scenarios or scenario.operation_budget != manifest.operation_budget:
                raise ValueError("duplicate scenario or inconsistent operation budget")
            seen_scenarios.add(scenario.scenario_id)
            validate_scenario(world, scenario)


def load_manifest(path: Union[str, Path]) -> Manifest:
    """Accept a third-party private path without publishing its seeds or instances."""
    manifest = Manifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def validate_splits(*manifests: Manifest) -> None:
    worlds, business = set(), set()
    for manifest in manifests:
        validate_manifest(manifest)
        current = {e.world_id for e in manifest.entries}
        semantic = {e.business_fingerprint for e in manifest.entries}
        if current & worlds or semantic & business:
            raise ValueError("unintended cross-split instance overlap")
        worlds.update(current)
        business.update(semantic)


def sample_train(start_seed: int = 0, config: Optional[GeneratorConfig] = None) -> Iterator[WorldState]:
    cfg = config or GeneratorConfig(profile="train_single")
    if cfg.profile != "train_single":
        raise ValueError("compositional training stream requires train_single profile")
    seed = start_seed
    while True:
        yield generate_world(seed, cfg)
        seed += 1


def assert_fair_comparison(left: Manifest, right: Manifest) -> None:
    validate_manifest(left)
    validate_manifest(right)
    if left.manifest_hash != right.manifest_hash:
        raise ValueError("methods must use exactly the same frozen evaluation manifest")
