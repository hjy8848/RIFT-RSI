"""Versioned deterministic indexes over independently registered frozen worlds.

This module returns two deliberately separate products: an internal index that
contains frozen objects and scopes, and a compact, identifier-free catalogue
that can safely be added to a Designer request.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from usami_harness.models import Model, digest
from usami_benchmark.contracts import DOMAINS
from usami_benchmark.public_policy import catalogue as native_catalogue
from usami_benchmark.splits import verified_world
from usami_benchmark.worlds import FrozenWorld, WorldPool

from .abstract_challenges import challenge_grammar


INDEX_PROTOCOL = "world-grounding-index-v1"
CATALOGUE_PROTOCOL = "world-grounded-domain-catalogue-v2"


class IndexedWorld(Model):
    registration_hash: str
    world_hash: str
    partition: str
    domain: str
    rules_version: str
    structure_signature: str
    permission_signature: str
    generator_source_hash: str
    rules_source_hash: str
    frozen: FrozenWorld
    entity_counts: dict[str, int]
    identity_profiles: dict[str, list[str]] = Field(default_factory=dict)
    relationship_edges: list[dict[str, str]] = Field(default_factory=list)
    external_events_by_line: dict[str, list[str]] = Field(default_factory=dict)


class PrivateWorldIndex(Model):
    protocol: Literal[INDEX_PROTOCOL] = INDEX_PROTOCOL
    pool_hash: str
    index_schema: str = INDEX_PROTOCOL
    worlds: list[IndexedWorld] = Field(min_length=1)
    index_hash: str = ""

    def body(self):
        return self.model_dump(mode="json", exclude={"index_hash"})

    def computed_hash(self):
        return digest(self.body())

    def checked(self):
        if self.index_hash != self.computed_hash():
            raise ValueError("private world index content hash differs")
        if digest(sorted(row.registration_hash for row in self.worlds)) != self.pool_hash:
            raise ValueError("private world index pool commitment differs")
        if len({row.registration_hash for row in self.worlds}) != len(self.worlds):
            raise ValueError("private world index has duplicate registrations")
        for row in self.worlds:
            frozen = verified_world(row.frozen)
            if (frozen.registration_hash, frozen.world_hash, frozen.spec.partition, frozen.spec.domain,
                    frozen.spec.rules_version, frozen.structure_signature, frozen.permission_signature,
                    frozen.generator_source_hash, frozen.rules_source_hash) != (
                    row.registration_hash, row.world_hash, row.partition, row.domain, row.rules_version,
                    row.structure_signature, row.permission_signature, row.generator_source_hash, row.rules_source_hash):
                raise ValueError("indexed world registration metadata differs")
            expected_counts, expected_identities, expected_edges, expected_events = _world_facts(frozen)
            if (row.entity_counts, row.identity_profiles, row.relationship_edges, row.external_events_by_line) != (expected_counts, expected_identities, expected_edges, expected_events):
                raise ValueError("private world relationship index differs from frozen world")
        return self


def _world_facts(frozen: FrozenWorld):
    world = frozen.world
    counts = {key: len(value) for key, value in sorted(world.items()) if isinstance(value, dict)}
    edges: list[dict[str, str]] = []
    events_by_line: dict[str, list[str]] = {}
    identity_profiles: dict[str, list[str]] = {}
    if frozen.spec.domain == "aftersales":
        for profile_id, profile in sorted(world.get("profiles", {}).items()):
            identity = profile.get("identity")
            if identity:
                identity_profiles.setdefault(identity, []).append(profile_id)
                edges.append({"from_type": "profile", "from_id": profile_id,
                    "relation": "authenticated_identity", "to_type": "customer", "to_id": identity})
        for line in sorted(world.get("lines", {}).values(), key=lambda r: r["id"]):
            edges += [
                {"from_type": "line", "from_id": line["id"], "relation": "references", "to_type": "order", "to_id": line["order_id"]},
                {"from_type": "line", "from_id": line["id"], "relation": "references", "to_type": "product", "to_id": line["product_id"]},
            ]
        for order in sorted(world.get("orders", {}).values(), key=lambda r: r["id"]):
            edges.append({"from_type": "order", "from_id": order["id"],
                "relation": "authenticated_customer", "to_type": "customer", "to_id": order["customer_id"]})
        for profile_id, profile in sorted(world.get("profiles", {}).items()):
            facts = profile.get("facts", {})
            for name, typ in (("line_id", "line"), ("order_id", "order"), ("evidence_id", "evidence")):
                if facts.get(name):
                    edges.append({"from_type": "profile", "from_id": profile_id, "relation": "references", "to_type": typ, "to_id": facts[name]})
        for evidence in sorted(world.get("evidence", {}).values(), key=lambda r: r["id"]):
            edges.append({"from_type": "evidence", "from_id": evidence["id"], "relation": "references", "to_type": "line", "to_id": evidence["line_id"]})
        for event in sorted(world.get("external_events", {}).values(), key=lambda r: r["id"]):
            edges.append({"from_type": "external_event", "from_id": event["id"], "relation": "references", "to_type": "line", "to_id": event["line_id"]})
            events_by_line.setdefault(event["line_id"], []).append(event["id"])
    else:
        # General indexability is retained; domain-specific relation typing is
        # provided by the corresponding native adapter when registered.
        for collection, rows in sorted(world.items()):
            if not isinstance(rows, dict):
                continue
            for entity_id, row in sorted(rows.items()):
                if not isinstance(row, dict):
                    continue
                for field, target in sorted(row.items()):
                    if field.endswith("_id") and isinstance(target, str) and target:
                        edges.append({"from_type": collection, "from_id": entity_id, "relation": field[:-3], "to_type": "unknown", "to_id": target})
    return counts, {k: sorted(v) for k, v in sorted(identity_profiles.items())}, sorted(edges, key=lambda e: (e["from_type"], e["from_id"], e["relation"], e["to_type"], e["to_id"])), {k: sorted(v) for k, v in sorted(events_by_line.items())}


def build_world_index(pool: WorldPool) -> PrivateWorldIndex:
    """Index a registered WorldPool only; mutable JSON-world inputs are refused."""
    if not isinstance(pool, WorldPool):
        raise TypeError("build_world_index requires a registered WorldPool")
    rows = []
    for registration in sorted(pool.entries):
        frozen = verified_world(pool.load(registration))
        if frozen.registration_hash != registration:
            raise ValueError("pool registration key differs from frozen world")
        counts, identities, edges, events = _world_facts(frozen)
        rows.append(IndexedWorld(registration_hash=registration, world_hash=frozen.world_hash,
            partition=frozen.spec.partition, domain=frozen.spec.domain, rules_version=frozen.spec.rules_version,
            structure_signature=frozen.structure_signature, permission_signature=frozen.permission_signature,
            generator_source_hash=frozen.generator_source_hash, rules_source_hash=frozen.rules_source_hash,
            frozen=frozen, entity_counts=counts, identity_profiles=identities,
            relationship_edges=edges, external_events_by_line=events))
    expected_pool_hash = digest(sorted(r.registration_hash for r in rows))
    if pool.pool_hash != expected_pool_hash:
        raise ValueError("WorldPool commitment differs from verified registrations")
    result = PrivateWorldIndex(pool_hash=pool.pool_hash, worlds=rows)
    result.index_hash = result.computed_hash()
    return result.checked()


def load_world_index(path) -> PrivateWorldIndex:
    """Load and independently regenerate every included frozen registration."""
    from pathlib import Path
    from usami.service.models import strict_json
    raw = strict_json(Path(path).read_text())
    row = PrivateWorldIndex.model_validate(raw)
    row.checked()
    return row


def verify_world_index(index: PrivateWorldIndex | dict, pool: WorldPool) -> bool:
    row = index if isinstance(index, PrivateWorldIndex) else PrivateWorldIndex.model_validate(index)
    expected = build_world_index(pool)
    if row != expected:
        raise ValueError("rebuilt world index differs from supplied index")
    return True


def public_domain_catalogue(index: PrivateWorldIndex, domain: str, *, partitions: tuple[str, ...] = ("train",)) -> dict[str, Any]:
    index = index.checked()
    if domain not in DOMAINS:
        raise ValueError("unregistered domain")
    if domain != "aftersales":
        raise ValueError("no symbolic Designer grammar/Grounder is registered for this domain")
    if not partitions or set(partitions) - {"train", "validation"}:
        raise ValueError("Designer catalogue access is restricted to explicit train/validation partitions")
    selected = [w for w in index.worlds if w.domain == domain and w.partition in partitions]
    if not selected:
        raise ValueError("no registered worlds in the requested domain/partition")
    versions = {(w.rules_version, w.frozen.spec.generator_version) for w in selected}
    if len(versions) != 1:
        raise ValueError("catalogue cannot mix business-rule or generator versions")
    rules_version, generator_version = next(iter(versions))
    base = native_catalogue(domain, rules_version)
    entity_names = sorted({key for w in selected for key in w.entity_counts})
    counts = {name: {"min": min(w.entity_counts.get(name, 0) for w in selected),
                     "max": max(w.entity_counts.get(name, 0) for w in selected)} for name in entity_names}
    body = {
        **challenge_grammar(),
        "protocol": CATALOGUE_PROTOCOL, "domain": domain, "rules_version": rules_version,
        "generator_version": generator_version, "native_policy": base,
        "world_pool_coverage": {"authorized_partitions": list(partitions), "world_count": len(selected),
            "distinct_structures": len({w.structure_signature for w in selected}),
            "distinct_permission_structures": len({w.permission_signature for w in selected}),
            "aggregate_entity_count_ranges": counts,
            "scope": "Aggregate structure only; no world IDs, object IDs, per-world states, witnesses, Oracle labels, or service outcomes."},
        "pool_commitment": digest(sorted(w.registration_hash for w in selected)),
    }
    body["catalogue_hash"] = digest(body)
    return body
