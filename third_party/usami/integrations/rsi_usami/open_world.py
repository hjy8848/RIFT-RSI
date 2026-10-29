"""Persistent open World populations and certified RSI challenge registration.

World generation is independent of Service results. A period publishes a
catalogue, while native IDs, bindings and proof witnesses stay on the Host.
"""
from copy import deepcopy
from pathlib import Path
import json

from pydantic import ValidationError

from rsi_core.archive import Archive
from rsi_core.period import run_period
from rsi_core.protocol import digest
from usami_v2.scenarios.builder import verify_build
from usami_v2.scenarios.designer import (
    OpenAIDesignerProvider, TaskCapabilityUnsupported, WorldDesignRun, design_world_slot,
    task_design_capabilities, _tool_schema,
)
from usami_v2.scenarios.distribution import DistributionSpec, sample_schedule
from usami_v2.scenarios.grounding import AbstractChallenge, GroundingBudget, ground_challenge, compile_grounded
from usami_v2.scenarios.task_certification import certify_task, verify_task_certificate
from usami_v2.runtime.runner import EpisodeBudget
from usami_v2.runtime.customer import bind_customer_profile
from usami_v2.scenarios.tasks import TaskSpecV3

from .adapter import NativeTask


def _challenge_failure_receipt(grounding, attempts):
    """Keep grounding, capability, compile, UNSAT and UNKNOWN outcomes distinct."""
    if not grounding.candidates:
        return {"status": "UNVERIFIED", "reason_code": grounding.status}
    statuses = [row["status"] for row in attempts]
    if statuses and all(status == "CAPABILITY_UNSUPPORTED" for status in statuses):
        codes = sorted({row["capability_code"] for row in attempts})
        return {"status": "UNVERIFIED", "reason_code": "CAPABILITY_UNSUPPORTED",
                "capability_codes": codes}
    if statuses and all(status in ("COMPILE_REJECTED", "CAPABILITY_UNSUPPORTED") for status in statuses):
        return {"status": "UNVERIFIED", "reason_code": "COMPILE_REJECTED"}
    proof_statuses = [status for status in statuses if status in ("UNSAT", "UNKNOWN")]
    if proof_statuses and len(proof_statuses) == len(statuses):
        reason = ("REFERENCE_UNSAT" if set(proof_statuses) == {"UNSAT"} else
                  "REFERENCE_UNKNOWN" if set(proof_statuses) == {"UNKNOWN"} else
                  "REFERENCE_INCOMPLETE")
        return {"status": "UNVERIFIED", "reason_code": reason}
    return {"status": "UNVERIFIED", "reason_code": "CHALLENGE_NOT_CERTIFIED"}


def _abstract_challenge_schema(capabilities):
    """Use the native Challenge schema, narrowed to this run's certifier."""
    schema = deepcopy(_tool_schema("submit_customer_task", AbstractChallenge)["function"]["parameters"])
    supported_remedies = list(capabilities["supported_remedies"])
    if not supported_remedies:
        raise ValueError("the active certifier supports no remedy goals")
    supported_target_stages = list(capabilities["supported_target_stages"])
    if not supported_target_stages:
        raise ValueError("the active certifier supports no target stages")
    remedy = schema["$defs"]["RemedyDraft"]["properties"]
    remedy["allowed_remedies"]["items"]["enum"] = supported_remedies
    remedy["target_stage"]["enum"] = supported_target_stages
    if not capabilities["return_receipt_event_configured"]:
        selector = schema["$defs"]["LineSelector"]
        selector["properties"]["return_required"] = {
            "type": "boolean", "const": False,
            "description": "This run has no return-receipt events; select only items not requiring return.",
        }
        selector["required"].append("return_required")
    if not capabilities["information_goals_certifiable"]:
        schema["properties"]["information_goals"]["maxItems"] = 0
    return schema


def world_structure(spec, build):
    """Alias/name/seed-independent business topology, not an ID-count claim."""
    families = list(dict.fromkeys(row.family for row in spec.skus))
    sku_index = {row.alias: i for i, row in enumerate(spec.skus)}
    customers = {name: i for i, name in enumerate(spec.customer_aliases)}
    ids, world = dict(build.private_aliases), build.world
    return {
        "customers": len(customers), "warehouses": len(spec.warehouses),
        "skus": [{"family_group": families.index(s.family), "attribute_keys": sorted(a.name for a in s.attributes),
                  "return_required": s.return_required, "return_days": s.return_days} for s in spec.skus],
        "orders": [{"owner": customers[o.customer_alias], "lines": [
            {"sku": sku_index[l.original_sku], "issue": l.issue,
             "quantity": world.lines[ids[l.alias]].quantity} for l in o.lines]} for o in spec.orders],
        "stock_availability": [[next(v.available > 0 for v in world.inventory.values()
                                     if v.sku_id == ids[s.alias] and v.warehouse_id == ids[w.alias])
                                for w in spec.warehouses] for s in spec.skus],
    }


class WorldPopulation:
    """A resumable sequence of native World-generation slots, never a task template.

    The provider implements the existing DesignerProvider World phase. Production
    uses OpenAIDesignerProvider; deterministic providers are for integration tests.
    """
    def __init__(self, root, distribution, provider, *, seed=0, build_budget=None, max_attempts=3):
        self.root = Path(root)
        self.archive = Archive(self.root / "records")
        self.distribution = DistributionSpec.model_validate(distribution).checked()
        self.provider, self.seed = provider, seed
        self.build_budget, self.max_attempts = build_budget, max_attempts
        self.state_path = self.root / "state.json"
        self.identity = {"distribution": self.distribution.model_dump(mode="json"), "seed": seed,
                         "build_budget": build_budget.model_dump(mode="json") if build_budget else None,
                         "max_attempts": max_attempts}
        self.period_refs = []
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            saved = self.archive.load(state["state_ref"])
            if saved["identity"] != self.identity:
                raise ValueError("World population resume changed its frozen distribution/configuration")
            self.period_refs = saved["period_refs"]
            for index, ref in enumerate(self.period_refs):
                period = self.load_period(ref)
                if period["index"] != index or period["previous_ref"] != (self.period_refs[index-1] if index else ""):
                    raise ValueError("World population period history differs")

    def load_period(self, ref):
        period = self.archive.load(ref)
        if period["distribution_hash"] != self.distribution.record_hash or period["partition"] != self.distribution.split:
            raise ValueError("World period belongs to another distribution/partition")
        schedule = sample_schedule(self.distribution, int(digest([self.seed, period["index"]])[:8], 16),
                                   len(period["schedule"]))
        if [slot.model_dump(mode="json") for slot in schedule] != period["schedule"]:
            raise ValueError("World sampling schedule differs from its registered distribution")
        if len(period["world_runs"]) != len(schedule):
            raise ValueError("World generation history omits sampling slots")
        signatures = []
        for slot, value in zip(schedule, period["world_runs"]):
            run = WorldDesignRun.model_validate(value).checked()
            if run.sampling_slot != slot or run.distribution != self.distribution:
                raise ValueError("generated World differs from its registered sampling slot")
            if run.status == "SAT":
                if run.scenario_spec.distribution_hash != self.distribution.record_hash:
                    raise ValueError("World ScenarioSpec belongs to another distribution")
                signatures.append(digest(world_structure(run.scenario_spec, run.build_result)))
        if signatures != period["structural_signatures"]:
            raise ValueError("recorded structural diversity differs from actual Worlds")
        return period

    def worlds(self, ref):
        period = self.load_period(ref)
        rows = []
        for index, value in enumerate(period["world_runs"]):
            run = WorldDesignRun.model_validate(value).checked()
            if run.distribution != self.distribution:
                raise ValueError("World source crosses a distribution/partition boundary")
            if run.status != "SAT":
                continue
            verify_build(run.scenario_spec, run.build_result)
            rows.append((f"period-{period['index']}-world-{index}", run.scenario_spec, run.build_result))
        return tuple(rows)

    def next_period(self, count):
        if count * self.max_attempts > self.distribution.max_generation_attempts:
            raise ValueError("World sampling slots exceed the registered generation-attempt budget")
        index = len(self.period_refs)
        period_seed = int(digest([self.seed, index])[:8], 16)
        schedule = sample_schedule(self.distribution, period_seed, count)
        runs = [design_world_slot(self.provider, self.distribution, slot,
                                 build_budget=self.build_budget, max_attempts=self.max_attempts)
                for slot in schedule]
        prior_hashes = {build.world_hash for ref in self.period_refs for _, _, build in self.worlds(ref)}
        current_hashes = [run.build_result.world_hash for run in runs if run.status == "SAT"]
        if len(set(current_hashes)) != len(current_hashes) or set(current_hashes) & prior_hashes:
            raise ValueError("World generator repeated a frozen World; no replacement was sampled")
        period = {"kind": "rsi-open-world-period-v1", "index": index,
                  "previous_ref": self.period_refs[-1] if self.period_refs else "",
                  "distribution_hash": self.distribution.record_hash,
                  "partition": self.distribution.split,
                  "schedule": [slot.model_dump(mode="json") for slot in schedule],
                  "world_runs": [run.model_dump(mode="json") for run in runs],
                  "structural_signatures": [digest(world_structure(run.scenario_spec, run.build_result))
                                            for run in runs if run.status == "SAT"]}
        ref = self.archive.save(period)
        self.period_refs.append(ref)
        state_ref = self.archive.save({"identity": self.identity, "period_refs": self.period_refs})
        pending = self.state_path.with_suffix(".tmp")
        pending.write_text(json.dumps({"state_ref": state_ref}) + "\n")
        pending.replace(self.state_path)
        return ref

    def catalogue(self, period_ref, *, challenge_schema=None):
        worlds = self.worlds(period_ref)
        # Publish aggregate vocabulary plus anonymous topology; native IDs stay private.
        families, issues, attributes, line_kinds = set(), set(), set(), set()
        topologies = []
        for world_index, (_, spec, build) in enumerate(worlds):
            families.update(row.family for row in spec.skus)
            issues.update(line.issue for order in spec.orders for line in order.lines)
            attributes.update((a.name, a.value) for sku in spec.skus for a in sku.attributes)
            aliases = dict(build.private_aliases)
            sku_by_alias = {row.alias: row for row in spec.skus}
            customer_rows = []
            customer_aliases = {alias: f"customer_{index + 1}"
                                for index, alias in enumerate(spec.customer_aliases)}
            order_index = 0
            line_index = 0
            for customer in spec.customer_aliases:
                customer_orders = []
                for order in spec.orders:
                    if order.customer_alias != customer:
                        continue
                    order_index += 1
                    lines = []
                    for line in order.lines:
                        line_index += 1
                        sku = sku_by_alias[line.original_sku]
                        native_line = build.world.lines[aliases[line.alias]]
                        age_days = (build.world.clock -
                                    build.world.orders[aliases[order.alias]].delivered_at).days
                        lines.append({
                            "alias": f"line_{line_index}",
                            "family": sku.family,
                            "issue": line.issue,
                            "attributes": [{"name": item.name, "value": item.value}
                                           for item in sku.attributes],
                            "quantity": native_line.quantity,
                            "age_days": age_days,
                            "return_required": sku.return_required,
                        })
                        line_kinds.add((sku.family, line.issue, native_line.quantity,
                                        sku.return_required))
                    customer_orders.append({"alias": f"order_{order_index}", "lines": lines})
                customer_rows.append({"alias": customer_aliases[customer], "orders": customer_orders})
            topologies.append({"alias": f"world_{world_index + 1}", "customers": customer_rows})
        return {"protocol": "commerce-open-world-catalogue-v1", "world_count": len(worlds),
                "distinct_structures": len(set(self.load_period(period_ref)["structural_signatures"])),
                "families": sorted(families), "issues": sorted(issues),
                "observed_line_kinds": [{"family": f, "issue": i, "quantity": q, "return_required": r}
                                        for f, i, q, r in sorted(line_kinds)],
                "world_topologies": topologies,
                "attributes": [{"name": k, "value": v} for k, v in sorted(attributes)],
                "request_contract": "Challenges contain structured Customer intents. The Host binds each intent "
                                    "to owned frozen objects and renders the public request from the same intent. "
                                    "No keyword or substring matching is used to authorize scored goals.",
                "challenge_schema": challenge_schema if challenge_schema is not None else
                    _tool_schema("submit_customer_task", AbstractChallenge)["function"]["parameters"]}


class OpenChallengeBridge:
    """The existing RSI certify_challenge callback over one frozen population period."""
    def __init__(self, population, period_ref, adapter, verifier, customer_binding, *,
                 budget=None, episode_budget=None, mode="reference", allowed_training_hashes=(),
                 proof_budget=None, capability_options=None, external_event_factory=None):
        if population.distribution.split != "train":
            raise ValueError("RSI training challenges cannot use validation/heldout World populations")
        self.population, self.period_ref = population, period_ref
        self.worlds = population.worlds(period_ref)
        self.adapter, self.verifier, self.customer_binding = adapter, verifier, customer_binding
        self.budget = GroundingBudget.model_validate(budget or {})
        if self.budget.partition != "train":
            raise ValueError("RSI training Grounder requires the train partition")
        self.episode_budget, self.mode = episode_budget or EpisodeBudget(), mode
        self.allowed_training_hashes, self.proof_budget = tuple(allowed_training_hashes), proof_budget
        self.capability_options = dict(capability_options or {})
        self.external_event_factory = external_event_factory
        self.attempt_refs = []
        self.archive = verifier.archive

    def public_view(self):
        # Reuse the compiler's actual capabilities, including explicit false
        # flags. World hashes/scopes stay private; they are unnecessary for an
        # ID-free challenge that will be grounded by the Host.
        manifests = [task_design_capabilities(spec, build, **self.capability_options).model_dump(mode="json",
            exclude={"record_hash", "scenario_hash", "world_hash", "customer_scopes"})
            for _, spec, build in self.worlds]
        if manifests and any(row != manifests[0] for row in manifests[1:]):
            raise ValueError("one open challenge catalogue requires a consistent native certifier contract")
        capability_manifest = manifests[0] if manifests else None
        return {"domain": self.adapter.domain_id,
                "open_world_catalogue": self.population.catalogue(
                    self.period_ref,
                    challenge_schema=_abstract_challenge_schema(capability_manifest)
                    if capability_manifest else None),
                "task_certification_capabilities": capability_manifest}

    def certify(self, proposal):
        record = {"kind": "rsi-open-challenge-attempt-v1", "period_ref": self.period_ref,
                  "proposal": proposal, "proof_attempts": [], "selected": None}
        try:
            challenge = AbstractChallenge.model_validate(proposal).checked()
        except ValidationError as exc:
            record["receipt"] = {"status": "INVALID", "reason_code": "CHALLENGE_SCHEMA_INVALID"}
            record["validation_error"] = str(exc)
            return self._save(record)
        if not set(challenge.parent_training_evidence_hashes) <= set(self.allowed_training_hashes):
            record["receipt"] = {"status": "INVALID", "reason_code": "UNAUTHORIZED_TRAINING_EVIDENCE"}
            return self._save(record)
        grounding = ground_challenge(challenge, self.worlds, self.budget)
        record["grounding"] = grounding.model_dump(mode="json")
        for candidate in grounding.candidates:
            _, spec, build = next(row for row in self.worlds if row[0] == candidate.world_ref)
            try:
                task = compile_grounded(candidate, spec, build, self.population.distribution,
                                        self.allowed_training_hashes, capability_options=self.capability_options)
            except TaskCapabilityUnsupported as exc:
                record["proof_attempts"].append({"candidate_hash": candidate.record_hash,
                    "status": "CAPABILITY_UNSUPPORTED", "capability_code": exc.code, "reason": str(exc)})
                continue
            except ValueError as exc:
                record["proof_attempts"].append({"candidate_hash": candidate.record_hash,
                    "status": "COMPILE_REJECTED", "reason": str(exc)})
                continue
            profile, scenario, customer_factory = self.customer_binding(build.world, task)
            if isinstance(task, TaskSpecV3):
                profile = bind_customer_profile(profile, task)
            certificate = certify_task(build.world, task, scenario, self.proof_budget)
            attempt = {"candidate_hash": candidate.record_hash, "status": certificate.status,
                       "task": task.model_dump(mode="json"), "scenario": scenario.model_dump(mode="json"),
                       "certificate": certificate.model_dump(mode="json")}
            record["proof_attempts"].append(attempt)
            if certificate.status != "CERTIFIED":
                continue
            verify_task_certificate(build.world, task, scenario, certificate)
            task_ref = "open-" + digest([self.period_ref, challenge.record_hash, candidate.record_hash])
            origin = "native-world/" + build.world_hash
            events = tuple(self.external_event_factory(build.world, task, scenario)) if self.external_event_factory else ()
            native = NativeTask(build.world, task, profile, customer_factory, self.episode_budget,
                                external_events=events, mode=self.mode)
            if task_ref not in self.adapter.tasks:
                # Both public registries must agree before a handle can reach RSI.
                if task_ref in self.verifier.tasks:
                    raise ValueError("partial native task registration")
                proof = self.verifier.register(task_ref, native, origin, scenario, certificate)
                self.adapter.register_task(task_ref, native)
            else:
                if self.adapter.tasks[task_ref] != native:
                    raise ValueError("same abstract challenge changed its frozen runtime binding")
                proof = self.verifier.issue(task_ref, origin, scenario, certificate)
            record["selected"] = candidate.record_hash
            record["receipt"] = {"status": "CERTIFIED", "task_ref": task_ref,
                "certificate_ref": proof.certificate_ref,
                "certification_scope": "usami_v2.native_reference_task_certification", "frozen": True}
            return self._save(record)
        record["receipt"] = _challenge_failure_receipt(grounding, record["proof_attempts"])
        return self._save(record)

    def _save(self, record):
        self.attempt_refs.append(self.archive.save(record))
        return record["receipt"]

    def verify_attempt(self, ref):
        """Recompute matches, candidate order, native proofs and selected binding."""
        record = self.archive.load(ref)
        if record["period_ref"] != self.period_ref:
            raise ValueError("challenge attempt belongs to another World period")
        try:
            challenge = AbstractChallenge.model_validate(record["proposal"]).checked()
        except ValidationError:
            expected = {"status": "INVALID", "reason_code": "CHALLENGE_SCHEMA_INVALID"}
            if record["receipt"] != expected:
                raise ValueError("invalid proposal receipt differs")
            return {"verified": True, "status": "INVALID"}
        if not set(challenge.parent_training_evidence_hashes) <= set(self.allowed_training_hashes):
            if record["receipt"] != {"status": "INVALID", "reason_code": "UNAUTHORIZED_TRAINING_EVIDENCE"}:
                raise ValueError("training-evidence rejection differs")
            return {"verified": True, "status": "INVALID"}
        rebuilt = ground_challenge(challenge, self.worlds, self.budget)
        if rebuilt.model_dump(mode="json") != record["grounding"]:
            raise ValueError("grounding candidates or selection order differs")
        attempts, selected, selected_native = [], None, None
        for candidate in rebuilt.candidates:
            _, spec, build = next(row for row in self.worlds if row[0] == candidate.world_ref)
            try:
                task = compile_grounded(candidate, spec, build, self.population.distribution, self.allowed_training_hashes,
                                        capability_options=self.capability_options)
            except TaskCapabilityUnsupported as exc:
                attempts.append({"candidate_hash": candidate.record_hash,
                    "status": "CAPABILITY_UNSUPPORTED", "capability_code": exc.code, "reason": str(exc)})
                continue
            except ValueError as exc:
                attempts.append({"candidate_hash": candidate.record_hash, "status": "COMPILE_REJECTED", "reason": str(exc)})
                continue
            profile, scenario, factory = self.customer_binding(build.world, task)
            if isinstance(task, TaskSpecV3):
                profile = bind_customer_profile(profile, task)
            certificate = certify_task(build.world, task, scenario, self.proof_budget)
            verify_task_certificate(build.world, task, scenario, certificate)
            attempts.append({"candidate_hash": candidate.record_hash, "status": certificate.status,
                             "task": task.model_dump(mode="json"), "scenario": scenario.model_dump(mode="json"),
                             "certificate": certificate.model_dump(mode="json")})
            if certificate.status == "CERTIFIED":
                selected = candidate
                events = tuple(self.external_event_factory(build.world, task, scenario)) if self.external_event_factory else ()
                selected_native = NativeTask(build.world, task, profile, factory, self.episode_budget,
                                             external_events=events, mode=self.mode)
                break
        if attempts != record["proof_attempts"] or record["selected"] != (selected.record_hash if selected else None):
            raise ValueError("native proof attempts or selected candidate differ")
        if selected is not None:
            from .packages import _binding
            receipt = record["receipt"]
            expected_ref = "open-" + digest([self.period_ref, challenge.record_hash, selected.record_hash])
            saved = self.archive.load(receipt["certificate_ref"])
            if (receipt != {"status": "CERTIFIED", "task_ref": expected_ref,
                    "certificate_ref": receipt["certificate_ref"],
                    "certification_scope": "usami_v2.native_reference_task_certification", "frozen": True}
                    or saved["binding"] != _binding(selected_native)
                    or saved["task_ref"] != expected_ref
                    or saved["origin_ref"] != "native-world/" + selected.world_hash
                    or saved["certificate"] != attempts[-1]["certificate"]):
                raise ValueError("certified task registration differs from replayed native binding")
        else:
            expected = _challenge_failure_receipt(rebuilt, attempts)
            if record["receipt"] != expected:
                raise ValueError("uncertified challenge receipt differs")
        return {"verified": True, "status": record["receipt"]["status"]}


def run_open_period(population, world_count, host, service_evaluator, adaptation_evaluator,
                    designer, service, service_rounds, self_edit_context, customer_binding, *,
                    verifier, episode_budget=None, grounding_budget=None, proof_budget=None, mode="reference",
                    capability_options=None, external_event_factory=None, allowed_training_hashes=(), **kwargs):
    """Formal v0.2 entry: fresh Worlds each call, unchanged RSI algorithm/gates.

    Independent task sets/packages are supplied through the existing Core
    registries. They are never added to the open challenge catalogue.
    """
    if population.distribution.split != "train":
        raise ValueError("open RSI periods require a training World distribution")
    reserved_worlds = {native.world.state_hash for native in verifier.tasks.values()}
    period_ref = population.next_period(world_count)
    if any(build.world_hash in reserved_worlds for _, _, build in population.worlds(period_ref)):
        raise ValueError("open exploration World overlaps an independently registered task source")
    bridge = OpenChallengeBridge(population, period_ref, service_evaluator.adapter,
        verifier, customer_binding,
        budget=grounding_budget, episode_budget=episode_budget, proof_budget=proof_budget, mode=mode,
        capability_options=capability_options, external_event_factory=external_event_factory,
        allowed_training_hashes=allowed_training_hashes)
    rounds = [{**config, "certify_challenge": bridge.certify,
               "public_view": {**config.get("public_view", {}), **bridge.public_view()}}
              for config in service_rounds]
    result = run_period(host, service_evaluator, adaptation_evaluator, designer, service,
                        rounds, self_edit_context, **kwargs)
    for ref in bridge.attempt_refs:
        bridge.verify_attempt(ref)
    receipt = {"kind": "rsi-open-period-execution-v1", "world_period_ref": period_ref,
               "challenge_attempt_refs": bridge.attempt_refs, "rsi_period": result}
    return {**receipt, "archive_ref": host.archive.save(receipt)}


def propose_open_challenge(provider, bridge, *, previous_attempt_ref=None):
    """One actual Designer submission; explicit validation, no automatic retries.

    This tests autonomous proposal generation separately from scripted period
    integration. The model receives only this training population's catalogue.
    """
    public_view = bridge.public_view()
    catalogue_context = {key: value for key, value in public_view["open_world_catalogue"].items()
                         if key != "challenge_schema"}
    designer_context = {**public_view, "open_world_catalogue": catalogue_context}
    messages = [{"role": "system", "content":
        "You are the Usami RSI Challenge Designer. Propose one abstract business challenge using the supplied "
        "native schema and authorized training population. Inspect its anonymous world_topologies before choosing "
        "customer/order relations; use only combinations shown by the topology. Choose the business objective, "
        "constraints and "
        "number of related items yourself. line_selectors declare symbolic variables; remedy_goals.line_alias "
        "references those variables. customer_alias is a symbolic Customer variable, not a World entity. "
        "A Grounder will enumerate customer-owned bindings. Submit only structured remedy/information intent "
        "fields and constraints; do not write customer_request, public_request_span or subject_mention. The Host "
        "will render the Customer text from those same structured goals, so the scored goals are authorized by "
        "construction, not by matching prose. Use only published supported capabilities. Submit the challenge with "
        "the registered tool; do not supply IDs, proof witnesses, answers or World changes."},
        {"role": "user", "content": json.dumps(designer_context, ensure_ascii=False, sort_keys=True)}]
    if previous_attempt_ref is not None:
        previous = bridge.archive.load(previous_attempt_ref)
        if previous["period_ref"] != bridge.period_ref:
            raise ValueError("revision evidence belongs to another frozen World period")
        if previous["kind"] == "rsi-open-challenge-attempt-v1":
            bridge.verify_attempt(previous_attempt_ref)
            if previous["receipt"]["status"] == "CERTIFIED":
                raise ValueError("a certified challenge is not a rejected proposal to revise")
            prior_proposal, rejection = previous["proposal"], dict(previous["receipt"])
            # The native Task Designer already returns public compilation
            # errors. Publish these two contract diagnostics, never a private
            # Reference blocker, witness, object binding or Oracle judgment.
            capability_codes = sorted({row["capability_code"] for row in previous["proof_attempts"]
                                       if row["status"] == "CAPABILITY_UNSUPPORTED"})
            if capability_codes:
                rejection["capability_feedback"] = capability_codes
        elif (previous["kind"] == "rsi-autonomous-challenge-validation-v1"
              and previous.get("error_code") == "DESIGNER_SCHEMA_INVALID"):
            from usami_v2.runtime.agents import _one_tool_call, _json_arguments
            message = _one_tool_call(previous["call"]["response"])
            prior_proposal = _json_arguments(message["tool_calls"][0]["function"]["arguments"])
            rejection = {"status": "INVALID", "reason_code": "DESIGNER_SCHEMA_INVALID",
                         "schema_error": previous["error"]}
        else:
            raise ValueError("revision requires a recorded proposal/schema rejection")
        def without_hashes(value):
            if isinstance(value, dict):
                return {k: without_hashes(v) for k, v in value.items() if k != "record_hash"}
            if isinstance(value, list):
                return [without_hashes(v) for v in value]
            return value
        messages.append({"role": "user", "content": json.dumps({
            "instruction": "Revise your challenge using this controlled Host rejection and the authorized catalogue.",
            "prior_proposal": without_hashes(prior_proposal),
            "rejection": rejection}, ensure_ascii=False)})
    record = {"kind": "rsi-autonomous-challenge-validation-v1", "period_ref": bridge.period_ref,
              "parent_attempt_ref": previous_attempt_ref,
              "validation_scope": "Designer proposal/grounding/certification; no online Service or training"}
    try:
        if isinstance(provider, OpenAIDesignerProvider):
            challenge, call = provider.submit(
                "task", messages, AbstractChallenge,
                schema_override=public_view["open_world_catalogue"]["challenge_schema"])
        else:
            challenge, call = provider.submit("task", messages, AbstractChallenge)
    except Exception as exc:
        call = getattr(provider, "last_call", None)
        record.update(status="FAILED", error_code=getattr(exc, "code", type(exc).__name__),
                      error=str(exc), call=call.model_dump(mode="json") if call else None)
    else:
        record.update(call=call.model_dump(mode="json"), proposal=challenge.model_dump(mode="json"))
        receipt = bridge.certify(challenge.model_dump(mode="json"))
        record.update(status=receipt["status"], receipt=receipt, attempt_ref=bridge.attempt_refs[-1])
        bridge.verify_attempt(bridge.attempt_refs[-1])
    return {**record, "archive_ref": bridge.archive.save(record)}
