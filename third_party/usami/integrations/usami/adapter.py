"""Real four-domain Sessions behind an independently versioned algorithm boundary."""
from __future__ import annotations

import hashlib
import copy
import json
import re
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

from pydantic import JsonValue

from vast_coevo.models import (ArtifactRef, Budget, Consumption, EpisodeConfig, EpisodeEvidence,
                               InjectionRecord, ModelCall, ModelSpec, PublicContext, ReplayResult,
                               VerifiedEvent, VerifiedFailure)
from vast_coevo.schema import FrozenModel, atomic_json, digest
from vast_coevo.evidence.signatures import failure_fingerprint
from vast_coevo.skills.library import LibrarySnapshot
from vast_coevo.skills.retriever import SkillContext

from .domains import (ServiceTaskProfile, UsamiProfile, customer_records_of, execution_of,
                      failure_inputs, run_native_failure, replay_native_failure,
                      initial_world, interaction_session, model_settings, native_interaction_config,
                      native_service_config, native_task, native_trace, public_observation,
                      public_policy, replay_native, run_native_service, task_score, tool_names)
from .evidence import failure_details, operation_rows, reflection_view

ROOT = Path(__file__).resolve().parents[2]


def source_hashes():
    """Outer source pins include new runtime code without rewriting old manifests."""
    packages = ("usami", "usami_cloudops", "usami_fulfillment", "usami_stayops", "usami_harness",
                "vast_coevo", "integrations")
    paths = sorted(path for package in packages for path in (ROOT / package).rglob("*.py"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


class NativeArtifact(FrozenModel):
    schema_version: Literal[1] = 1
    episode_id: str
    profile: UsamiProfile
    task_profile: Optional[ServiceTaskProfile] = None
    config: EpisodeConfig
    track: Literal["discovery", "service_task", "failure_regression"]
    origin: Optional[ArtifactRef] = None
    origin_evidence: Optional[EpisodeEvidence] = None
    native_config: Dict[str, JsonValue]
    native_trace: Dict[str, JsonValue]
    # Some native observations deliberately enumerate insertion-ordered world
    # mappings. Keep their exact serialization even in a canonical outer file.
    native_trace_json: str
    customer_snapshot: LibrarySnapshot
    service_snapshot: LibrarySnapshot
    model_calls: Tuple[ModelCall, ...]
    injections: Tuple[InjectionRecord, ...]
    source_hashes: Dict[str, str]


def _request_context(domain, tools, request):
    """Retrieve exclusively from already model-visible, permission-filtered text."""
    messages = request.get("messages", [])
    visible = [message.get("content", "") for message in messages
               if message.get("role") in {"user", "tool"} and isinstance(message.get("content"), str)]
    last_user = next((message.get("content", "") for message in reversed(messages)
                      if message.get("role") == "user"), "")
    intents = tuple(name for name in tools if re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", last_user))
    return SkillContext(domain=domain, public_text="\n".join(visible), intents=intents,
                        available_tools=tuple(tools))


def _native_hash(trace, track):
    return trace.interaction_hash if track == "discovery" else trace.execution_hash


def _recorded_wire_client(settings, records, calls):
    """Original response replay, including native transport exception semantics."""
    from usami.service.client import RecordedCompletionClient

    class WireJournal(RecordedCompletionClient):
        def complete(self, request):
            if self.index >= len(self.records):
                raise ValueError("missing recorded wire response")
            original = self.records[self.index]
            call = calls[self.index]
            if request != original.request:
                raise ValueError("actual augmented model request differs from the frozen wire journal")
            if original.error.get("stage") == "transport" and call.error:
                error = call.error
                if error.get("type") == "KeyboardInterrupt":
                    self.index += 1
                    raise KeyboardInterrupt()
                if error.get("type") != "ServiceError":
                    # Recreate only an inert exception's name/data; no imported
                    # provider code, arbitrary code or network call is executed.
                    self.index += 1
                    exception_type = type(str(error.get("type", "Exception")), (Exception,), {})
                    exception = exception_type(str(error.get("message", "")))
                    exception.code = error.get("code", "client_error")
                    exception.response = copy.deepcopy(call.response)
                    raise exception
            return super().complete(request)

    return WireJournal(settings, records)


class UsamiAdapter:
    """A profile registry and transport factory around unchanged native Sessions.

    ``client_factory(spec, role, config, episode_id)`` must return an ordinary
    Usami CompletionClient with settings exactly equal to the frozen ModelSpec.
    Its default uses the original Usami environment/Keychain transport. Tests
    may supply deterministic responses; the business Engine and Oracle always
    remain the original implementations.
    """

    def __init__(self, root, profiles, service_tasks=(), client_factory=None):
        self.root = Path(root).resolve()
        profiles = [UsamiProfile.model_validate(profile.model_dump(mode="json")) for profile in profiles]
        tasks = [ServiceTaskProfile.model_validate(task.model_dump(mode="json")) for task in service_tasks]
        if len({profile.config_id for profile in profiles}) != len(profiles):
            raise ValueError("duplicate Usami profile IDs")
        if len({task.task_ref for task in tasks}) != len(tasks):
            raise ValueError("duplicate Service task IDs")
        self._profiles = {profile.config_id: profile for profile in profiles}
        self._tasks = {task.task_ref: task for task in tasks}
        if any(task.config_id not in self._profiles for task in tasks):
            raise ValueError("Service task references a missing profile")
        self.client_factory = client_factory or self._default_client
        self._examiner_history = {}

    def examiner_enabled(self, config):
        return self._profiles[config.config_id].examiner_protocol is not None

    def _history_item(self, evidence):
        evidence = EpisodeEvidence.model_validate(evidence.model_dump(mode="json"))
        if evidence.config.partition != "development" or not evidence.replay.verified:
            raise ValueError("examiner may learn only from verified development evidence")
        artifact = NativeArtifact.model_validate_json(Path(evidence.trace.path).read_text())
        if artifact.profile.split == "heldout" or not self.replay(evidence, artifact.customer_snapshot, artifact.service_snapshot).verified:
            raise ValueError("examiner history failed independent source replay")
        trace = native_trace(evidence.config.domain,json.loads(artifact.native_trace_json),artifact.track)
        execution = execution_of(evidence.config.domain,trace,artifact.track)
        return {"evidence":evidence.model_dump(mode="json"), "service_execution":execution.model_dump(mode="json"),
                "events":[row.model_dump(mode="json") for row in evidence.events]}

    def prepare_examiner_generation(self, journal, run, generation, customer, service, previous, prefix):
        domains = {c.config.domain for c in run.exploration_cases if self.examiner_enabled(c.config)}
        if not domains:
            return ()
        if run.schema_version not in (3, 4) or any(self.examiner_enabled(c.config) for c in run.validation_cases):
            raise ValueError("examiner needs recursive training and frozen, non-authoring Service validation worlds")
        seeds = ()
        if not previous:
            cases = tuple(c for c in run.service_task_cases if c.config.domain in domains)
            if domains - {c.config.domain for c in cases}:
                raise ValueError("each examiner domain needs an initial development Service task")
            seeds = tuple(journal.run_service_task(c.config,c.task_ref,customer,service,prefix+".seed-task."+c.id) for c in cases)
        source = seeds or previous
        self._examiner_history = {domain:[self._history_item(row) for row in source if row.config.domain==domain][-8:] for domain in domains}
        if any(not self._examiner_history[d] for d in domains):
            raise ValueError("missing verified previous task trajectory for an examiner domain")
        return seeds

    def assert_frozen_service_validation(self, cases):
        if any(self.examiner_enabled(case.config) for case in cases):
            raise ValueError("Service validation must use frozen non-authoring profiles")

    def _check_examiner_history(self, history, domain=None):
        rebuilt=[]
        for item in history:
            if domain is not None and item["evidence"]["config"]["domain"] != domain:
                raise ValueError("examiner history belongs to another domain")
            rebuilt.append(self._history_item(EpisodeEvidence.model_validate(item["evidence"])))
        if rebuilt != history:
            raise ValueError("examiner history differs from verified source trajectories")

    @staticmethod
    def _default_client(spec, role, config, episode_id):
        from usami.service.client import OpenAICompatibleClient, key_from_env
        return OpenAICompatibleClient(model_settings(spec), key_from_env())

    def config_for(self, config_id, customer_model: ModelSpec, service_model: ModelSpec,
                   budget: Optional[Budget] = None, *, task_ref=None, tools=None, identifier=None):
        profile = self._profiles[config_id]
        task = None
        if task_ref is None:
            world, identity = initial_world(profile), profile.identity
        else:
            task_profile = self._tasks[task_ref]
            world, task, _ = native_task(profile, task_profile)
            identity = task.identity
        config = EpisodeConfig(id=identifier or (config_id + ("." + task_ref if task_ref else "")),
                               config_id=config_id, domain=profile.domain,
                               partition="heldout" if profile.split == "heldout" else "development",
                               initial_world_hash=world.state_hash, identity=identity, mode=profile.mode,
                               customer_model=customer_model, service_model=service_model,
                               budget=budget or Budget(), tools=tuple(tool_names(profile.domain, world) if tools is None else tools))
        self._resolve(config, task_ref)
        return config

    def _resolve(self, config, task_ref=None, *, public_only=False):
        config = EpisodeConfig.model_validate(config.model_dump(mode="json"))
        if config.environment != "usami" or config.config_id not in self._profiles:
            raise ValueError("foreign or unregistered environment configuration")
        profile = self._profiles[config.config_id]
        task_profile = task = generator = None
        world, identity = initial_world(profile), profile.identity
        if task_ref is not None:
            task_profile = self._tasks.get(task_ref)
            if task_profile is None or task_profile.config_id != config.config_id:
                raise ValueError("Service task does not belong to this frozen profile")
            world, task, generator = native_task(profile, task_profile)
            identity = task.identity
        elif public_only and (world.state_hash != config.initial_world_hash or identity != config.identity):
            # Public summaries may be requested for a task config. Resolve its
            # frozen world without making its certificate visible to an Agent.
            matches = []
            for candidate in self._tasks.values():
                if candidate.config_id == config.config_id:
                    candidate_world, candidate_task, candidate_generator = native_task(profile, candidate)
                    if candidate_world.state_hash == config.initial_world_hash and candidate_task.identity == config.identity:
                        matches.append((candidate, candidate_world, candidate_task, candidate_generator))
            if not matches:
                raise ValueError("configuration has no registered frozen world")
            task_profile, world, task, generator = matches[0]
            identity = task.identity
        partition = "heldout" if profile.split == "heldout" else "development"
        if (config.domain != profile.domain or config.partition != partition or config.mode != profile.mode
                or config.initial_world_hash != world.state_hash or config.identity != identity):
            raise ValueError("world, identity, domain, partition or mode differs from frozen profile")
        if set(config.tools) - set(tool_names(profile.domain, world)):
            raise ValueError("configured tools are outside the native interface")
        if identity not in world.users:
            raise ValueError("identity is absent from the frozen world")
        model_settings(config.customer_model)
        model_settings(config.service_model)
        return profile, world, task_profile, task, generator

    def public_capabilities(self, config):
        self._resolve(config, public_only=True)
        return tuple(config.tools)

    def initial_observation(self, config):
        profile, world, _, _, _ = self._resolve(config, public_only=True)
        observed_profile = UsamiProfile.model_validate({**profile.model_dump(mode="json"), "identity": config.identity})
        return PublicContext(domain=config.domain, public_text=json.dumps(public_observation(observed_profile, world),
                                                                         sort_keys=True, ensure_ascii=False),
                             available_tools=config.tools)

    def reflection_view(self, evidence):
        """Post-run only; callers supply independently replay-verified evidence.

        Historical experiments must be replayed under their archived sources,
        rather than pretending their source pins match the current adapter.
        """
        if not evidence.replay.verified or evidence.config.partition != "development":
            raise ValueError("reflection projection requires replay-verified development evidence")
        artifact = NativeArtifact.model_validate_json(Path(evidence.trace.path).read_text())
        if (digest(artifact) != evidence.trace.sha256 or artifact.episode_id != evidence.episode_id or
                artifact.config != evidence.config or artifact.model_calls != evidence.model_calls or
                artifact.customer_snapshot.snapshot_hash != evidence.customer_snapshot_hash or
                artifact.service_snapshot.snapshot_hash != evidence.service_snapshot_hash):
            raise ValueError("reflection artifact differs from the verified evidence")
        trace = native_trace(artifact.profile.domain, json.loads(artifact.native_trace_json), artifact.track)
        if _native_hash(trace, artifact.track) != evidence.native_trace_hash:
            raise ValueError("reflection native trace hash mismatch")
        return reflection_view(evidence, trace, artifact.profile.domain, artifact.track)

    def public_policy(self, config):
        _, world, _, _, _ = self._resolve(config, public_only=True)
        return public_policy(config.domain, world)

    def customer_probe_source_hashes(self):
        return source_hashes()

    def customer_probe_protocol(self, config):
        profile, _, _, _, _ = self._resolve(config, public_only=True)
        return 'cloudops-probes-v2' if profile.read_probe_protocol else 'cloudops-probes-v1'

    def customer_business_shape(self, config):
        if config.domain != "cloudops":
            raise ValueError("QGD v1 currently supports CloudOps Discovery only")
        from .customer_probes import business_shape
        _, world, _, _, _ = self._resolve(config, public_only=True)
        return business_shape(world, config.identity)

    def customer_probes(self, evidence):
        """Post-run only, with full native, injection and source-pin replay."""
        if evidence.config.domain != "cloudops" or evidence.track != "discovery":
            raise ValueError("QGD v1 currently supports CloudOps Discovery only")
        artifact = NativeArtifact.model_validate_json(Path(evidence.trace.path).read_text())
        replay = self.replay(evidence, artifact.customer_snapshot, artifact.service_snapshot)
        if not replay.verified:
            raise ValueError("Customer probe evidence failed exact replay: " + replay.reason)
        from .customer_probes import evaluate_cloudops
        trace = native_trace("cloudops", json.loads(artifact.native_trace_json), "discovery")
        return evaluate_cloudops(trace, evidence)

    def service_scope_risks(self, snapshot):
        from .service_audit import scope_audit
        return scope_audit(snapshot)

    def service_decision_audit(self, evidence):
        if evidence.config.domain != 'cloudops': raise ValueError('write audit supports CloudOps only')
        artifact = NativeArtifact.model_validate_json(Path(evidence.trace.path).read_text())
        replay = self.replay(evidence, artifact.customer_snapshot, artifact.service_snapshot)
        if not replay.verified: raise ValueError('Service audit failed exact replay: ' + replay.reason)
        from .service_audit import audit_writes
        trace = native_trace('cloudops', json.loads(artifact.native_trace_json), artifact.track)
        return audit_writes(evidence, execution_of('cloudops', trace, artifact.track), artifact.service_snapshot)

    def forbidden_identifiers(self, config):
        _, world, _, _, _ = self._resolve(config, public_only=True)
        identifiers = set()
        for value in world.model_dump(mode="json").values():
            if isinstance(value, dict):
                for key, entity in value.items():
                    if isinstance(entity, dict) and "id" in entity:
                        identifiers.add(key)
                        identifiers.add(entity["id"])
                        # Preserve ordinary business concepts such as project
                        # names "production" and "development" as usable
                        # generic terms. Generated identifying display names
                        # contain a number and are checked alongside exact IDs.
                        if isinstance(entity.get("name"), str) and any(character.isdigit() for character in entity["name"]):
                            identifiers.add(entity["name"])
        return tuple(sorted(identifiers))

    def _runtime(self, inner, role, library, config, episode_id, *, journal=False):
        from vast_coevo.agents.runtime import RuntimeClient
        spec = config.customer_model if role == "customer" else config.service_model
        if inner.settings != model_settings(spec):
            raise ValueError("completion client differs from the frozen model configuration")
        def observe(stage, call):
            folder = self.root / "episodes" / digest({"episode_id": episode_id})[:32] / "calls"
            payload = {"schema_version": 1, "episode_id": episode_id, "stage": stage,
                       "config_hash": config.config_hash, "snapshot_hash": library.snapshot_hash,
                       "call": call.model_dump(mode="json"),
                       "injection": client.injections[-1].model_dump(mode="json")}
            atomic_json(folder / (digest(call.id) + "." + stage + ".json"), payload, immutable=True)

        client = RuntimeClient(inner, role, library, config.domain, tools=config.tools, model=spec,
                             episode_id=episode_id,
                             observer=observe if journal else None,
                             context_factory=lambda request: _request_context(config.domain, config.tools, request))
        return client

    def run_episode(self, config, customer, service, episode_id):
        profile, _, _, _, _ = self._resolve(config)
        native_config = native_interaction_config(profile, config)
        if profile.examiner_protocol is not None:
            native_config.history = copy.deepcopy(self._examiner_history.get(profile.domain, []))
        return self._run(config, profile, native_config, customer, service, episode_id, "discovery")

    def run_service_task(self, config, task_ref, customer, service, episode_id):
        profile, world, task_profile, task, generator = self._resolve(config, task_ref)
        native_config, message, operations = native_service_config(profile, config, world, task, generator)
        return self._run(config, profile, native_config, customer, service, episode_id, "service_task",
                         task_profile, task, message, operations)

    def _origin(self, source):
        source = EpisodeEvidence.model_validate(source.model_dump(mode="json"))
        if source.config.partition != "development" or source.track != "discovery" or not source.failures:
            raise ValueError("known-failure regression requires verified development Discovery evidence")
        raw = json.loads(Path(source.trace.path).read_text(encoding="utf-8"))
        if digest(raw) != source.trace.sha256:
            raise ValueError("known failure source trace hash differs")
        artifact = NativeArtifact.model_validate(raw)
        verified = self.replay(source, artifact.customer_snapshot, artifact.service_snapshot)
        if not verified.verified:
            raise ValueError("known failure source is not replayable: " + verified.reason)
        return artifact, native_trace(source.config.domain, json.loads(artifact.native_trace_json), "discovery")

    def run_failure_case(self, source, customer, service, episode_id):
        _, original = self._origin(source)
        profile, _, _, _, _ = self._resolve(source.config)
        native_config = execution_of(source.config.domain, original).experiment.model_copy(deep=True)
        inputs = failure_inputs(source.config.domain, original)
        return self._run(source.config, profile, native_config, customer, service, episode_id, "failure_regression",
                         operations=inputs, origin=source)

    def _run(self, config, profile, native_config, customer, service, episode_id, track,
             task_profile=None, task=None, message=None, operations=None, origin=None):
        customer = LibrarySnapshot.model_validate(customer.model_dump(mode="json"))
        service = LibrarySnapshot.model_validate(service.model_dump(mode="json"))
        if customer.role != "customer" or service.role != "service":
            raise ValueError("customer and service snapshots are reversed")
        path = self.root / "episodes" / digest({"episode_id": episode_id})[:32] / "native.json"
        if path.exists():
            raise ValueError("episode ID is already published; use Replay or an explicit new ID")
        start = path.with_name("online-start.json")
        if start.exists():
            raise ValueError("episode has an in-flight marker with unknown outcome; no duplicate online execution")
        pins = source_hashes()
        atomic_json(start, {"schema_version": 1, "episode_id": episode_id, "config": config.model_dump(mode="json"),
                           "customer_snapshot_hash": customer.snapshot_hash, "service_snapshot_hash": service.snapshot_hash,
                           "track": track, "source_hashes": pins}, immutable=True)
        service_inner = self.client_factory(config.service_model, "service", config, episode_id)
        customer_inner = None
        try:
            service_client = self._runtime(service_inner, "service", service, config, episode_id, journal=True)
            customer_client = None
            if track == "discovery":
                customer_inner = self.client_factory(config.customer_model, "customer", config, episode_id)
                customer_client = self._runtime(customer_inner, "customer", customer, config, episode_id, journal=True)
                if profile.examiner_protocol is not None:
                    from .examiner import run_exam
                    self._check_examiner_history(native_config.history,profile.domain)
                    trace = run_exam(native_config,customer_client,service_client,
                        freeze_observer=lambda payload: atomic_json(path.with_name("exam-frozen.json"),payload,immutable=True))
                else:
                    trace = interaction_session(profile, native_config, customer_client, service_client).run()
            elif track == "service_task":
                trace = run_native_service(profile, native_config, service_client, message, operations)
            else:
                trace = run_native_failure(profile, native_config, service_client, operations)
        finally:
            for inner in (customer_inner, service_inner):
                if inner is not None and hasattr(inner, "close"):
                    inner.close()
        calls = tuple(customer_client.records if customer_client else ()) + tuple(service_client.records)
        injections = tuple(customer_client.injections if customer_client else ()) + tuple(service_client.injections)
        artifact = NativeArtifact(episode_id=episode_id, profile=profile, task_profile=task_profile, config=config,
                                  track=track, native_config=native_config.model_dump(mode="json"),
                                  origin=origin.trace if origin else None, origin_evidence=origin,
                                  native_trace=trace.model_dump(mode="json"), native_trace_json=trace.model_dump_json(), customer_snapshot=customer,
                                  service_snapshot=service, model_calls=calls, injections=injections, source_hashes=pins)
        atomic_json(path, artifact.model_dump(mode="json"), immutable=True)
        ref = ArtifactRef(path=str(path), sha256=digest(artifact))
        evidence = self._evidence(artifact, trace, ref, task)
        replay = self.replay(evidence, customer, service)
        payload = evidence.model_dump(mode="json", exclude={"evidence_hash"})
        payload["replay"] = replay.model_dump(mode="json")
        evidence = EpisodeEvidence.model_validate(payload)
        atomic_json(path.with_name("evidence.json"), evidence.model_dump(mode="json"), immutable=True)
        if profile.domain == 'cloudops' and profile.read_probe_protocol is not None:
            from .service_audit import audit_writes
            atomic_json(path.with_name('service-decision-audit.json'),
                audit_writes(evidence, execution_of('cloudops', trace, track), service), immutable=True)
        return evidence

    def _evidence(self, artifact, trace, ref, task=None):
        domain, track = artifact.config.domain, artifact.track
        execution = execution_of(domain, trace, track)
        rows = operation_rows(domain, trace, track)
        events = tuple(VerifiedEvent(event_id=row["event_id"], operation=row["operation"],
                                     classification=row["classification"],
                                     reference_codes=tuple(sorted(set(row["violations"] + ([row["reference_error_code"]] if row["reference_error_code"] else [])))),
                                     successful_execution=row["success"], authorized=row["authorized"],
                                     operation_correct=row["operation_correct"], before_hash=row["before_hash"],
                                     after_hash=row["after_hash"]) for row in rows)
        failures = []
        for details in failure_details(domain, trace, track):
            row = details["row"]
            codes = tuple(sorted(set(row["violations"] + ([row["reference_error_code"]] if row["reference_error_code"] else []))))
            fingerprint = failure_fingerprint(domain, row["operation"], codes, details["relevant_prestate"],
                                              details["permission_scope"], details["causal_prerequisites"],
                                              details["expected"], details["actual"], details["normalization_scope"])
            failures.append(VerifiedFailure(episode_id=artifact.episode_id, event_id=row["event_id"], domain=domain,
                                            operation=row["operation"], reference_codes=codes,
                                            relevant_prestate=details["relevant_prestate"], permission_scope=details["permission_scope"],
                                            causal_prerequisites=tuple(details["causal_prerequisites"]),
                                            expected=details["expected"], actual=details["actual"],
                                            successful_business_violation=True, trace=ref, fingerprint=fingerprint,
                                            normalization_scope=details["normalization_scope"]))
        customer_calls = len(customer_records_of(domain, trace)) if track == "discovery" else 0
        consumption = Consumption(customer_llm_calls=customer_calls, service_llm_calls=len(execution.llm_records),
                                  tool_calls=len(execution.tool_records))
        if track == "discovery":
            termination = getattr(trace, "termination_reason", None) or trace.status
        else:
            termination = execution.status
        metrics = task_score(domain, task, execution, artifact.config.budget) if task is not None else {}
        if getattr(trace,"version",None)=="usami-examiner-v1":
            from .examiner import score
            metrics = score(trace)
        if artifact.profile.fixed_customer_brief is not None:
            from .designer.dialogue import score_execution
            metrics = score_execution(execution, artifact.profile.fixed_task_spec)
        if domain=='cloudops' and ((artifact.profile.fixed_task_spec or {}).get('task_protocol')=='frozen-task-discovery-v1' or
                (artifact.task_profile is not None and (artifact.task_profile.frozen_task or {}).get('family','').startswith('frozen_'))):
            from .frozen.task_oracle import decisions
            metrics['task_decisions']=decisions(execution,artifact.model_calls)
        if artifact.profile.examiner_protocol is not None:
            from .examiner import binding
            initial=execution.world_trace.initial_state
            initial=binding(domain).World.model_validate(initial) if isinstance(initial,dict) else initial
            identity=execution.experiment.customer.identity if domain=="saas" else execution.experiment.identity
            metrics.update(execution_initial_world_hash=initial.state_hash,execution_identity=identity)
        return EpisodeEvidence(episode_id=artifact.episode_id, config=artifact.config, track=track,
                               task_ref=artifact.task_profile.task_ref if artifact.task_profile else None,
                               customer_snapshot_hash=artifact.customer_snapshot.snapshot_hash,
                               service_snapshot_hash=artifact.service_snapshot.snapshot_hash,
                               model_calls=artifact.model_calls, injections=artifact.injections, events=events,
                               failures=tuple(failures), consumption=consumption, termination=termination,
                               trace=ref, native_trace_hash=_native_hash(trace, track),
                               replay=ReplayResult(verified=False, reason="pending native and augmented request replay"), **metrics)

    def replay(self, evidence, customer, service):
        """Offline replay checks both original Session bodies and transformed wire bodies."""
        try:
            evidence = EpisodeEvidence.model_validate(evidence.model_dump(mode="json"))
            customer = LibrarySnapshot.model_validate(customer.model_dump(mode="json"))
            service = LibrarySnapshot.model_validate(service.model_dump(mode="json"))
            raw = json.loads(Path(evidence.trace.path).read_text(encoding="utf-8"))
            if digest(raw) != evidence.trace.sha256:
                raise ValueError("native artifact differs from its external content hash")
            artifact = NativeArtifact.model_validate(raw)
            if artifact.source_hashes != source_hashes():
                raise ValueError("source pin differs; replay requires the original algorithm and native runtime")
            if (artifact.episode_id != evidence.episode_id or artifact.config != evidence.config
                    or artifact.track != evidence.track or artifact.model_calls != evidence.model_calls
                    or artifact.injections != evidence.injections
                    or artifact.customer_snapshot != customer or artifact.service_snapshot != service
                    or customer.snapshot_hash != evidence.customer_snapshot_hash
                    or service.snapshot_hash != evidence.service_snapshot_hash):
                raise ValueError("episode, frozen models, skills or augmented request journal differs")
            profile, world, task_profile, task, generator = self._resolve(evidence.config, evidence.task_ref)
            if profile != artifact.profile or task_profile != artifact.task_profile:
                raise ValueError("registered native profile or task differs")
            original_payload = json.loads(artifact.native_trace_json)
            if original_payload != artifact.native_trace:
                raise ValueError("native ordered serialization differs from its inspected payload")
            trace = native_trace(evidence.config.domain, original_payload, artifact.track)
            if _native_hash(trace, artifact.track) != evidence.native_trace_hash:
                raise ValueError("native trace differs from the outer evidence hash")
            execution = execution_of(evidence.config.domain, trace, artifact.track)
            from usami.service.models import LLMRecord
            clients = {}
            for role, snapshot, original_records, settings in (
                ("customer", customer, customer_records_of(evidence.config.domain, trace) if artifact.track == "discovery" else [], model_settings(evidence.config.customer_model)),
                ("service", service, execution.llm_records, model_settings(evidence.config.service_model))):
                calls = [call for call in artifact.model_calls if call.role == role]
                if len(calls) != len(original_records):
                    raise ValueError("native and wire model call counts differ")
                records = []
                for original, call in zip(original_records, calls):
                    if original.request != call.base_request or original.response != call.response:
                        raise ValueError("base request or response differs from the original Session journal")
                    records.append(LLMRecord(request=call.request, response=original.response, error=original.error))
                inner = _recorded_wire_client(settings, records, calls)
                clients[role] = (inner, self._runtime(inner, role, snapshot, evidence.config, evidence.episode_id))
            if artifact.track == "discovery":
                native_config = native_interaction_config(profile, evidence.config)
                if profile.examiner_protocol is not None:
                    from .examiner import run_exam
                    self._check_examiner_history(trace.config.history,profile.domain)
                    frozen=json.loads(Path(evidence.trace.path).with_name("exam-frozen.json").read_text())
                    if digest(frozen)!=trace.construction_hash:
                        raise ValueError("pre-execution frozen exam differs from construction record")
                    native_config.history = copy.deepcopy(trace.config.history)
                    rebuilt = run_exam(native_config,clients["customer"][1],clients["service"][1])
                else:
                    rebuilt = interaction_session(profile, native_config, clients["customer"][1], clients["service"][1]).run()
            elif artifact.track == "service_task":
                native_config, message, operations = native_service_config(profile, evidence.config, world, task, generator)
                rebuilt = run_native_service(profile, native_config, clients["service"][1], message, operations)
            else:
                if artifact.origin_evidence is None or artifact.origin != artifact.origin_evidence.trace:
                    raise ValueError("known failure regression has no hash-bound original evidence")
                if artifact.origin_evidence.config != evidence.config:
                    raise ValueError("known failure regression changed original world, identity, models or budget")
                _, original = self._origin(artifact.origin_evidence)
                native_config = execution_of(evidence.config.domain, original).experiment.model_copy(deep=True)
                rebuilt = run_native_failure(profile, native_config, clients["service"][1], failure_inputs(evidence.config.domain, original))
            if native_config.model_dump(mode="json") != artifact.native_config or rebuilt != trace:
                raise ValueError("original Session, scoped observations, grants, state transitions or Oracle differs")
            rebuilt_calls = tuple(clients["customer"][1].records) + tuple(clients["service"][1].records)
            rebuilt_injections = tuple(clients["customer"][1].injections) + tuple(clients["service"][1].injections)
            if rebuilt_calls != artifact.model_calls or rebuilt_injections != artifact.injections:
                raise ValueError("actual augmented wire request or exact skill selection differs")
            if any(inner.index != len(inner.records) for inner, _ in clients.values()):
                raise ValueError("recorded responses were not consumed exactly once")
            interruption_bridge = evidence.config.domain in {"fulfillment", "stayops"} and any(
                call.role == "customer" and call.error and call.error.get("type") == "KeyboardInterrupt"
                for call in artifact.model_calls)
            if artifact.track == "failure_regression" or interruption_bridge or (
                    artifact.track == "service_task" and evidence.config.domain == "saas"):
                replay_native_failure(evidence.config.domain, execution)
            else:
                replay_native(evidence.config.domain, trace, artifact.track)
            expected = self._evidence(artifact, trace, evidence.trace, task)
            # Replay status is the only field that differs between provisional
            # execution evidence and its verified published form.
            compare = {"replay", "evidence_hash"}
            if expected.model_dump(mode="json", exclude=compare) != evidence.model_dump(mode="json", exclude=compare):
                raise ValueError("generic event, failure, task score or budget evidence differs from native evidence")
            reason = "Original CustomerSession and native Service/World replay verified using the recorded interruption." if interruption_bridge else ""
            return ReplayResult(verified=True, reason=reason, request_count=len(artifact.model_calls), native_trace_hash=evidence.native_trace_hash)
        except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
            return ReplayResult(verified=False, reason=str(exc))
