"""Recorded completion fixtures exercise the actual four native business worlds."""
import copy
import json

import pytest

from integrations.usami import ServiceTaskProfile, UsamiAdapter, UsamiProfile
from integrations.usami.domains import model_settings
from vast_coevo.models import Budget, EpisodeConfig, EpisodeEvidence, ModelSpec
from vast_coevo.schema import atomic_json, digest
from vast_coevo.skills.library import LibrarySnapshot, empty_snapshot
from vast_coevo.skills.models import DecisionSkill, ExplorationSkill, Scope, Trigger


SPEC = ModelSpec(model="recorded-integration-fixture", base_url="https://fixture.invalid/v1")
LIMITS = Budget(max_customer_turns=5, max_customer_llm_calls=6,
                max_service_llm_calls=14, max_service_tool_calls=12, max_format_recoveries=1)


class ScriptedClient:
    def __init__(self, spec, responses):
        self.settings = model_settings(spec)
        self.responses = iter(response if isinstance(response, BaseException) else copy.deepcopy(response) for response in responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(copy.deepcopy(request))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def answer(content=None, calls=()):
    return {"choices": [{"finish_reason": "tool_calls" if calls else "stop",
                          "message": {"role": "assistant", "content": content, "tool_calls": list(calls)}}]}


def call(identifier, name, args):
    return {"id": identifier, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def customer_action(domain, identifier, message, operation=None):
    if domain in {"saas", "cloudops"}:
        return answer(json.dumps({"type": "send_message", "message": message,
                                  "requested_operations": [operation] if operation else []}))
    args = {key: value for key, value in operation.items() if key != "name"} if operation else {}
    args["message"] = message
    return answer(calls=[call(identifier, "intend_" + operation["name"] if operation else "ask_service", args)])


def stopped(domain):
    if domain in {"saas", "cloudops"}:
        return answer('{"type":"stop","reason":"Finished my chosen exploration."}')
    return answer(calls=[call("customer_stop", "stop_exploration", {"reason": "Finished my chosen exploration."})])


def service_arguments(domain, operation):
    args = {key: value for key, value in operation.items() if key not in {"name", "params"}}
    # SaaS exposes flattened tool parameters, unlike the other three Sessions.
    if domain == "saas":
        args.update(operation["params"])
    else:
        args["params"] = operation["params"]
    return args


def operations(domain):
    if domain == "saas":
        return [dict(name="change_plan", org_id="org_001", resource_id="sub_001", params={"plan_id": "team"}),
                dict(name="change_seats", org_id="org_001", resource_id="sub_001", params={"seats": 1}),
                dict(name="cancel_renewal", org_id="org_001", resource_id="sub_001", params={})]
    if domain == "cloudops":
        return [dict(name="start_instance", project_id="p1", resource_id="p1_db", params={}),
                dict(name="start_instance", project_id="p0", resource_id="p0_db", params={}),
                dict(name="stop_instance", project_id="p1", resource_id="p1_db", params={})]
    if domain == "fulfillment":
        return [dict(name="capture_payment", scope_id="w0", resource_id="order0",
                     params={"amount_cents": 5300, "key": "fixture_charge", "token": "card_approved"}),
                dict(name="reserve_inventory", scope_id="w0", resource_id="order0",
                     params={"reservation_id": "fixture_reservation", "line_id": "line0", "lot_id": "lot0", "quantity": 4}),
                dict(name="confirm_order", scope_id="w0", resource_id="order0", params={})]
    return [dict(name="create_hold", scope_id="p0", resource_id="p0",
                 params={"reservation_id": "b0", "guest_id": "g0", "room_type_id": "p0_standard",
                         "rate_plan_id": "p0_flex", "arrival": "2026-01-15", "departure": "2026-01-17",
                         "occupants": 2, "hold_minutes": 30}),
            dict(name="confirm_reservation", scope_id="p0", resource_id="b0", params={}),
            dict(name="capture_payment", scope_id="p0", resource_id="b0",
                 params={"amount_cents": 2400, "key": "fixture_deposit", "token": "card_approved"})]


def scripts(domain):
    ops = operations(domain)
    customer = [customer_action(domain, "intent_legal", "Explore a chosen subscription project order reservation request.", ops[0]),
                customer_action(domain, "intent_invalid", "Explore a chosen subscription project order reservation boundary.", ops[1]),
                customer_action(domain, "question", "Inspect the subscription project order reservation; this question grants no writes."),
                stopped(domain)]
    service = []
    for index, op in enumerate(ops):
        service.extend([answer(calls=[call("operation_" + str(index), op["name"], service_arguments(domain, op))]),
                        answer("Actual result for " + op["resource_id"] + " received.")])
    return {"customer": customer, "service": service}


def libraries(domain):
    scope = Scope(domains=(domain,))
    trigger = Trigger(keywords=("initial_observation", "permission-filtered", "subscription", "project", "order", "reservation"))
    customer = ExplorationSkill(id="customer.inspect_boundaries", scope=scope, trigger=trigger,
                                hypotheses=("State prerequisites may be relevant to the next request.",),
                                exploration=("Inspect permitted observations and choose an independent request.",),
                                feedback=("Use the actual Service reply when choosing the next request.",),
                                stop=("Stop when the exploration budget is exhausted.",))
    service = DecisionSkill(id="service.check_prerequisites", scope=scope, trigger=trigger,
                            preconditions=("Require an exact customer write grant and applicable resource permission.",),
                            procedure=("Query relevant state before choosing an existing tool.",),
                            postconditions=("Report actual tool results.",),
                            recovery=("Report a failed tool without claiming success.",))
    return LibrarySnapshot(role="customer", skills=(customer,)), LibrarySnapshot(role="service", skills=(service,))


def adapter_for(tmp_path, domain, mode="reference", responses=None, tasks=()):
    identity = "user_001" if domain == "saas" else "u2" if domain == "cloudops" else "u0"
    profile = UsamiProfile(config_id=domain + "_dev", domain=domain, identity=identity, mode=mode)
    replies = responses or scripts(domain)
    used_clients = []

    def factory(spec, role, config, episode_id):
        client = ScriptedClient(spec, replies[role])
        used_clients.append((role, client))
        return client

    adapter = UsamiAdapter(tmp_path, [profile], tasks, factory)
    return adapter, adapter.config_for(profile.config_id, SPEC, SPEC, LIMITS), used_clients


@pytest.mark.parametrize("domain", ["saas", "cloudops", "fulfillment", "stayops"])
def test_four_domain_real_sessions_skills_authority_oracle_and_offline_replay(tmp_path, domain):
    adapter, config, clients = adapter_for(tmp_path, domain)
    customer, service = libraries(domain)
    evidence = adapter.run_episode(config, customer, service, "episode." + domain)
    assert evidence.replay.verified, evidence.replay.reason
    assert evidence.consumption.customer_llm_calls == 4
    assert evidence.consumption.service_llm_calls == 6
    assert evidence.consumption.tool_calls == 3
    classes = [event.classification for event in evidence.events]
    assert classes == ["legal_execution", "rejected_invalid_attempt", "authorization_only"]
    assert not evidence.failures
    assert evidence.customer_snapshot_hash == customer.snapshot_hash
    assert evidence.service_snapshot_hash == service.snapshot_hash
    assert any(record.role == "customer" and record.hits for record in evidence.injections)
    assert any(record.role == "service" and record.hits for record in evidence.injections)
    for model_call in evidence.model_calls:
        if any(record.call_id == model_call.id and record.hits for record in evidence.injections):
            assert "<VAST_SKILLS>" in json.dumps(model_call.request)
            assert "<VAST_SKILLS>" not in json.dumps(model_call.base_request)
            assert model_call.request_hash != digest(model_call.base_request)
        assert model_call.request.get("tools") == model_call.base_request.get("tools")
    clients_before = len(clients)
    assert adapter.replay(evidence, customer, service).verified
    assert len(clients) == clients_before  # Recorded replay never contacts a model.
    native = json.loads(open(evidence.trace.path, encoding="utf-8").read())
    for model_call in evidence.model_calls:
        assert model_call.model == SPEC
    assert native["source_hashes"] and native["native_trace"]
    assert "operation_correct" not in adapter.initial_observation(config).public_text


@pytest.mark.parametrize("domain", ["saas", "cloudops", "fulfillment", "stayops"])
def test_four_domain_sandbox_reports_actual_business_failure_separate_from_authorization(tmp_path, domain):
    adapter, config, _ = adapter_for(tmp_path, domain, mode="sandbox")
    customer, service = libraries(domain)
    evidence = adapter.run_episode(config, customer, service, "sandbox." + domain)
    assert evidence.replay.verified, evidence.replay.reason
    # These fixtures independently request a policy-invalid write. The unchanged
    # Sandbox's deliberate defects, rather than a fake Oracle result, determine
    # which domain actually succeeds in violating its Reference specification.
    for failure in evidence.failures:
        assert failure.successful_business_violation and not failure.authorization_only
        assert failure.reference_codes
        assert next(event for event in evidence.events if event.event_id == failure.event_id).successful_execution
    assert "authorization_only" in [event.classification for event in evidence.events] or "unscored" in [event.classification for event in evidence.events]
    if domain in {"saas", "cloudops", "stayops"}:
        assert evidence.failures


def test_replay_rejects_skill_and_actual_wire_request_tampering(tmp_path):
    adapter, config, _ = adapter_for(tmp_path, "cloudops")
    customer, service = libraries("cloudops")
    evidence = adapter.run_episode(config, customer, service, "tamper")
    changed_skill = customer.skills[0].model_dump(mode="json", exclude={"content_hash"})
    changed_skill["exploration"] = ["Choose another publicly permitted request."]
    changed = LibrarySnapshot(role="customer", skills=(ExplorationSkill.model_validate(changed_skill),))
    assert not adapter.replay(evidence, changed, service).verified
    raw = json.loads(open(evidence.trace.path, encoding="utf-8").read())
    raw["model_calls"][0]["request"]["messages"][0]["content"] += " Changed native system body."
    raw["model_calls"][0]["request_hash"] = ""
    atomic_json(evidence.trace.path, raw)
    result = adapter.replay(evidence, customer, service)
    assert not result.verified and "content hash" in result.reason


def test_frozen_config_cannot_change_identity_world_tools_or_discovery_task_conditioning(tmp_path):
    adapter, config, _ = adapter_for(tmp_path, "cloudops")
    raw = config.model_dump(mode="json", exclude={"config_hash"})
    raw["identity"] = "u0"
    with pytest.raises(ValueError, match="frozen profile"):
        adapter.run_episode(EpisodeConfig.model_validate(raw), *libraries("cloudops"), "foreign")
    with pytest.raises(ValueError, match="fixed task"):
        UsamiProfile(config_id="leak", domain="saas", identity="user_000", generator={"workflow": "seat_activate"})


@pytest.mark.parametrize("domain,scenario", [("saas", "seat_activate"), ("cloudops", "dependency_boot"),
                                              ("fulfillment", "paid_fulfillment"), ("stayops", "booking_confirmation")])
def test_native_legal_service_goal_regressions_and_exact_replay(tmp_path, domain, scenario):
    from integrations.usami.domains import native_task
    profile = UsamiProfile(config_id=domain + "_task", domain=domain,
                           identity="user_000" if domain == "saas" else "u0")
    task_profile = ServiceTaskProfile(task_ref=domain + ".legal", config_id=profile.config_id, scenario=scenario)
    _, task, _ = native_task(profile, task_profile)
    path = task.witnesses[0].operations
    service_responses = []
    for index, operation in enumerate(path):
        raw = operation.model_dump(mode="json") if hasattr(operation, "model_dump") else operation
        service_responses.append(answer(calls=[call("task_" + str(index), raw["name"], service_arguments(domain, raw))]))
    service_responses.append(answer("Verified the requested business outcome."))
    adapter = UsamiAdapter(tmp_path, [profile], [task_profile],
                           lambda spec, role, config, episode_id: ScriptedClient(spec, service_responses))
    limits = Budget(max_customer_turns=1, max_customer_llm_calls=0, max_service_llm_calls=32, max_service_tool_calls=48)
    config = adapter.config_for(profile.config_id, SPEC, SPEC, limits, task_ref=task_profile.task_ref)
    customer, service = libraries(domain)
    evidence = adapter.run_service_task(config, task_profile.task_ref, customer, service, "task." + domain)
    assert evidence.replay.verified, evidence.replay.reason
    assert evidence.task_success and evidence.legal_task_success
    assert not evidence.correct_refusal
    assert evidence.consumption.customer_llm_calls == 0
    assert evidence.goal_satisfied and evidence.frame_preserved
    assert not evidence.failures
    assert adapter.replay(evidence, customer, service).verified
    # A conditioned task world cannot be repurposed as open Discovery. Native
    # Fulfillment/StayOps first-step goals may share the unconditioned initial
    # world, but no goal or witness information enters their scoped observation.
    assert "witnesses" not in adapter.initial_observation(config).public_text
    assert "goal_satisfied" not in adapter.initial_observation(config).public_text
    if domain in {"saas", "cloudops"}:
        with pytest.raises(ValueError, match="frozen profile"):
            adapter.run_episode(config, customer, service, "leaked.task." + domain)


def test_public_capabilities_resolve_refusal_task_with_same_world_but_independent_actor(tmp_path):
    from integrations.usami.domains import initial_world
    profile = UsamiProfile(config_id="cloud-owner", domain="cloudops", identity="u0", seed=42, mode="sandbox")
    task = ServiceTaskProfile(task_ref="viewer-denied", config_id=profile.config_id, scenario="viewer_refusal")
    adapter = UsamiAdapter(tmp_path, (profile,), (task,))
    config = adapter.config_for(profile.config_id, SPEC, SPEC, LIMITS, task_ref=task.task_ref)
    assert config.initial_world_hash == initial_world(profile).state_hash
    assert config.identity == "u3" and config.identity != profile.identity
    assert adapter.public_capabilities(config) == config.tools
    assert adapter.public_policy(config)
    assert adapter.initial_observation(config).domain == "cloudops"
    assert adapter.forbidden_identifiers(config)
    with pytest.raises(ValueError, match="frozen profile"):
        adapter.run_episode(config, *libraries("cloudops"), "must-not-use-task-identity-for-discovery")


def test_no_match_uses_exact_original_base_request(tmp_path):
    adapter, config, _ = adapter_for(tmp_path, "saas")
    evidence = adapter.run_episode(config, empty_snapshot("customer"), empty_snapshot("service"), "no-skills")
    assert evidence.replay.verified
    assert all(call.request == call.base_request for call in evidence.model_calls)
    assert all(not record.hits and not record.rendered for record in evidence.injections)


@pytest.mark.parametrize("domain", ["saas", "cloudops", "fulfillment", "stayops"])
def test_native_provider_failure_is_archived_and_replayed(tmp_path, domain):
    from usami.service.models import ServiceError
    adapter, config, _ = adapter_for(tmp_path, domain, responses={
        "customer": [ServiceError("PROVIDER_TIMEOUT", "Recorded transport timeout.")], "service": []})
    evidence = adapter.run_episode(config, *libraries(domain), "provider." + domain)
    assert evidence.consumption.customer_llm_calls == 1 and evidence.consumption.service_llm_calls == 0
    assert evidence.replay.verified, evidence.replay.reason
    assert evidence.model_calls[0].error["code"] == "PROVIDER_TIMEOUT"
    assert not evidence.events


@pytest.mark.parametrize("domain", ["saas", "cloudops", "fulfillment", "stayops"])
def test_known_failure_regression_uses_original_customer_grants_and_new_service(tmp_path, domain):
    adapter, config, _ = adapter_for(tmp_path, domain, mode="sandbox")
    customer, service = libraries(domain)
    source = adapter.run_episode(config, customer, service, "known." + domain)
    assert source.failures and source.replay.verified
    baseline = adapter.run_failure_case(source, customer, service, "known.baseline." + domain)
    assert baseline.replay.verified, baseline.replay.reason
    assert baseline.failures and baseline.consumption.customer_llm_calls == 0
    assert {failure.fingerprint for failure in baseline.failures} == {failure.fingerprint for failure in source.failures}
    guard = DecisionSkill(id="service.regression_guard", scope=Scope(domains=(domain,)),
                          trigger=Trigger(keywords=("subscription", "project", "order", "reservation")),
                          preconditions=("Check current business prerequisites and require exact Customer grants.",),
                          procedure=("Refuse a requested write if a business prerequisite is unsatisfied.",),
                          postconditions=("Preserve world state when refusing.",),
                          recovery=("Explain a missing grant without issuing a write.",))
    candidate = LibrarySnapshot(role="service", skills=service.skills + (guard,))
    first = scripts(domain)["service"][:2]
    corrected = first + [answer("The required business prerequisite is unsatisfied."), answer("This question gives no exact write grant.")]
    service_calls = []

    def corrected_factory(spec, role, frozen_config, episode_id):
        assert role == "service"  # A fixed failing testcase makes zero Customer LLM calls.
        client = ScriptedClient(spec, corrected)
        service_calls.append(client)
        return client

    adapter.client_factory = corrected_factory
    selected = adapter.run_failure_case(source, customer, candidate, "known.candidate." + domain)
    assert selected.replay.verified, selected.replay.reason
    assert not selected.failures and all(event.classification == "legal_execution" for event in selected.events)
    assert selected.config == source.config and selected.customer_snapshot_hash == customer.snapshot_hash
    assert selected.service_snapshot_hash == candidate.snapshot_hash
    assert selected.consumption.customer_llm_calls == 0 and len(service_calls) == 1
    assert any(hit.id == guard.id for injection in selected.injections for hit in injection.hits)
    source_artifact = json.loads(open(source.trace.path, encoding="utf-8").read())
    selected_artifact = json.loads(open(selected.trace.path, encoding="utf-8").read())
    assert selected_artifact["origin"] == source.trace.model_dump(mode="json")
    original_trace = json.loads(source_artifact["native_trace_json"])
    corrected_trace = json.loads(selected_artifact["native_trace_json"])
    if domain == "saas":
        original_grants = original_trace["turns"][0]["prepared"]["customer"]["authorizations"]
        corrected_grants = [value["operation"] for value in corrected_trace["trusted_requests"].values() if value["authorized_operations"]]
        assert original_grants[0] in corrected_grants
    else:
        source_inputs = original_trace["service_execution" if domain == "cloudops" else "service"]["inputs"]
        assert corrected_trace["inputs"] == source_inputs


def test_replay_rejects_wire_tampering_even_when_outer_hashes_are_resealed(tmp_path):
    from vast_coevo.models import ArtifactRef, InjectionRecord, ModelCall
    adapter, config, _ = adapter_for(tmp_path, "cloudops")
    customer, service = libraries("cloudops")
    evidence = adapter.run_episode(config, customer, service, "resealed-tamper")
    raw = json.loads(open(evidence.trace.path, encoding="utf-8").read())
    changed = raw["model_calls"][0]
    changed["request"]["messages"][1]["content"] += " Corrupted injected skill text."
    changed["request_hash"] = ""
    changed = ModelCall.model_validate(changed)
    raw["model_calls"][0] = changed.model_dump(mode="json")
    raw["injections"][0]["request_hash"] = changed.request_hash
    atomic_json(evidence.trace.path, raw)
    payload = evidence.model_dump(mode="json", exclude={"evidence_hash"})
    payload["trace"] = ArtifactRef(path=evidence.trace.path, sha256=digest(raw)).model_dump(mode="json")
    payload["model_calls"][0] = raw["model_calls"][0]
    payload["injections"][0] = raw["injections"][0]
    resealed = EpisodeEvidence.model_validate(payload)
    result = adapter.replay(resealed, customer, service)
    assert not result.verified and "original Session" in result.reason


@pytest.mark.parametrize("domain", ["saas", "cloudops", "fulfillment", "stayops"])
def test_raw_invalid_model_output_and_unexpected_transport_error_remain_diagnostic(tmp_path, domain):
    for suffix, raw in [("invalid", "invalid raw wire body"), ("client-error", RuntimeError("Recorded client fault."))]:
        adapter, config, _ = adapter_for(tmp_path / suffix, domain, responses={"customer": [raw], "service": []})
        evidence = adapter.run_episode(config, *libraries(domain), suffix + "." + domain)
        assert evidence.replay.verified, evidence.replay.reason
        assert evidence.consumption.customer_llm_calls == 1 and not evidence.events
        if suffix == "invalid":
            assert evidence.model_calls[0].response == raw
        else:
            assert evidence.model_calls[0].error["type"] == "RuntimeError"


@pytest.mark.parametrize("domain,scenario", [("saas", "owner_refusal"), ("cloudops", "viewer_refusal"),
                                              ("fulfillment", "stock_refusal"), ("stayops", "deposit_refusal")])
def test_native_policy_refusal_regressions_and_no_over_refusal_credit(tmp_path, domain, scenario):
    from integrations.usami.domains import native_task
    profile = UsamiProfile(config_id=domain + "_refusal", domain=domain,
                           identity="user_000" if domain == "saas" else "u0")
    task_profile = ServiceTaskProfile(task_ref=domain + ".refusal", config_id=profile.config_id, scenario=scenario)
    _, task, _ = native_task(profile, task_profile)
    op = task.capabilities[0]
    raw = op.model_dump(mode="json") if hasattr(op, "model_dump") else op
    code = task.policy_certificate.error_code if domain == "saas" else task.refusal_code
    responses = [answer(calls=[call("policy", raw["name"], service_arguments(domain, raw))]), answer(code)]
    adapter = UsamiAdapter(tmp_path, [profile], [task_profile],
                           lambda spec, role, config, episode_id: ScriptedClient(spec, responses))
    config = adapter.config_for(profile.config_id, SPEC, SPEC, LIMITS, task_ref=task_profile.task_ref)
    evidence = adapter.run_service_task(config, task_profile.task_ref, *libraries(domain), "refusal." + domain)
    assert evidence.replay.verified, evidence.replay.reason
    assert evidence.task_success and evidence.correct_refusal and not evidence.legal_task_success
    assert evidence.frame_preserved and not evidence.failures
    assert [event.classification for event in evidence.events] == ["rejected_invalid_attempt"]


def test_heldout_evidence_cannot_become_known_failure_regression(tmp_path):
    from integrations.usami.domains import UsamiProfile
    p = UsamiProfile(config_id="cloud_heldout", domain="cloudops", identity="u2", mode="sandbox", split="heldout")
    responses = scripts("cloudops")
    adapter = UsamiAdapter(tmp_path, [p], client_factory=lambda spec, role, config, episode_id: ScriptedClient(spec, responses[role]))
    config = adapter.config_for(p.config_id, SPEC, SPEC, LIMITS)
    source = adapter.run_episode(config, *libraries("cloudops"), "heldout")
    assert source.replay.verified and source.failures
    with pytest.raises(ValueError, match="development"):
        adapter.run_failure_case(source, *libraries("cloudops"), "leaked")


@pytest.mark.parametrize("domain", ["saas", "cloudops", "fulfillment", "stayops"])
def test_native_interruption_keeps_termination_and_full_wire_record(tmp_path, domain):
    adapter, config, _ = adapter_for(tmp_path, domain, responses={"customer": [KeyboardInterrupt()], "service": []})
    evidence = adapter.run_episode(config, *libraries(domain), "interrupted." + domain)
    assert evidence.consumption.customer_llm_calls == 1 and not evidence.events
    assert evidence.model_calls[0].error["type"] == "KeyboardInterrupt"
    assert evidence.replay.verified, evidence.replay.reason


def _actual_capacity_failure(active_count, prefix=False):
    """Two valid scale variants executed by real SaaS Engine/Oracle."""
    from types import SimpleNamespace
    from usami import WorldEngine, generate_world, make_request, evaluate_trace, Operation
    from usami.world.generator import GeneratorConfig
    from usami.world.models import Membership, RulesConfig
    from integrations.usami.evidence import failure_details
    world = generate_world(42, GeneratorConfig(version="saas-generator-0.2.0", user_count=24,
                                               rules=RulesConfig(version="saas-rules-0.2.0")))
    for key in [key for key, member in world.memberships.items() if member.org_id == "org_000"]:
        del world.memberships[key]
    for number in range(active_count):
        identifier = "active_member_" + str(number)
        world.memberships[identifier] = Membership(id=identifier, org_id="org_000", user_id="user_" + str(number).zfill(3),
                                                   role="owner" if number == 0 else "member", status="active")
    world.memberships["pending_target"] = Membership(id="pending_target", org_id="org_000", user_id="user_023", role="member", status="pending")
    subscription = world.subscriptions["sub_000"]
    subscription.plan_id, subscription.seats, subscription.status, subscription.auto_renew = "business", active_count, "active", True
    from datetime import timedelta
    subscription.period_start, subscription.period_end = world.clock, world.clock + timedelta(days=30)
    engine, authority, tools = WorldEngine(world), {}, []
    path = ([Operation(name="cancel_renewal", org_id="org_000", resource_id="sub_000")] if prefix else []) + [
        Operation(name="activate_member", org_id="org_000", resource_id="pending_target")]
    for index, operation in enumerate(path):
        request = make_request("actual_capacity_" + str(index), "user_000", operation)
        authority[request.request_id] = request
        engine.register_request(request)
        event = engine.execute("user_000", operation, request.request_id, "sandbox")
        tools.append(SimpleNamespace(event_id=event.event_id, call_id="capacity_" + str(index)))
    trace = engine.finish()
    oracle = evaluate_trace(trace, authority)
    execution = SimpleNamespace(world_trace=trace, oracle=oracle, tool_records=tools, trusted_requests=authority)
    result = failure_details("saas", execution, "service_task")
    assert len(result) == 1 and result[0]["row"]["success"]
    assert "NO_AVAILABLE_SEATS" in result[0]["row"]["violations"]
    return result[0]


def _detail_fingerprint(details):
    from vast_coevo.evidence.signatures import failure_fingerprint
    row = details["row"]
    return failure_fingerprint("saas", row["operation"], row["violations"], details["relevant_prestate"],
                               details["permission_scope"], details["causal_prerequisites"], details["expected"],
                               details["actual"], details["normalization_scope"])


def test_single_step_full_capacity_scale_variants_share_policy_relative_signature():
    small, large = _actual_capacity_failure(10), _actual_capacity_failure(20)
    assert small["normalization_scope"] == large["normalization_scope"] == "policy_relative_features"
    assert small["relevant_prestate"]["policy_version"] == "saas-rules-0.2.0"
    assert not small["causal_prerequisites"] and not large["causal_prerequisites"]
    assert _detail_fingerprint(small) == _detail_fingerprint(large)


def test_unproved_prior_write_causality_keeps_exact_prestate_and_distinct_signatures():
    small, large = _actual_capacity_failure(10, prefix=True), _actual_capacity_failure(20, prefix=True)
    assert small["normalization_scope"] == large["normalization_scope"] == "pending_causal_normalization"
    assert small["relevant_prestate"]["exact_prestate_hash"] != large["relevant_prestate"]["exact_prestate_hash"]
    assert all(prerequisite["causal_status"] == "observed_prefix_not_counterfactually_verified"
               for prerequisite in small["causal_prerequisites"] + large["causal_prerequisites"])
    assert _detail_fingerprint(small) != _detail_fingerprint(large)
