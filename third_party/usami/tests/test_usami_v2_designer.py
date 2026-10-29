"""P5 focused tests: two-stage proposal, provenance, proof and diagnostics."""
import json
from datetime import timedelta

import pytest

from vast_coevo.schema import digest
from usami_v2.runtime.agents import AgentCall
from usami.service.models import ModelSettings, ServiceError
from usami_v2.scenarios.distribution import (
    BusinessSource, BusinessStratum, DistributionSpec, sample_schedule,
)
from usami_v2.scenarios.designer import (
    DesignerTrainingEvidence, InformationDraft, OpenAIDesignerProvider, RemedyDraft, ScenarioDraft, TaskDraft,
    _chat, compile_task, design_and_certify,
)
from usami_v2.scenarios.task_certification import CustomerScenario, ProofBudget
from usami_v2.scenarios.builder import build_world
from usami_v2.scenarios.schema import SKUVariable
from test_usami_v2_builder import scene


def distribution_and_slot(seed):
    source = BusinessSource(source_id="unit-test-source", kind="synthetic_assumption",
                            reference="synthetic test fixture", version="1",
                            supported_claims=("offline designer protocol test",))
    stratum = BusinessStratum(stratum_id="single-order", weight=1,
                              weight_basis_source_id=source.source_id,
                              contexts=("damaged-item-remedy",))
    distribution = DistributionSpec(distribution_id="offline-designer-test", version=1,
        population_scope="one synthetic business fixture", sources=(source,), strata=(stratum,),
        split="train")
    return distribution, sample_schedule(distribution, seed, 1)[0]


def scenario_draft(name="scenario", natural_product_names=False):
    base = scene()
    skus = base.skus
    if natural_product_names:
        natural_skus = []
        for row in base.skus:
            if row.alias == "cable":
                natural_skus.append(row)
                continue
            data = row.model_dump(mode="json", exclude={"record_hash"})
            # Broad retail category belongs in customer-visible attributes;
            # the family field is the narrower replacement-compatibility key.
            data["family"] = "laptop"
            data["attributes"] += [{"name": "brand", "value": "TechPro"},
                                   {"name": "product", "value": "laptop"},
                                   {"name": "category", "value": "electronics"}]
            natural_skus.append(type(row).model_validate(data))
        skus = tuple(natural_skus)
    return ScenarioDraft(
        scenario_id=name, merchant_alias=base.merchant_alias,
        customer_aliases=base.customer_aliases, skus=skus,
        warehouses=base.warehouses, orders=base.orders, stock=base.stock,
        default_stock=base.default_stock,
        refund_approval_limit_cents=base.refund_approval_limit_cents,
        constraints=base.constraints)


def task_draft(evidence_hashes=()):
    return TaskDraft(
        task_id="designer-exchange", customer_alias="customer",
        remedy_goals=({
            "goal_id": "replace-device", "line_alias": "damaged", "quantity": 1,
            "allowed_remedies": ["exchange"],
            "required_attributes": [{"name": "interface", "value": "USB_C"}],
            "maximum_extra_charge_cents": 2000,
            "latest_arrival_within_days": 3,
        },), parent_training_evidence_hashes=evidence_hashes)


def replace_sealed(row, **updates):
    data = row.model_dump(mode="json", exclude={"record_hash"})
    data.update(updates)
    return type(row).model_validate(data)


class RecordedDesigner:
    def __init__(self, world, task):
        self.world, self.task = world, task
        self.contexts = []

    def submit(self, phase, messages, model):
        self.contexts.append((phase, messages))
        output = self.world if phase == "world" else self.task
        request = {"phase": phase, "messages": messages}
        response = {"choices": [{"message": {"role": "assistant", "content": "offline fixture"}}]}
        return output, AgentCall(role="designer", request=request, response=response)


class CorrectingDesigner:
    def __init__(self, world, invalid_task, valid_task):
        self.world, self.tasks, self.contexts = world, [invalid_task, valid_task], []

    def submit(self, phase, messages, model):
        self.contexts.append((phase, messages))
        output = self.world if phase == "world" else self.tasks.pop(0)
        raw = {"phase": phase, "attempt": len(self.contexts)}
        return output, AgentCall(role="designer", request=raw,
            response={"choices": [{"message": {"role": "assistant", "content": "offline correction fixture"}}]})


class TimeoutDesigner:
    def __init__(self):
        self.calls = 0
        self.last_call = None

    def submit(self, phase, messages, model):
        self.calls += 1
        request = {"phase": phase, "messages": messages}
        self.last_call = AgentCall(role="designer", request=request,
            response={"error": {"code": "PROVIDER_TIMEOUT",
                                "provider_response": {"timeout_phase": "ReadTimeout"}}})
        raise ServiceError("PROVIDER_TIMEOUT", "LLM request timed out; no retry performed",
                           {"timeout_phase": "ReadTimeout"})


class FixedCompletionClient:
    def __init__(self, arguments):
        self.settings = ModelSettings(model="offline-test", base_url="http://localhost/v1")
        self.response = {"choices": [{"message": {"role": "assistant", "tool_calls": [{
            "id": "call-1", "type": "function", "function": {
                "name": "submit_customer_task", "arguments": arguments,
            },
        }]}, "finish_reason": "tool_calls"}]}

    def complete(self, request):
        return self.response


def proof_customer(world, task):
    return CustomerScenario(customer_id=task.customer_id,
                           delegated_line_ids=tuple(row.line_id for row in task.remedy_goals),
                           maximum_extra_charge_cents=2000,
                           latest_arrival=world.clock + timedelta(days=3))


def run_designer(seed, evidence):
    distribution, slot = distribution_and_slot(seed)
    training_hash = evidence.source_record_hash
    provider = RecordedDesigner(scenario_draft(), task_draft((training_hash,)))
    run = design_and_certify(provider, distribution, slot,
        proof_budget=ProofBudget(max_plan_candidates=128, max_operation_steps=24, max_witnesses=2),
        allowed_training_hashes=(training_hash,), training_evidence=(evidence,),
        customer_scenario_factory=proof_customer, max_attempts=2)
    return run, provider


def test_two_stage_designer_builds_frozen_world_and_certified_task_with_provenance():
    evidence = DesignerTrainingEvidence(
        source_record_hash="a" * 64, partition="train",
        summary="Service asked for confirmation before committing the quoted remedy.",
        observed_events=({"kind": "service_reply", "text": "Please confirm the quote."},))
    run, provider = run_designer(21, evidence)
    assert run.status == "CERTIFIED"
    assert len(run.calls) == 2 and [call.phase for call in run.calls] == ["world", "task"]
    assert all(len(call.prompt_hash) == 64 for call in run.calls)
    assert len(run.world_attempts) == len(run.build_attempts) == 1
    assert run.world_attempts[0].status == "SAT"
    assert run.task.partition == "train"
    assert run.task.protocol == "commerce-task-v3"
    assert "replacement" in run.task.customer_request
    assert run.task.request_bundle.request_hash == digest(run.task.customer_request)
    assert run.task.source_distribution_hash == run.distribution.record_hash
    assert run.task.parent_training_evidence_hashes == (evidence.source_record_hash,)
    assert run.certification.business_reachability_certified
    assert run.pre_run_quality.business_valid and run.pre_run_quality.diagnosability
    assert run.pre_run_quality.decision_power == "diagnostic_only"
    task_context = json.loads(provider.contexts[1][1][1]["content"])
    rendered = json.dumps(task_context, sort_keys=True)
    assert evidence.source_record_hash in rendered
    capabilities = task_context["available_context"]["task_certification_capabilities"]
    assert capabilities["scenario_hash"] == run.scenario_spec.record_hash
    assert capabilities["world_hash"] == run.build_result.world_hash
    assert run.calls[1].capability_manifest_hash == digest(capabilities)
    assert capabilities["information_goals_certifiable"] is False
    assert capabilities["interaction_certifier_available"] is False
    assert capabilities["delivery_confirmation_available"] is False
    assert capabilities["supported_target_stages"] == ["settled"]
    assert capabilities["return_receipt_event_configured"] is False
    assert capabilities["policy_review_event_configured"] is False
    sku_context = next(row for row in task_context["available_context"]["skus"] if row["alias"] == "original")
    line_context = next(line for order in task_context["available_context"]["orders"]
                        for line in order["lines"] if line["alias"] == "damaged")
    assert sku_context["price_cents"] > 0 and sku_context["return_days"] == 30
    assert line_context["line_paid_cents"] == 20000 and "delivered_days_ago" in task_context["available_context"]["orders"][0]
    assert task_context["available_context"]["merchant_policy"]["refund_approval_limit_cents"] == 50000
    def has_internal_id_key(value):
        if isinstance(value, dict):
            return any(key == "id" or has_internal_id_key(item) for key, item in value.items())
        if isinstance(value, list):
            return any(has_internal_id_key(item) for item in value)
        return False
    assert not has_internal_id_key(task_context["available_context"])
    assert task_context["active_stratum"]["stratum_id"] == "single-order"
    assert task_context["active_stratum"]["design_focus"] == ["damaged-item-remedy"]
    assert "candidate_hash" not in rendered and "witness_hashes" not in rendered
    assert run.pre_run_quality.novelty_status == "NOT_EVALUATED"
    assert run.pre_run_quality.novel_against_history is None
    assert run.pre_run_quality.candidate_consequence_diagnostics
    assert run.pre_run_quality.quality_status == "business_certified_interaction_pending"


def test_id_and_seed_renaming_do_not_create_new_decision_structure():
    evidence = DesignerTrainingEvidence(source_record_hash="b" * 64, partition="train",
                                        summary="A training-only interaction observation.")
    left, _ = run_designer(31, evidence)
    right, _ = run_designer(32, evidence)
    assert left.build_result.world_hash != right.build_result.world_hash
    assert set(dict(left.build_result.private_aliases).values()) != set(dict(right.build_result.private_aliases).values())
    assert left.pre_run_quality.structural_signature == right.pre_run_quality.structural_signature
    assert left.pre_run_quality.world_hash != right.pre_run_quality.world_hash


def test_task_compiler_feedback_is_bounded_and_does_not_rebuild_frozen_world():
    evidence = DesignerTrainingEvidence(source_record_hash="e" * 64, partition="train",
                                        summary="A Service decision from a training episode.")
    valid_template = task_draft((evidence.source_record_hash,))
    goal_data = valid_template.remedy_goals[0].model_dump(mode="json", exclude={"record_hash"})
    goal_data["line_alias"] = "unknown-line"
    invalid = TaskDraft(task_id=valid_template.task_id, customer_alias=valid_template.customer_alias,
        remedy_goals=(RemedyDraft.model_validate(goal_data),),
        parent_training_evidence_hashes=valid_template.parent_training_evidence_hashes)
    provider = CorrectingDesigner(scenario_draft(), invalid,
                                  task_draft((evidence.source_record_hash,)))
    distribution, slot = distribution_and_slot(63)
    run = design_and_certify(provider, distribution, slot,
        proof_budget=ProofBudget(max_plan_candidates=128, max_operation_steps=24, max_witnesses=2),
        allowed_training_hashes=(evidence.source_record_hash,), training_evidence=(evidence,),
        customer_scenario_factory=proof_customer, max_attempts=2)
    assert run.status == "CERTIFIED"
    assert len(run.calls) == 3 and [row.phase for row in run.calls] == ["world", "task", "task"]
    assert len(run.build_attempts) == len(run.world_attempts) == 1
    assert [row.status for row in run.task_attempts] == ["REJECTED", "CERTIFIED"]
    correction = json.loads(provider.contexts[2][1][1]["content"])
    assert any("unknown order-line alias" in row for row in correction["validator_feedback"])


def test_provider_timeout_blocks_without_spending_proposal_correction_attempts():
    provider = TimeoutDesigner()
    distribution, slot = distribution_and_slot(64)
    run = design_and_certify(provider, distribution, slot,
        customer_scenario_factory=proof_customer, max_attempts=3)

    assert run.status == "BLOCKED"
    assert provider.calls == 1
    assert len(run.calls) == len(run.world_attempts) == 1
    assert run.world_attempts[0].status == "ERROR"
    assert "world provider failed: PROVIDER_TIMEOUT" in run.validation_errors[0]
    assert "proposal rejected" not in run.validation_errors[0]
    assert run.calls[0].validation_feedback == ""


@pytest.mark.parametrize("arguments,code,feedback", [
    ("{}", "EMPTY_TOOL_ARGUMENTS", "empty object"),
    ('{"task_id":"only"}', "DESIGNER_SCHEMA_INVALID", "customer_alias"),
])
def test_empty_and_incomplete_task_tool_calls_are_classified_and_preserved(arguments, code, feedback):
    client = FixedCompletionClient(arguments)
    provider = OpenAIDesignerProvider(client)
    with pytest.raises(ServiceError) as error:
        provider.submit("task", [{"role": "user", "content": "test"}], TaskDraft)
    assert error.value.code == code
    assert feedback in str(error.value)
    assert provider.last_call is not None
    assert provider.last_call.response == client.response
    from usami_v2.scenarios.designer import _call_provenance
    provenance = _call_provenance("task", provider.last_call)
    assert len(provenance["prompt_hash"]) == 64
    assert len(provenance["submission_schema_hash"]) == 64


def test_post_run_quality_is_separate_diagnostic_not_a_selection_rule():
    from usami_v2.scenarios.designer import post_run_quality
    evidence = DesignerTrainingEvidence(source_record_hash="c" * 64, partition="train",
                                        summary="Training sample")
    run, _ = run_designer(41, evidence)
    fake_episode = type("Episode", (), {
        "record_hash": "d" * 64,
        "replay_verified": True,
        "outcome": type("Result", (), {"strict_task_success": False, "final_goal_reached": False})(),
        "tool_executions": (), "status": "agent_error", "model_calls": (), "total_tokens": 0,
    })()
    vector = post_run_quality(run.pre_run_quality, fake_episode)
    assert vector.decision_power == "diagnostic_only"
    assert vector.strict_task_success is False and vector.model_calls == 0


def test_task_intent_is_structured_and_the_host_renders_the_public_request():
    evidence = DesignerTrainingEvidence(source_record_hash="f" * 64, partition="train",
                                        summary="Training evidence.")
    draft = task_draft((evidence.source_record_hash,))
    schema = TaskDraft.model_json_schema()
    assert "customer_request" not in schema["properties"]
    remedy_schema = schema["$defs"]["RemedyDraft"]["properties"]
    assert "public_request_span" not in remedy_schema
    assert "subject_mention" not in remedy_schema
    assert "allowed_remedies" in remedy_schema
    distribution, slot = distribution_and_slot(65)
    scenario = scenario_draft("structured-intent").compile(distribution)
    built = build_world(scenario, slot.seed)
    task = compile_task(draft, scenario, built, distribution,
                        allowed_training_hashes=(evidence.source_record_hash,))
    assert task.protocol == "commerce-task-v3"
    assert "replacement" in task.customer_request
    assert task.request_bundle.intent_hash
    assert task.request_bundle.request_hash == digest(task.customer_request)
    from usami_v2.scenarios.tasks import check_task_scope
    check_task_scope(built.world, task)


def test_structured_information_objective_does_not_need_a_quoted_question():
    draft = TaskDraft(task_id="owned-line-info", customer_alias="customer", information_goals=({
        "goal_id": "status", "subject_alias": "damaged", "fact_kind": "refund_status",
    },))
    distribution, slot = distribution_and_slot(66)
    scenario = scenario_draft("structured-information").compile(distribution)
    built = build_world(scenario, slot.seed)
    task = compile_task(draft, scenario, built, distribution,
                        interaction_certifier_available=True, information_goals_certifiable=True)
    assert task.information_goals[0].subject_id == dict(built.private_aliases)["damaged"]
    assert "refund status" in task.customer_request


def test_task_schema_and_prompt_do_not_ask_for_natural_language_matching():
    schema = TaskDraft.model_json_schema()["$defs"]["RemedyDraft"]["properties"]
    assert "public_request_span" not in schema
    assert "subject_mention" not in schema
    assert "structured" in schema["allowed_remedies"]["description"]
    assert "actual receipt" in schema["target_stage"]["description"]

    from usami_v2.scenarios.designer import _tool_schema
    task_schema = _tool_schema("submit_customer_task", TaskDraft)["function"]["parameters"]
    assert "record_hash" not in json.dumps(task_schema)
    assert "replacement-compatibility group" in SKUVariable.model_json_schema()["properties"]["family"]["description"]

    distribution, slot = distribution_and_slot(65)
    messages = _chat("task", {"world_hash": "a" * 64}, distribution, slot,
                     errors=("unknown order-line alias",))
    instruction = messages[0]["content"]
    assert "do not write customer_request" in instruction
    assert "without matching keywords or substrings" in instruction


def test_natural_product_names_and_replacement_intent_compile_against_frozen_world():
    draft = TaskDraft(task_id="natural-laptop-replacement", customer_alias="customer",
        remedy_goals=({
            "goal_id": "replace-laptop", "line_alias": "damaged", "quantity": 1,
            "allowed_remedies": ["exchange", "reship"],
            "required_attributes": [{"name": "interface", "value": "USB_C"}],
            "maximum_extra_charge_cents": 2000, "latest_arrival_within_days": 3,
            "target_stage": "settled",
        },))
    distribution, slot = distribution_and_slot(67)
    scenario = scenario_draft("natural-product", natural_product_names=True).compile(distribution)
    built = build_world(scenario, slot.seed)
    assert built.status == "SAT"
    task = compile_task(draft, scenario, built, distribution)
    assert task.remedy_goals[0].line_id == dict(built.private_aliases)["damaged"]
    assert task.remedy_goals[0].target_stage == "settled"
    assert task.remedy_goals[0].allowed_remedies == ("exchange", "reship")

    provider = RecordedDesigner(scenario_draft("natural-product", natural_product_names=True), draft)
    run = design_and_certify(provider, distribution, slot,
        proof_budget=ProofBudget(max_plan_candidates=128, max_operation_steps=24, max_witnesses=2),
        customer_scenario_factory=proof_customer, max_attempts=1)
    assert run.status == "CERTIFIED"
    context = json.loads(provider.contexts[1][1][1]["content"])["available_context"]
    assert context["task_certification_capabilities"]["delivery_confirmation_available"] is False
    assert context["task_certification_capabilities"]["information_goals_certifiable"] is False


def test_broad_electronics_category_does_not_make_laptop_and_mouse_compatible():
    from usami_v2.scenarios.task_certification import certify_task

    distribution, slot = distribution_and_slot(70)
    draft = scenario_draft("separate-compatibility-families", natural_product_names=True)
    skus = []
    for row in draft.skus:
        data = row.model_dump(mode="json", exclude={"record_hash"})
        if row.alias == "original":
            data["family"] = "laptop"
        elif row.alias == "wrong":
            data["family"] = "mouse"
            data["attributes"] = [
                ({**attribute, "value": "mouse"} if attribute["name"] == "product" else attribute)
                for attribute in (value.model_dump(mode="json", exclude={"record_hash"})
                                  for value in row.attributes)
            ]
        skus.append(type(row).model_validate(data))
    draft = replace_sealed(draft, skus=tuple(skus))
    spec = draft.compile(distribution)
    built = build_world(spec, slot.seed)
    assert built.status == "SAT"
    world, aliases = built.world, dict(built.private_aliases)
    draft_task = TaskDraft(task_id="laptop-only", customer_alias="customer",
        remedy_goals=({"goal_id": "replace", "line_alias": "damaged", "quantity": 1,
                       "allowed_remedies": ["exchange"]},))
    task = compile_task(draft_task, spec, built, distribution)
    proof = CustomerScenario(customer_id=aliases["customer"], delegated_line_ids=(aliases["damaged"],),
                             maximum_extra_charge_cents=100000)
    certificate = certify_task(world, task, proof)
    assert certificate.status == "CERTIFIED"
    assert all(row.target_sku_id != aliases["wrong"] for attempt in certificate.attempts
               for row in attempt.candidates if row.target_sku_id)


def test_task_compiler_gives_actionable_feedback_for_uncertifiable_targets_and_bad_info_aliases():
    distribution, slot = distribution_and_slot(68)
    scenario_draft_value = scenario_draft()
    scenario = scenario_draft_value.compile(distribution)
    built = build_world(scenario, slot.seed)

    delivered = task_draft()
    goal = replace_sealed(delivered.remedy_goals[0], target_stage="delivered")
    delivered = replace_sealed(delivered, remedy_goals=(goal,))
    with pytest.raises(ValueError, match="outside this run's certified capabilities.*settled"):
        compile_task(delivered, scenario, built, distribution)

    information = TaskDraft(task_id="customer-alias-info", customer_alias="customer",
        information_goals=({"goal_id": "status", "subject_alias": "customer",
                            "fact_kind": "refund_status"},))
    with pytest.raises(ValueError, match="selected Customer's owned order or order-line alias"):
        compile_task(information, scenario, built, distribution, interaction_certifier_available=True,
                     information_goals_certifiable=True)

    line_information = TaskDraft(task_id="owned-line-info", customer_alias="customer",
        information_goals=({"goal_id": "status", "subject_alias": "damaged",
                            "fact_kind": "refund_status"},))
    compiled_information = compile_task(
        line_information, scenario, built, distribution, interaction_certifier_available=True,
        information_goals_certifiable=True)
    assert compiled_information.information_goals[0].subject_id == dict(built.private_aliases)["damaged"]

    supported_alias_goal = replace_sealed(information.information_goals[0], subject_alias="purchase")
    unsupported_information = replace_sealed(information, information_goals=(supported_alias_goal,))
    with pytest.raises(ValueError, match="explicitly enabled natural-runtime interaction certification capability"):
        compile_task(unsupported_information, scenario, built, distribution)


def test_capability_manifest_blocks_unconfigured_return_and_policy_review_events():
    from usami_v2.scenarios.designer import task_design_capabilities

    distribution, slot = distribution_and_slot(71)
    base = scenario_draft()

    return_skus = [replace_sealed(row, return_required=True) if row.alias == "original" else row
                   for row in base.skus]
    return_draft = replace_sealed(base, skus=return_skus)
    return_spec = return_draft.compile(distribution)
    return_build = build_world(return_spec, slot.seed)
    assert return_build.status == "SAT"
    return_capabilities = task_design_capabilities(return_spec, return_build)
    with pytest.raises(ValueError, match="return-receipt event that is not configured"):
        compile_task(task_draft(), return_spec, return_build, distribution, capabilities=return_capabilities)

    review_draft = replace_sealed(base, refund_approval_limit_cents=1)
    review_spec = review_draft.compile(distribution)
    review_build = build_world(review_spec, slot.seed)
    assert review_build.status == "SAT"
    refund = TaskDraft(task_id="needs-policy-review", customer_alias="customer",
        remedy_goals=({
            "goal_id": "refund", "line_alias": "damaged", "quantity": 1,
            "allowed_remedies": ["refund"], "maximum_extra_charge_cents": 0,
        },))
    review_capabilities = task_design_capabilities(review_spec, review_build)
    with pytest.raises(ValueError, match="policy-review event that is not configured"):
        compile_task(refund, review_spec, review_build, distribution, capabilities=review_capabilities)
    no_refund = replace_sealed(review_capabilities, supported_remedies=("exchange",))
    with pytest.raises(ValueError, match="unsupported remedy method.*refund"):
        compile_task(refund, review_spec, review_build, distribution, capabilities=no_refund)


def test_host_renders_all_scored_bounds_from_structured_intent():
    distribution, slot = distribution_and_slot(69)
    scenario = scenario_draft().compile(distribution)
    built = build_world(scenario, slot.seed)
    draft = task_draft()
    task = compile_task(draft, scenario, built, distribution)
    assert task.protocol == "commerce-task-v3"
    assert "CNY 20.00" in task.customer_request
    assert "within 3 days" in task.customer_request
    assert task.request_bundle.intent_hash
    assert task.request_bundle.request_hash == digest(task.customer_request)
