"""CPU contract tests for RIFT-facing Usami World and task compilation."""
import pytest

from vast_coevo.schema import digest
from usami_v2.scenarios.distribution import (
    BusinessSource, BusinessStratum, DistributionSpec, sample_schedule,
)
from usami_v2.scenarios.designer import (
    InformationDraft, RemedyDraft, ScenarioDraft, TaskDraft, compile_task,
    task_design_capabilities,
)
from usami_v2.scenarios.task_certification import CustomerScenario
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


def test_task_intent_is_structured_and_the_host_renders_the_public_request():
    evidence_hash = "f" * 64
    draft = task_draft((evidence_hash,))
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
                        allowed_training_hashes=(evidence_hash,))
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


def test_task_schema_is_structured_and_has_no_internal_record_fields():
    schema = TaskDraft.model_json_schema()
    remedy_schema = schema["$defs"]["RemedyDraft"]["properties"]
    assert "customer_request" not in schema["properties"]
    assert "public_request_span" not in remedy_schema
    assert "subject_mention" not in remedy_schema
    assert "structured" in remedy_schema["allowed_remedies"]["description"]
    assert "actual receipt" in remedy_schema["target_stage"]["description"]
    assert "replacement-compatibility group" in SKUVariable.model_json_schema()["properties"]["family"]["description"]


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
