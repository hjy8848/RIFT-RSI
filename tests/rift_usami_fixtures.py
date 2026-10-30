"""Shared deterministic fixtures for RIFT tests of the embedded Usami runtime."""
from datetime import timedelta

from usami_harness.models import digest
from usami_v2.commerce.models import Operation, case_id, quote_id
from usami_v2.runtime.agents import ServiceAction
from usami_v2.runtime.customer import CustomerDecision, CustomerProfileV2
from usami_v2.scenarios.builder import build_world
from usami_v2.scenarios.designer import ScenarioDraft
from usami_v2.scenarios.distribution import (
    BusinessSource, BusinessStratum, DistributionSpec, sample_schedule,
)
from usami_v2.scenarios.schema import (
    Attribute, BuildBudget, IntegerRange, LineVariable, LinearExpression, LinearTerm,
    OrderVariable, RelationConstraint, ScenarioSpec, SKUVariable, StockConstraint,
    WarehouseVariable,
)
from usami_v2.scenarios.task_certification import CustomerScenario
from usami_v2.scenarios.tasks import GroundedRemedyGoal, RemedyGoal, TaskSpec, TaskSpecV2



def fixed(value):
    return IntegerRange(low=value, high=value)



def term(alias, field, coefficient=1, warehouse=None):
    return LinearTerm(variable=alias, field=field, coefficient=coefficient, warehouse=warehouse)



def expression(*terms, constant=0):
    return LinearExpression(terms=terms, constant=constant)



def scene():
    skus = tuple(SKUVariable(
        alias=name, family="device" if name != "cable" else "cable",
        attributes=(Attribute(name="interface", value="USB_A" if name == "wrong" else "USB_C"),),
        price_cents=fixed(price))
        for name, price in (("original", 20000), ("wrong", 20000), ("late", 18000),
                            ("expensive", 24000), ("correct_choice_D", 22000),
                            ("correct_choice_E", 21000), ("cable", 2000)))
    return ScenarioSpec(
        scenario_id="business-world-test", provenance_kind="offline_fixture",
        customer_aliases=("customer",), skus=skus,
        warehouses=(WarehouseVariable(alias="fast", delivery_days=fixed(2), delivery_fee_cents=fixed(0)),
                    WarehouseVariable(alias="slow", delivery_days=fixed(6), delivery_fee_cents=fixed(0))),
        orders=(OrderVariable(alias="purchase", customer_alias="customer", delivered_days_ago=fixed(1),
                              lines=(LineVariable(alias="damaged", original_sku="original", quantity=fixed(1),
                                                  paid_cents=fixed(20000)),
                                     LineVariable(alias="accessory", original_sku="cable", quantity=fixed(1),
                                                  paid_cents=fixed(2000)))),),
        default_stock=fixed(0),
        stock=tuple(StockConstraint(sku_alias=name, warehouse_alias=warehouse, quantity=fixed(3))
                    for name, warehouse in (("wrong", "fast"), ("late", "slow"), ("expensive", "fast"),
                                            ("correct_choice_D", "fast"), ("correct_choice_E", "fast"))),
        constraints=(
            RelationConstraint(constraint_id="D_extra_under_budget",
                               left=expression(term("correct_choice_D", "price_cents"),
                                               term("damaged", "paid_cents", -1)),
                               operator="le", right=expression(constant=3000)),
            RelationConstraint(constraint_id="cheap_is_late", left=expression(term("slow", "delivery_days")),
                               operator="gt", right=expression(constant=3)),
        ))



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



def frozen_world(seed=13):
    result = build_world(scene(), seed, BuildBudget(max_nodes=128))
    assert result.status == "SAT"
    return result.world, dict(result.private_aliases)



def make_task(world, aliases, *, attributes=(), remedy=("exchange",), return_goal=False):
    line_id = aliases["damaged"]
    return TaskSpec(
        task_id="certified-refund-or-replacement", world_hash=world.state_hash,
        customer_id=aliases["customer"], customer_request="Please resolve the damaged item.",
        remedy_goals=(RemedyGoal(
            goal_id="damaged-line", line_id=line_id, quantity=1, allowed_remedies=remedy,
            required_attributes=attributes,
            maximum_extra_charge_cents=2000 if "exchange" in remedy else 0,
            latest_arrival=world.clock + timedelta(days=3) if "exchange" in remedy else None),),
        partition="train", source_distribution_hash=digest("offline task-certificate test"))



def customer_scenario(aliases, **updates):
    data = {
        "customer_id": aliases["customer"],
        "delegated_line_ids": (aliases["damaged"],),
        "maximum_extra_charge_cents": 2000,
        "latest_arrival": "2026-01-18T12:00:00+00:00",
    }
    data.update(updates)
    return CustomerScenario(**data)



def frozen(seed=17):
    result = build_world(scene(), seed)
    assert result.status == "SAT"
    return result.world, dict(result.private_aliases)



def refund_task(world, aliases):
    line_id = aliases["damaged"]
    request = "Please refund my damaged device."
    return TaskSpecV2(
        task_id="runtime-refund", world_hash=world.state_hash,
        customer_id=aliases["customer"], customer_request=request,
        remedy_goals=(GroundedRemedyGoal(
            goal_id="refund-damaged", line_id=line_id, quantity=1,
            public_request_span=request, subject_mention="damaged device",
            allowed_remedies=("refund",), maximum_extra_charge_cents=0),),
        partition="train", source_distribution_hash=digest("P3/P4 focused runtime"))



def refund_profile(aliases):
    line_id = aliases["damaged"]
    request = "Please refund my damaged device."
    return CustomerProfileV2(customer_id=aliases["customer"], delegated_line_ids=(line_id,),
                             delegation_request_spans={line_id: request})



class ScriptedRefundService:
    last_call = None
    uses_model = False

    def __init__(self, world, aliases):
        self.world, self.aliases, self.index = world, aliases, 0
        self.case = case_id(aliases["damaged"], (0,))
        self.quote = quote_id(self.case, 1)
        self.evidence = next(row.id for row in world.evidence.values()
                             if row.line_id == aliases["damaged"])

    def decide(self, prompt):
        merchant = next(iter(self.world.merchants))
        line = self.aliases["damaged"]
        actions = [
            ServiceAction(kind="operation", operation=Operation(
                name="open_remedy_case", scope_id=merchant, resource_id=line,
                params={"quantity": 1, "issue": "damaged"})),
            ServiceAction(kind="operation", operation=Operation(
                name="verify_submitted_evidence", scope_id=merchant, resource_id=self.case,
                params={"evidence_id": self.evidence})),
            ServiceAction(kind="operation", operation=Operation(
                name="prepare_remedy_quote", scope_id=merchant, resource_id=self.case,
                params={"remedy": "refund", "target_sku_id": None,
                        "warehouse_id": None, "pickup_slot": None})),
            ServiceAction(kind="reply", message="Please confirm this exact refund quote.",
                          present_quote_id=self.quote, present_for="commit_accepted_quote"),
            ServiceAction(kind="operation", operation=Operation(
                name="commit_accepted_quote", scope_id=merchant, resource_id=self.quote, params={})),
            ServiceAction(kind="reply", message=line + ": refund status is released.",
                          subject_ids=(line,)),
        ]
        action = actions[self.index]
        self.index += 1
        return action



class ScriptedCustomer:
    last_call = None
    uses_model = False

    def decide(self, prompt, profile):
        if prompt.offer:
            return CustomerDecision(message="I accept this refund quote.",
                                    accept_quote_id=prompt.offer.quote_id,
                                    accept_operation=prompt.offer.operation, finish=True)
        return CustomerDecision(message="Thank you.", finish=True)



class RejectedThenRefundService:
    uses_model = False
    last_call = None

    def __init__(self, world, aliases):
        self.world, self.aliases = world, aliases
        self.refund = ScriptedRefundService(world, aliases)
        self.first = True

    def decide(self, prompt):
        if self.first:
            self.first = False
            return ServiceAction(kind="operation", operation=Operation(
                name="open_remedy_case", scope_id=next(iter(self.world.merchants)),
                resource_id="invented-line", params={"quantity": 1, "issue": "damaged"}))
        return self.refund.decide(prompt)



class NoConsentSandboxService:
    last_call = None
    uses_model = False

    def __init__(self, world, aliases):
        self.aliases, self.index = aliases, 0
        self.merchant = next(iter(world.merchants))
        self.case = case_id(aliases["damaged"], (0,))
        self.quote = quote_id(self.case, 1)
        self.evidence = next(row.id for row in world.evidence.values()
                             if row.line_id == aliases["damaged"])

    def decide(self, prompt):
        line = self.aliases["damaged"]
        actions = [
            ServiceAction(kind="operation", operation=Operation(
                name="open_remedy_case", scope_id=self.merchant, resource_id=line,
                params={"quantity": 1, "issue": "damaged"})),
            ServiceAction(kind="operation", operation=Operation(
                name="verify_submitted_evidence", scope_id=self.merchant, resource_id=self.case,
                params={"evidence_id": self.evidence})),
            ServiceAction(kind="operation", operation=Operation(
                name="prepare_remedy_quote", scope_id=self.merchant, resource_id=self.case,
                params={"remedy": "refund", "target_sku_id": None,
                        "warehouse_id": None, "pickup_slot": None})),
            ServiceAction(kind="operation", operation=Operation(
                name="commit_accepted_quote", scope_id=self.merchant, resource_id=self.quote, params={})),
            ServiceAction(kind="reply", message=line + ": refund status is released.", subject_ids=(line,)),
        ]
        action = actions[self.index]
        self.index += 1
        return action
