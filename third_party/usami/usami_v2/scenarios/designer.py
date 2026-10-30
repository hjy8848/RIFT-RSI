"""RIFT-facing structured World and task drafts with native task compilation."""
from datetime import timedelta
from typing import Literal, Optional, Tuple

from pydantic import Field, model_validator

from usami_harness.models import digest
from usami_v2.commerce.models import unit_credit
from usami_v2.contracts import Sealed
from usami_v2.scenarios.builder import BuildResult
from usami_v2.scenarios.distribution import DistributionSpec
from usami_v2.scenarios.schema import (
    Attribute, IntegerRange, LineVariable, OrderVariable, RelationConstraint,
    SKUVariable, StockConstraint, WarehouseVariable, ScenarioSpec,
)
from usami_v2.scenarios.task_certification import CustomerTaskScope, TaskDesignCapabilities
from usami_v2.scenarios.tasks import (
    CustomerRequestBundle, RemedyGoal, StructuredInformationObjective, TaskSpecV3,
    check_task_scope, render_customer_request, request_intent_hash,
)
class ScenarioDraft(Sealed):
    scenario_id: str
    merchant_alias: str = "merchant"
    customer_aliases: Tuple[str, ...]
    skus: Tuple[SKUVariable, ...]
    warehouses: Tuple[WarehouseVariable, ...]
    orders: Tuple[OrderVariable, ...]
    stock: Tuple[StockConstraint, ...] = ()
    default_stock: IntegerRange = IntegerRange(low=0, high=5)
    refund_approval_limit_cents: int = Field(default=50000, ge=0, strict=True)
    constraints: Tuple[RelationConstraint, ...] = ()

    def compile(self, distribution: DistributionSpec):
        return ScenarioSpec(
            scenario_id=self.scenario_id, merchant_alias=self.merchant_alias,
            customer_aliases=self.customer_aliases, skus=self.skus, warehouses=self.warehouses,
            orders=self.orders, stock=self.stock, default_stock=self.default_stock,
            refund_approval_limit_cents=self.refund_approval_limit_cents,
            constraints=self.constraints, provenance_kind="designer_exploration",
            distribution_hash=distribution.record_hash)



class RemedyDraft(Sealed):
    goal_id: str
    line_alias: str
    quantity: int = Field(ge=1, strict=True,
                          description="Number of purchased units covered by this structured Customer intent.")
    allowed_remedies: Tuple[Literal["refund", "exchange", "reship"], ...] = Field(
        min_length=1,
        description="Customer-authorized outcomes. The Host renders these structured outcomes into the request.",
    )
    required_attributes: Tuple[Attribute, ...] = Field(
        default=(), description="Required outcome attributes, checked against frozen SKU facts.")
    maximum_extra_charge_cents: Optional[int] = Field(
        default=None, ge=0, strict=True,
        description="Maximum authorized extra charge in cents; the Host renders the exact amount.")
    latest_arrival_within_days: Optional[int] = Field(
        default=None, ge=0, strict=True,
        description="Latest permitted arrival in whole days from frozen world time; rendered by the Host.")
    target_stage: Literal["settled", "delivered"] = Field(
        default="settled",
        description=("Use 'settled' for the currently supported Reference proof. 'delivered' means actual receipt, "
                     "which this run can certify only when a natural interaction certifier is explicitly available."),
    )



class InformationDraft(Sealed):
    goal_id: str
    subject_alias: str = Field(
        description=("Alias of an existing order or order line owned by the selected Customer, exactly as listed "
                     "in that Customer's scope. Never use a Customer, SKU, or another Customer's object."),
    )
    fact_kind: Literal["refund_amount", "extra_charge", "arrival_date", "refund_status",
                       "shipment_status", "quantity", "currency"] = Field(
        description="The structured business fact requested; the Host renders its public question.",
    )



class TaskDraft(Sealed):
    task_id: str
    customer_alias: str
    remedy_goals: Tuple[RemedyDraft, ...] = ()
    information_goals: Tuple[InformationDraft, ...] = Field(
        default=(),
        description=("Only use when task_certification_capabilities says a natural interaction certifier is available; "
                     "the bounded native Reference proof certifies remedy goals, not communication goals."),
    )
    parent_training_evidence_hashes: Tuple[str, ...] = ()

    @model_validator(mode="after")
    def unique_goal_ids(self):
        rows = self.remedy_goals + self.information_goals
        if not rows or len({row.goal_id for row in rows}) != len(rows):
            raise ValueError("TaskDraft needs unique business or information goals")
        if any(len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
               for value in self.parent_training_evidence_hashes):
            raise ValueError("training evidence references must be SHA-256 hashes")
        return self



class TaskCapabilityUnsupported(ValueError):
    """The native proof contract cannot certify a structurally valid goal."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code



def task_design_capabilities(spec: ScenarioSpec, build: BuildResult, *,
                             interaction_certifier_available=False,
                             information_goals_certifiable=False,
                             delivery_confirmation_available=False,
                             return_receipt_event_configured=False,
                             policy_review_event_configured=False):
    """Construct the exact capability manifest shared by prompt and compiler."""
    aliases = dict(build.private_aliases)
    scopes = []
    for customer_alias in spec.customer_aliases:
        orders = tuple(row.alias for row in spec.orders if row.customer_alias == customer_alias)
        lines = tuple(line.alias for row in spec.orders if row.customer_alias == customer_alias
                      for line in row.lines)
        scopes.append(CustomerTaskScope(customer_alias=customer_alias,
                                        order_aliases=orders, line_aliases=lines))
    required_aliases = {spec.merchant_alias, *spec.customer_aliases}
    required_aliases.update(row.alias for row in spec.orders)
    required_aliases.update(line.alias for row in spec.orders for line in row.lines)
    required_aliases.update(row.alias for row in spec.skus)
    required_aliases.update(row.alias for row in spec.warehouses)
    if not required_aliases.issubset(aliases):
        raise ValueError("the frozen world alias map is unavailable")
    interaction = bool(interaction_certifier_available)
    information = bool(information_goals_certifiable)
    delivery = bool(delivery_confirmation_available)
    return TaskDesignCapabilities(
        scenario_hash=spec.record_hash,
        world_hash=build.world_hash,
        supported_target_stages=("settled", "delivered") if delivery else ("settled",),
        interaction_certifier_available=interaction,
        information_goals_certifiable=information,
        delivery_confirmation_available=delivery,
        return_receipt_event_configured=bool(return_receipt_event_configured),
        policy_review_event_configured=bool(policy_review_event_configured),
        customer_scopes=tuple(scopes),
    )



def compile_task(draft: TaskDraft, spec: ScenarioSpec, build: BuildResult,
                 distribution: DistributionSpec, allowed_training_hashes=(), *,
                 interaction_certifier_available: Optional[bool] = None,
                 information_goals_certifiable: Optional[bool] = None,
                 delivery_confirmation_available: Optional[bool] = None,
                 capabilities: Optional[TaskDesignCapabilities] = None):
    draft, spec, build, distribution = draft.checked(), spec.checked(), build.checked(), distribution.checked()
    if build.status != "SAT" or not build.world:
        raise ValueError("a SAT frozen world is required before task design")
    aliases = dict(build.private_aliases)
    capabilities = (capabilities.checked() if capabilities is not None else
                    task_design_capabilities(spec, build,
                        interaction_certifier_available=bool(interaction_certifier_available),
                        information_goals_certifiable=information_goals_certifiable,
                        delivery_confirmation_available=bool(delivery_confirmation_available)))
    if capabilities.scenario_hash != spec.record_hash or capabilities.world_hash != build.world_hash:
        raise ValueError("Task capability manifest belongs to a different frozen world")
    if (interaction_certifier_available is not None
            and capabilities.interaction_certifier_available != bool(interaction_certifier_available)):
        raise ValueError("Task capability manifest differs from the configured interaction certifier")
    if (delivery_confirmation_available is not None
            and capabilities.delivery_confirmation_available != bool(delivery_confirmation_available)):
        raise ValueError("Task capability manifest differs from configured delivery confirmation")
    if (information_goals_certifiable is not None
            and capabilities.information_goals_certifiable != bool(information_goals_certifiable)):
        raise ValueError("Task capability manifest differs from configured information certification")
    scopes = {row.customer_alias: row for row in capabilities.customer_scopes}
    customer_scope = scopes.get(draft.customer_alias)
    if customer_scope is None:
        raise ValueError("TaskDraft references a Customer alias outside the declared capability scope")
    if draft.customer_alias not in aliases:
        raise ValueError("TaskDraft references a Customer alias outside the frozen world")
    if not set(draft.parent_training_evidence_hashes).issubset(set(allowed_training_hashes)):
        raise ValueError("TaskDraft references training evidence outside the authorized input set")
    if draft.information_goals and not capabilities.information_goals_certifiable:
        raise TaskCapabilityUnsupported("INFORMATION_CERTIFIER_UNAVAILABLE",
            "information goals require an explicitly enabled natural-runtime interaction certification capability"
        )
    if draft.remedy_goals and not capabilities.native_reference_certifies and not capabilities.interaction_certifier_available:
        raise TaskCapabilityUnsupported("TASK_CERTIFIER_UNAVAILABLE",
                                        "this run has no configured certifier for remedy goals")
    remedies, information = [], []
    for row in draft.remedy_goals:
        unsupported_remedies = set(row.allowed_remedies) - set(capabilities.supported_remedies)
        if unsupported_remedies:
            raise TaskCapabilityUnsupported(
                "REMEDY_UNSUPPORTED", "unsupported remedy method(s): {}".format(
                    ", ".join(sorted(unsupported_remedies))))
        if row.line_alias not in customer_scope.line_aliases:
            raise ValueError("TaskDraft references an unknown order-line alias")
        original_id = aliases[row.line_alias]
        if row.target_stage not in capabilities.supported_target_stages:
            raise TaskCapabilityUnsupported("TARGET_STAGE_UNSUPPORTED",
                "target_stage '{}' is outside this run's certified capabilities; supported stages are {}".format(
                    row.target_stage, ", ".join(capabilities.supported_target_stages))
            )
        line = build.world.lines[original_id]
        original_sku = build.world.skus[line.sku_id]
        if row.quantity > line.quantity:
            raise ValueError("TaskDraft quantity exceeds the purchased line quantity")
        if (original_sku.return_required and not capabilities.return_receipt_event_configured):
            raise TaskCapabilityUnsupported(
                "RETURN_RECEIPT_UNAVAILABLE",
                "this task requires a return-receipt event that is not configured for certification")
        if "refund" in row.allowed_remedies:
            credit = unit_credit(line, tuple(range(row.quantity)))
            merchant_id = build.world.orders[line.order_id].merchant_id
            approval_limit = build.world.merchants[merchant_id].refund_approval_limit_cents
            if credit > approval_limit and not capabilities.policy_review_event_configured:
                raise TaskCapabilityUnsupported(
                    "POLICY_REVIEW_UNAVAILABLE",
                    "this refund requires a policy-review event that is not configured for certification")
        arrival = (build.world.clock + timedelta(days=row.latest_arrival_within_days)
                   if row.latest_arrival_within_days is not None else None)
        compiled_goal = RemedyGoal(
            goal_id=row.goal_id, line_id=original_id, quantity=row.quantity,
            allowed_remedies=row.allowed_remedies, required_attributes=row.required_attributes,
            maximum_extra_charge_cents=row.maximum_extra_charge_cents,
            latest_arrival=arrival, target_stage=row.target_stage)
        remedies.append(compiled_goal)
    for row in draft.information_goals:
        if row.subject_alias not in aliases:
            raise ValueError("TaskDraft references an unknown public order or line alias")
        if row.subject_alias not in set(customer_scope.order_aliases) | set(customer_scope.line_aliases):
            raise ValueError(
                "information subject must be the selected Customer's owned order or order-line alias"
            )
        information.append(StructuredInformationObjective(
            goal_id=row.goal_id, subject_id=aliases[row.subject_alias], fact_kind=row.fact_kind))
    if any(len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
           for value in draft.parent_training_evidence_hashes):
        raise ValueError("invalid training evidence hash")
    customer_id = aliases[draft.customer_alias]
    customer_request = render_customer_request(build.world, customer_id, tuple(remedies), tuple(information))
    request_bundle = CustomerRequestBundle(
        customer_id=customer_id,
        goal_ids=tuple(row.goal_id for row in tuple(remedies) + tuple(information)),
        intent_hash=request_intent_hash(customer_id, tuple(remedies), tuple(information)),
        request_hash=digest(customer_request))
    task = TaskSpecV3(
        task_id=draft.task_id, world_hash=build.world_hash, customer_id=aliases[draft.customer_alias],
        customer_request=customer_request, remedy_goals=tuple(remedies),
        information_goals=tuple(information), partition=distribution.split,
        source_distribution_hash=distribution.record_hash,
        parent_training_evidence_hashes=draft.parent_training_evidence_hashes,
        request_bundle=request_bundle)
    return check_task_scope(build.world, task)
