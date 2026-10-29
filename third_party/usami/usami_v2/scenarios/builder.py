"""Construct legal frozen commerce worlds from business constraints."""
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional, Tuple

from pydantic import Field, model_validator

from vast_coevo.schema import digest
from usami_v2.contracts import Sealed
from usami_v2.commerce.invariants import invariants
from usami_v2.commerce.models import (
    Customer, DeliveryRoute, Evidence, Inventory, Merchant, Order, OrderLine, SKU, Warehouse, World,
)
from .schema import BuildBudget, ScenarioSpec
from .solver import assignments, expression_value, impossible_by_interval, relation_satisfied, variable_ranges


class ConstraintProof(Sealed):
    constraint_id: str
    left: int = Field(strict=True)
    operator: str
    right: int = Field(strict=True)


class BuildAttempt(Sealed):
    node: int = Field(ge=0, strict=True)
    assignment_hash: str
    rejected_constraints: Tuple[str, ...]
    world_invariant_errors: Tuple[str, ...] = ()


class BuildResult(Sealed):
    protocol: Literal["commerce-world-build-v1"] = "commerce-world-build-v1"
    status: Literal["SAT", "UNSAT", "UNKNOWN"]
    scenario_hash: str
    seed: int = Field(strict=True)
    budget: BuildBudget
    reason: str
    world: Optional[World] = None
    world_hash: Optional[str] = None
    private_aliases: Tuple[Tuple[str, str], ...] = ()
    assignment: Tuple[Tuple[str, str, str, int], ...] = ()
    proofs: Tuple[ConstraintProof, ...] = ()
    attempts: Tuple[BuildAttempt, ...] = ()

    @model_validator(mode="after")
    def result_binding(self):
        if self.status == "SAT":
            if self.world is None or self.world.state_hash != self.world_hash or invariants(self.world):
                raise ValueError("SAT requires a valid frozen world with matching hash")
        elif self.world is not None or self.world_hash is not None:
            raise ValueError("failed/uncertain construction cannot present a certified world")
        return self


def _identifier(seed, kind, alias):
    # A Designer alias such as "correct_choice" must not become the public SKU ID.
    return kind + "_" + digest([seed, kind, alias])[:12]


def _construct(spec, seed, values):
    aliases = {}
    for kind, names in (
        ("merchant", (spec.merchant_alias,)), ("customer", spec.customer_aliases),
        ("sku", tuple(row.alias for row in spec.skus)),
        ("warehouse", tuple(row.alias for row in spec.warehouses)),
        ("order", tuple(row.alias for row in spec.orders)),
        ("line", tuple(line.alias for order in spec.orders for line in order.lines)),
    ):
        for alias in names:
            aliases[alias] = _identifier(seed, kind, alias)
    merchant = aliases[spec.merchant_alias]
    clock = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
    skus = {}
    for row in spec.skus:
        key = aliases[row.alias]
        skus[key] = SKU(
            id=key, merchant_id=merchant, product_family=row.family,
            attributes={item.name: item.value for item in row.attributes},
            price_cents=values[(row.alias, "price_cents", "")],
            return_days=row.return_days, return_required=row.return_required)
    warehouses, routes, inventory = {}, {}, {}
    for row in spec.warehouses:
        key = aliases[row.alias]
        warehouses[key] = Warehouse(id=key, merchant_id=merchant)
        rid = _identifier(seed, "route", row.alias)
        routes[rid] = DeliveryRoute(id=rid, warehouse_id=key,
                                    days=values[(row.alias, "delivery_days", "")],
                                    fee_cents=values[(row.alias, "delivery_fee_cents", "")])
        for sku in spec.skus:
            inv_id = _identifier(seed, "stock", sku.alias + "/" + row.alias)
            amount = values[(sku.alias, "stock", row.alias)]
            inventory[inv_id] = Inventory(id=inv_id, sku_id=aliases[sku.alias], warehouse_id=key,
                                           opening_quantity=amount, available=amount)
    orders, lines, evidence = {}, {}, {}
    for order in spec.orders:
        order_id, customer = aliases[order.alias], aliases[order.customer_alias]
        for row in order.lines:
            key = aliases[row.alias]
            lines[key] = OrderLine(
                id=key, order_id=order_id, sku_id=aliases[row.original_sku],
                quantity=values[(row.alias, "quantity", "")],
                allocated_paid_cents=values[(row.alias, "paid_cents", "")])
            eid = _identifier(seed, "evidence", row.alias)
            evidence[eid] = Evidence(id=eid, customer_id=customer, line_id=key,
                                       issue=row.issue, valid=row.evidence_valid)
        orders[order_id] = Order(
            id=order_id, merchant_id=merchant, customer_id=customer,
            paid_cents=sum(lines[aliases[row.alias]].allocated_paid_cents for row in order.lines),
            payment_channel="original-card", delivered_at=clock - timedelta(
                days=values[(order.alias, "delivered_days_ago", "")]))
    return World(
        seed=seed, clock=clock,
        customers={aliases[name]: Customer(id=aliases[name], name="Customer " + str(index + 1))
                   for index, name in enumerate(spec.customer_aliases)},
        merchants={merchant: Merchant(id=merchant, refund_approval_limit_cents=spec.refund_approval_limit_cents)},
        skus=skus, warehouses=warehouses, routes=routes, inventory=inventory,
        orders=orders, lines=lines, evidence=evidence), tuple(sorted(aliases.items()))


def _assignment_hash(values):
    return digest([(*key, value) for key, value in sorted(values.items())])


def build_world(spec: ScenarioSpec, seed: int, budget: Optional[BuildBudget] = None):
    spec, budget = spec.checked(), (budget or BuildBudget()).checked()
    if type(seed) is not int:
        raise ValueError("integer world seed required")
    ranges = variable_ranges(spec)
    impossible = [row.constraint_id for row in spec.constraints if impossible_by_interval(row, ranges)]
    if impossible:
        return BuildResult(status="UNSAT", scenario_hash=spec.record_hash, seed=seed, budget=budget,
                           reason="interval contradiction:" + ",".join(impossible))
    trace = []
    complete = False
    for node, (values, complete) in enumerate(assignments(ranges, seed, budget.max_nodes)):
        rejected = tuple(row.constraint_id for row in spec.constraints if not relation_satisfied(row, values))
        if rejected:
            trace.append(BuildAttempt(node=node, assignment_hash=_assignment_hash(values), rejected_constraints=rejected))
            continue
        world, aliases = _construct(spec, seed, values)
        errors = invariants(world)
        trace.append(BuildAttempt(node=node, assignment_hash=_assignment_hash(values), rejected_constraints=(),
                                  world_invariant_errors=tuple(row["code"] for row in errors)))
        if errors:
            continue
        result = BuildResult(
            status="SAT", scenario_hash=spec.record_hash, seed=seed, budget=budget,
            reason="constructed business constraints satisfied; task/interaction not yet certified",
            world=world, world_hash=world.state_hash, private_aliases=aliases,
            assignment=tuple((*key, value) for key, value in sorted(values.items())),
            proofs=tuple(ConstraintProof(constraint_id=row.constraint_id,
                                        left=expression_value(row.left, values), operator=row.operator,
                                        right=expression_value(row.right, values)) for row in spec.constraints),
            attempts=tuple(trace))
        from .certification import verify_world_constraints
        # An implementation defect is an error, not a fabricated SAT or UNSAT.
        verify_world_constraints(spec, result)
        return result
    return BuildResult(status="UNSAT" if complete else "UNKNOWN",
                       scenario_hash=spec.record_hash, seed=seed, budget=budget,
                       reason="complete finite enumeration found no world" if complete else "construction node budget exhausted",
                       attempts=tuple(trace))


def verify_build(spec: ScenarioSpec, result: BuildResult):
    """Reconstruct the actual generation transcript; never trust its SAT flag."""
    spec, result = spec.checked(), result.checked()
    if result.scenario_hash != spec.record_hash:
        raise ValueError("world build belongs to another scenario")
    rebuilt = build_world(spec, result.seed, result.budget)
    if rebuilt != result:
        raise ValueError("world/constraint proof or generation transcript differs")
    certification = None
    if result.status == "SAT":
        from .certification import verify_world_constraints
        certification = verify_world_constraints(spec, result)
    return {"verified": True, "status": result.status, "world_hash": result.world_hash,
            "build_hash": result.record_hash, "model_calls": 0,
            "world_constraint_certificate_hash": certification.record_hash if certification else None,
            "task_reachability_certified": False, "interaction_certified": False}
