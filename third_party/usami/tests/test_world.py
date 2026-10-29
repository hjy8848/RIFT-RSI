from datetime import timedelta

import pytest
from pydantic import ValidationError

from usami import (GeneratorConfig, Operation, WorldEngine, clone_world, execute_operation,
                   generate_world, make_request, reset_world)
from usami.world.models import Plan, RulesConfig, WorldState, digest
from usami.world.rules import ReferenceRules, check_invariants


def operation(name, resource="sub_000", org="org_000", **params):
    return Operation(name=name, org_id=org, resource_id=resource, params=params)


def execute(engine, op, actor="user_000", request_id="r", **kwargs):
    engine.register_request(make_request(request_id, actor, op))
    return engine.execute(actor, op, request_id, **kwargs)


@pytest.mark.parametrize("profile", ["standard", "train_single", "test_composed"])
def test_many_seeds_deterministic_valid_and_diverse(profile):
    combinations = set()
    cfg = GeneratorConfig(profile=profile)
    for seed in range(150):
        a, b = generate_world(seed, cfg), generate_world(seed, cfg)
        assert a.model_dump_json() == b.model_dump_json()
        assert a.state_hash == b.state_hash
        assert not check_invariants(a)
        assert len(a.organizations) >= 2
        for org in a.organizations.values():
            sub = a.subscriptions[org.subscription_id]
            combinations.add((sub.plan_id, sub.seats, sub.status, sub.auto_renew))
        assert {i.status for i in a.invoices.values()} == {"unpaid", "paid", "refunded"}
    assert len(combinations) > 40


def test_config_order_does_not_change_generation():
    cfg = GeneratorConfig()
    reversed_cfg = cfg.model_copy(deep=True)
    reversed_cfg.rules.plans = dict(reversed(list(cfg.rules.plans.items())))
    assert digest(cfg) == digest(reversed_cfg)
    assert generate_world(42, cfg) == generate_world(42, reversed_cfg)


def test_generator_version_has_a_fixed_regression_world():
    assert generate_world(42).state_hash == "cbdc43984d682555281707a7431c7ed702f0c9bc4a3ebb3346b614f10936e0ec"


@pytest.mark.parametrize("window", [0, 1, 30])
def test_custom_rules_are_consistent(window):
    cfg = GeneratorConfig(rules=RulesConfig(refund_window_days=window, plans={
        "tiny": Plan(id="tiny", name="Tiny", rank=0, max_seats=2),
        "big": Plan(id="big", name="Big", rank=1, max_seats=9)}))
    for seed in range(20):
        assert not check_invariants(generate_world(seed, cfg))


def test_reference_is_pure_and_engine_transitions(world):
    original = world.state_hash
    engine = WorldEngine(world)
    upgrade = operation("change_plan", plan_id="team")
    decision = ReferenceRules().decide(world, "user_000", upgrade)
    assert decision.allowed and world.state_hash == original
    execute(engine, upgrade, request_id="plan")
    execute(engine, operation("change_seats", seats=8), request_id="seats")
    event = execute(engine, operation("cancel_renewal"), request_id="cancel")
    sub = engine.state.subscriptions["sub_000"]
    assert sub.plan_id == "team" and sub.seats == 8 and not sub.auto_renew
    assert sub.status == "active" and sub.period_end == world.subscriptions["sub_000"].period_end
    assert event.payload["result"]["success"]
    assert not check_invariants(engine.state)
    assert world.state_hash == original


@pytest.mark.parametrize("op,code", [
    (operation("change_seats", seats=6), "SEAT_LIMIT_EXCEEDED"),
    (operation("change_seats", seats=0), "SEATS_BELOW_MEMBERS"),
    (operation("change_plan", plan_id="missing"), "INVALID_PLAN"),
    (operation("cancel_renewal", org="org_001", resource="sub_000"), "CROSS_ORG_RESOURCE"),
    (operation("cancel_renewal", resource="missing"), "RESOURCE_NOT_FOUND"),
    (operation("cancel_renewal", org="missing"), "ORG_NOT_FOUND"),
])
def test_reference_rejection_preserves_state(world, op, code):
    engine = WorldEngine(world)
    actor = world.organizations[op.org_id].owner_id if op.org_id in world.organizations else "user_000"
    event = execute(engine, op, actor)
    assert event.payload["result"]["error_code"] == code
    assert engine.state == world


def test_plan_downgrade_must_fit_seats(world):
    world.subscriptions["sub_000"].plan_id = "business"
    world.subscriptions["sub_000"].seats = 30
    decision = ReferenceRules().decide(world, "user_000", operation("change_plan", plan_id="team"))
    assert decision.error_code == "SEAT_LIMIT_EXCEEDED"


def test_expired_subscription_denied(world):
    decision = ReferenceRules().decide(world, "user_001", operation("cancel_renewal", org="org_001", resource="sub_001"))
    assert decision.error_code == "SUBSCRIPTION_NOT_ACTIVE"


def test_member_and_billing_permissions(world):
    rules = ReferenceRules()
    assert rules.decide(world, "user_002", operation("cancel_renewal")).allowed
    assert rules.decide(world, "user_003", operation("cancel_renewal")).error_code == "PERMISSION_DENIED"
    assert rules.decide(world, "user_003", operation("get_subscription")).allowed
    assert rules.decide(world, "user_003", operation("get_invoice", resource="invoice_000_recent")).error_code == "PERMISSION_DENIED"
    assert rules.decide(world, "user_002", operation("change_member_role", resource="membership_000_user_003", role="billing_admin")).error_code == "PERMISSION_DENIED"


@pytest.mark.parametrize("target,role", [("membership_000_user_000", "member"), ("membership_000_user_003", "owner")])
def test_owner_is_immutable(world, target, role):
    assert ReferenceRules().decide(world, "user_000", operation("change_member_role", resource=target, role=role)).error_code == "OWNER_IMMUTABLE"


def test_role_change_and_queries(world):
    engine = WorldEngine(world)
    execute(engine, operation("change_member_role", resource="membership_000_user_003", role="billing_admin"))
    assert engine.state.memberships["membership_000_user_003"].role == "billing_admin"
    for n, op in enumerate([operation("get_organization", resource="org_000"),
                           operation("list_members", resource="org_000"), operation("get_subscription"),
                           operation("get_invoice", resource="invoice_000_recent")]):
        previous = engine.state.state_hash
        event = execute(engine, op, request_id=str(n))
        assert event.payload["result"]["success"] and event.payload["result"]["data"]
        assert previous == engine.state.state_hash


@pytest.mark.parametrize("age,allowed", [(timedelta(days=14), True), (timedelta(days=14, microseconds=1), False)])
def test_refund_exact_boundary(world, age, allowed):
    inv = world.invoices["invoice_000_recent"]
    inv.paid_at = world.clock - age
    op = operation("refund_invoice", resource=inv.id, amount_cents=inv.amount_cents, payment_method=inv.payment_method)
    decision = ReferenceRules().decide(world, "user_000", op)
    assert decision.allowed == allowed


@pytest.mark.parametrize("kind,code", [("unpaid", "INVOICE_NOT_PAID"), ("stale", "REFUND_WINDOW_EXPIRED"), ("refunded", "DUPLICATE_REFUND")])
def test_refund_eligibility(world, kind, code):
    inv = world.invoices["invoice_000_" + kind]
    op = operation("refund_invoice", resource=inv.id, amount_cents=inv.amount_cents, payment_method=inv.payment_method or "card_visa")
    assert ReferenceRules().decide(world, "user_000", op).error_code == code


def test_clone_reset_and_snapshot_isolation(world, tmp_path):
    engine = WorldEngine(world)
    execute(engine, operation("cancel_renewal"))
    clone = clone_world(engine)
    execute(clone, operation("change_plan", plan_id="team"), request_id="other")
    assert engine.state.subscriptions["sub_000"].plan_id == "starter"
    exported = tmp_path / "snapshot.json"
    engine.export_snapshot(exported)
    assert WorldEngine.import_snapshot(exported).state == engine.state
    copy = clone_world(world)
    copy.users["user_000"].name = "changed"
    assert world.users["user_000"].name != "changed"
    exposed = engine.state
    exposed.subscriptions["sub_000"].seats = 100
    assert engine.state.subscriptions["sub_000"].seats != 100
    assert reset_world(engine) == world
    assert len(engine.trace.events) == 1
    assert clone.state != world


def test_request_context_is_copied(world):
    engine = WorldEngine(world)
    op = operation("cancel_renewal")
    req = make_request("r", "user_000", op)
    engine.register_request(req)
    req.authorized_operations.clear()
    event = execute_operation(engine, "user_000", op, "r")
    assert event.payload["result"]["success"]
    with pytest.raises(ValueError, match="already registered"):
        engine.register_request(req)


@pytest.mark.parametrize("params", [{"seats": True}, {"seats": "3"}, {"seats": 3.5}, {"seats": -1}, {"seats": 4, "extra": 1}])
def test_technical_parameter_validation(params):
    with pytest.raises(ValidationError):
        Operation(name="change_seats", org_id="org_000", resource_id="sub_000", params=params)


def test_invalid_initial_world_and_naive_clock_rejected(world):
    world.subscriptions["sub_000"].seats = 100
    with pytest.raises(ValueError):
        WorldEngine(world)
    with pytest.raises(ValidationError):
        GeneratorConfig(clock=world.clock.replace(tzinfo=None))


def test_budget_and_finalization(world):
    engine = WorldEngine(world, operation_budget=1)
    execute(engine, operation("cancel_renewal"))
    with pytest.raises(ValueError, match="budget"):
        engine.execute("user_000", operation("cancel_renewal"), "r")
    trace = engine.finish()
    assert engine.finish() == trace
    with pytest.raises(ValueError, match="finalized"):
        engine.register_request(make_request("r2", "user_000", operation("cancel_renewal")))


def test_mutated_input_parameters_are_revalidated(world):
    engine = WorldEngine(world)
    op = operation("change_seats", seats=4)
    engine.register_request(make_request("r", "user_000", op))
    op.params["seats"] = "not an integer"
    with pytest.raises(ValidationError):
        engine.execute("user_000", op, "r")
