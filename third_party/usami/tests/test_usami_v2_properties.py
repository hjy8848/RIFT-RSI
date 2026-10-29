"""Property/state-machine checks use real commerce transitions and native Replay."""
from datetime import timedelta

from hypothesis import given, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from test_usami_v2_commerce import world, execute, prepared, accept
from usami_v2.commerce import adapter
from usami_v2.commerce.models import ExternalEvent, OrderLine, unit_credit
from usami_v2.commerce.external import apply_event, reference_event


@settings(max_examples=100, deadline=None, database=None)
@given(quantity=st.integers(1, 30), cents=st.integers(0, 10000000),
       first_size=st.integers(0, 30))
def test_integer_allocation_conserved_under_arbitrary_partial_partition(quantity, cents, first_size):
    line = OrderLine(id="line", order_id="order", sku_id="sku", quantity=quantity, allocated_paid_cents=cents)
    cut = min(quantity, first_size)
    left, right = tuple(range(cut)), tuple(range(cut, quantity))
    assert unit_credit(line, left) + unit_credit(line, right) == cents
    assert sum(unit_credit(line, (i,)) for i in range(quantity)) == cents


@settings(max_examples=24, deadline=None, database=None)
@given(quantity=st.integers(1, 5), cents=st.integers(0, 50000))
def test_actual_partial_refund_sequences_conserve_original_payment(quantity, cents):
    value = world()
    value.lines["B"].quantity = quantity
    value.lines["B"].allocated_paid_cents = cents
    value.orders["order"].paid_cents = 20000 + cents
    for _ in range(quantity):
        value, cid, qid = prepared(value, line="B", remedy="refund")
        value = accept(value, qid)
        value, result, report, _ = execute(value, "commit_accepted_quote", qid)
        assert result["success"] and report["status"] == "pass"
        assert adapter.invariants(value) == []
    assert sum(row.amount_cents for row in value.ledger.values() if row.line_id == "B") == cents
    assert len({i for row in value.settlements.values() for i in row.unit_indices}) == quantity


class InventoryCommitmentMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.value = world()
        self.value.skus["original"].return_required = True
        self.value, self.cid, self.qid = prepared(self.value, pickup="morning")
        self.next_event = 0

    @precondition(lambda self: self.value.cases[self.cid].status == "verified"
                  and self.value.inventory["D_fast"].available > 0)
    @rule()
    def prepare(self):
        self.value, result, report, _ = execute(self.value, "prepare_remedy_quote", self.cid, {
            "remedy": "exchange", "target_sku_id": "D", "warehouse_id": "fast", "pickup_slot": "morning"})
        assert result["success"] and report["status"] == "pass"
        self.qid = result["data"]["quote"]["id"]

    @precondition(lambda self: self.value.cases[self.cid].status == "verified"
                  and self.value.quotes[self.qid].status == "prepared")
    @rule()
    def commit_or_observe_stale_quote(self):
        branch = accept(self.value, self.qid)
        self.value, result, report, _ = execute(branch, "commit_accepted_quote", self.qid)
        assert report["status"] == "pass"
        if not result["success"]:
            assert result["error_code"] in ("QUOTE_STALE", "INSUFFICIENT_STOCK")

    @precondition(lambda self: self.value.cases[self.cid].status == "awaiting_return")
    @rule()
    def cancel(self):
        self.value, result, report, _ = execute(self.value, "cancel_pending_commitment", self.qid)
        assert result["success"] and report["status"] == "pass"

    @rule(delta=st.integers(-1, 2))
    def external_stock(self, delta):
        inv = self.value.inventory["D_fast"]
        if inv.available + delta < 0:
            return
        event_id = "stock_" + str(self.next_event)
        self.next_event += 1
        event = ExternalEvent(id=event_id, kind="stock_adjustment", target_id=inv.id,
                              at=self.value.clock + timedelta(seconds=1), integer_value=delta)
        self.value.external_events[event_id] = event
        expected = reference_event(self.value, event_id)
        self.value = apply_event(self.value, event_id)
        assert self.value == expected

    @invariant()
    def stock_ledger_and_purchase_units_remain_consistent(self):
        assert adapter.invariants(self.value) == []
        inv = self.value.inventory["D_fast"]
        assert inv.opening_quantity + inv.adjustments == inv.available + inv.reserved + inv.shipped
        assert not self.value.ledger and not self.value.settlements


TestInventoryCommitment = InventoryCommitmentMachine.TestCase
TestInventoryCommitment.settings = settings(max_examples=25, stateful_step_count=18,
                                            deadline=None, database=None)
