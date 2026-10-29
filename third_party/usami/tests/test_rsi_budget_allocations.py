"""Whole-group reservations cannot be borrowed by root or inner evaluations."""
import pytest

from rsi_core.budget import Ledger
from rsi_core.protocol import BudgetExceeded, Limits


def test_group_preflight_is_atomic_and_runs_no_sampling():
    ledger = Ledger(Limits(episodes=5))
    started = []
    with pytest.raises(BudgetExceeded):
        with ledger.reserve_group({"root": {"episodes": 2}, "reward": {"episodes": 4}}):
            started.append(True)
    assert not started
    assert ledger.used["episodes"] == ledger.reserved["episodes"] == 0


def test_spent_allocation_preserves_other_phases_and_only_restores_unused():
    ledger = Ledger(Limits(episodes=10))
    with ledger.reserve_group({"root": {"episodes": 3}, "reward": {"episodes": 5}}) as group:
        assert ledger.remaining("episodes") == 2
        with group["root"].spend():
            assert ledger.remaining("episodes") == 3
            ledger.charge("root/child", episodes=2)
            assert ledger.remaining("episodes") == 1
        assert ledger.reserved["episodes"] == 6
        with group["root"].spend():
            ledger.charge("root/second-child", episodes=1)
        with group["reward"].spend():
            ledger.charge("evaluator/solver", episodes=5)
    assert ledger.used["episodes"] == 8
    assert ledger.reserved["episodes"] == 0
    with pytest.raises(ValueError, match="closed"):
        with group["reward"].spend():
            pass


def test_allocation_cannot_borrow_free_root_balance_after_overrun():
    ledger = Ledger(Limits(episodes=100))
    with ledger.reserve_group({"root": {"episodes": 2}, "reward": {"episodes": 5}}) as group:
        with pytest.raises(BudgetExceeded, match="allocation exceeded"):
            with group["root"].spend():
                ledger.charge("root", episodes=3)
        # Actual expenditure remains recorded even when the caller exceeded it.
        assert ledger.used["episodes"] == 3
        assert ledger.reserved["episodes"] == 5
        with pytest.raises(BudgetExceeded, match="already exceeded"):
            with group["root"].spend():
                pass
        with group["reward"].spend():
            ledger.charge("reward", episodes=5)
    assert ledger.used["episodes"] == 8 and ledger.reserved["episodes"] == 0


def test_unallocated_resource_and_nested_reservation_are_rejected():
    ledger = Ledger(Limits())
    with ledger.reserve(episodes=2) as allocation:
        with allocation.spend():
            with pytest.raises(ValueError, match="before starting"):
                with ledger.reserve(episodes=1):
                    pass
            with pytest.raises(BudgetExceeded, match="allocation exceeded"):
                ledger.charge("hidden-model-call", model_calls=1)
    assert ledger.used["model_calls"] == 1
    assert all(value == 0 for value in ledger.reserved.values())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1])
def test_invalid_usage_cannot_poison_budget(value):
    ledger = Ledger(Limits())
    with pytest.raises(ValueError, match="measured usage"):
        ledger.charge("candidate", cpu_seconds=value)
    assert not ledger.events
