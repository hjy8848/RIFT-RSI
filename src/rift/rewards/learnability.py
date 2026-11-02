"""Dense, source-separated statistics from complete Service GRPO groups."""
from dataclasses import dataclass
from math import isfinite

from rift.rewards.learning_value import InsufficientEvidenceError
from rift.training.contracts import UpdateRequest
from rift.training.package import TrainingPackage


@dataclass(frozen=True)
class TaskLearnability:
    task_ref: str
    source: str
    successes: int
    group_size: int
    learnability: float

    @property
    def informative(self):
        return 0 < self.successes < self.group_size


@dataclass(frozen=True)
class LearnabilityStats:
    tasks: tuple[TaskLearnability, ...]
    designer_mean: float
    benchmark_mean: float | None
    designer_informative_fraction: float
    designer_valid_groups: int
    designer_expected_groups: int

    @property
    def scalar(self):
        return self.designer_mean


def score_learnability(package, request, *, min_valid=0.5):
    """Compute from the frozen allocation; absent groups count against coverage."""
    if not isinstance(package, TrainingPackage) or not isinstance(request, UpdateRequest):
        raise ValueError("a training package and Service update request are required")
    if (request.package_id != package.package_id or request.policy_before != package.service_checkpoint_before
            or request.algorithm.lower() != "grpo"):
        raise ValueError("request does not match the frozen package")
    if not isinstance(min_valid, (int, float)) or not isfinite(min_valid) or not 0 < min_valid <= 1:
        raise ValueError("min_valid must be in (0, 1]")
    groups = {group.group_id: group for group in package.task_groups}
    observed = {}
    for rollout in request.rollouts:
        group = groups.get(rollout.group_id)
        if group is None or rollout.task_id != group.task_ref or rollout.reward not in (0, 1):
            raise ValueError("rollout does not match a package group or binary outcome")
        observed.setdefault(group.group_id, []).append(rollout)
    rows = []
    for group in package.task_groups:
        rollouts = observed.get(group.group_id)
        if rollouts is None:
            continue
        if len(rollouts) != group.group_size or len({r.episode_id for r in rollouts}) != group.group_size:
            raise ValueError("incomplete or duplicate Service group")
        k = sum(int(r.reward) for r in rollouts)
        p = k / group.group_size
        rows.append(TaskLearnability(group.task_ref, group.source, k, group.group_size, 4 * p * (1 - p)))
    designer = [row for row in rows if row.source == "designer"]
    benchmark = [row for row in rows if row.source == "benchmark"]
    expected = sum(group.source == "designer" for group in package.task_groups)
    if not expected or len(designer) / expected < min_valid:
        raise InsufficientEvidenceError("Designer group coverage is below the configured floor")
    return LearnabilityStats(tuple(rows), sum(r.learnability for r in designer) / len(designer),
        sum(r.learnability for r in benchmark) / len(benchmark) if benchmark else None,
        sum(r.informative for r in designer) / len(designer), len(designer), expected)


class LearnabilityVerifier:
    """Compare supplied statistics with a trusted package and update request."""
    def __init__(self, *, min_valid=0.5):
        self.min_valid = min_valid

    def verify_reward(self, stats, package, request):
        try:
            return isinstance(stats, LearnabilityStats) and stats == score_learnability(
                package, request, min_valid=self.min_valid)
        except (ValueError, TypeError, AttributeError):
            return False
