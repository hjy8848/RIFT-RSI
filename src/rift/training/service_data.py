"""Construct Service requests from native replay-verified episodes and a trusted recorder."""
from rift.schemas import Outcome, Tokens
from rift.benchmark.private_evaluation import public_business_outcome
from rift.training.contracts import UpdateRequest
from rift.training.records import service_rollout

def prepare_service_update(package_id, certified_tasks, episodes, checkpoint, recorder, *, algorithm='policy_gradient'):
    from usami_v2.runtime.runner import replay_episode
    if not certified_tasks: raise ValueError('certified tasks required')
    episodes = tuple(episodes)
    episode_ids = [episode.record_hash for episode in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError('duplicate Service episode')
    rows = []
    for episode in episodes:
        task_id = episode.task.record_hash
        if task_id not in certified_tasks: raise ValueError('episode is not from a certified package task')
        verified = replay_episode(episode).verified
        if not verified: raise ValueError('episode replay failed')
        outcome = public_business_outcome(episode, verified)
        if outcome is Outcome.UNKNOWN: continue
        calls = [c for c in episode.model_calls if c.role == 'service']
        if not calls or recorder.verify_source(episode, checkpoint) is not True:
            raise ValueError('real Service model provenance required')
        for call in calls:
            tokens = recorder.tokens_for_call(call, checkpoint)
            if not isinstance(tokens, Tokens): raise ValueError('verified tokens required')
            rows.append(service_rollout(tokens, checkpoint, task_id, episode.record_hash, outcome, response_id=call.record_hash))
    if not rows: raise ValueError('no usable verified Service rollouts')
    if {row.task_id for row in rows} != set(certified_tasks):
        raise ValueError('every certified task must have a usable Service rollout')
    return UpdateRequest(checkpoint, package_id, tuple(rows), algorithm)

def prepare_service_trajectory_update(package_id, certified_tasks, episodes, checkpoint, capture_store, *,
                                     algorithm='grpo', allow_partial_groups=False):
    """Build whole-episode GRPO samples from verified, generation-time captures."""
    from usami_v2.runtime.runner import replay_episode
    if algorithm.lower() != 'grpo':
        raise ValueError('whole-episode Service training is configured for GRPO')
    episodes = tuple(episodes)
    if not certified_tasks or len({e.record_hash for e in episodes}) != len(episodes):
        raise ValueError('certified tasks and distinct episode provenance required')
    grouped, expected_sizes = {}, {}
    for episode in episodes:
        task_id = episode.task.record_hash
        if task_id not in certified_tasks: raise ValueError('episode is not from a certified package task')
        verified = replay_episode(episode).verified
        if not verified: raise ValueError('episode replay failed')
        outcome = public_business_outcome(episode, verified)
        if capture_store.verify_trajectory(episode, checkpoint) is not True:
            raise ValueError('generation-time trajectory evidence failed verification')
        tokens = capture_store.trajectory_for(episode, checkpoint)
        group = capture_store.group_for(episode, checkpoint)
        calls = [c for c in episode.model_calls if c.role == 'service']
        request = calls[0].request if calls else {}
        group_size = request.get('group_size')
        if (not group or group != request.get('group_id') or type(group_size) is not int or group_size < 2
                or any(c.request.get('group_id') != group or c.request.get('group_size') != group_size for c in calls)):
            raise ValueError('Service GRPO group identity and expected size must be recorded before sampling')
        if group in expected_sizes and expected_sizes[group] != group_size:
            raise ValueError('conflicting expected Service group sizes')
        expected_sizes[group] = group_size
        grouped.setdefault(group, []).append((episode, task_id, outcome, tokens, group_size))
    rows = []
    for group, members in grouped.items():
        if len(members) != expected_sizes[group]:
            raise ValueError('incomplete Service GRPO group; resample or discard the whole group')
        if len({item[1] for item in members}) != 1:
            raise ValueError('a Service GRPO group contains multiple certified tasks')
        if any(item[2] is Outcome.UNKNOWN for item in members):
            continue
        for episode, task_id, outcome, tokens, _ in members:
            rows.append(service_rollout(tokens, checkpoint, task_id, episode.record_hash, outcome,
                group, response_id=episode.record_hash))
    if (not rows or (not allow_partial_groups
                     and {row.task_id for row in rows} != set(certified_tasks))):
        raise ValueError('every certified task needs a complete known-outcome GRPO group')
    return UpdateRequest(checkpoint, package_id, tuple(rows), algorithm)

def prepare_package_service_update(package, task_registry, episodes, checkpoint, capture_store):
    """Prepare complete known-outcome GRPO groups from one frozen mixed package.

    A task whose outcomes are UNKNOWN is dropped as a whole GRPO group by the
    underlying trajectory builder. The package records the realized source mix
    from the remaining verified groups.
    """
    from rift.training.package import TrainingPackage
    if not isinstance(package, TrainingPackage):
        raise TypeError('frozen TrainingPackage required')
    refs = set(package.task_refs)
    records = {ref: task_registry.resolve_record(ref) for ref in refs}
    sources = {row.task_ref: row for row in package.task_sources}
    if any(record.identity != sources[ref].identity
            or record.certificate.record_hash != sources[ref].certificate_hash
            for ref, record in records.items()):
        raise ValueError('package TaskRefs differ from certified task identities')
    request = prepare_service_trajectory_update(package.package_id, refs, episodes,
        checkpoint, capture_store, algorithm='grpo', allow_partial_groups=True)
    return request, package.record_service_request(request)
