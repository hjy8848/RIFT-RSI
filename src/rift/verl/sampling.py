"""Deterministic per-rollout seeds for valid, independent policy groups."""
import hashlib


LOGPROB_FLOOR = -1.0e4


def finite_logprobs(values):
    """Clamp -inf to a finite floor; NaN or +inf still fails loudly.

    vLLM can report -inf for a sampled token (e.g. one cut by top-k/top-p). The
    generation ledger forbids non-finite JSON and a -inf would turn the policy
    gradient into NaN, so one such token must not abort a whole 352-episode batch.
    """
    if values is None:
        return None
    result = []
    for value in values:
        value = float(value)
        if value != value or value == float('inf'):
            raise ValueError('rollout log probability is NaN or +inf')
        result.append(max(value, LOGPROB_FLOOR))
    return result


def rollout_seed(base_seed, rollout_id, turn=0):
    if type(base_seed) is not int or base_seed < 0 or not rollout_id or type(turn) is not int or turn < 0:
        raise ValueError('nonnegative base seed, rollout identity and turn are required')
    material = f'rift-verl-rollout-seed-v1:{base_seed}:{rollout_id}:{turn}'.encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], 'big') & 0x7fffffff
