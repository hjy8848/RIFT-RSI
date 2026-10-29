"""Deterministic per-rollout seeds for valid, independent policy groups."""
import hashlib


def rollout_seed(base_seed, rollout_id, turn=0):
    if type(base_seed) is not int or base_seed < 0 or not rollout_id or type(turn) is not int or turn < 0:
        raise ValueError('nonnegative base seed, rollout identity and turn are required')
    material = f'rift-verl-rollout-seed-v1:{base_seed}:{rollout_id}:{turn}'.encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], 'big') & 0x7fffffff
