"""Exact per-decision capture during a VERL continuous-token rollout."""
from dataclasses import dataclass
from math import isfinite
from rift.schemas import Checkpoint, Tokens

@dataclass(frozen=True)
class CapturedDecision:
    sequence: int
    tokens: Tokens
    logprobs: tuple[float, ...] | None
    checkpoint: Checkpoint

class ContinuousCapture:
    def __init__(self, initial_prompt, checkpoint):
        self.initial_prompt = tuple(initial_prompt)
        if not self.initial_prompt: raise ValueError('empty initial prompt')
        self.checkpoint = checkpoint
        self.current = self.initial_prompt
        self.mask = ()
        self.decisions = []

    def add_generation(self, actual_prompt, output_ids, output_logprobs, merged_ids, merged_mask):
        prompt, action = tuple(actual_prompt), tuple(output_ids)
        merged, mask = tuple(merged_ids), tuple(merged_mask)
        if prompt != self.current or not action: raise ValueError('generation did not use the captured prefix')
        if merged != prompt + action: raise ValueError('continuous merge changed generated token identity')
        if mask != self.mask + (1,) * len(action): raise ValueError('generation mask differs from actual action tokens')
        logs = None if output_logprobs is None else tuple(output_logprobs)
        if logs is not None and (len(logs) != len(action) or any(not isfinite(x) or x > 0 for x in logs)):
            raise ValueError('generation log probabilities are invalid')
        self.decisions.append(CapturedDecision(len(self.decisions), Tokens(prompt, action, (True,) * len(action)), logs, self.checkpoint))
        self.current, self.mask = merged, mask

    def add_context(self, merged_ids, merged_mask):
        merged, mask = tuple(merged_ids), tuple(merged_mask)
        if merged[:len(self.current)] != self.current: raise ValueError('context merge rewrote previous tokens')
        added = len(merged) - len(self.current)
        if mask != self.mask + (0,) * added: raise ValueError('environment tokens must be masked')
        self.current, self.mask = merged, mask

    def output(self):
        response = self.current[len(self.initial_prompt):]
        if not self.decisions or len(response) != len(self.mask): raise ValueError('incomplete trajectory')
        return Tokens(self.initial_prompt, response, tuple(bool(x) for x in self.mask))
