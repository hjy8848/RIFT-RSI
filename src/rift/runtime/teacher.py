"""The shared-weight Teacher: one model writes plans, the Solver trains on them, the plans are scored.

The same Qwen weights that solve tasks also write task plans (``rift.designer.format``) when
they are prompted with the Solver's training failures. A plan only reaches training after the
environment has certified it, so the Teacher cannot reward itself with impossible tasks.

Reward (dense, per plan): the mean learnability ``4p(1-p)`` of the plan's certified tasks, where
p is the Solver's success rate on the task in the training samples that used it. A plan the
environment rejects, or one copied from a prompt example, scores ``invalid_reward``; a plan whose
decision structure was already written (this round or earlier) scores 0. Plans written from one
prompt form a GRPO group, so each plan is judged against its siblings. The matched-arm
reward in ``matched_arms`` is not used for the update: it is the slow, noisy check that this
dense proxy really tracks learning.

Plans are sampled before the update, so the update is off-policy by the few Solver steps in
between: it uses the sampling policy's own log-probabilities (``behavior_logprobs``) in the PPO
ratio. See docs/PLAN_SELF_TAUGHT_TEACHER.md.
"""
import asyncio
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from rift.schemas import Checkpoint, Tokens
from rift.training.contracts import Rollout, UpdateRequest

REWARD_SOURCE = 'teacher_learnability_v1'
LOGPROB_SLACK = 1e-6          # sampled-token log-probabilities are <= 0 up to float noise


def learnability(successes, valid):
    """``4p(1-p)``: 1 at p = 0.5, 0 when the Solver always or never succeeds."""
    if valid <= 0:
        raise ValueError('learnability needs at least one valid outcome')
    p = successes / valid
    return 4.0 * p * (1.0 - p)


@dataclass
class TeacherSample:
    """One plan written by the policy, with everything the update and the audit need."""
    sample_id: str
    group_id: str
    checkpoint: Checkpoint
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    text: str
    plan_id: str
    world_seed: int
    row: dict = field(default_factory=dict)       # certification row (stage0.certify_designer_plan)

    def usable(self, copy_threshold):
        row = self.row
        return (row.get('status') in ('certified', 'partly_certified') and bool(row.get('task_refs'))
                and (row.get('copy_score') or 0) < copy_threshold)

    def selected_refs(self):
        flags = self.row.get('selected') or []
        return [ref for ref, keep in zip(self.row.get('task_refs', ()), flags) if keep]


def sample_id_for(group_id, index):
    return '%s-p%02d' % (group_id, index)


class TeacherLedger:
    """Write-once record of every sample, so an update can be checked against what was generated."""
    def __init__(self, root):
        self.root = Path(root)

    def _path(self, sample_id):
        if not sample_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'
                                for c in sample_id):
            raise ValueError('invalid Teacher sample id')
        return self.root / (sample_id + '.json')

    def record(self, sample):
        self.root.mkdir(parents=True, exist_ok=True)
        with self._path(sample.sample_id).open('x', encoding='utf-8') as stream:
            json.dump(asdict(sample), stream, sort_keys=True, ensure_ascii=False)

    def update_row(self, sample):
        """Certification and selection happen after generation; the generation fields never change."""
        path = self._path(sample.sample_id)
        stored = json.loads(path.read_text(encoding='utf-8'))
        stored['row'] = sample.row
        path.write_text(json.dumps(stored, sort_keys=True, ensure_ascii=False), encoding='utf-8')

    def get(self, sample_id):
        raw = json.loads(self._path(sample_id).read_text(encoding='utf-8'))
        raw['checkpoint'] = Checkpoint(**raw['checkpoint'])
        for key in ('prompt_ids', 'response_ids', 'behavior_logprobs'):
            raw[key] = tuple(raw[key])
        return TeacherSample(**raw)


# -- prompt and generation ----------------------------------------------------

def teacher_prompt_ids(tokenizer, failures, *, seed, examples=2, focus=True):
    """Token ids of the chat prompt that asks for one plan; ``seed`` picks the examples and the focus."""
    from rift.designer.prompt import designer_messages
    messages = designer_messages(failures, seed=seed, examples=examples, focus=focus)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    return list(tokenizer.encode(text, add_special_tokens=False))


def check_generation(output):
    """A usable generation has response ids and one finite sampled-token log-probability for each."""
    ids, logprobs = list(output.token_ids), output.log_probs
    if not ids or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError('the Teacher generation has no valid response tokens')
    if logprobs is None or len(logprobs) != len(ids):
        raise ValueError('the Teacher generation is missing per-token log-probabilities')
    if any(not isfinite(x) or x > LOGPROB_SLACK for x in logprobs):
        raise ValueError('Teacher log-probabilities must be finite and non-positive')
    return ids, [min(0.0, float(x)) for x in logprobs]


def vllm_structured_outputs(schema):
    """The object vLLM's sampling parameters take for JSON-schema decoding (a plain dict is rejected)."""
    try:
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError:                       # no vLLM in this process (unit tests)
        return {'json': schema}
    return StructuredOutputsParams(json=schema)


async def generate_group(client, prompt_ids, *, count, seed, group_id, max_tokens=4096, temperature=1.0,
                         structured_schema=None, structured_factory=vllm_structured_outputs):
    """``count`` independent plans for one prompt through the live vLLM server of the shared weights."""
    async def one(index):
        params = {'temperature': temperature, 'top_p': 1.0, 'max_tokens': max_tokens, 'logprobs': True}
        if structured_schema is not None:
            params['structured_outputs'] = structured_factory(structured_schema)
        output = await client.generate(request_id='%s-%d-%d' % (group_id, seed, index),
                                       prompt_ids=list(prompt_ids), sampling_params=params)
        return check_generation(output)
    return await asyncio.gather(*(one(i) for i in range(count)))


def make_samples(group_id, checkpoint, prompt_ids, generations, decode, *, seed):
    samples = []
    for index, (ids, logprobs) in enumerate(generations):
        sid = sample_id_for(group_id, index)
        samples.append(TeacherSample(sid, group_id, checkpoint, tuple(prompt_ids), tuple(ids), tuple(logprobs),
                                     decode(ids), 'teacher-' + sid, seed + index))
    return samples


# -- certification ------------------------------------------------------------

def certify_samples(world_kit, samples, *, prompt_seed, examples=2, copy_threshold=None, seen=None):
    """Certify every plan, flag copies and repeated decision structures; fills ``sample.row``."""
    from rift.designer.format import PlanError, extract_json
    from rift.designer.prompt import copy_score, example_plans
    from rift.designer.stage0 import COPY_THRESHOLD, certify_designer_plan, select_tasks
    threshold = COPY_THRESHOLD if copy_threshold is None else copy_threshold
    rows = []
    for sample in samples:
        row = certify_designer_plan(world_kit, sample.text, sample.plan_id, sample.world_seed, signatures=True)
        try:
            row['copy_score'] = copy_score(extract_json(sample.text), example_plans(prompt_seed, examples))
        except PlanError:
            row['copy_score'] = None
        sample.row = row
        rows.append(row)
    select_tasks(rows, threshold, seen)
    return samples


# -- reward -------------------------------------------------------------------

def plan_rewards(samples, task_counts, *, copy_threshold=None, invalid_reward=-0.1, repeat_reward=0.0):
    """``{sample_id: reward or None}``; None means its tasks were not measured, so it is left out.

    ``task_counts`` maps a task ref to ``(successes, valid)`` from the Solver samples that trained
    on it. A plan is scored by the mean learnability of its selected (new, uncopied) tasks.
    """
    from rift.designer.stage0 import COPY_THRESHOLD
    threshold = COPY_THRESHOLD if copy_threshold is None else copy_threshold
    rewards = {}
    for sample in samples:
        if not sample.usable(threshold):
            rewards[sample.sample_id] = invalid_reward
            continue
        refs = sample.selected_refs()
        if not refs:
            rewards[sample.sample_id] = repeat_reward
            continue
        measured = [learnability(*task_counts[ref][:2]) for ref in refs
                    if ref in task_counts and task_counts[ref][1] > 0]
        rewards[sample.sample_id] = sum(measured) / len(measured) if measured else None
    return rewards


# -- update request -----------------------------------------------------------

def build_teacher_update(samples, rewards, policy_before, ledger, *, world_size=1):
    """The off-policy GRPO request: every rewarded plan, in groups of plans written from one prompt.

    The FSDP world needs a batch it can split evenly, so the last plan of the largest group is
    dropped until the count divides ``world_size`` (a group never goes below two plans).
    """
    chosen = [s for s in samples if rewards.get(s.sample_id) is not None]
    by_group = {}
    for sample in sorted(chosen, key=lambda s: s.sample_id):
        by_group.setdefault(sample.group_id, []).append(sample)
    by_group = {key: group for key, group in by_group.items() if len(group) >= 2}
    while sum(map(len, by_group.values())) % world_size:
        largest = max(by_group.values(), key=len, default=[])
        if len(largest) <= 2:
            raise ValueError('cannot split the Teacher batch evenly over %d workers' % world_size)
        largest.pop()
    chosen = [s for group in by_group.values() for s in group]
    if not chosen:
        raise ValueError('no Teacher group has two rewarded plans to learn from')
    for sample in chosen:
        ledger.update_row(sample)
    rollouts = tuple(Rollout(
        tokens=Tokens(prompt=s.prompt_ids, response=s.response_ids, mask=(True,) * len(s.response_ids)),
        policy=s.checkpoint, task_id=s.group_id, episode_id=s.sample_id, reward=float(rewards[s.sample_id]),
        reward_source=REWARD_SOURCE, group_id=s.group_id, response_id=s.sample_id, reward_ref=s.sample_id,
        behavior_logprobs=s.behavior_logprobs) for s in chosen)
    behaviors = tuple(dict.fromkeys(s.checkpoint for s in chosen))
    package = sha256(('|'.join(sorted(s.sample_id for s in chosen)) + '@' + policy_before.weights_digest
                      ).encode()).hexdigest()[:32]
    return UpdateRequest(policy_before, 'teacher-' + package, rollouts, 'teacher_grpo',
                         reward_protocol='teacher_learnability_v1', behavior_policies=behaviors)


class TeacherRolloutVerifier:
    """Checks an update row against the sample the ledger wrote when the plan was generated."""
    def __init__(self, ledger):
        self.ledger = ledger

    def verify_rollout(self, row, policy):
        try:
            sample = self.ledger.get(row.response_id)
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return (sample.checkpoint == row.policy and sample.prompt_ids == row.tokens.prompt
                and sample.response_ids == row.tokens.response
                and sample.behavior_logprobs == row.behavior_logprobs and sample.group_id == row.group_id)

    def verify_group(self, group, policy):
        return len({r.tokens.prompt for r in group}) == 1 and len({r.policy for r in group}) == 1
