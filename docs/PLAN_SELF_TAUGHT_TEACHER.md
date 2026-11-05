Status: active plan — implemented, not yet run on GPU
Owner: RIFT-RSI maintainers
Last verified: 2026-11-05
Scope: shared-weight Self-Taught Teacher: one model writes certified task plans and learns from how the Solver does on them

# Self-Taught Teacher

Supersedes decision 1 (two weight sets) and section 2.5 (learnability reward on a separate Designer) of
[PLAN_DESIGNER_STAGE.md](PLAN_DESIGNER_STAGE.md). The rest of that file stays valid.

## Adopted design

One Qwen3-4B plays both roles. Each *action* starts from weights on disk:

1. The policy, prompted with recent TRAIN-pool failures, writes `groups x plans` plans (default 2 x 8).
2. The environment certifies them (`designer.stage0.certify_designer_plan`). Copies of prompt examples and
   decision structures already seen (this action or earlier) are set aside.
3. The Solver trains `steps` (K = 4) GRPO steps on the new tasks mixed with TRAIN tasks.
4. **Each plan is rewarded with the mean learnability `4p(1-p)` of its selected tasks**, p being the Solver's success
   rate on the task in those training samples. Invalid or copied plans get `invalid_reward` (-0.1); repeated
   structures get 0; plans whose tasks were never trained on are left out.
5. The policy is updated on the plans (`teacher_grpo`): off-policy GRPO, the plans of one prompt form a group,
   PPO ratio against the *sampling* policy's own log-probabilities. The result is written to disk as the next fork.

**Why the dense reward trains the Teacher (changed from the first draft).** The first draft rewarded a whole action
with one number, the difference between a Teacher arm and a random arm on frozen probes (`R_dual`). Thirty
actions would give the Teacher about thirty noisy scalars. The per-plan reward gives dozens per action. `R_dual`
is kept, but as calibration: every `--calibrate-every` actions a random-task control arm is trained from the same
fork and both arms are scored on the probes (`matched_arms`), so the log shows whether the dense reward tracks
real learning. It does not enter the update.

## Where the code is

| Piece | File |
| --- | --- |
| Probes, dual-bank score, paired bootstrap, fork-and-restore runner | `src/rift/runtime/matched_arms.py` |
| `ExperimentContext.restore/resync`, `RoundBackend(persist=, rollout_verifier=, mini_batch_size=)`, `power-check`, `self-teach` commands | `src/rift/runtime/curriculum_run.py` |
| Plan generation, certification, per-plan reward, update request, ledger and verifier | `src/rift/runtime/teacher.py` |
| The action loop, fork pruning | `src/rift/runtime/self_teach.py` |
| `Ascend910B3RoleRuntime.run_teacher` (one sleep after a whole batch of plan generations) | `src/rift/runtime/ascend_910b3.py` |
| `teacher_grpo`, `Rollout.behavior_logprobs`, `UpdateRequest.behavior_policies` | `src/rift/training/contracts.py` |
| `teacher_grpo` branch (behavior log-probs as `old_log_probs`, drift metrics) | `src/rift/verl/actor_backend.py` |
| Configs for the power check | `configs/experiments/power_{train,probes,arms}.json` |
| Tests | `tests/test_matched_arms.py`, `test_teacher.py`, `test_self_teach.py` |

## Verified and not verified

Verified on CPU: the pure logic, the contracts, the loop orchestration with fakes, certification of the pools
(496 tasks, no overlap), the whole suite (516 passed).

**Not verified on GPU, each can break the first run:**

- Whether `structured_outputs={'json': schema}` reaches vLLM through verl's client. If not, run `--unconstrained`:
  unparseable plans then score as invalid.
- Whether the log-probabilities vLLM returns are the raw (unprocessed) ones. With constrained decoding they must be
  for the PPO ratio to mean anything. Also the samples come from the constrained distribution, not the policy's, so the
  estimator is biased by an unknown amount.
- The `teacher_grpo` branch of `VerlGRPOBackend.train` on a real FSDP worker.
- Cost per action: not measured. The earlier estimate of one hour is probably low; measure it in the first run.

## Known limits

- `--drop-references delivery_age` removes the whole near-distractor F1 class from both TRAIN and probes (it is 0 of
  16 in each) until the renderer fix (v3, "Today is ...") lands. That class is the Service's weakest.
- Resuming a half-finished run is not supported; use a new run root. `--keep-checkpoints 0` (default) deletes nothing
  and each fork is about 23 GB.
- The probe pool keeps 50 / 43 / 47 tasks of F6 / F7 / F9 after dropping delivery_age. If the Solver masters more of
  them, `select_probes` will say the pool is too small; enlarge those classes in `power_probes.json`.

## Run order

1. `power-check` (3 seeds): the noise floor of `R_dual` and the effect of a Teacher arm. Two random arms give the floor.
   If a Teacher arm cannot be told from a random one at K = 4, raise K to 8 or the probes to 96 x 8 before relying on
   the calibration number.
2. `self-teach --actions 3 --calibrate-every 1` on one GPU host: checks the three unverified items above.
3. `self-teach --actions 30` with 3 seeds.

```bash
python -m rift.runtime.curriculum_run self-teach \
  --service-model /home/shenyl/hf/model/Qwen/Qwen3-4B --run-root $RUNS/self-teach-s1 \
  --verl-config-dir configs/verl --train-specs configs/experiments/power_train.json \
  --probe-specs configs/experiments/power_probes.json --drop-references delivery_age \
  --actions 3 --calibrate-every 1 --init-from-run $RUNS/rift-base-eval-1gpu-20261104a --seed 2000
```
