Status: active research protocol
Owner: RIFT-RSI maintainers
Last verified: 2026-10-29
Scope: Designer learning value from Service capability acquisition and retention

This research specification decides how RIFT measures whether Designer-generated training packages teach Service new, independently tested abilities without erasing established abilities. The implementation boundary and versioned data protocol are in [DUAL_BANK_DESIGN.md](DUAL_BANK_DESIGN.md) and `src/rift/`.

## Research question and motivation

RIFT studies a two-model learning system. Service learns from real multi-turn Usami Customer, Engine, and Tool interactions. Designer observes allowlisted public Failure Bank records, explicitly chooses an existing `ability_id` or `NEW`, and proposes structured Usami tasks. Host records only the selected ID and task references; it does not infer semantic ability matches. Designer receives credit only after Service has actually updated its full parameter set. The research question is whether Designer can choose tasks that improve independent Recovery success while preserving independent Retention success.

The primary reward is the change in task-averaged business success on frozen probes:

`G = SR(θ_after, Recovery) - SR(θ_before, Recovery)`

`H = SR(θ_after, Retention) - SR(θ_before, Retention)`

`R_dual = G - λ max(0, -H)`

This directly rewards acquired capability and penalizes forgetting. It does not reward a task for being difficult, for producing a failure, or for being successfully generated. The Recovery and Retention tasks are independently sampled, random-frozen before the Service update, and replayed with the same task, seeds, checkpoint-relative protocol, and business-outcome definition before and after. They are excluded from the Service training package. UNKNOWN or invalid evidence is not treated as failure; if valid coverage falls below the configured floor, the update receives no scalar Designer reward.

The legacy Replay LogP reward remains available as an explicit experimental comparison: it measures change in the probability of observed action tokens on a frozen replay set. It is not the primary measure of new ability because likelihood gain does not establish that a complete task can be solved, and successful trajectories may include poor or unnecessary actions. Neither proxy replaces independent final-test rollouts.

`AbilityRegistry` records Designer-selected ability hypotheses, explicit Failure Bank references, and later package/update/Probe history. A new ability gets a stable Host ID. An existing ability can be reused and its links explicitly revised. Recovery samples linked historical failures first, then fills from other eligible failures; a prior Designer exposure does not disqualify a task. Recovery still excludes this round's Service training tasks and real data overlap. Every historical candidate needs a trusted baseline at the current Service checkpoint. These associations change Probe selection only; they add no ability-count, confidence, or difficulty term to `R_dual` and add no extra Probe.

## Learning flow and information boundary

Bootstrap evaluates trusted task instances under the initial Service checkpoint and records task identity, `BANK_CANDIDATE` purpose, protocol, seed, evidence level, final business outcome, and checkpoint. Failure Bank entries need no ability label and remain available until selected explicitly. SuccessBank tracks mastered candidates; UNKNOWN observations stay unresolved. Designer sees only an opaque failure TaskRef and allowlisted `PublicSummary`; private identity, evidence, hidden Oracle labels, and Probe membership stay Host-side. A failure task previously shown to Designer may later be scored.

Designer makes one on-policy structured choice containing the selected ability ID, hypothesis, diagnosis TaskRefs, one native Usami `ScenarioDraft`, and multiple native `AbstractChallenge` records. Host builds and verifies one World, then independently grounds and certifies each challenge in it. Certified Designer tasks are mixed at the configured ratio with a reproducibly sampled set from a pre-frozen Usami TRAIN pool; the default package has 8 tasks from each source. One GRPO group belongs to each TaskRef. UNKNOWN removes a complete group, and package preparation rejects a realized mix that no longer matches its configured ratio. Service executes the resulting package through the existing VERL 0.9.1 multi-turn Agent Loop and receives only verified final business reward. Before the update, Host freezes an independent Recovery/Retention pair, using a fixed seed and current-checkpoint baselines. The package's original Designer action is held until its Service update and dual-bank probes are verified. This delayed package-level reward updates Designer once.

Both policies use full-parameter RL. Service keeps GRPO with complete groups of independent rollouts for the same exact task/prompt and policy. Sequential dual-bank rewards from changing Service checkpoints are not valid GRPO comparisons; Designer therefore uses ungrouped REINFORCE with a configured scalar baseline and no batch whitening in this version. The previous Replay LogP path retains its own reward protocol and grouping rules.

The Host may inspect private Usami worlds, certification records, Reference witnesses, and Oracle evaluations. These stay outside model input and reward calculation. The usable reward field is the replay-verified final `business_goal_success`, not `strict_task_success` or process diagnostics. Tool failures remain public observations for the Service and do not become hidden training labels. Usami remains the benchmark; RIFT owns the outer bank management, policy updates, reward attribution, and round schedule.

## Reward comparisons

| Signal | What it measures | Role in this study | Main limitation |
|---|---|---|---|
| Dual-bank business outcome | Independent Recovery success gain with a Retention regression penalty | Primary Designer objective | Requires valid before/after real Usami episodes and frozen task identities |
| Replay LogP change | Probability shift on demonstrated action tokens | Preserved legacy baseline | Action likelihood is not whole-task capability |
| Difficulty or failure rate | Task challenge or current weakness | Diagnostic-only comparator | Does not measure learning value |
| Independent TEST success | End-of-study task performance | Final evaluation only | Must never flow back into Designer reward or bank selection |

## Prior work and evaluation

[R-Zero](https://arxiv.org/abs/2508.05004) uses a challenger/solver curriculum with difficulty-oriented feedback. [Teacher-Student Curriculum Learning](https://arxiv.org/abs/1707.00183) and [Learning to Teach](https://arxiv.org/abs/1805.03643) optimize teaching from learner feedback. RIFT's intended distinction is delayed package credit from independent task-success changes paired with an explicit retention set. This is a research hypothesis, not an established novelty or result.

Compare dual-bank Designer with matched random task selection, difficulty-only selection, learning-progress selection, and the retained Replay LogP objective under equal Service rollout, optimizer-update, probe, and evaluation budgets. Report Recovery and Retention task-averaged success separately, UNKNOWN/invalid rates, reward uncertainty, and independent TEST results. Ablate `λ`, probe size, seed policy, trial budget, split policy, and delayed-reward baseline. Diagnose whether dual-bank reward predicts later capability transfer and forgetting; do not tune against final TEST results.

## Evidence and implementation boundary

The pinned Usami source is `third_party/usami` at commit `fc8014690e4461e176ba5da7d8122f4cb99c961a`. The selected VERL source is v0.9.1 commit `1876b06d0a3e4e71e06230be10af14492ca8a75b`. `configs/dual_bank.yaml` records the current experiment defaults. CPU tests exercise native Usami World build, multi-challenge grounding and certification, native episode cloning/replay, package mixing and state recovery. Mock capture and verifier paths cover the Service request boundary only. No actual model training, capability gain, retention result, or Ascend compatibility has been demonstrated.
