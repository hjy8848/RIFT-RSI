# RIFT 双池能力获得与能力保持设计

Status: active design
Owner: RIFT-RSI maintainers
Last verified: 2026-10-29
Scope: 双池数据协议、独立能力 Probe、Designer 延迟奖励及其与旧 Replay LogP 的兼容边界

本设计冻结 RIFT 的主学习目标与信息边界：Designer 的主奖励来自 Service 在独立 Recovery Probe 上的真实业务成功率增益，并扣除 Retention Probe 上的能力退化；成功轨迹 LogP 只保留为兼容对照和辅助诊断。代码实现位于 `src/rift/banks/`、`src/rift/probes/`、`src/rift/rewards/`、`src/rift/evolution/` 和 `src/rift/verl/`。

## 研究目标

RIFT 要训练 Designer 找到能帮助 Service 获得新能力、同时保留已有能力的训练任务。只提高历史成功动作的概率并不能证明 Service 学会了新的完整任务：成功轨迹可能包含绕路，失败任务也可能存在偶然失败。因此主要结果必须来自 Service 更新前后对同一批独立真实任务的完整 Usami Rollout，使用经过 replay 校验的最终 `business_goal_success`。`strict_task_success`、Oracle 过程诊断、witness 和逐步操作标签不得进入训练 Reward、Probe 筛选或 Designer 可见信息。

## 两个池和三种记录

| 记录 | 内容 | 用途 | 可见性 |
|---|---|---|---|
| `FailureObservation` | 失败任务 opaque `TaskRef` 与 `PublicSummary` | Designer 显式选择 ability ID/`NEW`，关联失败任务并提出新任务 | 仅显式公开投影 |
| `TaskEvidenceRecord` | Host 的 TaskRef、来源身份、统一 `BANK_CANDIDATE` 用途、Checkpoint、试次结果和可信证据引用 | Failure/Success Bank 与 Recovery/Retention Probe | Host 私有 |
| `Demonstration` | 成功 Episode 的精确上下文、Action Token 与 mask | 旧 Replay LogP 对照 | Host Scorer 使用 |

`FailureBank` 保存各 Service Checkpoint 下的真实薄弱候选，未归因任务可长期保留。Bootstrap 不要求 ability 标签；Designer 后续可显式追加 `ability_id → failure_task_refs`。Recovery 与 Failure Bank 共用任务来源，不再按 Diagnosis/Recovery 划分任务；`RECOVERY_FEEDBACK` 和 `RETENTION_FEEDBACK` 只标识冻结快照的评分角色。UNKNOWN 单独记录，不能作为失败样本。

Failure/Success 是随 Checkpoint 和协议变化的历史状态。后续任务结果可以新增 `RECOVERED` 或候选状态事件，不覆盖旧记录。任务引用、World、来源 lineage、证据引用及 Probe 具体成员都留在 Host。Designer 输入通过 allowlist 只包含公开事件和允许公开的最终业务结果；不序列化私有记录。

## Ability Registry 与 Failure Bank 的关联

`AbilityRegistry` 保存能力 ID、描述、显式失败任务引用，以及对应的 Service 更新和评测历史。Designer 每次从 Host 展示的已有 ID 中选择一个，或输出 `NEW`，并提交 hypothesis、诊断任务引用和训练任务提案。Host 校验 ID、引用是否出现在本次公开上下文、选择是否绑定原始生成记录，以及新增引用是否对应 Failure Bank 中真实 Service 失败证据；`NEW` 的稳定 ID 由 Host 分配。Host 不根据文本相似度推断归属。选中既有能力时，Designer 可明确增删关联失败任务或修订描述。Ability 状态只参考 Recovery 快照中的目标关联任务；Registry 不代表 Usami Oracle 标签或已知的真实能力本体。

Designer 的完整结构化选择必须能追溯到捕获的 on-policy 输出。Identity、EvidenceRef 和 Oracle 信息留在 Host；公开 FailureObservation 只包含 opaque TaskRef 与已允许公开的摘要。`AbilityRecord` 保留 ID、描述、失败任务引用、训练历史和评测历史，显式关联修订也记入记录。当前没有旧 Registry 迁移分支：仓库没有实际使用的旧 Registry 数据需要兼容。

Recovery 先从目标 ability 关联且当前符合资格的历史失败任务中按冻结 seed 随机抽取，不足时从其他合格失败任务随机补足，任务不重复。Designer 曾见过任务不构成排除条件。当前 Service 训练包及按 split policy 判定的实质重叠实例必须排除；最终 TEST 永不进入训练或奖励。历史失败只证明任务曾在旧 Checkpoint 下失败，不作为本轮评分基线。每个入选 TaskRef 都必须配有本轮 Service Checkpoint 与评测协议下可信的当前基线记录；缺少基线时不能进入冻结快照。快照保存目标 ability ID、优先任务引用、当前基线和内容 digest。Retention 从历史 mastered 任务中独立随机抽样，并同样使用当前 Checkpoint 基线。

Ability Registry 不改变双池 Reward 公式，不添加能力数量、置信度或难度奖励，也不增设第三类 Probe。旧 Replay LogP 继续作为单独标记的实验对照。Registry 默认持久化在 scheduler 的 state root 下 `ability-registry.json`，JSON 内容有 checksum；可通过运行时指定独立路径。配置见 [`configs/dual_bank.yaml`](../configs/dual_bank.yaml)，实现见 `src/rift/abilities/`。

## 每轮协议

1. Bootstrap 用 θ₀ 对可信任务按固定协议重复 Rollout，保存每次 `SUCCESS` / `FAILURE` / `UNKNOWN`、Checkpoint 和证据。默认只建池和基线，不训练 Service，也不生成 Designer 奖励。
2. Designer 从统一 Failure Bank 的公开失败轨迹中，选择已有 ability ID 或 `NEW`，给出 hypothesis、诊断任务引用和结构化 Usami 训练任务提案。Host 校验 ID、公开引用和原始生成记录；`NEW` 由 Host 分配稳定 ID。Grounding、Certification 和 TaskRegistry 在 Host 完成。
3. Service 更新前冻结 `RECOVERY_FEEDBACK` 与 `RETENTION_FEEDBACK`：Recovery 先随机抽取目标能力关联的合格失败任务，再随机补齐其他合格失败任务；Retention 独立随机抽取。Designer 历史上见过的失败任务允许评分。每个候选必须先取得本轮 θ_before 与协议下的可信 baseline。快照保存 split policy 版本、seed、目标 ability、优先任务引用、当前基线、协议 digest、θ_before 和内容 digest。排除本轮训练包、实质重叠实例和 TEST。
4. Probe 快照只使用 θ_before 下、同 TaskRef、同评测协议和固定 seeds 的已验证当前基线。历史 θ₀ 失败可以在后续 θ₂ 轮继续提供 Recovery 候选资格，但必须先读取或重新执行 θ₂ 的 baseline；不能把 θ₀ 的分数当作 θ₂ 基线，也不能缺失时默认成 0。
5. Service 在新训练包上通过 verl 0.9.1、Usami 多轮 Agent Loop 和全参数 GRPO 更新 θ_before→θ_after。UNKNOWN 或验证失败的 GRPO 成员使整个组失效。
6. 在相同冻结任务、seed 和协议下用 θ_after 独立重新执行 Probe。Probe seed 传入 CustomerFactory；Service 每轮生成使用由 TaskRef、seed、protocol 组成且不依赖 checkpoint 的随机种子范围，原始 sampling 参数写入生成证据。保留新的 Episode 和 replay 证据。任务先算自己的有效试次成功率，再对任务等权平均；UNKNOWN 不计成功也不计失败，但超出有效覆盖阈值则拒绝 Reward。
7. `G = SR(θ_after, F) - SR(θ_before, F)`，`H = SR(θ_after, S) - SR(θ_before, S)`，`R = G - λ max(0, -H)`。独立保存 G、H、R、有效率、试次和 evidence refs。
8. 把 `R` 绑定 Designer 原始 on-policy Token、结构化能力选择、包 ID、目标 ability、Service update、两个冻结 Probe、Checkpoint 和预算。对应 ability 追加真实更新与 Recovery/Retention 评测历史。默认采用 `sequential_curriculum`：每个包从当前 θ 更新，不伪造同起点 GRPO 组；Designer 使用带显式标量基线、不开 batch whitening 的 `reinforce_baseline`。若要用 GRPO，必须执行 matched-baseline 候选并共享 θ_before、冻结 Probe、协议和预算。
9. 完成本轮证据持久化后，更新后续候选池；Recovery/Retention 具体题面仍不向 Designer 暴露。最终 TEST 永不产生 Designer Reward。

## 数据分区与策略版本

任务用途为 `TRAIN`、`BANK_CANDIDATE`、`REPLAY_LOGP_FEEDBACK`、`VALIDATION`、`TEST`；Recovery 和 Retention 是 Probe 快照角色，不再分割来源任务。现有 `Split` 语义保持不变，`Identity.purpose` 用于任务来源。`strict_family_disjoint_v1` 按 World、Task、Family 或 Lineage 判定跨 split 冲突；`instance_disjoint_within_family_v1` 允许不同任务实例共享宽泛 family，但仍隔离 World、Task 和 source lineage。

旧 SplitRegistry JSON 数组按 `strict_family_disjoint_v1` 加载；新 envelope 记录 policy 版本。任何变更 policy 的实验需显式新建 Registry，不能在读取旧档时自动放宽隔离。Probe、奖励和实验 manifest 都记录该版本；同一 Failure Bank 任务可以在不同轮次作为训练候选或 Probe，但不能同时进入本轮 Service 训练包和评分快照。

## Reward、Verifier 与恢复

双池 Reward 只能由可信 Host 在已验证 `CompletedUpdate`、冻结快照和完整 before/after 试次证据上重算。奖励 verifier 从可信 evidence resolver 取回 Snapshot 和两次结果，验证 hash、用途、split、Checkpoint、task 对齐、试次数、结果来源和公式。Probe 数据不足时结果状态是 `INSUFFICIENT_EVIDENCE`，不是零奖励。

旧 `ReplayReward` 继续由 `replay_logp_v1` 路径产生。Reward 类型有明确 protocol tag；混合批次必须拒绝。旧 LoopState JSON 若无协议字段，显式按 `replay_logp_v1` 加载；旧奖励不会自动迁移成双池奖励。双池状态按冻结、基线、Service 更新、Probe 完成、奖励、Designer 更新的阶段落盘，恢复时重新验证 update、snapshot 和 evidence，不能换 Probe、重复计分或重复训练。

## 算法和未实现边界

Service 保留 VERL 0.9.1 FSDP2 全参数 GRPO 与 vLLM 权重同步。Designer 双池默认使用无分组 REINFORCE baseline：跨不同 Service 起点的序列奖励不能作为同一个 GRPO 对比组，也不能使用隐含 batch whitening。该实现提供算法与证据接口，但基线的真实方差、稳定性和学习效果须通过真实训练验证。旧 Replay LogP Designer 路径保持原有 matched-source 约束。

Usami Benchmark 的认证、业务规则、Oracle 和 Reference 不改写。Grounding 按原生固定 seed 的候选顺序逐一尝试，次数由 native `GroundingBudget.max_matches` 限定。默认 Customer 工厂可以构造信息-only/mixed profile；但固定版本的原生 proof certifier 会将信息目标标为 UNKNOWN，直到接入交互式认证器，因此当前不能把这类任务当作已认证训练任务。真实 Ray/FSDP2 Worker Launcher、持久化认证任务及 Episode Store、完整 Probe Runtime Factory、Ascend 兼容栈仍由目标运行环境提供；CPU/Scripted 测试只验证协议，不证明模型能力进步。本文不报告训练结果。
