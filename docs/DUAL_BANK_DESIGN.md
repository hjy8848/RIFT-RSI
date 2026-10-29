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
| `DiagnosisExperience` | 失败/薄弱 Episode 的 `PublicSummary` 与公开结果 | Designer 形成可检验 hypothesis 并提出新任务 | 仅显式公开投影 |
| `TaskEvidenceRecord` | Host 的 opaque TaskRef、来源身份、用途分区、Checkpoint、试次结果和可信证据引用 | 持久化双池、重放任务、选择 Probe | Host 私有 |
| `Demonstration` | 成功 Episode 的精确上下文、Action Token 与 mask | 旧 Replay LogP 对照 | Host Scorer 使用 |

`FailureBank` 保存特定 Service Checkpoint 下的薄弱候选，不能把失败 Token 当作正向动作标签。它区分可见 `DIAGNOSIS` 与保密 `RECOVERY_FEEDBACK`；两个分区不能包含相同任务、World 或 source lineage。`SuccessBank` 保存能力估计较高的任务候选；其任务可独立抽入 `RETENTION_FEEDBACK`。同一用途内的多个试次估计成功率，UNKNOWN 单独记录，不能作为失败样本。

Failure/Success 是随 Checkpoint 和协议变化的历史状态。后续任务结果可以新增 `RECOVERED` 或候选状态事件，不覆盖旧记录。任务引用、World、来源 lineage、证据引用及 Probe 具体成员都留在 Host。Designer 输入通过 allowlist 只包含公开事件和允许公开的最终业务结果；不序列化私有记录。

## 每轮协议

1. Bootstrap 用 θ₀ 对可信任务按固定协议重复 Rollout，保存每次 `SUCCESS` / `FAILURE` / `UNKNOWN`、Checkpoint 和证据。默认只建池和基线，不训练 Service，也不生成 Designer 奖励。
2. Designer 从随机选择的公开诊断轨迹提出 hypothesis 与结构化 Usami Proposal。Grounding、Certification 和 TaskRegistry 在 Host 完成；失败提案不获得虚构的能力收益。
3. Service 更新前冻结随机 `RECOVERY_FEEDBACK` 与 `RETENTION_FEEDBACK` 快照，保存 split policy 版本、seed、任务引用、样本权重、协议 digest、θ_before 和内容 digest。训练包、诊断实例、两个 Probe 池与 TEST 必须满足所选 split-policy。
4. 在 θ_before 下真实评测 Probe；若无同一完整 Checkpoint、TaskRef、环境/采样协议及 seed 的已验证缓存，则先执行 baseline Rollout。缺失 baseline 不能默认成 0。
5. Service 在新训练包上通过 verl 0.9.1、Usami 多轮 Agent Loop 和全参数 GRPO 更新 θ_before→θ_after。UNKNOWN 或验证失败的 GRPO 成员使整个组失效。
6. 在相同冻结任务、seed 和协议下用 θ_after 独立重新执行 Probe，保留新的 Episode 和 replay 证据。任务先算自己的有效试次成功率，再对任务等权平均；UNKNOWN 不计成功也不计失败，但超出有效覆盖阈值则拒绝 Reward。
7. `G = SR(θ_after, F) - SR(θ_before, F)`，`H = SR(θ_after, S) - SR(θ_before, S)`，`R = G - λ max(0, -H)`。独立保存 G、H、R、有效率、试次和 evidence refs。
8. 把 `R` 绑定 Designer 的原始 on-policy Token、包 ID、Service update、两个冻结 Probe、Checkpoint 和预算。默认采用 `sequential_curriculum`：每个包从当前 θ 更新，不伪造同起点 GRPO 组；Designer 使用带显式标量基线、不开 batch whitening 的 `reinforce_baseline`。若要用 GRPO，必须执行 matched-baseline 候选并共享 θ_before、冻结 Probe、协议和预算。
9. 完成本轮证据持久化后，更新后续候选池；Recovery/Retention 具体题面仍不向 Designer 暴露。最终 TEST 永不产生 Designer Reward。

## 数据分区与策略版本

逻辑用途为 `TRAIN`、`DIAGNOSIS`、`RECOVERY_FEEDBACK`、`RETENTION_FEEDBACK`、`REPLAY_LOGP_FEEDBACK`、`VALIDATION`、`TEST`。现有 `Split` 语义保持兼容，新增 `Identity.purpose` 明确细分用途。旧默认 `strict_family_disjoint_v1` 保持原行为：World、Task、Family 或 Lineage 任一相同即跨分区冲突。`instance_disjoint_within_family_v1` 明确允许不同任务实例共享宽泛 family，但仍禁止 World、Task 或具体 source lineage 跨分区重叠。不同 family 本身不能替代 source lineage 的可信粒度。

旧 SplitRegistry JSON 数组按 `strict_family_disjoint_v1` 加载；新 envelope 记录 policy 版本。任何变更 policy 的实验需显式新建 Registry，不能在读取旧档时自动放宽隔离。Probe、奖励和实验 manifest 都记录该版本。Diagnosis 与 Recovery 既不能共享任务/World/lineage，也不能只凭不同 `task_ref` 被视为独立。

## Reward、Verifier 与恢复

双池 Reward 只能由可信 Host 在已验证 `CompletedUpdate`、冻结快照和完整 before/after 试次证据上重算。奖励 verifier 从可信 evidence resolver 取回 Snapshot 和两次结果，验证 hash、用途、split、Checkpoint、task 对齐、试次数、结果来源和公式。Probe 数据不足时结果状态是 `INSUFFICIENT_EVIDENCE`，不是零奖励。

旧 `ReplayReward` 继续由 `replay_logp_v1` 路径产生。Reward 类型有明确 protocol tag；混合批次必须拒绝。旧 LoopState JSON 若无协议字段，显式按 `replay_logp_v1` 加载；旧奖励不会自动迁移成双池奖励。双池状态按冻结、基线、Service 更新、Probe 完成、奖励、Designer 更新的阶段落盘，恢复时重新验证 update、snapshot 和 evidence，不能换 Probe、重复计分或重复训练。

## 算法和未实现边界

Service 保留 VERL 0.9.1 FSDP2 全参数 GRPO 与 vLLM 权重同步。Designer 双池默认使用无分组 REINFORCE baseline：跨不同 Service 起点的序列奖励不能作为同一个 GRPO 对比组，也不能使用隐含 batch whitening。该实现提供算法与证据接口，但基线的真实方差、稳定性和学习效果须通过真实训练验证。旧 Replay LogP Designer 路径保持原有 matched-source 约束。

Usami Benchmark 的认证、业务规则、Oracle 和 Reference 不改写。真实 Ray/FSDP2 Worker Launcher、持久化认证任务及 Episode Store、完整 Probe Runtime Factory、Ascend 兼容栈仍由目标运行环境提供；CPU/Scripted 测试只验证协议，不证明模型能力进步。本文不报告训练结果。
