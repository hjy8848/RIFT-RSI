# Usami v2 业务交互简化：现状核查与改造方案

> 状态：**方案评估稿（未改任何代码，未跑测试）**。所有结论均来自对本地 `D:\RealLab\RSI\RIFT-RSI`（HEAD `ae3a4a9`）源码的静态通读。
> 目标：把 Usami 的 Agent-facing 交互简化到 τ-bench 那样"少工具、一步一业务动作"，同时保留动态 World/Task 构造与 Engine/Reference/Oracle/Replay 独立验证。

---

## 0. 一句话结论

**Smoke 全 0 奖励不是模型能力问题，是任务在当前工具面下基本不可达**：certifier 证出来的最短合法退款路径需要 8–10 个 Service 操作、2 次结构化客户同意；而 smoke 预算只有 4 个 customer turn，模型每发一句 `respond_to_customer` 就消耗一个 turn，且它拿不到 `evidence_id`（observation 层故意隐藏 `Evidence.valid` 和 case 上的 evidence 引用细节）。改造方向正确，但必须同步改 **Reference 闸门 + Certification 见证路径 + 状态目标**，否则会出现"删了工具但闸门还在"的死锁。

---

## 1. 现状：一个退款要经过什么

### 1.1 Service 工具面：18 个

`third_party/usami/usami_v2/runtime/agents.py:143-183`（`service_tools()`）暴露 17 个业务工具 + 1 个 `respond_to_customer`：

| 类别 | 工具 |
|---|---|
| 查询（8） | `list_customer_orders` `get_order_details` `search_catalog` `get_sku_details` `get_inventory` `estimate_delivery` `get_refundable_balance` `get_case_status` |
| 预处理（5，全部受 delegation 门控） | `open_remedy_case` `verify_submitted_evidence` `request_policy_review` `prepare_remedy_quote` `cancel_pending_commitment` |
| 提交/结算（4） | `commit_accepted_quote` `schedule_authorized_return` `release_accepted_refund` `ship_accepted_replacement` |
| 对话（1） | `respond_to_customer` |

参数外壳统一为 `{scope_id, resource_id, params}`（`agents.py:168-172`），`scope_id` 必须是 merchant id，`resource_id` 随操作类型变化（`commerce/transitions.py:11-37` 的 `_hard_scope`）。模型必须自己搞清"哪类操作用哪类 resource_id"。

### 1.2 最短合法退款路径（`return_required=False`）

```
Customer profile 自动注入 Delegation（不可见）
  → open_remedy_case {quantity, issue}          # 需要 valid evidence 存在
  → get_case_status                              # 拿 evidence_id（否则无从填参数）
  → verify_submitted_evidence {evidence_id}
  → prepare_remedy_quote {remedy: refund}
  → 普通 reply（runtime 自动把 quote 条款渲染给客户）
  → Customer 回结构化 consent（accept_quote_id + accept_operation）
  → commit_accepted_quote                       # 立即 _settle：Ledger + Settlement
  → 普通 reply + Customer finish
```

- `return_required=True` 时再加 `schedule_authorized_return` → `return_received` 外部事件 → 第二次 consent → `release_accepted_refund`，共 7 步 + 2 次 consent。
- 金额超过 `merchant.refund_approval_limit_cents` 时还要 `request_policy_review` + 外部 `review_result` 事件。

### 1.3 预算与真实消耗

| 预算 | 默认值（`runtime/runner.py:39-43`） | RIFT smoke 实际值（`src/rift/runtime/ascend_910b3.py:600-601`） |
|---|---|---|
| `max_customer_turns` | 8 | **4** |
| `max_service_decisions` | 64 | 10 |
| `max_service_tool_calls` | 64 | 8 |
| `max_model_calls` | 96 | 10 |

每次 Service reply 触发一次 customer turn（`runner.py:698-703`），`respond_to_customer` 也算一个 decision。4 个 turn 只够"请求 → 追问 → 回答 → 结束"，留不出 case/quote/consent 的空间。**Certification 只证明"存在一条合法路径"，不检查这条路径能否在 episode 预算内走完**，这是当前设计与奖励 0 之间的结构性断点。

### 1.4 为什么模型会卡在"要求客户提交证据"

- `Evidence` 只在建世界时按 order line 预置（`scenarios/builder.py:103-105`），**全仓库没有任何提交/上传 evidence 的工具**（`submit_evidence|add_evidence|create_evidence|upload_evidence` 零命中）。
- Agent 看不到 `Evidence.valid`，也看不到 case 上的 evidence 绑定（`commerce/observations.py:40-42` 明确剥离），只能先 `open_remedy_case` 再 `get_case_status` 才能拿到 `evidence_id`。
- 于是模型的自然行为就是"让客户去交证据"，而环境根本没有这个动作 → 反复 `respond_to_customer` → 预算耗尽。

### 1.5 Customer 侧协议

`agents.py:186-200`：只有 1 个 `customer_turn`，强制 9 个字段 —— `message`、`delegate_line_ids`、`allowed_preparations`、`accept_quote_id`、`accept_operation`、`reject_quote_id`、`withdraw_delegation_ids`、`disclose_fact_ids`、`finish`。Customer 模型实际上在替环境管理授权生命周期，而不是"扮演客户"。

### 1.6 奖励链路（关键约束）

```
World 状态变化
  → evaluate_state_goals (scenarios/tasks.py:610-677)   # 必须找到该 line 的新 Settlement
  → OutcomeV2.business_goal_success (evaluation/outcomes.py:124-136)
  → public_business_outcome (src/rift/benchmark/private_evaluation.py:10-18)  # True→SUCCESS / False→FAILURE / None→UNKNOWN
  → service_reward (src/rift/benchmark/public_feedback.py:18-22)             # 1.0 / 0.0 / None(不能训练)
```

`evaluate_state_goals` 的 PASS 条件：settlement 的 unit 集合 == `goal.quantity`、remedy ∈ `allowed_remedies`、`final.quotes[settlement.quote_id]` 存在、refund ledger 总额 == `unit_credit`、channel == `order.payment_channel`。**奖励直接绑死 Settlement+Quote**，这是简化工具时必须一起改的地方。

---

## 2. 与改造大纲的逐项对照

| 大纲目标 | 现状 | 差距 |
|---|---|---|
| 保留动态 World 构造 | ✅ `ScenarioDraft → ScenarioSpec → build_world`，区间约束搜索（`scenarios/schema.py`、`builder.py:124`、`solver.py:50`） | 无需改动 |
| 保留动态 Task 构造 | ✅ `TaskDraft → compile_task → TaskSpecV3`，`render_customer_request` 由结构反向渲染并字节校验（`tasks.py:436,542`） | 无需改动 |
| 简化 Agent 工具面 | ❌ 18 个工具、嵌套参数、必须按序调用 | 目标 4 个（Phase 1） |
| 取消强制 Evidence | ❌ `reference.py:207-208` 硬闸门 + 认证器 `task_certification.py:194-197` | 需同时改两处 |
| 取消 Case/Quote/Consent 多阶段 | ❌ quote 是 settlement 的外键，consent 是 commit/release 的硬闸门 | 需引入原子操作 + 改状态目标 |
| 简化 Customer 协议 | ❌ 9 字段结构化决策 | 目标 2–3 字段 |
| 保留 Engine/Reference/Oracle/Replay | ✅ 结构完整 | 原子操作必须同时写两份独立实现 |
| 难度来自业务决策 | ⚠️ 只有 1 个 smoke 族（refund × damaged） | 需任务族矩阵（见 §5） |

---

## 3. 目标架构

### 3.1 工具面：18 → 4（+ 后续扩展）

| 工具 | 参数 | 语义 |
|---|---|---|
| `list_customer_orders` | `{}` | 当前客户的订单摘要列表 |
| `get_order_details` | `{order_id}` | 订单 + 行 + 可退余额 + 支付方式 |
| `return_delivered_order_items` | `{line_id, quantity, payment_method="original"}` | **原子退款**：校验 + 建退款申请 + 记账 |
| `respond_to_customer` | `{message}` | 唯一对话出口 |

Phase 2 再补 `cancel_pending_order`、`exchange_delivered_order_items`、`search_catalog`/`get_sku_details`/`get_inventory`（换货才需要）。

### 3.2 原子操作的内部语义

`return_delivered_order_items` 在一次 transition 里完成（agent 不可见）：

1. 归属校验：line → order.customer_id == actor，order.merchant_id == scope（`OBJECT_ACCESS_DENIED`）
2. 状态校验：`order.delivered_at` 非空（`ORDER_NOT_DELIVERED`）
3. 期限校验：`delivered_at + sku.return_days >= clock`（`RETURN_WINDOW_EXPIRED`）
4. 数量校验：`quantity <= line.quantity - already_returned_units`（`QUANTITY_EXCEEDS_PURCHASED`）
5. 重复校验：该 unit 未被其他退款占用（`ALREADY_RETURNED`）
6. 方式校验：`payment_method == "original"` 且 `order.payment_channel` 合法（`PAYMENT_METHOD_NOT_ALLOWED`）
7. 金额：`unit_credit(line, units)`（`commerce/models.py:298`，逐 unit 分摊，不新写算法）
8. 落账：写 Settlement + LedgerEntry（复用 `_settle` 的记账规则），并记录 `ReturnRequest(status="return_requested")`
9. 返回 `{success: true, status: "return_requested", quantity, refund_cents}`

失败返回 `{success: false, error_code: "..."}`，与"工具格式错误""越权"分开编码（大纲 §4）。

### 3.3 两种落地方式（需决策）

**方案 A（推荐先做）——内部复用 Case/Quote/Settlement，agent 不可见**

- 原子操作内部创建 `RemedyCase(status=resolved)` + `Quote(status=committed)` + Settlement + Ledger，`evaluate_state_goals` **完全不改**，奖励语义不变。
- 优点：改动面小、奖励/不变量/Oracle 全部复用，最快跑通真机 smoke。
- 代价：World 里仍然存在 case/quote 行（仅内部记账用），与大纲"基础路径只需退款申请"的文字不完全一致。

**方案 B——引入 `ReturnRequest` 一等实体，Settlement 降级为内部实现**

- 新增 `ReturnRequest{line_id, units, quantity, payment_method, status: requested|settled|rejected, refund_cents}`；`evaluate_state_goals` 改判 `ReturnRequest.status`；Outcome/reward 契约随之变化（`commerce-outcome-v2` → v3）。
- 优点：与大纲完全对齐，彻底去掉 Quote/Consent/Delegation 在基础路径上的存在。
- 代价：`tasks.py`、`task_certification.py`、`invariants.py`（40+ 条）、RIFT 侧协议字符串与哈希固定测试全部要改，工作量约为方案 A 的 2.5 倍。

> 建议：Phase 1 用 A 跑通闭环与真机 smoke，Phase 2 再决定是否迁移到 B（A 的原子操作接口在 B 下不需要变）。

### 3.4 Customer 侧

- `CustomerDecision` 收敛为 `{message: str, finish: bool}`（可选 `{answer: "yes"|"no"|None}` 供后续判断题）。
- 授权不再走 Quote-bound Consent：Delegation 由 profile 在 episode 初始化时自动铸造（agent 不可见，`runtime/customer.py:420-443` 已实现），这正是"客户已把这件事委托给你"的现实语义，不需要模型理解。
- 删除 `accept_quote_id`/`accept_operation`/`reject_quote_id`/`withdraw_delegation_ids`/`disclose_fact_ids` 的 agent-facing 部分；`FactEvidence` 保留给 information goals（v3）使用。
- 客户确认退款的语义由对话本身承载（τ-bench 风格），**不做关键词匹配伪造授权**。

### 3.5 业务实体：保留/降级/删除

| 实体 | 处置 | 理由 |
|---|---|---|
| Customer/Merchant/SKU/Warehouse/Inventory/DeliveryRoute/Order/OrderLine | 保留 | 全部业务规则依赖；World 动态性来源 |
| Evidence | 保留但降级为"世界属性" | `_line_description` 用它渲染客户请求里的故障词（`tasks.py:424-433`）；不再要求 agent 校验 |
| RemedyCase | 方案 A 保留（内部）；方案 B 删除 agent-facing 语义 | 认证器与 settlement 外键 |
| Quote / Consent / Delegation | 方案 A 内部保留；方案 B 删除 | 基础路径不再需要；exchange/reship 前保留代码 |
| Return / Fulfillment | 保留 | 换货/退货签收在 Phase 3 使用 |
| LedgerEntry / Settlement | 保留 | 账务不变量与奖励锚点 |
| ExternalEvent | 保留 | 政策复核、签收、库存调整仍需要 |

---

## 4. 文件级改动清单（Phase 1，方案 A）

| 文件 | 改动 |
|---|---|
| `third_party/usami/usami_v2/commerce/models.py` | 新增 `ReturnItemsInput{line_id, quantity, payment_method}`、`PARAMETERS` 增 `return_delivered_order_items`；新增 `ReturnRequest` 记录（若走 B）；`PREPARATIONS` 保持不含原子操作 |
| `.../commerce/transitions.py` | 新增原子分支：归属/状态/期限/数量/重复/方式校验 → 建 case+quote → `_settle` → 写 `ReturnRequest`；返回 `{status, refund_cents}` |
| `.../commerce/reference.py` | **独立实现**同一套规则（不得调用 transitions），返回 granular `error_code` 与 `expected_state`；`PREPARATIONS` 的 delegation 检查保留但不再涉及 evidence |
| `.../commerce/observations.py` | 新增 `return_request` 查询字段（订单详情里带 `returned_units`、`refundable`），移除 case/quote 暴露（或保留只读给诊断） |
| `.../commerce/invariants.py` | 加 `RETURN_EXCEEDS_PURCHASED`、`RETURN_LEDGER_MISMATCH`；`CASE_*` 系列在方案 B 下删减 |
| `.../runtime/agents.py` | `service_tools()` 改为 4 个；参数扁平化；`respond_to_customer` 的 `subject_ids` 约束放宽或移除 |
| `.../runtime/runner.py` | 移除 quote 自动渲染与 pending-consent 提示；reply 后直接进 customer turn；预算语义保持不变 |
| `.../runtime/customer.py` | `CustomerDecision` 收敛；删除 consent 铸造与消费；delegation 仅初始化铸造 |
| `.../scenarios/task_certification.py` | `_run_candidate` 见证路径改为 delegation + 原子操作 + `evaluate_state_goals`；删除 evidence/review/return 分支（保留 exchange/reship 分支） |
| `.../scenarios/tasks.py` | 方案 B 才改 `RemedyGoal`/`evaluate_state_goals`；方案 A 只补 `ReturnRequest` 一致性检查 |
| `src/rift/verl/usami_session.py` | 工具协议提示语改为新工具集；`ServiceAction.from_tool_message` 适配扁平参数 |
| `src/rift/benchmark/usami.py` | `_customer_limits`/`default_customer_scenario` 去掉 reviews/pickup 依赖；customer 场景不再需要第二笔同意 |
| `src/rift/verl/task_registry.py` | `_check` 中 `remedy_goals` 行绑定逻辑不变；如协议版本变化需同步 |
| `tests/rift_usami_fixtures.py` | `ScriptedRefundService` 改为 2 步（查询 + 原子退款）；`ScriptedCustomer` 改为自然语言 + finish |
| `tests/test_usami_native.py`、`tests/test_usami_v2_task_certification.py` | 哈希固定的冻结用例需重新基线 |

---

## 5. 任务族与难度阶梯（改造后）

| 难度 | 任务 | 业务能力 | 当前状态 |
|---|---|---|---|
| L1 | 已送达订单，退 1 件，原路退款 | 基本工具使用 | Phase 1 |
| L1 | 取消待处理订单 | 状态判断 | Phase 2 |
| L2 | 多件订单只退其中 1 件 | 数量/行识别 | Phase 1（参数已支持） |
| L2 | 超退货期限 → 拒绝 | 规则判断 | Phase 1（拒绝码已定） |
| L2 | 未送达 → 拒绝 | 状态判断 | Phase 1 |
| L2 | 已部分退款，再次申请 | 历史状态 | Phase 1 |
| L3 | 多订单只有 1 个符合条件 | 检索 | Phase 2 |
| L3 | 换货但库存不足 | 约束 + 替代方案 | Phase 3 |
| L3 | 多商品不同处理方式 | 多目标规划 | Phase 3 |

生成侧仍由 `ScenarioDraft` 参数化：`delivered_days_ago`、`return_days`、`quantity`、`paid_cents`、`default_stock`、`StockConstraint`、`RelationConstraint` 已经足够表达上述全部变体，**不需要新增 DSL**。

---

## 6. 测试计划

1. **工具一致性**：Prompt 文本、OpenAI schema、`PARAMETERS`、`Operation.typed_parameters`、Engine、Reference 六处同名同参（现有测试可扩展）。
2. **真实退款**：成功 / 未送达 / 超期 / 数量超买 / 越权 / 重复 / 部分退款，7 个用例走真实 Engine + Oracle + Replay。
3. **独立性**：Reference 不 import transitions/observations（现有断言保留）；篡改最终状态后 Oracle 必须报警。
4. **端到端**：Designer → World → Task → Certification → Registry → Episode → Replay → Outcome → Reward。
5. **公开信息基线**：`PublicToolBaseline` 只用 `list/get/return` 与公开 observation，至少跑通成功、拒绝、部分退款三条。
6. **回归红线**：`test_native_episode_execution_replay_and_projection`、`test_designer_diagnostic_removal_preserves_frozen_usami_results` 等固定哈希用例必须重新基线并记录原因。

---

## 7. 风险与依赖

1. **哈希连锁**：`World.state_hash`、`TaskSpec.record_hash`、`Quote.terms_hash`、`TaskCertification.witness_hashes` 全部由内容派生，任何实体增删都会改变冻结测试基线；需一次性重算并保留旧值对照。
2. **奖励语义**：方案 A 下奖励仍是 Settlement 二值；如果模型只是"说"退款成功而没调工具，`evaluate_state_goals` 会 FAIL，这是正确行为。若走方案 B，必须同步 `commerce-outcome-*` 协议字符串（`src/rift/replay/reward.py:41`、`src/rift/verl/actor_backend.py:29` 等 5 处硬编码）。
3. **远端状态**：`npu30001:/home/hejingyu/projects/RIFT-RSI` 落后本地 21 个提交，且有未提交修改（`runtime/ascend_910b3.py`、`verl/agent_loop.py`）。真机验证前需先决策同步策略，否则会把脏改动覆盖掉。
4. **Certification 与预算脱节**：认证只证明可达性，不证明在预算内可达。建议 Phase 1 后在 `certify_task` 里记录"最短见证步数"，episode 预算低于该步数时直接拒绝注册，避免再造 0 奖励 smoke。
5. **不可退单元记账**：`unit_credit` 按 unit index 分摊尾差，部分退款必须复用该函数，不能另写四舍五入。

---

## 8. 验收对照（大纲 §12）

- [ ] Agent-facing 工具 18 → 4
- [ ] 普通退款不再经过 Evidence/Case/Quote/Consent（方案 A：agent 不可见；方案 B：不存在）
- [ ] Customer 协议 9 字段 → ≤3
- [ ] World/Task 动态构造保持不变
- [ ] Engine/Reference/Oracle/Replay 职责独立，Reference 不 import Engine
- [ ] 基础退款能跑出 reward=1.0 的真机 episode
- [ ] 超期/未送达/超额/越权四类拒绝有独立 error_code 且不误判成功
- [ ] 失败路径不会因缺少证据工具而空转

---

## 9. 待决策

1. Phase 1 用方案 A（内部 Case/Quote 记账）还是直接上方案 B（ReturnRequest 一等实体）？
2. 是否保留"自动铸造的 Delegation"作为授权依据（agent 不可见）？
3. 远端 checkout 如何处理（备份脏文件后强制同步 / 先 merge / 只在本地改）？
4. Phase 1 完成后是否立即在 npu30001 跑一次真机 smoke 验证 reward 非零？
