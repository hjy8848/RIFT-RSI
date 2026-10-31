# Usami Phase 1：原子退款闭环交付报告

日期：2026-10-31　范围：`third_party/usami/usami_v2` + `src/rift` 接线 + 测试

## 1. 结论

Phase 1 的"普通退款"闭环已跑通：Service 侧只有 4 个公开工具，按
`list_customer_orders → get_order_details → return_delivered_order_items` 的公开路径即可发现订单行并
一次完成校验与退款记账；Engine 执行、Reference 独立重算、Oracle 复核、Replay 全部一致；
RIFT `service_reward` 实测 **1.0**。
本地 venv 全量测试：**60 passed, 1 skipped**（22 个收集错误均为缺少 numpy/verl 等可选依赖，
与本次改动无关），新增 11 个退款测试全部通过，其中 `test_public_tool_path_discovers_the_line_and_refunds`
不向 Service 提供任何 line id 或私有 alias。

## 2. 架构前后对比

| 维度 | 改造前 | 改造后 |
| --- | --- | --- |
| Agent 可见工具 | 18 个（17 业务 + respond） | 4 个（list_customer_orders / get_order_details / return_delivered_order_items / respond_to_customer） |
| 参数外壳 | `{scope_id, resource_id, params}` 三层 | 扁平参数；scope_id 由 Host 按"唯一 merchant"注入 |
| 一次退款 | delegation → open_case → verify_evidence → prepare_quote → 展示 quote → consent → commit → schedule_return → return_received → 二次 consent → release（约 9 个操作、3+ customer turn） | delegation（自动、不可见）→ list_orders → order_details → 原子退款 → 1 个自然语言确认 |
| Evidence | 任何退款路径都必须先有 valid evidence | 普通退款不读、不写 Evidence；`RemedyCase.issue/evidence_id` 均为 None |
| Customer 协议 | 9 个结构化字段（delegation/consent 生命周期） | `{message, finish}`，自然对话 |
| 内部记账 | Case/Quote/Settlement 全链路 | 保持同一数据结构，由原子工具内部一次写入 |
| 认证 Witness | 9 步固定序列 | 3 步公开路径（list → details → refund）+ state-goal 校验，成本计入 witness_cost |
| 预算 | 固定预算，不检查可达性 | 认证产出 `witness_cost`，注册前与 EpisodeBudget 比对 |

## 3. 工具列表前后对比

保留并重新描述（Agent 可见）：
- `list_customer_orders`（无参）
- `get_order_details {order_id}`
- `return_delivered_order_items {line_id, quantity, payment_method="original"}`
- `respond_to_customer {message}`

内部保留但不暴露（供 archived replay / 脚本夹具 / Phase 2-3 使用）：
`open_remedy_case`、`verify_submitted_evidence`、`request_policy_review`、
`prepare_remedy_quote`、`commit_accepted_quote`、`schedule_authorized_return`、
`release_accepted_refund`、`ship_accepted_replacement`、`cancel_pending_commitment`、
`search_catalog`、`get_sku_details`、`get_inventory`、`estimate_delivery`、
`get_refundable_balance`、`get_case_status`。

## 4. 实体保留 / 删除理由

| 实体 | 处理 | 理由 |
| --- | --- | --- |
| World / Merchant / SKU / Order / Line / Inventory / Payment | 保留 | 动态生成与可复现 seed 的基础，不因工具简化而删除 |
| RemedyCase | 保留（issue 可选） | 退款记账仍需要 case 作为 unit 占用与结算绑定；但普通退款不伪造 issue |
| Quote | 保留 | Settlement 必须绑定 committed refund quote（`SETTLEMENT_BINDING_INVALID`） |
| Settlement / LedgerEntry | 保留 | `evaluate_state_goals` 与 reward 的成功判据，Phase 1 不迁移 Outcome v3 |
| Delegation | 保留，Agent 不可见 | 只做身份与 line 范围授权；不再承担"对某次退款点头"的语义 |
| Consent | 保留代码路径 | 新流程不触发；Phase 2 如引入"必须客户确认才可执行"的分支再启用 |
| Evidence | 保留生成器 | Phase 1 普通退款不依赖；Phase 2 政策类任务再使用 |
| Return / Fulfillment | 保留未用 | 物理退货属 Phase 3；Phase 1 明确不把"退货申请"当作"退款到账" |
| 旧 17 个 PARAMETERS 操作 | 保留 | archived Episode / 脚本夹具仍可重放 |

## 5. 逐文件变化

| 文件 | 变化 |
| --- | --- |
| `third_party/usami/usami_v2/commerce/models.py` | `RemedyCase.issue` 改为可选；新增 `ReturnItemsInput`；PARAMETERS 增加原子退款 |
| `third_party/usami/usami_v2/commerce/transitions.py` | `_hard_scope` 支持 line 解析；`apply` 新增原子退款分支（支付方式/送达/期限/实物退货/unit 占用/审批上限 → 建 case+quote → `_settle`） |
| `third_party/usami/usami_v2/commerce/reference.py` | 同规则独立实现；delegation 仅校验 line 范围；投影 settlement 后返回与 Engine 逐字节一致的结果 |
| `third_party/usami/usami_v2/commerce/invariants.py` | **未改**：evidence_id=None 的 resolved case 本就合法 |
| `third_party/usami/usami_v2/runtime/agents.py` | 公开工具 4 个、扁平参数解析、`CUSTOMER_TOOLS={message,finish}`、两段 SYSTEM 提示词重写 |
| `third_party/usami/usami_v2/scenarios/task_certification.py` | 候选只枚举 refund；见证路径单步；新增 `witness_cost` 与 `certification_cost` |
| `src/rift/benchmark/usami.py` | 认证后计算 `certification_cost`；`_budget_covers` 拒绝超出预算的任务 |
| `src/rift/verl/usami_session.py` | 协议提醒改写为扁平参数契约；解析时传入 merchant scope |
| `src/rift/runtime/ascend_910b3.py` | smoke 假设更新；脚本 Customer 改为自然对话 + finish |
| `tests/test_service_tool_turn_protocol.py` | 断言新契约（无 params/scope_id） |
| `tests/test_usami_native.py` | 任务改用 refund 候选；重新冻结 task/certificate 哈希 |
| `tests/test_usami_atomic_refund.py` | 新增 10 个测试 |

## 6. World / Task 生成示例（smoke，本地实跑）

```
scenario_id = rift-ascend910b3-smoke
skus        = original(device/USB_C, 20000), alternate(22000), cable(2000)
order       = purchase, delivered_days_ago=1, qty=1, paid=20000
warehouse   = fast(delivery 2 天, fee 0), alternate 库存 3
refund_approval_limit_cents = 50000
witness     = tool_calls 1, service_decisions 2, customer_turns 1, model_calls 2
task hash   = 8605f5f042a1…（commerce-task-v3）
```

## 7. 完整退款调用记录（本地 e2e，RIFT 侧投影）

```
user      | Please help me with the following request(s): Please arrange a refund … damaged USB C device
tool_call | {"name":"list_customer_orders","resource_id":"merchant"}
tool_result | {"data":{"orders":[{"id":"order_…","customer_id":"customer_…",…}]}}
tool_call | {"name":"get_order_details","resource_id":"order_…"}
tool_result | {"data":{"order":{…},"lines":[{"line_id":"line_4316957ed71b","sku":{"product_family":"device",
                       "attributes":{"interface":"USB_C"},"price_cents":20000},…}]}}
tool_call | {"name":"return_delivered_order_items",
             "resource_id":"line_4316957ed71b",
             "params":{"quantity":1,"payment_method":"original"}}
tool_result | {"data":{"line_id":"line_4316957ed71b","payment_channel":"original-card",
                       "quantity":1,"refund_cents":20000,"status":"refund_issued","success":true}}
assistant | Refund issued to your original card.
user      | The device I received is damaged and I want a refund.
user      | Thanks, that is all.
```

## 8. Engine / Reference 验证

- 两个实现互不 import：Engine 走 `transitions.apply` 改状态；Reference 在 `world.clone()`
  上独立重算 case/quote/settlement 并返回 `expected_state` 与 `data`。
- Oracle 同时比较 `after == decision.expected_state` 与 `result.data == decision.data`，全部通过。
- 原子退款后 `invariants(world) == []`：refund quote 无替换条款、settlement 与 ledger 一一对应、
  case 已 resolved、unit 未被重复占用。
- 拒绝码覆盖：`PAYMENT_METHOD_UNSUPPORTED`、`ORDER_NOT_DELIVERED`、`RETURN_WINDOW_EXPIRED`、
  `RETURN_REQUIRED`、`PURCHASED_QUANTITY_UNAVAILABLE`、`ALREADY_REFUNDED`、
  `APPROVAL_REQUIRED`、`BUSINESS_DELEGATION_REQUIRED`。

## 9. Replay / Outcome / Reward

- `replay_episode(episode).verified == True`
- `episode.status == "customer_finished"`，`outcome.business_goal_success == True`
- RIFT：`public_business_outcome → Outcome.SUCCESS → service_reward == 1.0`
- 状态变化是真实的：final world 里新增 1 条 refund settlement、1 条 refund ledger（20000 分），
  case 为 resolved 且 `evidence_id is None`。

## 10. 预算一致性检查

- `certification_cost(certificate)` 从 witness 步骤推导
  `{tool_calls, service_decisions, customer_turns, model_calls}`。
- 认证 witness 现在先执行 `list_customer_orders` 与 `get_order_details`，再执行原子退款，
  因此成本是 **3/4/1/4**（此前只记 1 次操作，低估了真实路径）。
- `CommerceHost._budget_covers(cost)` 与 `EpisodeBudget` 逐项比较；超预算时不注册任务，
  attempt 记为 `UNKNOWN`，理由 `certified_witness_cost_exceeds_episode_budget`。
- 当前 smoke 预算 4/10/8/10，witness 成本 3/4/1/4，仍有余量。

## 11. 测试结果

```
.venv/Scripts/python.exe -m pytest -q --continue-on-collection-errors
60 passed, 1 skipped, 22 errors   # 22 个错误全部是缺少 numpy/verl 的既有收集错误
.venv/Scripts/python.exe -m pytest -q tests/test_usami_atomic_refund.py
11 passed
```

覆盖：正常退款、公开工具发现路径（无私有 id）、Engine/Reference 一致、超期、未送达、
数量超买、重复退款、审批上限、支付方式限制、公开 schema 一致性、witness 成本。

## 12. 对照用户六条要求

1. **方案 A**：只新增一个原子业务工具，Case/Quote/Settlement 保留在内部；Outcome 与
   `evaluate_state_goals` 未迁移。
2. **Delegation**：Profile 自动创建、Agent 不可见，只校验 line 范围；Customer 协议
   `{message, finish}`，没有关键词匹配。
3. **工具接口**：Phase 1 只暴露四个工具；退款工具一次完成校验 + 记账；归属/商品/数量/
   期限/重复退款/支付方式检查全部保留。
4. **预算可行性**：认证 witness 记录成本，注册前比对预算，不可达任务不进入训练。
5. **不伪造 Evidence**：普通退款不读不写 evidence，case.evidence_id 为 None。
6. **语义区分**：返回 `refund_issued`（退款已记账），不声称物理退货已完成；`Return` 实体
   留到 Phase 3。

## 13. 评审后的两处修正

评审指出两个问题，已修：

1. **测试绕过了公开查询**：原测试直接把 fixture 的私有 line id 交给脚本 Service。
   现新增 `test_public_tool_path_discovers_the_line_and_refunds`：Service 不接收任何
   line id / world / alias，只凭 `list_customer_orders → get_order_details` 的返回结果
   （line 的 `sku.product_family` 与 `sku.attributes.interface`）选出目标行再退款；
   为支撑这一点，`get_order_details` 的公开结果里补充了嵌套的 SKU 描述符
   （Engine 的 `observations.query_data` 与 Reference 的 `_expected_query` 同步修改）。
2. **Witness Cost 低估**：认证 witness 原来直接用 `goal.line_id` 一步退款，成本记 1/2/1/2。
   现在 witness 先执行 `list_customer_orders` 与 `get_order_details`，再退款，成本变为
   3/4/1/4，与真实公开路径一致；证书哈希重新冻结为
   `623283f88d162f978f05254331ba88b4a57a3446ff00253f6c1666fe505118a3`。

## 14. 未解决问题

- 远端 `npu30001` 的 checkout 仍落后 21 个提交且有未提交改动，尚未同步本次改动；
  NPU smoke 未运行。
- `tests/test_verl_bridge.py` 因缺 verl 依赖无法收集，其中的旧 `{scope_id,...}` 工具调用
  夹具需要在新环境里重跑并更新。
- 部分退款（多 unit line 只退 1 件）与多订单/多商品任务尚未有测试；Phase 2 补。
- `ReturnRequest` / 物理退货、换货、取消订单留待 Phase 3。
