# Usami Phase 3：取消订单、换货/补发、实物退货

日期：2026-11-01　基线：`master` = `82b95d994d543b802c34dc47d2b676d77b3f5965`　工作分支：`phase3-business-types`

## 1. 结论

Phase 3 在 Phase 1/2 的原子退款之上增加三种业务能力，Agent 侧仍然是"一个业务动作 = 一次工具调用"，
没有重新暴露 Case / Quote / Evidence / Settlement，也没有新增 Consent 协议：

| 业务 | 新增公开工具 | 复杂度来自哪里 |
| --- | --- | --- |
| 取消订单 | `cancel_order(order_id)` | 订单必须未发货；整单取消；只释放真实预留的库存 |
| 换货 | `list_exchange_options(line_id)`（只读）+ `exchange_delivered_order_items(line_id, quantity, target_sku_id)` | 兼容性、真实库存、差价、时效 |
| 补发 | `reship_order_items(line_id, quantity)` | 已送达、窗口内、已核实的送达问题、真实库存 |
| 实物退货 | `request_item_return(line_id, quantity)`，退款仍用 `return_delivered_order_items` | 申请受理 → 外部收货事件 → 才能退款 |

Agent 可见工具共 9 个（Phase 1/2 的 4 个不变）。所有参数扁平，没有 `params` / `scope_id`。

## 2. 新增状态与业务不变量

### 2.1 数据结构（均为"默认值时不序列化"，旧记录哈希不变）

| 结构 | 新增字段 | 说明 |
| --- | --- | --- |
| `Order` | `fulfillment_status: unshipped / shipped / delivered`，默认 `delivered` | 非 `delivered` 时 `delivered_at` 表示预计送达时间 |
| `OrderLine` | `reserved_inventory_id: Optional[str]` | 未发货订单行真实占用的库存余额；取消时只释放它 |
| `Quote.remedy` / `Settlement.remedy` | 取值增加 `cancel` | 取消沿用 Case/Quote/Settlement/Ledger 记账，但对 Agent 不可见 |
| `TaskSpec` | `cancellation_goals: Tuple[OrderCancellationGoal]` | 仅 `commerce-task-v3`；为空时不出现在序列化中 |
| `OrderVariable`（World Builder） | `fulfillment_status` | `unshipped` 订单在第一个仓库为每个行预留库存 |
| `LineSelector`（Grounding） | `fulfillment_status` | 让 Designer 能选"未发货订单" |
| `TaskDraft` | `cancellation_goals: Tuple[CancellationDraft]` | 与 remedy / information 目标并列 |

`Return`（既有模型）已足以表达"受理 / 收货"，**没有引入 `ReturnRequest`**：

- 退货申请被受理 = `Return.status == "scheduled"` 且 case 为 `awaiting_return`，无 Ledger、无 Settlement；
- 商品已退回 = 外部事件 `return_received` 把 `Return` 置为 `received`、case 置为 `ready`；
- 退款已完成 = 之后的 `return_delivered_order_items` 写 Settlement + Ledger，case 变 `resolved`。

Agent 的任何工具都不能把 `Return` 改成 `received`；该状态只能由 `external.py` 的冻结事件产生。

### 2.2 新增/调整的不变量（`commerce/invariants.py`）

- `LINE_RESERVATION_INVALID`：只有未发货订单的行才能持有预留，且必须指向同一 SKU 的真实库存。
- 库存守恒：未取消的未发货行计入 `reserved`；取消后预留归还 `available`，总量不变（不会凭空增加库存）。
- `CANCELLED_ORDER_NOT_CANCELLABLE` / `ORDER_CANCELLATION_PARTIAL`：取消必须是整单、全部数量、且订单必须是未发货。
- `CANCELLATION_TERMS_INVALID`、`REFUND_HAS_REPLACEMENT_TERMS`：取消与退款一样不能带替换条款。
- 换货/补发的 `Fulfillment` 与 `Settlement` 一一对应；退款/取消不允许有 Fulfillment。

### 2.3 拒绝码

取消：`ORDER_ALREADY_CANCELLED`、`ORDER_NOT_CANCELLABLE`、`CANCELLATION_CONFLICT`、
`BUSINESS_DELEGATION_REQUIRED`、`OBJECT_ACCESS_DENIED`。
换货/补发：`EXCHANGE_TARGET_NOT_FOUND`、`REPLACEMENT_FAMILY_MISMATCH`、`EXCHANGE_TARGET_SAME_AS_ORIGINAL`、
`RESHIP_NOT_ELIGIBLE`、`INSUFFICIENT_STOCK`、`PURCHASED_QUANTITY_UNAVAILABLE`、`ALREADY_SETTLED`、
`APPROVAL_REQUIRED`、`ORDER_NOT_DELIVERED`、`RETURN_WINDOW_EXPIRED`、`RETURN_REQUIRED`。
退货：`RETURN_NOT_REQUIRED`、`RETURN_ALREADY_REQUESTED`、`RETURN_NOT_RECEIVED`、`RETURNED_QUANTITY_MISMATCH`。

## 3. 各业务规则

- **取消**：订单属于当前客户且为 `unshipped`；其行未被任何 Settlement/Case 占用；整单每行按实付金额退回原支付渠道；
  行若有 `reserved_inventory_id`，对应数量从 `reserved` 回到 `available`。已发货/已送达的订单不能取消，也不能被当成"已送达商品退款"
  （`return_delivered_order_items` 现在要求 `fulfillment_status == delivered`）。
- **换货**：目标 SKU 必须同商户、同 `product_family`、不同于原 SKU，且在某仓库有足够 `available` 并有配送路线；
  选"最快、最便宜"的仓库；差价 = 目标价 × 数量 + 运费 − 原行按 unit 分摊的实付，正数记 `extra_charge`，负数记 `refund`（受审批上限约束）。
- **补发**：同 SKU 免费补发；要求该行存在对该客户有效的送达问题记录（商户内部事实，不对 Agent 公开，Agent 只会得到 `RESHIP_NOT_ELIGIBLE`）。
- **换货/补发都**要求：已送达、在退货窗口内、不是 `return_required` 商品、数量未被占用。
- **实物退货**：`return_required=True` 的商品，`return_delivered_order_items` 在收货事件前返回 `RETURN_NOT_REQUIRED/RETURN_REQUIRED/RETURN_NOT_RECEIVED`；
  收货后按已受理的数量和金额退款；窗口内受理的退货在窗口外收货仍可退款（以受理时间为准）。

## 4. Engine 与 Reference 的独立实现

- `transitions.py`（Engine）直接修改状态；`reference.py` 在 `world.clone()` 上用自己的规则重新推导 `expected_state` 与 `data`，
  两个模块互不 import 对方的实现（Reference 只共享 schema、稳定 ID 与 `unit_credit` 整数分摊）。
- 取消、换货、补发、退货申请、收货后退款、`list_exchange_options` 都各写了两遍：例如换货的"最快仓库"Engine 用 `min`，
  Reference 用排序元组；`list_exchange_options` 在 `observations.py` 与 `reference._expected_query` 分别实现。
- Oracle 同时比较 `after == expected_state` 与 `result.data == decision.data`；每个新操作的测试都经过 Engine → Oracle → 轨迹 Replay，
  并另有 `engine_matches_reference` 直接比对两侧结果。
- World Builder 的独立审计 `scenarios/certification.verify_world_constraints` 同步理解了未发货订单与预留库存，
  对不一致的世界仍会拒绝。

## 5. Task / Certification / Reward

- **Designer**：`TaskDraft.cancellation_goals` + `LineSelector.fulfillment_status`，可生成单订单取消、单商品换货、单商品补发、
  实物退货，以及它们的任意多目标/多订单组合。公开请求仍由 Host 从结构化目标渲染（"Please cancel my order with the … before it ships."），
  没有任何自然语言关键词/正则判断授权。
- **每个目标独立判定**：取消目标检查"每行全部数量已 `cancel`、退款金额等于实付、同一支付渠道、无 Fulfillment"；
  换货/补发沿用 `RemedyGoal` 判定（SKU 属性、额外费用上限、到达期限）；退货退款沿用退款判定。
  整体成功仍是全部注册目标 PASS，且 `unrequested_settlements` 合规项通过。
- **越权**：Customer 的委托范围 = 全部 remedy 行 + 被取消订单的所有行（`delegated_line_ids(task)`）。范围外的工具调用被 Reference 拒绝
  （`BUSINESS_DELEGATION_REQUIRED`），状态不变，并计入过程错误。
- **Certification**：候选只列举"补救方式 + 兼容目标 SKU"；witness 全程通过公开工具（`list_customer_orders → get_order_details →
  [list_exchange_options] → 业务动作`），参数取自公开结果；库存、差价、时效由 Engine/Reference 在执行时决定，不预先伪造。
  `witness_cost` 现在按"业务动作次数"计算确认回合，已受理并已收货的退货的退款不再额外计一次确认。
- **UNKNOWN 而非伪造**：`return_required` 退款没有冻结的收货事件时，认证返回 `UNKNOWN`（blocker `return_receipt_event_not_configured`），
  Designer 侧也会因 `RETURN_RECEIPT_UNAVAILABLE` 拒绝编译，除非 Host 显式配置 `return_receipt_after_days`。
  配置后 Host 会把冻结的收货事件随认证记录一起登记，rollout 时按计划触发；默认 Customer 委托有效期会自动延长到覆盖收货延迟（≤ 28 天）。
- **Reward**：未改动。`business_goal_success → Outcome.SUCCESS/FAILURE → service_reward`，Replay 不通过则为 UNKNOWN。
  RIFT 侧公开转录对收货通知与 Service 所见一致，只给 `line_id` / `quantity`，不暴露内部 case ID。
- Service 系统提示词从"退款流程"改为"业务动作流程"：先说明动作、数量、差价与渠道 → 等客户用自然语言确认 → 才调用对应工具；
  对需要退回的商品必须等环境报告收货后才能退款。环境不建模结构化 Consent。

## 6. 测试结果

命令（工作目录 `C:\reallab\Rift-rsi`，Python 3.12.16，venv 内装 pydantic 2.14 / pytest 8.4 / numpy / PyYAML / jsonschema / hypothesis）：

```
.venv\Scripts\python.exe -m pytest -q --continue-on-collection-errors -p no:cacheprovider
```

| 范围 | 结果 |
| --- | --- |
| `tests/test_usami_phase3.py`（新增） | **56 passed** |
| Phase 1 `tests/test_usami_atomic_refund.py` | 21 passed, 1 xfailed（既有的"客户拒绝"设计边界） |
| Phase 2 `tests/test_usami_phase2.py` | 17 passed |
| `tests/test_usami_native.py`（含冻结哈希） | 16 passed |
| RIFT 全量 | **195 passed, 1 skipped, 1 xfailed, 5 failed** |
| 改动前同一环境基线 | 139 passed, 1 skipped, 1 xfailed, **同样的 5 failed** |
| 内置 Usami 自带测试（`third_party/usami`） | 118 passed, 10 failed；基线 115 passed, 13 failed，**无新增失败** |

5 个失败在改动前就存在，原因与本次无关：`test_core::test_imports_are_inert` 的子进程没有 `rift` 的 `PYTHONPATH`；
4 个 `test_verl_bridge` 用例仍在发送 Phase 1 之前的 `{scope_id, params}` 工具调用夹具（Phase 1 报告 §15 已记录）。
Usami 自带测试的失败是 Windows 下 `atomic_json` 的 `PermissionError` 等，基线 13 个、现在 10 个，**我不把差的 3 个算作修复**，
它们疑似不稳定。

Phase 3 覆盖（每个新操作）：成功路径；规则不满足；越权（他人订单/行、缺委托）；重复执行；库存/金额边界
（库存不足、无法整除分摊、审批上限、数量超买）；Engine/Reference 一致；Oracle 通过；轨迹 Replay；
最终 Goal / Reward（成功为 1.0，缺一个目标、提前退款、超额外费用上限均为 0.0）；篡改的 Episode 无法通过 Replay。

## 7. 历史协议兼容

没有升级任何协议字符串（`commerce-2.0.0`、`commerce-task-v3`、`commerce-episode-v1`、`commerce-outcome-v2`）。
新增字段在默认值时不参与序列化与哈希，所以旧记录的哈希不变：

- `test_usami_native.py` 里冻结的世界/任务/证书/Episode/Outcome 哈希全部仍然匹配；
- 用**未修改的基线代码**（`82b95d9`）真实写出的 Phase 1 与 Phase 2 Episode JSON，用 Phase 3 代码做离线
  `verify_episode_artifact`，两者均 `verified=True` 且哈希相同（Phase 1 `c9211ae1…`、Phase 2 `a0732050…`）；
- `test_pre_phase3_records_keep_their_exact_shape_and_hashes` 固化"旧世界/任务不含新键且可由 `parse_task_spec` 无损读回"。

反向限制：含 `cancellation_goals`、`fulfillment_status != delivered` 或 `remedy == "cancel"` 的新记录，**旧代码读不了**（模型 `extra=forbid`）。
对公开订单行，`fulfillment_status` 只在订单未送达时出现；没有该字段表示已送达，工具描述里已写明（这样保证旧查询输出逐字节不变，旧 Episode 仍能 Replay）。

## 8. 未解决问题

1. **NPU Smoke 没有运行，也没有任何真实模型 Reward 提升的证据。** 所有 Episode 都是脚本化 Service/Customer 驱动，只证明环境、认证与 Replay 正确，
   不能当作模型效果。
2. Service 工具从 4 个增加到 9 个、提示词变长，对真实模型的上下文预算与工具选择准确率**未测量**。
3. 换货/补发没有建模"旧商品退回"；为此对 `return_required` 商品一律拒绝换货/补发（`RETURN_REQUIRED`），偏保守。
4. 补发资格依赖商户内部的送达问题记录（对 Agent 不可见），Agent 只能从拒绝码得知，可能出现试错。
5. 环境仍不校验客户是否口头确认（延续 Phase 1/2 的设计边界与既有 xfail）。
6. 取消只支持整单；部分行取消、取消后部分退款留待后续。
7. `tests/test_verl_bridge.py` 的 4 个旧夹具失败仍待在装有 verl 的环境里更新。
8. 收货事件由 Host 配置的固定延迟冻结；不支持"迟迟不到货/丢件"之类的随机外部事件。

## 9. Git

- 分支 `phase3-business-types`，基于 `master@82b95d9`，已本地提交（提交哈希见 `git log`）。
- **推送状态：未推送。** 本机没有已登录的 GitHub 凭据，且推送会改动远程仓库，需要你确认后再做。
- 工作区未改动、未清理任何用户文件；测试用 Python / uv 装在 `C:\reallab\.tools`、`.venv`（已被忽略，不在提交里）。
