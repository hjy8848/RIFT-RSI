# Phase 2：部分退款、多商品与多订单，多目标 Certification / Reward / Replay

日期：2026-10-31
分支：`master`，本地提交 `7e01024`（`origin/master` 仍是 `7ee0ace`，推送受阻，见 F 节）

## 结论摘要

- 退款仍是**一次调用的原子操作**，未新增实体、工具、状态机或多阶段事务。
- 审计结果表明：Phase 1 的单位记账（`unit_credit` 按 unit 余数分摊）与 Engine/Reference 的
  `available` 单位选择已经能表达部分退款、同订单多行、多订单组合；Phase 2 只需补业务级
  验证与认证成本核算，**未修改 invariants、未放宽任何业务规则**。
- 多目标任务沿用既有 `TaskSpec`/`RemedyGoal`/`evaluate_state_goals`/`Outcome` 架构：
  逐目标独立求值，`business_goal_success` 只有在全部目标 PASS 时才为 True。
- 唯一的生产代码改动：认证 Witness 成本改为按“实际退款次数 / 确认回合数”推导，
  消除多目标任务里固定 2 个 customer turn 的虚假成本。
- 新增 `tests/test_usami_phase2.py`（17 个用例），并补充 `tests/test_usami_atomic_refund.py`
  的回归用例；全量 `87 passed, 1 skipped, 1 xfailed, 22 errors`，其中 22 个 errors
  是环境缺 `numpy`/`verl` 造成的既有收集错误（与本次改动无关，见 C 节）。

## A. 实施总结

### Step 1：代码审计（已有能力 / 缺口）

| 能力 | 状态 | 依据 |
| --- | --- | --- |
| 部分退款金额分摊 | 已有 | `commerce/models.py:306-311` `unit_credit()` 用 `divmod` 余数分摊，10001/3 → 3334/3334/3333 |
| 单位占用与重复退款 | 已有 | `commerce/transitions.py:83-134` 从 settlements/cases 汇总已占 unit，`ALREADY_REFUNDED` / `PURCHASED_QUANTITY_UNAVAILABLE` |
| 同订单多行独立退款 | 已有 | 以 line 为资源；`_hard_scope` 校验 line→order 的客户与商家 |
| 多订单组合 | 已有 | 每个 goal 绑定一个 line，同一客户不同订单互不影响；跨客户走 `OBJECT_ACCESS_DENIED` |
| 多目标 Goal Evaluation | 已有 | `scenarios/tasks.py:610-677` 逐 goal 收集新增 settlement 并独立判 PASS/FAIL |
| 多目标 Certification 组合枚举 | 已有 | `task_certification.py` `product(*option_sets)` 枚举每个 goal 的合法候选 |
| 认证成本 | **缺口** | 旧 `_CONFIRMATION_TURNS = 2` 固定值，多目标时低估 customer turns 与 model calls |
| 公开定位（不暴露 Hidden Goal） | 已有（Phase 1 评审修复） | `get_order_details` 每行附公开 `sku{product_family,attributes,price_cents,return_days}` |

结论：Phase 2 的主要工作是**验证 + 修正认证成本**，而非重写退款语义。

### Step 2：P2.1 部分退款

- 复用 `unit_credit()`，未引入 `paid // quantity` 之类取整逻辑。
- 新增用例覆盖：3 件分三次（1+1+1）、一次性 2 件再 1 件、退 4 件被拒、
  不可整除金额（10001 分 3 次）、累计金额等于 `allocated_paid_cents`、单位不重叠。
- 认证侧新增 `test_certification_uses_remaining_unit_credit_after_partial_refund`：
  先部分退款，再认证剩余数量，审批上限按剩余 unit credit 重新计算。

### Step 3：P2.2 同订单多商品

- 构造三行订单（laptop×2 / mouse×1 / cable×2），SKU 属性（product/interface/category）
  用于公开区分两个同类设备。
- 用例验证：只退选中行、两行分别退、错误商品不满足目标、目标数量与退款单位数必须一致、
  超量被拒；公开 schema 中不出现 case/quote/settlement 等内部对象。

### Step 4：P2.3 多订单组合

- 同一客户两个订单（第一单 laptop+cable，第二单 mouse），另有一个别的客户的订单。
- 用例验证：两单分别退款且各自 goal PASS；只完成一单时 `business_goal_success=False`；
  跨客户行被拒（`OBJECT_ACCESS_DENIED`）；把订单 id 当 line id 传入返回 `RESOURCE_NOT_FOUND`；
  两单端到端 episode 的 Replay 校验通过（operation_count=5）。

### Step 5：P2.4 多目标 Task / Certification / Reward / Replay

- 认证 Witness 真正走公开工具：`list_customer_orders → get_order_details → return_delivered_order_items`，
  每个 goal 一次，订单/行 id 均取自公开结果，找不到对应公开行则认证失败
  （`certified_target_order_not_publicly_visible` / `certified_target_line_not_publicly_visible`）。
- `_witness_cost` 改为按操作与确认回合推导：

  ```
  operations      = witness 中 kind == "operation" 的步骤数
  refunds         = operations 中原子退款次数
  confirmations   = refunds + consents + 1          # 每次退款一次确认 + 一次收尾回复
  tool_calls      = operations
  service_decisions = operations + confirmations
  customer_turns  = confirmations
  model_calls     = operations + 2 * confirmations
  ```

  单目标退款：`3/5/2/7`；双目标退款：`6/9/3/12`。
- 单目标不可达 → `UNKNOWN`（blocker），proof budget 耗尽 → `UNKNOWN`（不是 `UNSAT`）。
- 新增 v3（RIFT 原生训练协议）双目标端到端用例：认证 CERTIFIED → `CustomerProfileV3` 绑定 →
  两个脚本退款 → `status=customer_finished`、两个 goal 均 PASS、`business_goal_success=True`、
  Replay verified。
- `v3` 任务使用普通 `RemedyGoal`（v3 的 `remedy_goals` 不承载 `public_request_span`/`subject_mention`），
  公开文本由 `render_customer_request` 从结构化目标渲染，`CustomerRequestBundle` 绑定渲染后文本的哈希。
  这是既有协议行为，本次只按它构造测试，未改动协议。

## B. 修改文件清单

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `third_party/usami/usami_v2/scenarios/task_certification.py` | 重写 `_witness_cost`，删除固定 `_CONFIRMATION_TURNS` | 多目标时固定 2 个 customer turn 会低估预算；改为按退款次数/确认回合推导 |
| `tests/test_usami_phase2.py`（新增） | 17 个用例 + 多行/多订单 world 与脚本 Agent | P2.1–P2.4 的业务、认证、Runtime、Replay 覆盖 |
| `tests/test_usami_atomic_refund.py` | 补回归用例（跨客户、公开路径、Replay 断言、单目标认证成本、认证预算耗尽、部分退款后的剩余额度、公开参数来源、双行/双单认证）；1 个 xfail 显式标注“对话确认是 Prompt 契约、环境只验证业务规则” | 锁定 Phase 1 行为，并为 Phase 2 边界提供对照 |

未修改：`commerce/{models,transitions,reference,invariants}.py`、`runtime/{agents,customer,runner}.py`、
`scenarios/tasks.py`、RIFT `src/rift/**`、reward/outcome 路径、任何协议字符串与锁文件。

## C. 测试结果

命令（工作目录 `D:\RealLab\RSI\RIFT-RSI`）：

```
.venv/Scripts/python.exe -m pytest -q tests/test_usami_phase2.py
.venv/Scripts/python.exe -m pytest -q tests/test_usami_atomic_refund.py
.venv/Scripts/python.exe -m pytest -q --continue-on-collection-errors
```

| 范围 | 结果 |
| --- | --- |
| `tests/test_usami_phase2.py` | 17 passed |
| `tests/test_usami_atomic_refund.py` | 21 passed, 1 xfailed |
| 全量 | **87 passed, 1 skipped, 1 xfailed, 22 errors** |

22 个 errors 均为收集期错误，原因是本机 venv 缺 `numpy`/`verl` 等可选依赖（`test_ability_registry`、
`test_core`、`test_dual_bank`、`test_production_runtime`、`test_training_package`、`test_verl_bridge`），
与本次改动前基线一致（改动前为 `62 passed, 1 skipped, 22 errors`）。

未运行/无法运行：上述 6 个文件在安装 `verl`/`numpy` 的环境里需要重跑；未在 NPU 上跑真实模型 smoke。

xfail 说明：`test_customer_refusal_prevents_atomic_refund` 断言的是“客户拒绝后仍会执行退款”这一
**当前设计边界**——对话确认由 Service Prompt 约束，Environment 只验证业务规则（评审明确要求不加
结构化 Consent）。客户拒绝时 Service 不调用退款的行为，由 `tests/test_usami_phase2.py::test_a_customer_refusal_prevents_the_refund`
在 Agent 层验证。

## D. 业务正确性

- **单位分配**：每次都取当前未被 settlement 占用、且不属于进行中 case 的最小 unit 索引；
  重复/越界索引在 `unit_credit` 内抛错。
- **金额分摊**：`divmod(allocated_paid_cents, quantity)` 的余数分给低位 unit，多次退款之和
  严格等于 `allocated_paid_cents`（10001 → 3334+3334+3333）。
- **多商品定位**：只依赖公开 `get_order_details` 返回的 `sku{product_family, attributes, price_cents, return_days}`
  与行 id；区分同 family 的两个设备靠 attributes，歧义时 Agent 应向客户确认（Prompt 要求），
  不读取 Hidden Goal 的内部 id。
- **多订单隔离**：`_hard_scope` 校验 `order.customer_id == actor` 且 `order.merchant_id == scope`，
  跨客户被 `OBJECT_ACCESS_DENIED` 拒绝；未知行 id 返回 `RESOURCE_NOT_FOUND`。
- **全目标判定**：`evaluate_state_goals` 逐 goal 独立求值；未请求行上的 settlement 计入
  `evaluate_task_compliance` 的 `unrequested_settlements`；`business_goal_success` 为所有目标
  PASS 的合取，因此“部分成功”必然是非 SUCCESS，RIFT 侧 reward 相应为 0。
- **幂等/重复**：已全额退款后再退返回 `ALREADY_REFUNDED`；数量超过剩余单位返回
  `PURCHASED_QUANTITY_UNAVAILABLE`。

## E. 架构影响

- 新增实体：无。工具接口：无新增，仍是 `list_customer_orders` / `get_order_details` /
  `return_delivered_order_items` / `respond_to_customer`。
- Task 协议：未改动；v2 仍用 `GroundedRemedyGoal`，v3 仍用普通 `RemedyGoal` + 编译生成的
  `CustomerRequestBundle`。历史 Task/Episode/Certificate/Outcome 的读取路径未变。
- Reward：未改动；`business_goal_success → Outcome.SUCCESS/FAILURE` 的二值映射保持原样，
  Replay 失败仍不能产生成功。
- RIFT 核心（RSI/GRPO/训练循环）：未触及。

## F. Git 状态

- 分支 `master`，最终本地提交：`7e01024 Support partial, multi-line and multi-order atomic refunds`
  （3 个文件，+901/−15）。
- 提交内容：`tests/test_usami_phase2.py`（新增）、`tests/test_usami_atomic_refund.py`、
  `third_party/usami/usami_v2/scenarios/task_certification.py`。
- **推送状态：未推送。** `origin/master` 仍为 `7ee0ace`。本机到 `github.com:443` 的 TLS 握手被中断
  （`git push`/`curl`/Python `urllib` 均报 `unexpected eof while reading`），HTTP/SOCKS5 实验室代理
  需要认证（407 / SOCKS5 auth rejected），SSH 身份是 `hongxingli319` 且对 `hjy8848/RIFT-RSI` 无推送权限。
  网络恢复后执行：`git -c credential.helper=helper-selector push origin master`。
- 未提交项：无源码改动；工作区仅剩历史遗留的未跟踪目录（`.pytest-*`、`.test-codex-*`、
  `third_party/usami/build/`），本次未清理、未纳入提交。

## 未解决问题

1. 推送受阻（F 节），需要网络恢复或用户本机推送。
2. 真实模型 smoke 未跑（需要远端 NPU 环境与模型服务）。
3. 未安装 `verl`/`numpy` 的 6 个测试文件未能在本机验证。
4. v3 协议不承载 grounding 字段：若未来需要 v3 任务“公开定位可审计”，需要单独评审
   `TaskSpecV3.remedy_goals` 的类型（本次未改，以保持协议兼容）。
