# Usami 世界与任务生成：修改方案

（2026-11-03。依据：通读 `third_party/usami/usami_v2/scenarios/`、`commerce/transitions.py`、`commerce/observations.py`、`runtime/agents.py`、`src/rift/benchmark/families.py`、`src/rift/benchmark/usami.py`。本文是 `PLAN_DESIGNER_STAGE.md` 第 4 节的展开。）

本文只讨论 Usami **能生成**什么样的世界和任务，不讨论某个固定题库是否多样。前半部分解释为什么要改，后半部分给出每项具体怎么改。

---

## 0. 约束：无论怎么改都不能破坏的东西

1. **判分权在 Usami。** 出题方（F 生成器或 Designer）只描述世界和客户的请求，不能决定什么算对。判分仍由 Engine、Reference、Oracle 和 `evaluate_state_goals` 完成。
2. **世界合法。** 每个世界都要通过 `invariants`，并由 `verify_world_constraints` 独立复核。
3. **任务可识别。** 只凭公开信息就能唯一确定要处理哪一行、几件（`PLAN_DESIGNER_STAGE.md` 2.2）。
4. **独立验证。** 任务要有 Reference 见证或拒绝证明，并且证书可以重算。
5. **旧记录可读、哈希不变。** 沿用 Usami 已有的做法：新字段在默认值时不进入序列化（`omit_default_fields`）；语义有变化时升协议版本号，旧版本继续可解析。X2、X5 的任务和证书必须还能重放。

---

## 1. 为什么要改

### 1.1 现在的题目实质上只是换名字

`families.py` 的 `_construct` 把所有数值都写成了固定值：

| 量 | 值 |
| --- | --- |
| 商品价格、已付金额 | 8000 |
| 审批上限 | 10000 |
| 送达天数 | 1 |
| 仓库 | fast 2 天、slow 5 天，运费都是 0 |
| 默认库存 | 4 |
| 退货期限 | 30 天（默认值） |
| 品类名、干扰品 | `product_0`、`distractor_i` |

种子只做两件事：给型号加后缀 `_variant_{seed}`，以及改变内部 ID。`_expand_instances` 生成多任务世界时，也只是把同一世界复制 N 份，再加一个 `instance` 属性。近似题过滤 `fingerprint.py` 专门忽略 `model` 和 `instance`，说明作者也清楚这些只是名字。

**后果**：同一 F 类的所有题，客服要做的判断完全相同。训练时“新题”其实是同一道题；X5 里老虎机反复选同一类、出现负迁移，与此直接相关。

### 1.2 正确结果由出题方预先声明，所以数值不敢随机

- 出题时必须先声明这道题是“应该成功”（`remedy_goals` / `cancellation_goals`），还是“应该拒绝，原因是 X”（`refusal_goals` 的 `reason_code`）。
- 而正确结果其实取决于数值：退款额是否超过审批上限、购买天数是否超过退货期限、库存够不够、加价是否超过客户上限等等。
- 所以 F 生成器只能手工保证一致，例如 `(amount > limit) != refusal` 时直接报错。一旦随机化数值，声明的标签就可能和世界对不上，认证失败。
- Designer 也会遇到同样的问题：它必须自己算对边界，否则出的题通不过认证。这件事本该由 Usami 负责。

### 1.3 请求文字泄露了答案

`render_customer_request`（`tasks.py:492`）对成功目标和拒绝目标用了**不同的句式**：

| 目标 | 渲染结果 |
| --- | --- |
| 退款，应成功 | `Please arrange a refund for the … item in my order.` |
| 退款，应拒绝 | `Please refund the … in my order.` |
| 取消，应成功 | `Please cancel my order with the … before it ships.` |
| 取消，应拒绝 | `Please cancel the … in my order.` |

模型只看句式就能知道该执行还是该拒绝，不需要去读订单、比较日期或金额。这是**标签泄露**。对已有结果的影响：X2、X5 的拒绝类成绩可能部分来自句式，结论需要打折看待；但“老虎机在拒绝类上负迁移”这一现象方向不变。

### 1.4 很多真实情形根本表达不了

| 情形 | 卡在哪里 |
| --- | --- |
| “之前已经退过一件，现在要退剩下的” | `verify_world_constraints` 和 `_bind_scenario` 要求初始世界的 cases、settlements、returns、ledger 全为空。业务规则本身能处理这种情况（`_free_units`、`ALREADY_REFUNDED` 都在） |
| “3 件里退 1 件、换 1 件” | `TaskSpec.actual_goals` 规定一行只能有一个目标 |
| “三周前买的那副耳机”“贵的那个” | 唯一性检查和请求都只用“问题类型 + 属性 + 品类”（`_line_description`）描述商品，客服从不需要靠日期、价格、所在订单来分辨 |
| “几个款式里只有一个满足颜色、加价和到货期限” | F6、F7 只有一个候选款，Usami 不会生成款式网格 |
| “同款商品，一单过期了一单没过期” | 干扰品都是不同品类的 `distractor_i`，一眼就能排除 |

### 1.5 求解器在范围变宽时会失败

`solver.py:50` 在取值组合超过 256 种时，只是随机抽 256 次，没有约束传播。Designer 一旦写出宽范围加紧约束，就基本只能得到 UNKNOWN，失败原因也只记录在单次尝试里，没有汇总。

### 1.6 无法度量“是否真的多样”

现有指纹只比较表面字段：品类名、属性、数值分桶。它既会把“只换品类名”误判为新题，也看不出两道外表不同的题其实考的是同一个判断。没有度量，后面所有改动都无法验收。

---

## 2. 具体怎么改

按推荐的实施顺序排列。每项给出目的、改动、位置、兼容性和验收方法。

### D5 判断签名（先做，用来验收其余改动）

**目的**：给每道题算一个与名字、ID 无关的签名，表示“客服需要做的判断”。签名相同的两道题就视为只换了名字。

**签名内容**（对每个目标）：
1. 请求的动作：refund / exchange / reship / cancel。
2. 正确结果：成功，或拒绝码。
3. 见证路径的抽象工具序列，例如 `list → details → request_item_return → [receipt] → return_delivered`。只记工具名，不记 ID。
4. **关键事实集合**：把世界里的每个数值逐个扰动，看正确结果是否翻转；会翻转的字段类型就是关键事实，比如 `{line.age vs sku.return_days}`、`{credit vs approval_limit}`。
5. 指代方式（见 U5）。
6. 可行候选数，适用于换货目标。

题目层面再加上：目标数量、目标之间是否耦合（共享库存或同一行）、近似干扰项的个数。

**扰动方法**：对 `BuildResult.assignment` 里的每个数值，以及审批上限、退货期限、客户加价上限和期限，尝试几种取值：`x−1`、`x+1`、`0`、`x/2`、`2x`，以及“正好越过最近阈值”的值。用 U1 的结果推导重新算正确结果。Reference 是确定性的、在 CPU 上运行，每道题大约几十到一百多次单步执行，开销可以忽略。

**位置**：新文件 `usami_v2/scenarios/signature.py`，输出一个 `DecisionSignature(Sealed)`。`src/rift/probes/fingerprint.py` 增加按签名去重的入口，旧接口保留。

**验收**：
- 对现有 F1–F11 各生成 50 个种子：每个 F 类应只出现 1 个签名（证实 1.1）。
- 人工构造两道“名字不同、判断相同”的题和两道“名字相同、判断不同”的题，签名要分别相同、不同。

### U1 先定请求，再由 Usami 推出正确结果

**目的**：出题方只写客户的请求；Usami 在具体世界上算出每个请求应该成功还是被拒绝。数值因此可以自由随机，标签跟着世界走。

**新的出题格式**（加在 `TaskDraft` / `AbstractChallenge` 里，与原有目标字段并存）：

```text
RequestDraft:
  goal_id
  line_alias              # 或 AbstractChallenge 里的选择器变量
  action: refund | exchange | reship | cancel
  quantity
  required_attributes     # 仅 exchange / reship
  maximum_extra_charge_cents
  latest_arrival_within_days
```

`TaskDraft` 增加 `requests: Tuple[RequestDraft, ...] = ()`。这个字段与 `remedy_goals`、`refusal_goals` 等**互斥**：新格式只用 `requests`，旧格式保持不变。

**推导算法**（新函数 `derive_outcomes(world, customer_id, requests)`，放在 `designer.py` 或新文件 `scenarios/outcomes.py`）：

1. 对每个请求，把它当作“允许该动作的目标”，用现有的候选语法 `_base_candidate_options` 和 `_run_candidate` 在初始世界上执行：
   - 有见证 → 生成 `RemedyGoal`；动作是 cancel 时生成 `OrderCancellationGoal`。
2. 没有见证时，用 `refusal_certification._operations` 构造该动作的原子操作，取 `adapter.decide` 的 `error_code`：
   - 原因码在 `RefusalGoal.reason_code` 允许的集合里 → 生成 `RefusalGoal(requested_action=action, reason_code=…)`。
   - 操作本身能成功、但报价超出客户的加价上限或期限 → `CUSTOMER_DECLINED`，同时客户配置 `accepts_proposals=False`。这一点已由 `src/rift/benchmark/usami.py:97`、`:107` 根据拒绝目标自动设置，无需另改。
   - 其他错误码（如 `MUTATION_NOT_REPRESENTABLE`、`EXCHANGE_TARGET_NOT_FOUND`）→ **不出题**，返回原因 `UNSUPPORTED_OUTCOME:<code>`。
3. 多目标时，先在初始世界上逐个独立推导，再整体跑一遍原来的 `certify_task` / `certify_refusal`。
   - 逐个能成功、整体却失败（如两个换货抢同一份库存）→ 返回 `GOALS_CONFLICT`，不出题。
   - 冲突题本身也是有价值的题型，留到 D4 再支持。
4. 推导出的目标交给原来的 `compile_task` 和认证流程。**认证仍然独立重算**，推导只负责给出标签，本身不当作证明。

**同时修复 1.3 的标签泄露**：请求的渲染只看请求本身（动作、商品、件数、约束），**与推出的结果无关**，同一个请求无论成功还是被拒都渲染成同一句话。这需要新版渲染器 `commerce-request-renderer-v2`（见 U6）；v1 渲染器保留给旧任务。

**位置**：`designer.py`（`TaskDraft`、`compile_task`）、`grounding.py`（`AbstractChallenge` 的“每个选择器都有目标”检查要把 `requests` 算进去）、新的推导模块、`tasks.py`（渲染器 v2）。

**兼容性**：旧格式任务、旧证书完全不变；新格式任务使用 v2 渲染器。

**验收**：
1. 把 F1–F11 的每个默认题和拒绝版改写成 `requests` 形式：推出的目标必须与原来声明的完全一致，F3、F4、F5、F6、F8 的拒绝版要推出相同的原因码。
2. 对同一个请求扫一遍数值（例如退款额从上限以下扫到上限以上），推出的结果要在阈值处正确翻转。
3. 同一请求的成功版和拒绝版，渲染出的文字逐字相同。

### D1 按边界随机取值

**目的**：让数值真正变化，并且集中在决定结果的阈值附近。

**改动**：新增随机化生成器，不再写死数值。F1–F11 的固定版本保留，作为基线和冒烟测试。

| 阈值 | 涉及的量 |
| --- | --- |
| 审批上限 | 退款额 = 单件分摊额 × 件数（`unit_credit`）与 `refund_approval_limit_cents` 的比较 |
| 退货期限 | 购买天数（`delivered_days_ago`）与 `return_days` 的比较 |
| 库存 | 各仓库库存与请求件数的比较 |
| 加价上限 | 目标价 × 件数 + 运费 − 分摊额，与客户上限的比较 |
| 到货期限 | 最快有货仓库的送达天数与客户期限的比较（引擎自动选最快仓库，`_replacement_source`） |

**采样规则**（每个阈值独立）：
- 约 40% 落在阈值的 −1、0、+1 处（金额以分为单位，天数以天为单位）；
- 约 60% 在合理宽范围内均匀取值。
- 退货期限、审批上限、仓库数（2–4 个）、每个仓库的送达天数和运费、件数（1–5）、价格都随世界变化。

**做法**：随机数在生成器里用种子算好，再以单点范围写入 `ScenarioSpec`，不依赖求解器。这样可复现、可控，世界构建走完全枚举分支，与现有路径一致。

**位置**：新文件 `src/rift/benchmark/generator.py`（RIFT 侧）。阈值和采样规则写成数据表，便于以后给 Designer 当作参考。

**验收**：
- 每个原子题型生成 200 个种子，用 D5 统计：不同签名数应远大于 1。
- 成功和拒绝两种结果都应占到合理比例（例如各不低于 25%）。

### D2 款式网格

**目的**：换货题需要在多个候选里做比较推理。

**改动**（生成器侧）：
1. 每个品类有 1–3 个属性轴，例如颜色 3 个值、容量 3 个值、尺寸 2 个值，从中抽 k 个变体（k = 3–8）。
2. 价格 = 基价 + 各属性档位溢价 + 小幅随机。
3. 每个变体在每个仓库的库存随机，包括 0。
4. 客户约束 = 若干指定属性、可选的加价上限、可选的到货期限。
5. 生成器用与引擎相同的规则，即 `list_exchange_options` 的候选和 `_replacement_source` 的选仓，算出每个变体可不可行、违反哪条约束，以此控制题目结构：
   - “恰好 1 个可行，其余各违反一条约束”；
   - “没有可行项”，由 U1 推出拒绝结果；
   - “多个可行”。

生成器自己的判断只用于挑题，最终结果仍以 U1 和认证为准。

**Usami 侧的一处必要改动**：如果网格让同一客户名下出现多行描述相同的商品，唯一性检查会拒绝出题。这是正确的行为，生成器需要避免或配合 U5。

**位置**：`src/rift/benchmark/generator.py`。

**验收**：网格题的 D5 签名中，“可行候选数”和“关键事实集合”应覆盖多种组合；抽样人工读 10 道，确认要选对款式必须比较多个字段。

### U6 渲染器 v2：表述可变，且仍然确定、可验证

**目的**：一是修掉 1.3 的标签泄露；二是让请求不再是一个固定模板；三是为 U5 的新指代方式提供表达。

**改动**：
1. `CustomerRequestBundle.renderer_version` 增加 `commerce-request-renderer-v2`，任务记录增加 `style_seed`。
2. 渲染是 (结构化目标, 世界, `style_seed`) 的**确定性函数**。`check_task_scope` 用同样的输入重新渲染并逐字比较（现有机制），因此不需要反向解析就仍然可验证。
3. 可变的部分：
   - 开场句；
   - 是否编号、子句顺序；
   - 动作的同义说法，例如 refund 可以说成 “get my money back” 或 “return it for a refund”；
   - 指代方式（见 U5）；
   - 可选的一句无关背景。
4. **不可变的约束**：
   - 任何子句的措辞都不能依赖推出的结果（成功或拒绝）；
   - 件数、属性、加价上限、期限必须明确出现，现有的 `_validate_public_remedy_bounds` 规则继续适用。

**位置**：`tasks.py`（`render_customer_request` 增加 v2 分支、`CustomerRequestBundle`）、`designer.py`（`compile_task` 传入 `style_seed`）。

**验收**：
- 对每个新格式任务重新渲染，结果逐字一致；
- 成功版和拒绝版文字相同；
- 统计 1000 道题的请求，相同句式的比例低于设定阈值。

### U5 用其他事实指代商品

**目的**：让客服必须读订单详情、比较日期或价格，才能确定是哪一件。

**改动**：
1. 请求格式增加 `reference: attributes | delivery_age | price | order_date`，默认 `attributes`，也就是现在的行为。
2. `_line_description` 拆成按指代方式计算描述的函数：
   - `delivery_age` → 品类 + “大约 N 天前送达的”；
   - `price` → 品类 + “单价 CNY X 的”；
   - `order_date` → 品类 + “某日下单的那一单里的”。
   这些事实都在 `get_order_details` 的公开结果里：订单有 `delivered_at`，商品行有 `sku.price_cents`。
3. 唯一性检查（`compile_task` 末尾那段 `AMBIGUOUS_TARGET`，与 `PLAN_DESIGNER_STAGE.md` 2.2 合并实现）按同一指代方式计算：在该客户名下的所有商品行里，用这种指代方式得到的描述必须唯一。
4. 渲染器 v2 根据指代方式生成文字。

**限制**：未送达的订单没有“送达天数”，只能用其他指代方式；生成器要据此选择。

**位置**：`tasks.py`、`designer.py`。

**验收**：
- 构造“同款、不同送达日期”的两行：用 `attributes` 指代时必须被判为歧义，用 `delivery_age` 时必须通过；
- D5 签名里能区分出不同的指代方式。

### U4 一行商品可以有多个目标

**目的**：支持“3 件里退 1 件、换 1 件”。

**改动**：
1. `TaskSpec.actual_goals` 的“一行只能有一个目标”放宽为两条：
   - 同一行上各目标的件数之和不超过可用件数；
   - 同一行上各目标允许的补救方式**互不相交**。
2. `evaluate_state_goals`：同一行有多个目标时，按结算的 `remedy` 把结算记录归到对应目标，不再把这一行的全部结算都算给每个目标。由于补救方式互不相交，归属是唯一的。
3. `delegated_line_ids` 去重。
4. 认证：`_free_goal_units` 本来就按“当前空闲的前 n 件”取件，多个目标依次执行时不会重叠；`return_receipt_event` 用的也是空闲件，无需改动。
5. `AbstractChallenge`：放开“每个选择器恰好对应一个目标”的检查，允许一个选择器对应多个目标。
6. 渲染：同一行有多个目标时，每个子句都必须写明件数。

**兼容性**：放宽只影响新任务。为了不改动旧任务的语义，带多目标行的任务升协议为 `commerce-task-v4`，`TaskSpecV3` 保持原样。

**验收**：
- “买 3，退 1、换 1”可以认证通过；只退 1 件、或把 2 件都换掉，判为失败；
- 补救方式有重叠的两个目标被拒绝出题；
- 旧任务重放，结果不变。

### D3 近似干扰项

**目的**：干扰项只在关键事实上与目标不同，迫使客服去比较。

**改动**（生成器侧）：在同一客户名下加入下面这些行，每道题 0–3 个：

| 干扰类型 | 构造 | 依赖 |
| --- | --- | --- |
| 同款不同订单 | 同一 SKU 出现在另一张订单里，送达天数不同，一单在退货期内、一单已过期 | U5（否则会因描述相同而被判为歧义） |
| 同品类不同属性 | 只差一个属性值 | 无 |
| 同款但已处理过 | 同一 SKU 的另一行已经退过款 | U3 |

用 U5 的唯一性检查保证不产生歧义。用 D5 的关键事实集合确认干扰项确实是“近似”的：它和目标只在一个关键事实上不同。

**位置**：`src/rift/benchmark/generator.py`。

### D4 组合出题

**目的**：一道题由多个原子请求组合而成，组合空间远大于 11 个 F 类。

**改动**（生成器侧）：
1. 原子请求从“动作 × 关键阈值 × 指代方式 × 历史状态（U3）”中抽取，每道题 1–4 个。
2. 可选的耦合：
   - 两个换货指向同一个稀缺变体（共享库存）；
   - 同一行上有多个目标（U4）。
3. 有耦合的题，U1 会先返回 `GOALS_CONFLICT`。第二步再支持“冲突时的正确结果”：按请求顺序依次推导，后面的请求在前面请求执行后的状态上推导。这一步需要明确约定：客户先提的请求优先。约定写进渲染出的请求里，例如“先处理 A，再处理 B”，保证仍然可以从公开信息判断。

**验收**：D5 统计不同签名数随组合规模增长；抽样人工确认组合题的正确做法确实依赖子目标之间的关系。

### U3 允许由引擎执行出来的历史

**目的**：支持“已经部分退款”“退货已申请、尚未收到”“退货已收到、等待退款”“已经换过一次”等初始状态。

**改动**：
1. `ScenarioSpec` 增加 `history: Tuple[HistoryStep, ...] = ()`，默认值时不进入序列化，旧场景的哈希不变。
   `HistoryStep` 包含：`operation`（限 `BUSINESS_ACTIONS` 中的动作，或外部事件 `return_received`）、`line_alias` 或 `order_alias`、`quantity`、`target_sku_alias`（换货时）。
2. 构建：`_construct` 产出基础世界后，按顺序执行历史步骤。
   - 每一步的执行方式与 `task_certification._execute` 相同：Engine 的 reference 模式，加上 Oracle 和 `replay_trace`。
   - 每一步必须成功，否则构建失败，原因为 `HISTORY_STEP_FAILED:<i>:<error_code>`。
   - 历史执行需要的委托和授权，执行完后从世界里移除，不残留到初始状态。
3. `verify_world_constraints`：把“交易类集合必须为空”改成“必须等于从基础世界重放历史得到的状态”。授权类集合（delegations、consents）仍然必须为空。
4. `_bind_scenario`：同样只要求授权类集合为空，允许交易历史存在。`evaluate_state_goals` 本来就只统计初始世界之外的新增结算，`_free_goal_units` 会排除已经结算的件，这两处不用改。
5. **公开可见性**：客服现在没有工具能看到某一行已经退过几件。`get_order_details` 里的商品行要增加 `settled_quantity` 和 `return_status`，只在非零或非空时输出，这样没有历史的旧任务看到的内容逐字不变。

**位置**：`schema.py`、`builder.py`、`certification.py`、`task_certification.py`、`commerce/observations.py`。

**验收**：
- “买 3、已退 1、再退 2”可以认证通过，“再退 3”推出拒绝码 `PURCHASED_QUANTITY_UNAVAILABLE`。注意这个码目前不在 `RefusalGoal` 允许的集合里，需要一并加进去，或者由 U1 返回 `UNSUPPORTED_OUTCOME`，由出题方避开。
- 有历史的世界经 `verify_build` 重算后一致；
- 没有历史的旧世界，哈希和观察结果都不变。

### U2 求解器加约束传播

**目的**：Designer 写宽范围加约束时，求解器能找到解，失败时能说明原因。

**改动**：
1. `BuildBudget` 增加 `strategy: "sample" | "propagate"`，默认 `sample`，旧构建的哈希和结果不变。
2. `propagate` 分两步：
   - 先对每条线性约束做区间收窄，就是把 `impossible_by_interval` 的推理反过来用于缩小每个变量的范围，迭代到不再变化；
   - 再按变量逐个赋值，每赋一个值重新收窄一次，某个区间变空就回溯。仍受 `max_nodes` 限制，是确定性的。
3. 失败时，`BuildResult.reason` 汇总各约束被违反的次数，取前 3 条。这个信息作为给 Designer 的可读反馈，见 `PLAN_DESIGNER_STAGE.md` 2.3 第 3 条。

**位置**：`solver.py`、`builder.py`、`schema.py`。

**验收**：构造原方法会得到 UNKNOWN 的宽范围加紧约束例子，新方法能得到可行解；结果不满足约束时 `verify_build` 一定失败。

### U7 业务面扩展（最后单独讨论）

以下改动会改变所有基线，只列出来，暂不展开：
- 加入“申请审批”工具，并接上审批结果事件，让超额退款可以走审批后成功的路径；
- 对话过程中推进时间的事件（补货、路线变化）；
- 让 `issue` 和凭证影响更多规则。现在只有补发会检查凭证。

---

## 3. 实施顺序与验收

| 步骤 | 内容 | 是否改业务规则或工具 | 完成标准 |
| --- | --- | --- | --- |
| 1 | D5 | 否 | 现有 F 类各自只有 1 个签名；人工构造的对照例子签名判断正确 |
| 2 | U1 + U6（含修复标签泄露） | 否 | F1–F11 改写后推出的目标与原声明一致；成功版和拒绝版的请求文字相同 |
| 3 | D1、D2 | 否 | 新生成器下不同签名数显著增加；成功和拒绝两种结果都有合理占比 |
| 4 | U5、U4、D3 | 否（只改出题和判分时的归属） | 各自的反例和正例全部通过 |
| 5 | U3 | 只增加观察字段 | 有历史的题可认证；旧世界的哈希和观察不变 |
| 6 | D4 | 否 | 签名数随组合规模增长 |
| 7 | U2 | 否 | 宽范围约束能求解，失败原因可读 |
| 8 | U7 | 是 | 单独讨论 |

每一步都要满足：
- 旧任务和旧证书全部重放通过（X2、X5 的数据）；
- 新增单元测试先在 CPU 上全部通过；
- 提交前所有测试通过。

**对实验的影响**：
- 从第 2 步起，新生成的题和旧题不再可比，所有基线（随机、老虎机、原模型）都要在新生成器上重测。
- 由于 1.3 的标签泄露，X2、X5 中拒绝类的成绩可能偏高，报告时需要注明。

---

## 4. 风险

| 风险 | 应对 |
| --- | --- |
| U1 推导与认证结论不一致 | 认证始终独立重算；一旦不一致就不出题，并记录下来作为 bug 排查 |
| 题变难后，原模型成功率过低，学习信号消失 | D1 的边界采样比例和 D2 的可行候选数都可以调，用 X2 的方式先测原模型在各签名上的成功率，再定比例 |
| 渲染 v2 的同义说法与 `_positive_remedy_request` 等旧检查冲突 | v2 只用于新格式任务，旧检查只作用于 v1、v2 协议的任务，两者不交叉 |
| 新增拒绝码（如 `PURCHASED_QUANTITY_UNAVAILABLE`）要求客服能理解 | 只加入公开工具返回里已有的错误码，客服调用时能直接看到 |
| 工作量 | 前三步不改业务规则，可以先做先测；后面的步骤按需求推进 |
