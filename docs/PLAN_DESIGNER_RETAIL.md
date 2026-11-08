Status: implementation guide, nothing below is implemented yet (2026-11-08)
Owner: RIFT-RSI maintainers
Last verified: 2026-11-08 (against commit 7876a05)
Scope: let the Designer (the shared-weight Teacher) write the new retail tasks, and let those tasks train the Solver

# 让 Designer 出新格式（retail）题：实现指导书

这份文档写给实现者（Codex）。读完应当能不再追问就动手。每一节都写了：改哪个文件、新增什么、规则是什么、怎么测、做到什么程度算完成。

## 0. 先读这些

| 文件 | 是什么 |
| --- | --- |
| `third_party/usami/usami_v2/scenarios/retail_tasks.py` | 新题的任务规格 `RetailTask`、目标 `RetailGoal`、渲染、金标准推导 `gold_operations`、判分 `evaluate_retail_task`、认证 `certify_retail_task`、13 个题族的生成器 `generate_retail_task` |
| `third_party/usami/usami_v2/runtime/retail_runner.py` | 新题的回合运行器 `run_retail_episode` |
| `third_party/usami/usami_v2/commerce/*` | 引擎：地址、付款方式、`modify_order_items`（只能一次）已实现，`transitions.py` 执行、`reference.py` 独立判定 |
| `src/rift/designer/format.py` | 旧的 `DesignerPlan` 格式、解析、翻译成 Usami 草稿 |
| `src/rift/designer/prompt.py` | 旧的 Designer 提示词、示例、抄袭分 `copy_score` |
| `src/rift/designer/stage0.py` | `certify_designer_plan`（一份方案的认证结果行）、`select_tasks`（去重和选题） |
| `src/rift/designer/failures.py` | 把 Solver 的失败回合写成 Designer 能读的失败案例 |
| `src/rift/runtime/teacher.py` | 老师采样、`certify_samples`、`plan_rewards`（`4p(1-p)`）、老师 GRPO 更新 |
| `src/rift/runtime/self_teach.py` | 自教主循环：`generate_plans` → `certify_samples` → Solver 训练 → 老师更新 |
| `src/rift/runtime/curriculum_run.py` | `WorldKit`（任务登记簿、`certify_specs`）、`parse_task_specs`、`plan_identity`、CLI |
| `src/rift/verl/task_registry.py` | `CertifiedTaskRegistry`：只认旧任务（`commerce-task-v3/v4`） |
| `src/rift/verl/host_provider.py` | 训练回合开局：从登记簿取任务，建 `NativeUsamiSession` |
| `src/rift/verl/usami_session.py` | 把旧的 `run_episode` 包成逐步会话 |
| `src/rift/verl/agent_loop.py` | 会话结束后：持久化回合、`public_episode` 重放、算奖励 |
| `src/rift/verl/capture_store.py` | 保存和读回回合记录（只认 `EpisodeRecord`） |
| `docs/DISCUSSION_PENDING_CHANGES.md` 第 6 节 | 为什么做这件事、难度测量结果 |

现在的状态：新题能由程序按种子生成、认证、跑回合、判分，**但 Designer 写不了，训练链路也不认**。旧题的全部链路不动。

## 1. 目标和不做的事

**完成的样子：**
1. Designer 用一种新的、紧凑的方案格式 `RetailPlan` 写题；环境把它编译成世界和 `RetailTask`，认证后进登记簿。
2. 自教循环可以用 `--plan-format retail` 运行：老师只出新格式的题，Solver 的训练池、探针池、留出评测集也都是新格式的题（由 13 个题族生成）。
3. 老师的奖励、去重、抄袭检测、失败案例对新格式都成立。
4. 不加 `--plan-format retail` 时，所有旧行为（旧题、旧格式、`full9` 工具档位、旧登记簿）与现在逐字节一致。

**这次不做：**
- 旧格式和新格式混合（一次运行里只有一种格式）。
- 大模型客户（新题的客户仍是脚本：确认一次，然后结束）。
- 多客户、共享大世界、身份验证、物流单号、转人工。
- 在 GPU 或 NPU 上跑任何东西（第 7 节由我们自己跑）。

## 2. 总体设计：方案 B（Designer 只写决策变量）

旧格式让 Designer 写出完整世界（每个 SKU、仓库、库存、行、价格），4B 模型写长 JSON 容易出错，产出率只有约 13.5%。新格式只让 Designer 写**会影响正确答案的变量**，其余由引擎按种子填充：

| Designer 写 | 引擎填 |
| --- | --- |
| 有几个订单、每个订单的状态（未发货/已发货/已送达）、用什么付款 | 客户、商家、仓库、配送路线 |
| 每件商品的品类名和价格 | 商品的颜色和档位属性（保证每个变体能被唯一描述） |
| 每个可选变体：属于哪件商品、价格、有没有货 | 库存数量（有货 = 3，没货 = 0），原商品的预留库存 |
| 礼品卡余额 | 三种付款方式的 id（礼品卡、信用卡、PayPal 各一张） |
| 请求：改订单地址、改默认地址、改商品（首选和备选、付款偏好、都不行怎么办）、是否要求告知总额 | 所有地址（街道、城市、邮编），订单之间互不相同 |

**关键原则：** 一个编译器服务两个来源。13 个题族的生成器也要改写成"生成一份场景描述（`RetailScene`），再编译"。这样：
- Designer 和生成器产出的题完全同构，示例可以直接从生成器得到；
- 示例到方案的反向转换（`plan_from_scene`）自然存在；
- 只维护一套编译和认证。

分层（遵守"Usami 与 RIFT 分开"的约定）：
- **Usami 侧**（`third_party/usami`）：`RetailScene` 数据模型和编译器、请求到目标的翻译、有效性规则、决策签名、回合记录和重放。
- **RIFT 侧**（`src/rift`）：`RetailPlan`（给模型约束解码用的 JSON 格式）、方案翻译、认证结果行、抄袭分、提示词、登记簿、训练链路接入、CLI。

## 3. 阶段一（Usami 侧）：场景描述和编译

### 3.1 新文件 `usami_v2/scenarios/retail_scene.py`

数据模型全部用 `Sealed`（与 `retail_tasks.py` 相同的封存模式）：

```python
class SceneItem(Sealed):         # an item bought in an order
    ref: str                      # e.g. "A1"; unique in the scene
    product: str                  # product family name, e.g. "blender" (lowercase words, 1-3 words)
    price_cents: int              # 100..1_000_000

class SceneOrder(Sealed):
    ref: str                      # e.g. "A"
    status: Literal["unshipped", "shipped", "delivered"]
    pays_with: Literal["gift_card", "credit_card", "paypal"]
    items: Tuple[SceneItem, ...]  # 1..3

class SceneVariant(Sealed):      # another variant of one item's product
    ref: str                      # e.g. "X"
    of: str                       # an item ref
    price_cents: int
    in_stock: bool

class SceneRequest(Sealed):      # discriminated by "type"
    type: Literal["set_order_address", "set_default_address", "change_items"]
    order: Optional[str]          # order ref; required except for set_default_address
    to: Optional[str]             # address: "new" | "profile" | "order:<ref>"
    changes: Tuple[SceneChange, ...] = ()   # change_items: [{item, want: [variant refs, best first]}]
    pay_with: Tuple[Literal["original", "gift_card", "credit_card", "paypal"], ...] = ("original",)
    otherwise: Literal["nothing", "cancel_order"] = "nothing"

class SceneTask(Sealed):
    requests: Tuple[SceneRequest, ...]      # 1..3
    ask_total: bool = False

class RetailScene(Sealed):
    gift_card_balance_cents: int            # 0..1_000_000
    orders: Tuple[SceneOrder, ...]          # 1..4
    variants: Tuple[SceneVariant, ...] = () # 0..8
```

结构规则（违反就抛 `SceneError(code, message)`，`code` 用下表）：

| code | 条件 |
| --- | --- |
| `DUPLICATE_REF` | 订单、商品、变体的 ref 有重复（三者共用一个命名空间） |
| `UNKNOWN_REF` | 请求或变体引用了不存在的 ref |
| `ITEM_NOT_IN_ORDER` | `change_items` 改的商品不在该订单里 |
| `VARIANT_OF_OTHER_ITEM` | `want` 里的变体不是这件商品的变体 |
| `ADDRESS_SOURCE_INVALID` | `to` 不是 `new`/`profile`/`order:<ref>`；`set_default_address` 用了 `profile`；`order:<ref>` 指向自己 |
| `EMPTY_REQUEST` | `change_items` 没有 `changes`，或某个 `want` 为空 |
| `LIMIT` | 数量超过上面写的范围 |

### 3.2 编译 `compile_scene(scene, seed) -> (World, ids)`

- 用 `ScenarioSpec` + `build_world`（已支持付款方式、地址、订单付款方式）。一位客户 `me`；一个仓库 `w`（2 天、免运费）；三种付款方式各一张（礼品卡余额 = `gift_card_balance_cents`）。
- 每件商品是一个 SKU，品类 = `product`；每个变体是同品类的另一个 SKU。属性 `color` 和 `tier` 由 `random.Random(f"retail-scene:{seed}")` 分配，**同一品类内每个 SKU 的 (color, tier) 组合互不相同**，且每个变体至少有一个属性值与原商品不同。
- 每件商品数量 1、支付金额 = 价格；未发货订单的商品按现有规则在第一个仓库预留库存。变体 `in_stock` 为真则库存 3，否则 0；原商品默认库存 3。
- 每个订单一个收货地址，客户一个默认地址，互不相同；`to: "new"` 的目标地址再另外生成一个，也不与现有地址相同。地址从 `retail_tasks.py` 现有的城市和街道表里取。
- 两个订单的商品集合若完全相同（同品类同属性），渲染无法区分订单：编译阶段直接抛 `AMBIGUOUS_TARGET`。
- `ids`：场景 ref → 世界 id 的映射（订单、订单行、SKU、付款方式），再加 `"me"`。
- 同一 `(scene, seed)` 必须得到同一个世界（加测试）。

### 3.3 请求到目标 `scene_task(world, ids, scene_task, *, task_id) -> RetailTask`

一一对应到现有的 `RetailGoal`：
- `set_order_address` → `action="set_order_address"`，`address_source`：`new`→`stated`（目标地址填生成的新地址），`profile`→`profile`（`address` 填客户默认地址），`order:X`→`other_order`（`source_order_id`=X 的 id，`address` 填 X 的地址）。
- `set_default_address` → 同理，只允许 `new` 和 `order:X`。
- `change_items` → `wishes`（每件商品一个 `ItemWish`，`targets` = 变体 SKU id 按 `want` 顺序），`payment_preferences`（`original` 原样；三种种类映射到对应付款方式 id），`fallback`=`otherwise`。

### 3.4 有效性规则（新增，补在 `certify_retail_task` 里）

在现有检查之外加两条，失败抛 `ValueError`，消息以代码开头：
1. `TASK_NO_STATE_CHANGE`：金标准最终状态的 `projection` 等于初始世界的 `projection`。**理由：** 难度测量发现"已发货订单改地址"这类题，什么都不做就能拿满分。新规则下，"应当拒绝"的请求只能与至少一个"应当执行"的请求放在同一道题里。
2. `ENGINE_REFERENCE_MISMATCH`：金标准路径上执行与判定不一致（现有检查，只是给个固定代码）。

渲染无法唯一指代时，`render_retail_request` 已经抛 `AMBIGUOUS_TARGET`，保持不变。

### 3.5 13 个题族改写成场景

- 每个题族改成一个函数：`family_scene(family, rng) -> (RetailScene, SceneTask)`，数值仍由种子随机。保持 `generate_retail_task(seed, family)` 的签名和返回值不变，内部改为 `compile_scene` + `scene_task` + `certify_retail_task`。
- `address_delivered` 题族按 3.4 第 1 条必须改：在同一道题里加一个可执行的请求（例如同时改另一个未发货订单的地址）。改名为 `address_delivered_mixed`，更新 `FAMILIES`。
- 现有测试 `tests/test_usami_retail_tasks.py`、`test_usami_retail_runner.py`、`test_retail_model_service.py` 必须继续通过（允许修改断言具体世界内容的地方，不允许放宽"理想 Service 全部成功、什么都不做的失败、字面选第一偏好的在有陷阱的题族上失败"这几条性质）。
- 世界和任务的哈希会变化，这是允许的：新题还没有进入任何登记簿。

### 3.6 决策签名 `retail_signature(world, task) -> str`（新文件 `usami_v2/scenarios/retail_signature.py`）

用于去重。只描述**决策结构**，不含具体数值、名字、地址。对每个目标按顺序写出：
- 动作类型；所在订单的状态；地址来源（`stated`/`profile`/`other_order`）；
- 改商品：每件商品最终选了第几个偏好（0、1、2 或"都不行"）；
- 付款：差价方向（多付、退回、零）；最终用了第几个付款偏好，以及它的种类；礼品卡是否够付；
- 是否触发了 `otherwise`；
- 照字面执行第一偏好时引擎返回的拒绝代码集合（用 `certify_retail_task` 里已有的"朴素路径"算）；
- 这个目标是否改变了状态；
再加上 `ask_total`。把这个结构做成规范 JSON，返回其 SHA-256。测试：同一题族不同种子、数值不同但分支相同的题，签名相同；改变任一分支，签名不同。

### 3.7 回合记录与重放（阶段二要用，但属于 Usami 侧，放这里实现）

在 `retail_runner.py` 里：
- `run_retail_episode` 增加参数 `cancel_check=None`；Service 抛异常时，若 `cancel_check()` 为真，状态记为 `cancelled`，否则 `agent_error`（与 `run_episode` 一致）。
- 每次 Service 决策记录它的 `AgentCall`（`action.call` 或 `service.last_call`），每次工具调用保留 `_execution` 返回的 `ToolExecution`。
- 新增 `RetailEpisodeRecord(Sealed)`，`protocol="retail-episode-v1"`，字段至少有：`task`、`initial_world`（未加授权的世界）、预算参数、`model_calls`（只含 Service 的 `AgentCall`，顺序与决策一致）、`tool_executions`、`service_messages`（每条至少有 `text` 字段，`failures.py` 读 `row.text`）、`conversation`、`status`、`terminal_reason`、`final_world`、`outcome`（`state_match`、`told_total`、`success`）。`run_retail_episode` 返回值里带上这个记录。
- 新增 `replay_retail_episode(record) -> 对象(verified: bool, errors: list)`：从 `delegated(initial_world)` 出发，按顺序用 `_execution(..., "reference")` 重放每条工具调用，核对每条结果和前后状态哈希、核对最终世界、重新计算 `evaluate_retail_task` 并与 `outcome` 一致。

## 4. 阶段一（RIFT 侧）：Designer 能写、能认证

### 4.1 新文件 `src/rift/designer/retail_format.py`

`RetailPlan`（pydantic，`extra="forbid"`，结构与 3.1 一致，用 `_Strict` 基类，请求用 `type` 作判别字段，便于约束解码）：

```json
{
  "hypothesis": "The agent charges the gift card even when its balance cannot cover the difference.",
  "gift_card_balance_cents": 300,
  "orders": [
    {"ref": "A", "status": "unshipped", "pays_with": "credit_card",
     "items": [{"ref": "A1", "product": "blender", "price_cents": 5400},
               {"ref": "A2", "product": "sneaker", "price_cents": 6100}]},
    {"ref": "B", "status": "delivered", "pays_with": "gift_card",
     "items": [{"ref": "B1", "product": "kettle", "price_cents": 3900}]}
  ],
  "variants": [
    {"ref": "X", "of": "A1", "price_cents": 6600, "in_stock": true},
    {"ref": "Y", "of": "A1", "price_cents": 5000, "in_stock": false}
  ],
  "tasks": [
    {"requests": [
       {"type": "change_items", "order": "A", "changes": [{"item": "A1", "want": ["Y", "X"]}],
        "pay_with": ["gift_card", "paypal"], "otherwise": "nothing"},
       {"type": "set_order_address", "order": "B", "to": "new"}],
     "ask_total": false}
  ]
}
```

- `tasks`：1 到 3 道题，共用同一个世界（与旧格式一样，一份方案的奖励是其各题可学性的平均）。
- 提供 `retail_plan_schema()`（JSON Schema，给约束解码）、`parse_retail_plan(text_or_dict)`（复用 `format.extract_json`，校验失败抛 `PlanError('SCHEMA', ...)`）、`retail_plan_scene(plan) -> (RetailScene, [SceneTask])`（结构错误把 `SceneError` 的代码原样转成 `PlanError`）、`plan_from_scene(scene, scene_tasks, hypothesis) -> RetailPlan`（反向，用于示例）。

### 4.2 认证：`certify_retail_plan(world_kit, text, plan_id, world_seed, *, signatures=False) -> row`

放在 `src/rift/designer/retail_stage0.py`。返回的行**与 `stage0.certify_designer_plan` 的键完全相同**，这样 `select_tasks`、`TeacherSample.usable`、`plan_rewards`、自教日志都不用改：

| 情况 | `status` | `reasons` |
| --- | --- | --- |
| 解析或结构错误 | `format_rejected` | `[PlanError.code]` |
| 编译失败（`AMBIGUOUS_TARGET`、建世界失败） | `world_rejected` | `[代码]` |
| 有的题认证失败 | `partly_certified` / `tasks_rejected` | 每道失败题一个代码（`AMBIGUOUS_TARGET`、`TASK_NO_STATE_CHANGE`、`ENGINE_REFERENCE_MISMATCH`） |
| 全部通过 | `certified` | `[]` |

- `task_refs`：每道通过的题的 `RetailTask.record_hash`；`answers`：`'success'`（新题都要求改变状态）；`actions`：每道题的请求类型列表；`signatures`：`retail_signature`（`signatures=True` 时）。
- 每道通过的题注册到 4.4 的登记簿。
- `plan_id` 和 `world_seed` 决定 `compile_scene` 的种子：`seed = world_seed`。

### 4.3 抄袭分

`retail_copy_score(plan_value, examples)`：沿用 `prompt.copy_score` 的思路（方案里大于 100 的整数有多大比例出现在同一个示例里），再加一条：若方案的请求结构（去掉 ref 名字后的 `tasks`）与某个示例完全相同，且金额重合率不低于 0.5，记为 1.0。阈值仍用 `stage0.COPY_THRESHOLD = 0.5`。

### 4.4 登记簿 `src/rift/verl/retail_registry.py`

`RetailTaskRegistry`，接口与 `CertifiedTaskRegistry` 一致：`task_refs`、`register(record)`、`resolve(ref) -> (world, task, None, ())`、`resolve_record(ref)`、`purpose(ref)`、`verify_task(ref, world, task, profile, *, purpose=None)`（`profile` 必须为 `None`）、`save()`、`load(path, split_registry)`。
- 记录：`CertifiedRetailTask(world, task, certificate, identity)`，`certificate` 是 `certify_retail_task` 的返回字典。
- 保存到 `task_registry/retail-tasks.json`，格式仿照 `certified-tasks.json`（`schema_version`、`records_digest`）。`load` 时对每条记录重新运行 `certify_retail_task`，结果必须与保存的证书一致，否则拒绝加载。
- 身份：`Identity(world.state_hash, task.record_hash, family, lineage, Split.TRAIN, DataPurpose.TRAIN)`，在 `split_registry` 注册。`family`：Designer 的题用 `'retail:designer'`，生成器的题用 `'retail:gen:<题族>'`；`lineage` 是规范 JSON（Designer：`plan_id`、`world_seed`、方案哈希；生成器：题族、种子）。
- `WorldKit` 增加 `retail_registry` 属性（与旧登记簿并存，`save()` 一起保存）。再加一个 `TaskResolver`（组合解析器）：按 ref 在哪个登记簿里决定走哪个；旧登记簿的行为不变。

### 4.5 提示词 `src/rift/designer/retail_prompt.py`

`retail_designer_messages(failures, *, seed, examples=2, feedback=(), focus=False)`，结构仿照 `prompt.designer_messages`。系统提示要写明：
- 环境规则（与引擎一致，不能多写也不能少写）：
  - 只有未发货订单能改地址；默认地址随时能改。
  - `modify_order_items` 只能用于未发货订单，只能换成同一商品的其他有货变体，**每个订单只能调用一次**，调用后订单不能再改也不能取消，所以所有修改要一次提交。
  - 只有未发货且未锁定的订单能取消，取消是整单。
  - 差价：多付时可用客户任一付款方式，礼品卡余额必须够；退回时只能退到原付款方式或礼品卡。
  - 已发货、已送达的订单不能改商品、不能改地址。
- 方案格式说明（每个字段的含义和范围），以及"环境会自动生成名字、属性、地址和 id"。
- 你不需要说请求该不该成功，环境会推导；**每道题至少要有一个能执行的请求**，否则会被丢弃。
- 好题的特征：把数值放在规则边界上（礼品卡余额略低于或略高于差价）、首选缺货、把能做和不能做的请求放在一起、地址引用另一个订单或资料、一次要改多件商品、需要"都不行就取消"。
- 不要抄示例或失败案例；假设（`hypothesis`）用自己的话写要训练的弱点。只输出一个 JSON 对象。

示例：`example_retail_plans(seed, count)`，从 13 个题族里按种子轮换抽 `count` 个，随机数值，经 `plan_from_scene` 转成方案，`hypothesis` 写中性文字（参照 `prompt.example_plans` 的注释：题族名字会被模型抄进答案）。

### 4.6 失败案例 `src/rift/designer/failures.py`

- `goal_rows` 增加对 `RetailTask` 的分支 `retail_goal_rows(world, task)`：用金标准路径和朴素路径写出人能读懂的"隐藏答案"，例如：
  - `item: the order containing the red basic lamp | asked: change shipping address to my profile address | correct: change it to 701 Oak Street, Seattle`
  - `item: the grey basic blender | asked: switch to the white blender, paying with gift card or else PayPal | correct: white blender, paid with PayPal (gift card balance 3.00 < difference 12.00)`
- `failure_cases` 对两种登记簿都能工作（经 4.4 的组合解析器；回合记录经第 5.4 节的 `resolve_episode`）。`agent_steps` 读 `tool_executions`，新记录已兼容。
- 失败案例只能来自 TRAIN 划分（现有检查保留）。

### 4.7 开关

- `teacher.teacher_prompt_ids`、`teacher.certify_samples`、`self_teach.generate_plans` 增加参数 `plan_format='legacy'`；`'retail'` 时分别用 `retail_designer_messages`、`certify_retail_plan` + `retail_copy_score` + `example_retail_plans`、`retail_plan_schema()`。
- `stage0.py` 的 CLI 增加 `--plan-format`；`--dry-run` 在 retail 模式下用 `example_retail_plans` 代替模型输出。
- 默认值全部是 `legacy`，旧行为不变。

### 4.8 阶段一的测试和验收

新增测试（全部 CPU、无网络）：
1. `compile_scene` 确定性；编译出的世界通过 `invariants`；结构错误各给出对应代码（3.1 表中每个代码至少一个用例）。
2. 13 个题族经新编译器后：现有三个测试文件全部通过；`address_delivered_mixed` 的"什么都不做"不再成功。
3. `retail_signature`：同分支不同数值签名相同；改一个分支签名不同。
4. `RetailPlan` 往返：`plan_from_scene` → `parse_retail_plan` → `retail_plan_scene` 得到同一个场景。
5. `certify_retail_plan`：每个示例方案都 `certified`；故意损坏的方案得到预期的 `status` 和 `reasons`；抄示例的方案 `retail_copy_score >= 0.5`。
6. `RetailTaskRegistry`：保存后加载得到同样的记录；篡改文件后加载失败。
7. `stage0 --dry-run --plan-format retail` 以真实规模运行（64 份方案）通过，报告里认证率为 1.0。
8. 全量测试（`tests` 和 `third_party/usami/tests`）通过。

## 5. 阶段二（RIFT 侧）：接入训练链路

### 5.1 任务规格

`curriculum_run.parse_task_specs` 增加一种规格：`{"retail_family": "<题族名>|all", "instances": n, "seed": s}`。`WorldKit.certify_specs` 对它调用生成器，注册到 `retail_registry`，返回 `{spec_key: [task_ref]}`；`spec_key` 用 `'retail:<题族>'`。像现有的 `spec-refs.json` 一样缓存已认证的 `(题族, 种子)`。新增配置文件：
- `configs/experiments/retail_train.json`：13 个题族 × 16 个实例，种子从 20,000,000 开始；
- `configs/experiments/retail_probes.json`：每个题族 24 个，种子从 21,000,000 开始；
- `configs/experiments/retail_heldout.json`：每个题族 8 个，种子从 29,000,000 开始。
三者种子范围互不重叠（`self_teach` 已检查评测集与训练、探针不相交）。

### 5.2 会话

- `NativeUsamiSession.start()`：当 `self.task` 是 `RetailTask` 时，后台线程调用 `run_retail_episode(world, task, BlockingService(), customer, max_service_decisions=budget.max_service_decisions, max_tool_calls=budget.max_service_tool_calls, max_customer_turns=budget.max_customer_turns, cancel_check=...)`，`finished` 事件里放 `RetailEpisodeRecord`。其余逐步交互逻辑不变。
- 新题的客户：`ScriptedRetailCustomer`（`customer_factory` 为新题返回它；若设置了 `RIFT_CUSTOMER_URL`，本阶段直接报错"新题暂不支持大模型客户"，不要静默回退）。
- `host_provider.open_rollout`/`open_probe`：任务是 `RetailTask` 时要求 `tool_profile == 'retail'`，否则报错；授权检查走组合解析器。

### 5.3 判分和奖励

- `benchmark/usami.public_episode(record)`：遇到 `RetailEpisodeRecord` 时用 `replay_retail_episode`，再投影成 `PublicEpisode`（事件序列：客户请求、工具调用和结果、Service 回复）。结局：重放失败 → `UNKNOWN`；`status == 'cancelled'`（解析器失败、预算取消）→ `UNKNOWN`；`outcome.success` 为真 → `SUCCESS`；其他 → `FAILURE`。
- `agent_loop` 里读 `episode.replay_verified` 的地方，对新记录用重放结果。

### 5.4 回合存储

`capture_store.persist` 只读 `episode.model_calls` 和 `record_hash`，新记录已兼容；`resolve_episode` 按 `raw['episode_record']['protocol']` 分派，`retail-episode-v1` 时解析为 `RetailEpisodeRecord`。

### 5.5 自教 CLI

`curriculum_run.py` 的 `self-teach` 和 `stage0` 增加 `--plan-format {legacy,retail}`：
- `retail` 时必须同时给 `--tool-profile retail`，并且 `--train-specs`、`--probe-specs`、`--eval-specs` 里只能是 retail 规格；不满足就在开头报错。
- `--drop-references` 在 retail 模式下不适用，给了就报错。
- 新增 `--dedup-scope {action,run}`，默认 `action`：签名去重只在同一次 action 内进行。（现在的代码是跨整个运行累计去重，产出率因此逐次下降；这个默认值可以改，见第 8 节。）

### 5.6 阶段二的测试和验收

1. 用现有测试里的假运行时（参考 `tests/test_self_teach.py`、`tests/test_matched_arms.py` 的写法），以 `--plan-format retail` 跑 2 次 action：老师方案来自 `example_retail_plans`，认证、选题、Solver 训练请求、老师奖励和更新请求都正常产生。
2. 会话层：理想脚本 Service 经 `NativeUsamiSession` 跑完一局 retail 任务，持久化、读回、重放通过，奖励 1.0；什么都不做的得 0.0；模拟解析器失败得 `UNKNOWN`。
3. 失败案例：从一局失败的 retail 回合得到含隐藏答案的失败案例。
4. 旧路径：不带新参数时，所有现有测试和 `docs` 里记录的命令行为不变。
5. 全量测试通过。

## 6. 交付和提交

- 按阶段提交：阶段一 Usami 侧、阶段一 RIFT 侧、阶段二，各一个或几个提交，提交信息说明做了什么、怎么验证的。直接推到 `master`（`hjy8848/Rift-rsi`）。
- 每个阶段结束时更新 `docs/DISCUSSION_PENDING_CHANGES.md` 第 6 节的"仍未接入"清单。
- 在本文件末尾追加"实现记录"：实际改了哪些文件、与本文不同的决定及理由、未完成项。

## 7. 实现之后由我们自己做（不要实现，供参考）

1. 在 NPU 上做阶段零：冻结的 Qwen3-4B 用 `--plan-format retail` 写 256 份方案，量认证率、被拒原因分布、签名多样性。旧格式的认证率约 13.5%，这是对照。
2. 产出率合格后，跑短的自教（`--plan-format retail --tool-profile retail`），看留出集曲线和老师题的占比。
3. 再决定是否实现"Solver 只用 Designer 的题训练"（文档第 3 节已定的方向）。

## 8. 约束和默认值

**必须遵守：**
- 不改旧路径的行为、序列化和哈希。新代码全部在 `plan_format='retail'` 或新类型的分支里。
- 不在 GPU/NPU 上运行；不访问网络；不删除任何运行目录或检查点。
- 不改 Usami 引擎的业务规则（`commerce/*`）。如果发现引擎的问题，记在实现记录里，不要顺手修。
- 所有随机性来自显式种子；同样输入同样输出。
- 拒绝时给出固定的原因代码，不要只有自由文本。

**采用的默认值（可以由用户改，实现时做成参数）：**

| 项 | 默认 | 说明 |
| --- | --- | --- |
| 一份方案的题数 | 1 到 3 | 与旧格式（最多 4）相近，减少长度 |
| 去重范围 | 同一次 action 内 | `--dedup-scope run` 恢复现有的累计去重 |
| 有货变体的库存 | 3 | 没货为 0 |
| 每道题必须改变状态 | 是 | 堵住"什么都不做也得分"的漏洞 |
| 抄袭阈值 | 0.5 | 与旧格式相同 |
| 新题的客户 | 脚本客户 | 大模型客户以后再做 |
