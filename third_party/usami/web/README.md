# Usami World Explorer

本地交互式 SaaS 业务世界探索器。FastAPI 调用已有 Usami Python API，React 页面展示真实 WorldState、操作事件和独立 Oracle；前端不实现业务规则。桌面端优先。

## 安装与启动

从仓库根目录运行，需要 Python 3.9+、Node.js 20.19+（或 22.12+）与 npm：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[web,dev]' -c constraints-tested.txt
cd web/frontend
npm ci
cd ../..
python web/run.py
```

打开 <http://127.0.0.1:5173>。一键启动默认使用 API 8001、前端 5173；Ctrl+C 停止两项子进程。端口被占用时脚本会明确报错，可以设置 `USAMI_BACKEND_PORT` / `USAMI_FRONTEND_PORT`。

### 真人顾客实验台

这是一个本机运行的定性实验工具：真人扮演 Customer，真实 `OpenAIServiceAgent` 负责服务决策，业务操作仍走 `commerce-2.0.0` Engine、Oracle 和 Replay。当前提供一个已认证的“损坏耳机退款”场景。页面中的结束、取消、报价接受和报价拒绝都是显式 Customer 动作；普通文字不会创建报价同意。未完成会话按未决证据处理。

先在后端进程环境中设置模型和 OpenAI-compatible 服务地址，并按现有客户端配置后端凭据；不要把 Key 写进前端环境或浏览器。`web/run.py` 会将 UI 的本地来源加入 CSRF 白名单：

```bash
export USAMI_LLM_MODEL="your-model"
export USAMI_LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
# 凭据沿用 OpenAICompatibleClient 支持的后端环境 / 本机凭据配置
python web/run.py
```

打开 <http://127.0.0.1:5173/?mode=human>。未检测到真实模型配置时，页面会禁用开始按钮；不会切换到脚本客服。后端最多同时运行两个 Episode，每个浏览器最多一个，单次模型调用预算为 32，客服会话 30 分钟过期。运行数据写入仓库根目录 `human_records/<session-id>/`，目录权限限制为当前用户；此目录已加入 `.gitignore`。每次记录包括完整 Episode、任务与世界认证证书、Oracle、Outcome、成本和 Replay，并标记 `benchmark_eligible=false`、`training_evidence_eligible=false`。真人顾客数据只用于定性人工检查，不进入正式 Benchmark、训练或 Skill 进化。

关闭对话或取消时，系统先等待当前决策调用结束，再封存 Episode 并验证离线 Replay；因此界面会先显示“正在取消并封存”，随后才能下载完整报告。若模型调用一直没有返回，记录也会等待该调用结束，进程停止则内存中的活动会话无法恢复。

也可以在两个终端分别运行，支持后端热重载：

```bash
# Terminal 1: 仓库根目录，激活 .venv 后
uvicorn web.backend.app:app --reload --host 127.0.0.1 --port 8000

# Terminal 2
cd web/frontend
npm run dev
```

Vite 默认将 `/api` 代理至 8000。使用其他 API 端口时，例如：

```bash
USAMI_BACKEND_URL=http://127.0.0.1:8001 npm run dev
```

静态构建后也可以只运行一个 Python 服务（启动时检测 `dist`）：

```bash
cd web/frontend
npm run build
cd ../..
uvicorn web.backend.app:app --host 127.0.0.1 --port 8001
```

此时打开 <http://127.0.0.1:8001>。页面内部导航不依赖 URL 路由。停止再启动后端会清空所有内存会话。

### Research Studio 的反思证据

`?mode=research` 的 Stage 05 只消费人工明确选择、通过原生 Replay 的 train / S0 Episode。`usami-v2-reflection-evidence-v2` 为每次工具执行分配独立 Oracle 引用，并向 Reflector 提供原始业务结果、过程错误、安全和交互可靠性，以及过程/安全证据在 Episode 中的来源路径。`failures` 仍表示该字段中的已确认业务失败，不将所有工具拒绝改写成业务违规。实际展示给 Service 的工具 Schema 也从原始请求中提取。

反思视图用 `previous-request-prefix-v1` 对不断增长的 Service 请求去重：每条上下文保留前一条已还原请求的 `unchanged_prefix_messages` 条消息，再追加本条 `request_messages`，可精确还原每轮所见内容。完整请求与模型回复继续保存在原始 Episode 中。

旧版、未标记证据协议的 NOOP 可以在页面显式选择一条归档 Episode，用修正后的证据重新反思一次。该动作会调用模型；浏览页面不会自动调用。旧结果完整保存至 `reflection_history`，新版结果不能借此入口反复抽样。NOOP 始终是合法输出，静态通过不表示能力提高。

Reflector 单次等待默认 180 秒，页面可设置 1–600 秒；本次设置同时用于真实 HTTP 客户端和归档的模型配置，不改变原始 Service 的配置。超时会保存失败，不自动重试。

`evidence_ids` 引用整条 Episode，不引用其内部 Oracle 事件或内容哈希。原生反思请求通过 `allowed_evidence_ids` 和提交 Schema 的枚举提供精确白名单；事件级依据可写入 `rationale`。运行时仍拒绝错误引用，原始模型输出不会被自动替换。

## v0.2 对齐

网页默认创建 v0.2 世界；已有会话保留原版本。Advanced settings 可切换 v0.1 / v0.2，并选择 22 种业务 workflow。v0.2 至少需要 8 名目录用户。API 不带 config 的默认会话仍为 v0.1，兼容原调用方。

- Playground 支持 14 种写操作与 4 种只读查询：成员邀请、激活、停用、移除，模拟支付、开续期账单、续期、恢复和到期。支付和 v0.2 退款提供 idempotency key；返回数据与新账单 ID 在 **Tool / engine return data** 中。
- Organizations 展示 pending / inactive 成员、服务资金账单、剩余可退金额和支付、退款账本；关系图增加 Payment / Refund 节点和账单周期关联。
- **Benchmark v0.2** 读取实际发布的四份 Manifest，浏览 278 个 Service Task / 6 个 Discovery 世界，展示固定身份、seed、工具、预算、版本与 hash。Inspect 可查看 evaluator 的目标、frame、依赖证书与合法路径。
- **Load for manual exploration** 核验冻结初始 hash 后替换当前世界与历史。后续主体可人工选择，页面明确标记为手动探索；不产生 Benchmark 分数，不将 Playground 参数当作真实 Customer 授权。
- 真实模型证据区展示已发布的 6 次 Service / 3 次 Discovery pilot：Customer 多轮消息、Service LLM 请求响应、工具参数与返回、事件前后状态和 Oracle。**Replay recorded run** 使用冻结 profile、核心 `score_run()` 与记录客户端，完整核验模型请求、Engine、授权与预算，不调用外部模型。新实时实验继续通过 Benchmark CLI 执行。
- Scenarios 的 **v0.2 Reference Witness** 来自当前 workflow 的真实 goal generator，是人工规则路径演示，不是 Service 自主决策。

完整 release 的模型能力评测和外部人工 review 尚未完成；Pilot 的 100% / 33.3% 仅对应已发布的小规模联调。网页不读取钥匙串，不接收上传的 profile 或授权，私有测试实例不公开。

## 五分钟探索

1. **World Overview** 默认生成真实 v0.2 Seed 42 世界：3 个组织、8 个用户、1 个 active 订阅、12 张账单。组织卡片可进入详情；点击成员、账单和规则可查看原始实体。
2. **World Graph** 从当前世界生成节点与 Membership 角色边。可以筛选组织、拖动、缩放、适应画布，以及点击节点或成员关系查看数据。
3. **Operation Playground** 选择 Workspace 1 / Owner，执行 `change_plan → Team`。查看修改的字段、Before / After 和独立 Oracle。Reset 后执行 `change_seats → 6`，Reference 应拒绝；在 Sandbox 中重复，席位会改变，Oracle 报告违规。也可尝试退款、角色修改和关闭续费。
4. **Scenarios** 默认显示五步套餐组合：升级 → 增席 → 降级被拒 → 减席 → 再次降级。Run 后点选每一步，比较真实连续状态与由 Reference Rules 推导的预期结果。
5. **Event Timeline** 查看请求、操作、评分及其完整 JSON；Export Trace JSON 导出完整轨迹，Replay Trace 调用 Python 确定性重放并核对当前状态。
6. **Benchmark v0.2** 选择公开 suite，浏览冻结目标，载入实例进行手动探索；在真实 pilot 表中打开轨迹并离线重放。

生成器支持 Seed、用户数、组织数和已有 profile。相同 Seed、配置及版本产生相同初始 hash。Generate 会创建新的初始状态并清除操作历史；Reset 恢复本次生成的初始快照。模拟业务时间来自 WorldState 的冻结 UTC 时钟。

## 结果与边界

- Executed / Rejected 表示实际执行结果。`valid_request`、`operation_correct`、`violations` 和 `evidence_event_ids` 全部来自现有独立 Oracle；执行成功不等于业务正确。
- Sandbox 只复用核心实验执行器：可执行技术上支持但业务上违规的操作。它不是 Agent、攻击发现或真实支付系统。
- 违规状态可能使后续请求合法性为 Unknown；已证明的不变量违规仍可使 `operation_correct=False`。页面保留所有原始证据。
- 场景从本次生成的合法初始世界派生，Run 会替换当前历史，以保持场景边界和连续状态。不是追加到任意实验状态；界面对此有明确说明。
- Replay 使用当前会话导出的完整 Trace，通过 `replay_trace()` 实际执行；不接受浏览器上传授权或 Oracle。导出的 Trace 也可通过核心 CLI `python -m usami replay path/to/trace.json` 重放。
- 导出及重放使用引擎副本收尾，当前探索会话仍可继续操作。核心 ReferenceRules、Generator、Engine、Oracle 与 Replay 的业务语义均未修改。

## 会话与 API

每个浏览器 cookie 对应独立内存世界、事件链和后端权威请求表。后端根据用户明确选择构建 RequestContext，拒绝浏览器额外提交 context 或 Oracle 字段。cookie 为随机能力标识、HttpOnly、SameSite Strict，有效期六小时；服务端回收闲置六小时的会话，最多保留 64 个。当前会话过期后重新加载页面会创建默认新世界。

操作使用当前 state hash 检查陈旧状态；冲突返回 409 并刷新界面。错误返回 401 / 404 / 409 / 422 / 503，不注入示例数据。数据只存在单个 Python 进程内，因此使用一个 Uvicorn worker；不提供数据库、账号或公网部署。

| API | 用途 |
| --- | --- |
| `POST /api/sessions` | 建立独立会话，默认 Seed 42 |
| `GET /api/world` | 状态、组织关系、摘要、事件和 Oracle |
| `POST /api/world/generate`, `/api/world/reset` | 生成与恢复初始世界 |
| `GET /api/world/graph?org_id=...` | 真实实体与关系 |
| `POST /api/operations` | 14 种 v0.2 写操作与 4 种只读查询，reference / sandbox |
| `GET /api/catalog` | 版本与 workflow 列表 |
| `GET /api/benchmark` | 实际发布的 suite 摘要、coverage 与 pilot 分数 |
| `GET /api/benchmark/suites/{key}` | 公开冻结实例列表 |
| `GET /api/benchmark/suites/{key}/instances/{id}` | 完整目标与 evaluator 证书 |
| `POST /api/benchmark/load` | 核验初始 hash 后载入手动探索世界 |
| `GET /api/benchmark/pilots/{hash}` | 公开 run 的完整证据 |
| `POST /api/benchmark/pilots/{hash}/replay` | 通过当前核心模块离线重放并评分 |
| `GET /api/scenarios`, `POST /api/scenarios/run` | 派生和执行已有组合场景 |
| `GET /api/events/{event_id}` | 事件与完整前后快照 |
| `GET /api/export/world`, `/api/export/trace` | 下载 JSON |
| `POST /api/replay` | 确定性执行与 hash 核对 |

交互式 API schema 见后端 `/docs`。

## 测试

```bash
# 仓库根目录，安装 .[web,dev] 后
python -m pytest -q

cd web/frontend
npm run build
npm run test:e2e
```

端到端测试默认使用已安装的 Google Chrome，在独立浏览器 context 中实际操作；测试自动启动 8001 / 5173，或复用这两个地址的现有 Usami 服务。没有 Chrome 时可安装 Playwright Chromium：`npx playwright install chromium`，然后 `USAMI_BROWSER_CHANNEL=chromium npm run test:e2e`。测试启动的后端使用仓库根目录 `.venv/bin/python`。

验收覆盖真实实体、图筛选/拖动/缩放、合法转换、非法拒绝、Sandbox 检出、组织权限、多步组合、原始事件、下载与重放、Seed 一致性和浏览器会话隔离。实际测试结果与版本见 [VERIFICATION.md](VERIFICATION.md)。

## 四域版本对齐

左侧 **Business domain** 可切换 SaaS v0.2、CloudOps v0.1、FulfillmentOps v0.1 与 StayOps v0.1。选择保存在浏览器本地并同步 URL `?domain=...`。同一浏览器保留每个领域独立的内存会话；切换不会把业务状态、权限或工具串用。SaaS 原有操作、场景与 Benchmark 页面保留。

三个独立领域提供：

- **World Overview / Business Entities**：真实实体、状态、权限、四域发布指标与集合/scope/文本筛选。StayOps 保留每晚 Calendar 记录；仓储库存各阶段保持独立字段；完整实体及审计记录可检查。
- **World Graph**：实际 ID 引用、scope 筛选、拖动、缩放和关系检查。显示项目依赖、订单/SKU/库存/运输关联、酒店/房型/房间/住宿关联；StayOps 可包含每晚 Calendar。
- **Operation Playground**：参数表单来自领域强类型 Schema，调用真实 Reference / Sandbox Engine，显示实际返回、状态差异与独立 Oracle；可检查所选身份的权限过滤 Customer observation。
- **Benchmark & Pilots**：分别读取 45 / 338 / 302 个 Service 实例及独立 Discovery 世界，显示身份、工具、预算、版本、hash、目标与证书。实际中间状态实例有明确标记，基础结构指标不会混入这些实例或跨领域相加。
- **Run reference witness** 从冻结初态运行真实规则路径，零模型调用；结果进入 Event Timeline。它是离线规则示范，不是 Service 自主决策或模型成绩。
- **Real LLM pilot evidence** 保留真实模型请求/回复、Customer 动作、工具、实际世界与 Oracle，以及预算终止/协议错误/非法尝试/授权异常。CloudOps 历史格式与新 Harness 格式分别读取，原文件不改写。
- **Replay recorded run** 使用固定 profile、记录客户端和核心回放/评分，无新增模型调用。当前手动轨迹也可查看详情、导出 JSON 与实际重放。
- **Audit & release reports** 只显示实际存在的发布文档，保留人工 review、公开 Demo、Validation 数量限制和实测解释。

`GET /api/domains` 返回四域真实发布指标。三个独立领域的 API 前缀是 `/api/domains/{cloudops|fulfillment|stayops}`，包含 `/sessions`、`/world`、`/world/generate`、`/world/reset`、`/tools`、`/observation?identity=...`、`/operations`、`/world/graph`、`/events/{id}`、`/export/{world|trace}`、`/replay`、`/benchmark`、`/benchmark/suites/{split}`、`/benchmark/load`、`/benchmark/witness`、`/benchmark/pilots/{hash}` 及其 `/replay`、`/reports/{document}`。

完整 World 与 evaluator 证书供人工审计；正式 Customer 模型仍通过核心权限过滤 observation 运行。手动参数形成的授权仅属于手动探索，不会进入正式 Service/Customer 实验。网页不运行新实时模型、不读取钥匙串，不接受上传的 profile、授权或 Oracle；新实验通过 CLI 执行，网页展示已发布证据。桌面端优先。
