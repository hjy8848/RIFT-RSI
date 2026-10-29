# World Explorer 实际验收

## v0.2 对齐验收 · 2026-10-13

桌面 Chrome 1440 × 1000，继续按用户要求不验收手机端。

| 检查 | 实际结果 |
| --- | --- |
| `.venv/bin/python -m pytest -q` | **420 passed**，53.93s，含 25 项 Web 集成测试 |
| `cd web/frontend && npm run build` | TypeScript 和 Vite 构建成功，1744 modules |
| `cd web/frontend && npm run test:e2e` | **13 passed**，15.9s，7 项 v0.1 回归 + 6 项 v0.2 桌面流程 |
| 发布 pilot 的 HTTP 重放 | 全部 6 次 Service / 3 次 Discovery 的当前核心重放分数与发布记录逐项相等 |

新增验证包括默认 v0.2 世界、22 种 workflow 目录、成员席位拒绝与 Sandbox 不变量证据、真实 goal witness 连续执行、冻结世界和事件重放、失败支付恢复、幂等支付重试、部分退款余额、邀请 / 激活 / 停用 / 移除成员，以及新增账单支付后实际续期。关系图包含真实账本记录和账单订阅关联。

Benchmark 页的实例数量为 Train 170 / Validation 68 / Test demo 40 / Discovery 6。测试核验目标证书、预算、初始 hash、加载后的人工探索标记和实际 Discovery 成功违规的离线重放；陈旧 hash 被拒绝，任意 suite / run 文件路径不被接受。公开 pilot 内的 Customer 错误与仅授权异常仍保留；完整 release 能力评分和外部人工 review 未完成。

实际截图在忽略提交的 `outputs/web-v2/`，包含 benchmark-overview、benchmark-pilot、billing-graph、lifecycle-replay 和 frozen-world。人工检查了当前浏览器中的 v0.2 发布页。此次没有改动 `usami/` 核心规则或任何冻结 Manifest，原 source pins 仍有效。

## v0.1 原始验收记录

日期：2026-10-12。环境：macOS，Python 3.9.6，Node 26.8.2，npm 11.19.1，Google Chrome。桌面验收视口 1440 × 1000；按照用户后续要求，不继续验收手机端。

## 运行结果

| 检查 | 实际结果 |
| --- | --- |
| `.venv/bin/python -m pytest -q` | **106 passed**，16.05s；原有 87 项 + Web API 19 项 |
| `cd web/frontend && npm run build` | TypeScript 检查及 Vite 生产构建成功；1743 modules |
| `cd web/frontend && npm run test:e2e` | **7 passed**，7.4s；真实 Chrome、实际 FastAPI 调用 |
| `web/run.py` 启动与停止 | 自定义 API 8012 / 前端 5174 的 `/api/health` 成功；SIGTERM 后两项子进程退出且端口均释放 |

Python Web 依赖验收版本：FastAPI 0.128.8、Uvicorn 0.39.0、HTTPX 0.28.1、Starlette 0.49.3。核心约束仍为 Pydantic 2.13.5、pytest 8.4.2。前端使用提交的 package-lock.json（Vite 6.4.4）。

## 实际覆盖

- Seed 42 的 WorldState 与 `generate_world(42)` 完全一致。默认 world_id 为 `world_d7b758c82586402e848a`，初始 hash 为 `cbdc43984d682555281707a7431c7ed702f0c9bc4a3ebb3346b614f10936e0ec`。
- 默认统计为 3 个组织、8 个用户、1 个 active 订阅、12 张账单。页面切换组织，成员和账单与 API 的真实实体一致。
- 全世界图有 26 个真实实体节点；节点、Membership 角色与边均逐项对照 WorldState。组织筛选、节点拖动、点击检查、缩放和 Fit View 在浏览器中实际操作。
- Workspace 1 Owner 的 `change_plan → Team` 修改真实套餐字段，Oracle 判正确；Starter 的 `change_seats → 6` 在 Reference 中拒绝、hash 不变。
- 同一个请求在 Sandbox 中实际改变 seats，Oracle 判错误并提供 `SEAT_LIMIT_EXCEEDED` 证据。Before / After 展示引擎原始快照；Reset 恢复初始 hash。
- 先由 Owner 将有效 Billing Admin 改为 Member，再以该 Member 关闭续费：Reference 拒绝且状态不变；Sandbox 执行成功、auto_renew 变化，Oracle 独立检测 `PERMISSION_DENIED`。
- 默认五步套餐场景实际运行，第三步正确拒绝、第五步合法完成，最终 5 / 5 请求正确；API 另验多实体及组合退款场景，前后快照连续且终态与规则派生 hash 一致。
- 退款表单的账单、金额和原始渠道来自实际世界，合法全额退款生成真实 Refund 历史。
- Event Timeline 原始事件与数量匹配；浏览器实际下载含 final 事件的完整 Trace。Replay 调用核心重放、终态 hash 一致；导出及重放之后仍可继续操作。
- 同 Seed / 配置重生成完全一致；换 Seed 改变世界。两个独立浏览器 context 和两个 HTTP Session 的操作、生成和历史互不影响。
- 浏览器不能提交伪造 RequestContext / Oracle，额外字段返回 422；陈旧 hash 返回 409 且不创建事件；预算检查不写入多余请求；未知事件、资源和配置返回明确错误。
- 六个桌面页面无横向溢出。保存并实际查看了首页、关系图、Sandbox、场景与时间线截图；修正了筛选图的状态刷新、图标导航的可访问名称和说明文字对比度。

截图由测试保存到忽略提交的 `outputs/web/`。它们来自运行中的真实应用，不是设计稿。基准核心目录 `usami/` 未修改。

## 已知边界

场景 Run 从生成时的初始状态替换当前历史；Replay 只重放当前受信会话的完整 Trace，外部文件可使用核心 CLI。内存会话要求一个 Uvicorn worker，后端重启会清空世界。Sandbox 不支持核心未实现的技术操作，非法 pre-state 的 Unknown / 已知违规遵循已有 Oracle 语义。手机端保留基础响应式样式，未作为最终验收目标。

## 四域网页对齐（2026-10-14）

新增三域的独立会话、真实操作、实体/关系、冻结库、实测证据、报告与回放接口及桌面页面。SaaS 原页面保留；四域 Python 业务运行代码和所有发布 Manifest/Pilot 数据未修改。

- 新领域网页 API **17 passed**：确定性生成/重置、领域及浏览器会话隔离、陈旧 hash、真实权限拒绝、Sandbox 业务违规、酒店押金确认、错误参数与伪造 Oracle 拒绝、图引用、权限过滤 observation、冻结载入、真实 Witness 和已发布模型重放。
- 原有网页 API **25 passed**。
- 全仓库 **622 passed in 141.30s**，含此前 605 项测试。
- 全桌面 Chrome 交互 **18 passed**，含原有 13 项及新增 5 项；覆盖三域任务/图/Customer/模型回放、酒店状态持久化、切回 SaaS 和仓储 Sandbox 检出。
- TypeScript/Vite 构建通过；新领域按需加载。人工检查实际截图后调整面板间距、Oracle 排版和空字段显示，五项新领域交互复跑 **5 passed**。

工作区日志 `outputs/web-four-*.txt`，截图 `outputs/web-four-domains/`。网页没有调用新模型或读取凭证。真实模型成绩来自已发布证据；业务违规、授权异常、拒绝和预算终止分开。人工真实性审核待执行，公开 Test 为 Demo。
