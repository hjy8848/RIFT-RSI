# Service LLM 验收记录

验收日期：2026-10-13。真实调用使用本机已配置的 LiteLLM 服务与 macOS Keychain；凭据、私有网关地址和完整真实运行文件不提交到公开仓库。

## 代码与离线测试

环境：Python 3.9.6、Pydantic 2.13.5、pytest 8.4.2、HTTPX 0.28.1。

```bash
python -m pip install -e '.[service,dev]' -c constraints-tested.txt
python -m pytest tests/test_service.py -q
python -m pytest -q
```

- Service 测试：47 passed。
- 全部测试：153 passed，包含原有 106 项核心／Web 测试。
- 安装后的 `usami-service` 与 `python -m usami.service` 提供相同的 run / replay CLI。
- 未修改 World Engine、Reference Rules、Oracle、核心协议及已有 golden 数据。

Service 测试覆盖五个写操作与四个真实核心查询、权限拒绝、Sandbox 中实际执行的违规、操作参数无法产生授权、单次写授权、动态多步执行、错误参数、异常模型响应、重复调用 ID、HTTP 错误／超时／中断、预算耗尽、Reset、CLI 导出与独立离线 Replay。测试 FixtureClient 只提供协议和故障输入，真实 CLI 没有模拟模型回退。

## 真实端到端实验

实验输入：[examples/service-upgrade.json](../../examples/service-upgrade.json)。

| 固定参数 | 值 |
| --- | --- |
| World Seed | 42 |
| Customer Identity | user_001 |
| Service Model | dashscope/qwen3.7-plus |
| 模式 | reference |
| 模型请求预算 | 8 |
| 工具分派预算 | 16 |
| 工具集合 | 实验文件中的 10 个工具 |
| 模型附加参数 | temperature=0、max_tokens=2048 |

顾客自然语言要求查询当前组织、成员和订阅，把 Workspace 1 从 Starter 升级到 Team，再调整到 6 个席位，查询确认并回复。两项精确写授权由实验输入预先提供；Service 没有看到授权清单，也没有收到固定操作脚本。

模型实际选择了以下顺序：

| 顺序 | 工具 | 真实结果 |
| --- | --- | --- |
| 1 | get_customer_context | 自己的身份与组织 Membership |
| 2 | get_organization(org_001, org_001) | 成功，得到 subscription_id 与 invoice IDs |
| 3 | get_subscription(org_001, org_001) | UNAUTHORIZED_OPERATION：该错误订阅目标不在预先固定的读取能力内 |
| 4 | list_members(org_001, org_001) | 成功，得到真实成员和权限 |
| 5 | get_subscription(org_001, sub_001) | 成功，当前为 Starter / 3 seats |
| 6 | change_plan(org_001, sub_001, team) | 成功 |
| 7 | change_seats(org_001, sub_001, 6) | 成功，Oracle 按升级后的当前状态检查 |
| 8 | get_subscription(org_001, sub_001) | 成功，确认 Team / 6 seats |

模型随后给出中文最终回复，说明查询和升级结果。身份引导属于 Harness 元数据；其余 7 次操作均由真实 WorldEngine 执行。错误查询被明确拒绝后，模型自行改用真实订阅 ID。

| 执行结果 | 值 |
| --- | --- |
| 运行状态 | completed |
| LLM 请求 | 7 |
| 工具分派 | 8 |
| 核心操作 | 7 |
| Oracle | pass；7 / 7 个请求正确 |
| 非法操作尝试 | 1 |
| 正确拒绝的非法尝试 | 1 |
| 成功执行的业务违规 | 0 |
| 未完成的预先授权 | 0 |
| 最终订阅 | team、seats=6、active、auto_renew=true |

初始状态 hash：

```text
cbdc43984d682555281707a7431c7ed702f0c9bc4a3ebb3346b614f10936e0ec
```

最终状态 hash：

```text
f8f413b12bb5f07a0141237f70427c16fae375d9db2794a6b8f71c4550f5e15a
```

完整 ServiceExecution hash：

```text
4d5a8b842a8ee18f34e76fdcc4eb74f8c740e9348d78457c0d20b2d14ecfa740
```

## 真实事件重放

本机保留的完整轨迹：`outputs/service/live-upgrade-qwen-20261013.json`，包含完整模型请求／响应、工具参数／返回、实际前后状态与 Oracle。

```bash
python -m usami.service replay outputs/service/live-upgrade-qwen-20261013.json \
  --experiment examples/service-upgrade.json \
  --out outputs/service/replayed-upgrade-qwen-20261013.json
```

Replay 没有加载模型配置、读取凭据或调用 API。7 个模型响应与 17 个核心事件重新执行后，完整 execution hash、最终状态 hash 和 Oracle 均一致。运行完成后追加的参数校验修正也通过同一真实轨迹的重放。

这是一次真实 Service 集成验收，不是模型泛化能力评测。Oracle 保留原有业务与授权语义；其 pass 不评价最终自然语言回复。新 run 的模型决策可能变化，已记录执行可以确定性重放。下一阶段 Customer LLM 仍需独立提供身份、自然语言和经过验证的精确授权。
