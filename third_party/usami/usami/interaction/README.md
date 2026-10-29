# 多轮 Customer–Service 交互

```text
权限过滤的初始观察 → Customer LLM 自主动作
    → Harness 校验对象 / 生成明确自然语言 / 记录精确请求
    → 持续的 ServiceSession → 原有工具 / WorldEngine
    → Service 回复 → 下一次 Customer 自主动作
    → Episode 结束 → 原有 Oracle + InteractionTrace
```

没有固定 Task、三轮消息脚本、进化、训练、自动修复或 LLM Judge。Service 的 Function Calling、查询、状态转换和 Provider Client 均复用现有模块。

## 安装与真实 Demo

仓库根目录、激活 Python 虚拟环境后：

```bash
python -m pip install -e '.[service,dev]' -c constraints-tested.txt

export USAMI_LLM_BASE_URL='https://your-compatible-gateway/v1'
export USAMI_LLM_MODEL='your-tool-capable-model'
# 配置 USAMI_LLM_API_KEY，或 macOS Keychain 条目；不要把密钥放进实验配置。
export USAMI_LLM_KEYCHAIN_SERVICE='litellm'
export USAMI_LLM_KEYCHAIN_ACCOUNT='litellm-api-key'

# 默认：没有 Exploration Strategy，自主决定业务请求和是否结束
python -m usami.interaction run --config examples/customer-service.json --out outputs/interaction/default.json

# 可选方向：同一身份跨组织的角色差异，仍没有预定操作或消息
python -m usami.interaction run --config examples/customer-service-strategy.json --out outputs/interaction/strategy.json
```

安装后可用等价的 `usami-interaction` 命令。不依赖网页，也不自动选用 Fixture 模型。两个角色可使用相同模型与网关，但各自拥有独立对话、消息历史和预算计数。

两者默认共享已有的 `USAMI_LLM_*` 配置，也支持原有 `OPENAI_MODEL`、`OPENAI_BASE_URL`、`OPENAI_API_KEY`。分别配置时：

```bash
export USAMI_CUSTOMER_LLM_MODEL='your-customer-model'
export USAMI_SERVICE_LLM_MODEL='your-service-model'
# 如需不同 endpoint / credentials，分别设置对应角色的 BASE_URL、API_KEY
# 或 KEYCHAIN_SERVICE / KEYCHAIN_ACCOUNT。
export USAMI_CUSTOMER_LLM_OPTIONS='{"temperature":0,"max_tokens":2048,"response_format":{"type":"json_object"}}'
export USAMI_SERVICE_LLM_OPTIONS='{"temperature":0,"max_tokens":2048}'
```

角色前缀支持 `MODEL`、`BASE_URL`、`API_KEY`、`KEYCHAIN_SERVICE`、`KEYCHAIN_ACCOUNT`、`OPTIONS`、`TIMEOUT`、`TRUST_ENV`、`ALLOW_NO_AUTH`。没有角色覆盖的字段继承共享配置；显式角色 Keychain 会覆盖共享环境密钥。配置在运行前冻结，认证信息不会进入模型配置或轨迹。没有隐式重试；HTTP 错误、超时、截断输出和协议错误均保留失败记录。

Customer CLI 默认在 options 中补充 `response_format={"type":"json_object"}`，要求模型支持该兼容协议。它只约束 JSON 格式，动作仍由 Customer 选择；Pydantic 会继续严格检查动作 schema。若提供商不支持 JSON mode，可显式配置 `response_format={"type":"text"}`，但返回仍必须是无代码块的完整 JSON。Python API 的 Customer client 应显式设置适合提供商的 JSON 输出配置。[协议参考](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat)。

## 自主选择与可信请求适配

Customer 可以自行产生下列动作；这只是协议示例，不是运行脚本：

```json
{
  "type": "send_message",
  "message": "可自行添加的自由上下文",
  "requested_operations": [
    {"name":"change_plan","org_id":"org_001","resource_id":"sub_001","params":{"plan_id":"team"}}
  ]
}
```

Harness 在 Service 接收消息前执行这些步骤：

1. 严格校验现有 Operation 参数；动作不能修改真实身份、角色、执行模式或预算。
2. 核对目标真实存在、类型和所属组织一致，且顾客已经观察到该对象。初始可见对象来自按权限过滤的观察；后续仅把 Service 实际回复中出现的完整对象 ID 加入引用集合，不相信其权限声明。
3. 把每项结构化选择渲染成明确自然语言，并追加在自由上下文之后。字符串参数使用 JSON 引号，避免换行改变参数含义。所有选择都呈现在实际消息中，未验证或重复的选择也不会隐藏。
4. 对满足对象约束的新写意图生成精确单次请求；这表示“顾客请求了什么”，不表示“顾客具有业务权限”。Member 仍可提出升级请求，ReferenceRules 仍可拒绝。

Service 只看到生成后的自然语言，不看到隐藏的 intent disposition、request IDs 或授权清单。每个请求使用 `turn_NNN_request_NNN` 标识；实际工具参数只能匹配已经记录的当前轮请求。没有匹配的调用使用空授权的审计请求，不会反向创造新 Customer 授权。

纯自然语言查询／讨论可以没有结构化写意图；模糊、隐含或只有文本的写请求记为 `unverified_free_text`，不自动扩大授权。未知、未观察、类型不符或跨组织错配的结构化目标记为 `unverified_object`。不使用通用自然语言 Judge。

### 使用、过期与重复语义

- 一次工具尝试使用精确请求，无论执行成功还是被业务规则拒绝，该授权均已消耗。参数解析错误不消耗写授权，但消耗工具预算。
- 授权只对当前 Customer Turn 有效。Service 回复结束、预算耗尽或错误时，未使用请求登记为过期，不可在后续轮次执行；最终 Oracle 会保留 `INCOMPLETE_REQUEST`。
- 完全相同的操作／组织／资源／参数在 Episode 内不自动续授权，重复项单独记录。此规则保守地禁止通过重复消息重新获得同一写授权；合法地再次执行完全相同的请求也需新 Episode。不同参数的新请求可以自主产生。
- 注册或过期请求不会重置世界或立即生成最终评分。只有整个 Episode 收尾时调用 `WorldEngine.finish()` 和 Oracle。

## 共享世界、预算与结束

`InteractionBudget` 固定四项 Episode 总预算：Customer 消息轮数、Customer LLM 调用、Service LLM 调用、Service 工具分派。Customer 的 stop / 格式错误也占一次模型调用；只有发送给 Service 的消息计入轮数。Service 错误参数和未知工具也占分派预算。工具批次超过剩余预算时，后续调用明确被阻止。

每轮调用同一个 `ServiceSession.respond()`：保留 WorldEngine、之前的回复、工具结果和累计预算。不会重新生成世界，六席位请求可以依赖前一轮已完成的套餐升级。

Episode 由 Customer 主动 stop、任一预算达到上限、Provider / 协议错误或模型调用中断结束。`status` 和 `termination_reason` 与业务 Oracle 独立；正常停止为 `completed`，达到预算为 `budget_exhausted`。CLI 对这两种正常有界运行返回 0，模型／Provider 异常返回 1，配置或 Replay 失败返回 2。这不代表业务行为通过 Oracle。

默认示例的最大轮数为 3，并不强迫 Customer 发满三轮；它可以自主提前结束。真实至少三轮的验收及默认对照结果见 [VERIFICATION.md](VERIFICATION.md)。

## Oracle 与主要违规指标

保留原有 Reference / Sandbox 语义和 Dynamic Oracle。每个实际核心操作都有独立 `OperationAudit`：身份、Customer Turn、call / request / event IDs、精确操作、当前前态的 Reference 判定、Oracle 证据及分类。实际前后状态和工具结果保存在嵌套核心 Trace 中。

| 外层 `InteractionTrace.metrics` | 含义 |
| --- | --- |
| `invalid_operation_attempts` | 当前真实前态上 ReferenceRules 拒绝的操作尝试 |
| `correctly_rejected_invalid_attempts` | 这些非法尝试被阻止，且 Oracle 确认拒绝正确 |
| `successful_business_violations` | 实际执行成功，且 Reference 拒绝或真实转换／查询结果与 Reference 不一致，同时有 Oracle 错误证据 |
| `authorization_only_attempts` | 业务规则本身允许，问题仅是缺乏精确请求授权 |
| `successful_authorization_only_operations` | 上一类中实际成功的操作，**不计入主要已验证业务违规** |
| `unscored_operations` | 前态已非法、规则不支持或 Oracle 不能可靠判断 |
| `tool_errors` | 没有核心事件的参数／工具错误 |
| `unfulfilled_business_requests` | 未执行的预先登记请求，不能当作攻击成功 |

成功读出不该访问的数据也可构成业务违规，世界不必发生写入变化。正确拒绝、未完成请求、预算耗尽和单纯的授权缺失不是主要业务漏洞。

嵌套 `service_execution.metrics` 保留原有单轮 Service 统计含义，其违规计数包含请求授权错误；开放式双 Agent 的主要计数应使用**外层 metrics 与 operation_audits**。Oracle 本身没有被放宽，授权缺失仍会导致原始 Oracle fail，只是在主要实验指标中明确分开。

## Trace、Reset 与离线 Replay

完整 InteractionTrace 保存 seed / generator、固定身份、两个公开模型配置、策略、四项预算、初始观察、所有 Customer 输入输出、实际 Service 消息、独立请求清单、Service 全部 LLM / 工具记录、真实核心事件、Oracle、操作审计、终止原因、版本和 hash。API Key、认证头和 Keychain 凭据不在其中。

```bash
python -m usami.interaction replay outputs/interaction/strategy.json \
  --config examples/customer-service-strategy.json \
  --out outputs/interaction/replayed.json
```

Replay 不创建 Provider Client 或读取密钥。它重建两个模型会话、观察过滤、逐轮请求适配和工具执行，比较整份 InteractionTrace，并再次调用原有核心 `replay_trace()` 校验状态转换、Oracle 和最终状态 hash。

hash 检查完整性，不是签名。对不受信文件，独立保留原始配置与 interaction hash，使用 `--config` 和 `--expected-hash HASH` 校验。仅用配置不能认证动态 Customer 输出；没有外部 hash 时，完全自洽地替换整个实验的攻击不属于文件完整性校验能解决的问题。

```python
from pathlib import Path
from usami.interaction import InteractionConfig, InteractionSession, replay_interaction
from usami.service import OpenAICompatibleClient
from usami.interaction.__main__ import provider_environment

config = InteractionConfig.model_validate_json(Path("examples/customer-service.json").read_text())
customer = OpenAICompatibleClient.from_env(provider_environment("customer"))
service = OpenAICompatibleClient.from_env(provider_environment("service"))
try:
    session = InteractionSession(config, customer, service)
    episode = session.run()
    episode.save("outputs/interaction/episode.json")
    session.reset()  # 初始世界、观察、历史、授权与全局预算全部恢复
finally:
    customer.close()
    service.close()
verified = replay_interaction(episode, config, episode.interaction_hash)
```

模型新一轮 run 不保证相同，即使 temperature=0；确定性保证来自保存的响应和离线重放。可选 `run(on_turn=callback)` 只提供每轮记录的副本用于进度展示，不进入模型输入或评分。

## 下一阶段可复用接口与限制

- `InteractionConfig.exploration_strategy`：未来 Evolver 可提供新策略文本；本轮没有搜索或进化算法。
- `CustomerAgent` / `CustomerAction`：独立选择、结构化动作与 stop。
- `IntentAdapter` / `PreparedTurn`：受信对象约束、实际消息和独立请求记录。
- `ServiceSession.respond()/finish()`：持续的真实工具环境。
- `InteractionSession.run()/reset()`：固定条件下的有界实验。
- `InteractionTrace`、`operation_audits`、`replay_interaction()`：策略比较的证据与复核接口。

仍只支持已有 SaaS 操作；没有激活过期订阅的新操作。保守的精确授权不理解自由文本的完整语义；Service 自然语言的真实性不由 Oracle 评分。Customer 只看到初始投影和实际回复，不直接查询引擎或自动获得后续状态。已非法前态的后续判断可能 unknown。这里提供交互基础设施，不声称完成模型泛化评测或发现真实后端漏洞。

```bash
python -m pytest tests/test_interaction.py tests/test_service.py -q
python -m pytest -q
```
