# Customer LLM

Customer 没有领取预设 Task。`CustomerAgent` 根据权限过滤的初始业务信息、已发出的实际请求、Service 的真实最终回复及剩余预算，自主返回 `CustomerAction`。可以只提问、选择精确写操作、改变组织或话题、使用自由上下文，或者主动 `stop`。

```json
{
  "type": "send_message",
  "message": "顾客自行选择的自由上下文或查询",
  "requested_operations": []
}
```

写意图使用现有 `Operation` 结构：`name`、`org_id`、`resource_id`、`params`。这只是请求内容，不代表业务权限。`stop` 使用 `{"type":"stop","reason":"自主结束原因"}`；不能同时发送业务请求。所有实际身份由 Harness 固定，动作协议没有 actor、identity 或预算修改字段。错误 JSON、非法 schema、额外字段或 Customer 工具调用会结束实验并记录错误。

`initial_observation()` 使用原有 ReferenceRules 的读取权限过滤初始视图，只提供该身份允许读出的组织、订阅、成员和账单、自己的身份关系、相关公开用户资料、业务时钟与公开套餐 ID / 名称。拒绝原因、预期转换、整份 WorldState、Oracle 和 Service 私有提示词不会作为初始输入。初始观察是 Harness 的只读投影，不是 Service 业务操作，也不占用 Service 工具预算。

后续输入只含实际发给 Service 的消息、Service 最终回复及剩余预算；不复制 Service 的工具对话、世界快照或评分。初始视图不自动刷新，Customer 应通过 Service 获取变化后的信息。Service 自然语言可能错误，真实执行是否正确由独立 Oracle 判断。

`exploration_strategy=None` 是默认模式。可选策略只是 Customer 提示词中的探索方向，不控制具体目标、请求参数、消息顺序或最终回复。模型传输完全复用 `CompletionClient` / `OpenAICompatibleClient`；没有第二套业务执行器或 Provider Client。

安装、双模型环境配置、逐轮授权和完整 Replay 见 [interaction/README.md](../interaction/README.md)。
