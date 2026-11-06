Status: implemented, not yet validated on a GPU host
Owner: RIFT-RSI maintainers
Last verified: 2026-11-06
Scope: replacing the scripted customer with a model-backed one in training and evaluation

# 大模型客户

## 为什么
训练和评测里的客户现在是脚本（`_UsamiCustomer`，`uses_model = False`）：第一轮说 "Yes, please go ahead."，之后说 "Thanks, that is all."，只有按预设清单拒绝时才不同。它不闲聊、不改口、不隐瞒信息，所以 Service 平均每条对话只需对客户说约 1.9 次话，从不需要处理真正的对话。τ-bench 的客户是大模型（默认 GPT-4o，另有 react、verify、reflection 策略）。

## 做了什么
- Usami 本来就有大模型客户 `OpenAICustomerAgent`：按私有画像扮演、用自己的话回答、可以拒绝；每次调用被记录进回合记录，所以回放校验不需要再联网。
- `src/rift/runtime/llm_customer.py`：`RIFT_CUSTOMER_URL` 设置后，Host 提供者改用它。带重试的客户端（超时、连接错误、HTTP 5xx 最多 3 次；格式不对的回答不重试）；每个回合一个客户，种子取回合的客户种子；关闭 Qwen3 的思考模式。
- `scripts/customer_server.sh`：用 vLLM 在一张卡上起一个冻结的客户模型（只监听本机，显存占 0.15）。
- `scripts/probe_llm_customer.py`：不需要 Service，把已记录的回合里 Service 的最后一条回复喂给客户模型，查它的决定是否可用、是否与画像一致。

## 要先弄清楚的（未验证）
1. **客户模型出错的比例。** 客户的一次调用失败（超时、格式不对、与画像冲突）会让回合以 `customer_protocol_error` 结束，业务目标没达成就按失败计，等于把客户的错记在 Service 头上。用 `probe_llm_customer.py` 量；比例高就要换更强的客户模型、加提示约束，或把这类回合标为未知。
2. **客户模型的选择。** 默认是冻结的 Qwen3-4B（起点权重，绝不是正在训练的那份）。实验室网关里有更强的模型（如 qwen3.8-27b），但 a800 在公网，连不上内网网关（实测不通），而且需要管理员发的密钥。想用更强的客户，要么在 a800 上放一个更大的模型，要么换到能连网关的机器。
3. **对分数的影响。** 换客户会改变所有成功率，旧的基线和阶段 1 的结果不能直接比。要换，就用同一个检查点在两种客户下各测一遍。
4. **速度。** 每个回合多约 2 到 3 次客户调用，调用在回合线程里同步进行，几乎不拖慢生成，但客户服务器要有足够并发（`--max-num-seqs 128`）。
5. **占用的显存。** 客户服务器与训练共用 GPU，训练里的 vLLM 在更新时是休眠的，所以一般放得下；训练更新时如果没有检查点（关梯度检查点），显存紧，要观察。

## 启动
```bash
GPU=3 MODEL=$HOME/models/Qwen/Qwen3-4B bash scripts/customer_server.sh start
export RIFT_CUSTOMER_URL=http://127.0.0.1:8100/v1 RIFT_CUSTOMER_MODEL=customer
python scripts/probe_llm_customer.py --episodes $RUN/episodes --limit 200
# 通过检查后，再启动训练（环境变量会被转发给工人）
```
