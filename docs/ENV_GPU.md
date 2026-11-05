# GPU 环境（x86 + CUDA）

一个环境覆盖实验室全部 NVIDIA GPU 机器：`/home/hejingyu/miniconda3/envs/rift-gpu`（家目录共享，所有 GPU 机器可用）。
2026-11-04 建成，用时 5 分 23 秒，没有编译任何东西。

| 组件 | 版本 | 备注 |
| --- | --- | --- |
| Python | 3.12 | 与昇腾环境一致 |
| torch | 2.10.0+cu128 | PyPI 默认构建；显卡架构 sm_70/75/80/86/90/100/120（sm_89 由 sm_86 二进制兼容） |
| vLLM | 0.19.1 | 钉 torch==2.10.0；xgrammar 0.2.8 做受约束解码 |
| 其他 | transformers 5.22.1、flashinfer-python 0.6.6、numpy 2.2.6、pydantic 2.14.2 | |

## 为什么是这个组合
- vLLM 0.20 起 PyPI 默认是 CUDA 13 构建（依赖 `nvidia-*-cu13`，要求驱动 ≥ 580），实验室里驱动 555/560/570/575 的机器跑不了。
- vLLM 0.17–0.19.1 钉 torch 2.10.0，PyPI 的 torch 2.10.0 是 CUDA 12.8 构建，驱动 ≥ 525 即可（同一大版本内向后兼容），Blackwell（sm_120）也包含在内。
- 昇腾环境用 vLLM 0.23.0。要对齐的话需要 GitHub 发布页的 `vllm-0.23.0+cu129` 轮子（500 MB）加 torch 2.11.0+cu129；阶段 0 不需要，没做。

## 建环境的配方（快的原因）
1. **清华镜像直连**（约 41 MB/s；阿里云只有 3 MB/s，华为云 4.5 MB/s）：conda 用 `https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main`（`--override-channels`），PyPI 用 `https://pypi.tuna.tsinghua.edu.cn/simple`。
2. **uv 安装**：`python -m pip install uv`，再 `uv pip install --python <env>/bin/python --index-url <tuna> vllm==0.19.1`，并行下载，缓存放家目录（与环境同文件系统，可硬链接）。
3. 全部是预编译轮子；GitHub 只能经实验室代理 `http://10.205.2.46:7899`（逐条命令使用，不写入配置）。

```bash
ENV=/home/hejingyu/miniconda3/envs/rift-gpu
/home/hejingyu/miniconda3/bin/conda create -y -p $ENV python=3.12 --override-channels \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
$ENV/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple uv
UV_CACHE_DIR=/home/hejingyu/.cache/uv $ENV/bin/uv pip install --python $ENV/bin/python \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple "vllm==0.19.1"
```

## 已验证（2026-11-04，只用空闲卡）
| 机器 | 显卡 | 驱动 | 结果 |
| --- | --- | --- | --- |
| zju-57 | RTX PRO 6000 Blackwell（sm_120） | 580 | bf16 矩阵乘通过；Qwen3-4B + 受约束解码 + Designer schema 生成并解析成功 |
| zju-46 | RTX A6000（sm_86） | 575 | 矩阵乘通过 |
| zju-47 | RTX A6000（sm_86） | 580 | 矩阵乘通过 |
| zju-56 | RTX 3090（sm_86） | 580 | 矩阵乘通过 |

## 未验证
- 驱动 555/560/570 的机器（zju-17、zju-12、zju-54）：理论上靠 CUDA 12 的次版本兼容可用，当时没有空闲卡，没有测。
- A100（zju-12，sm_80）、4090（zju-17、zju-18，sm_89）、zju-48/49/53：同上。
- zju-55 连接被拒绝。
- 训练只在 zju-47 的 2 张 A6000 上验证过；Blackwell（96GB）、更多卡、其他机器的 NCCL 行为没测。
- **多轮训练还没有在 GPU 上验证**：冒烟只覆盖一次更新。运行时常驻、每轮权重同步、虚拟检查点、训练后原地评测（`train-pool`，配置见 `configs/experiments/cuda_port_*.json`）的第一次尝试因为 GPU 被别人占走而 OOM，之后整个实验室没有 2 张空闲卡，还没重跑。

## 训练环境（`rift-gpu-train`）与 CUDA 移植
2026-11-04 另建了 `/home/hejingyu/miniconda3/envs/rift-gpu-train`，**110 秒**（uv 缓存让 vLLM 几乎瞬间装完）：vLLM 0.19.1、torch 2.10.0+cu128、transformers 5.10.4（VERL 0.9.1 要求 <5.11，所以比 `rift-gpu` 低）、ray 2.60、tensordict 0.10。
VERL 用的是昇腾环境同一份源码（共享盘上 `verl-v091-cann91-py312/verl`，复制到 `/home/hejingyu/src/verl-v091-cuda` 后 `pip install --no-deps -e`，共享目录不动），代码里钉的 VERL 提交因此一致。

运行时 `rift.runtime.ascend_910b3` 由 `rift.hardware` 驱动（`RIFT_PLATFORM=cuda`、`RIFT_WORLD_SIZE=N`），昇腾路径的取值不变（测试逐项断言）。CUDA 上的差异：
- Ray 资源用内置 `num_gpus`，不是自定义 `NPU`；
- vLLM 张量并行度默认 1（Qwen3-4B 有 8 个 KV 头，只能取 1/2/4/8；每个 GPU 一个副本，卡数不必是 2 的幂），可用 `RIFT_ROLLOUT_TP` 改；
- vLLM 显存占比 `RIFT_VLLM_MEMORY_UTILIZATION`（默认 0.45）；
- 注意力用 **SDPA 且关闭 remove-padding**：torch 2.10 没有 flash-attn 预编译轮子（最新只到 torch 2.8），而去填充的打包输入靠 flash-attn 变长内核隔开样本，SDPA 下会悄悄串样本。更新时每卡微批次 1 条，没有填充浪费。`RIFT_ATTN_IMPLEMENTATION=flash_attention_2` 可改回。

### flash-attn（可选，已装进 `rift-gpu-train`）
官方没有 torch 2.10 的预编译轮子（最新到 torch 2.8，PyPI 只有源码包），装的是第三方预编译轮子：
`flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl`（253,572,977 字节），来自 GitHub `mjun0812/flash-attention-prebuild-wheels` 的 v0.9.0 发布，经实验室代理下载（约 6 分钟）。
SHA256 `a58c95a080363606e691c342d47dd173b510f4c013a8c5fcb8744def12e36a0f`，与 GitHub 接口报告的摘要一致。用 `uv pip install --no-deps` 只装这个轮子（torch 未改动），加 einops；文件缓存在 `/home/hejingyu/.cache/wheels/`。**这是社区个人维护的二进制，不是 Dao-AILab 官方发布。**

数值检查（2×RTX A6000，Qwen3-4B 的头配置 32 查询头 / 8 KV 头 / 头维 128，bf16）：与 PyTorch SDPA 的最大绝对偏差 0.0020；变长内核把两条样本打包计算，与分开计算的结果完全一致（偏差 0.0000），即样本之间不串。只在 sm_86 上测过，Blackwell 上未测。

用法：`RIFT_ATTN_IMPLEMENTATION=flash_attention_2`，此时 remove-padding 保持 VERL 的默认开启。**还没有用它跑过 VERL 的端到端更新**（已验证的冒烟用的是 SDPA），默认仍是 SDPA。

### 踩到的坑
| 现象 | 原因 | 处理 |
| --- | --- | --- |
| FSDP 初始化 `FlashAttention2 ... not installed` | VERL 默认 flash_attention_2 | 上面的 SDPA 设置 |
| 两个 rank 11 分钟 100% GPU/100% CPU 无日志 | **zju-47 的 GPU 0↔1 NCCL P2P 会死锁**（拓扑显示 P2P OK）。独立的 torchrun all_reduce 复现，`NCCL_P2P_DISABLE=1` 后 0.42 秒通过 | 启动时 `export NCCL_P2P_DISABLE=1`；运行时把 `NCCL_*` 环境变量转发给 Ray 工作进程。别的机器是否需要未测 |
| `torch.cuda` 偶发 `CUDA driver initialization failed`（zju-46 一次，重试三次都正常） | 那台机器 GPU 持久模式关闭，当时 CPU 负载约 73；`cuInit` 本身返回 0 | 偶发的宿主机现象，重试即可 |
| 单卡放不下 | 4B 的 FSDP2 训练：权重+梯度+Adam 状态约 48GB | A6000（48GB）至少 2 张；批次要能被卡数整除（8 条采样 → 1/2/4/8 张） |
| 多轮验证在 NCCL 初始化时 `Cuda failure 'out of memory'` | 别的用户的任务在我们启动后几分钟才开始占 GPU（他们的进程在 CPU 上加载了几分钟，启动检查时显存还是空的），把两张卡各占到 42GB。**"空闲"检查不是预约**，启动到真正占卡之间有几分钟窗口 | 不是代码问题。重跑时选更闲的机器，失败了换新标签重来；窗口越短越好（先把任务认证和基础检查点放在占卡之前不现实，只能重试） |

### 已验证
2026-11-04，zju-47，2×RTX A6000，`python -m rift.runtime.ascend_910b3`（冒烟）：FSDP2 初始化并保存检查点（约 2 分钟）→ vLLM 服务与权重同步 → 8 条滚动采样 → 1 次 GRPO 更新 → 保存并校验新检查点，`status: service_smoke_completed`。冒烟任务 8/8 成功，优势为零，只证明流程通，不证明学习。

2026-11-05，a800-1（`node054`），2×A800 80G（3、5 号卡），同一冒烟：`status: service_smoke_completed`，退出码 0，用的是默认 SDPA。
同日，zju-53 的冒烟**没有**测到 Blackwell：混装机上 `CUDA_VISIBLE_DEVICES=0,1` 与 nvidia-smi 的 0、1 号不是同一批卡，进程落到了别人几乎占满的卡上并卡住，已停止。Blackwell 上的训练仍未验证。

## 混装 GPU 机器：先设 `CUDA_DEVICE_ORDER=PCI_BUS_ID`
CUDA 默认按速度给卡排序，与 `nvidia-smi` 的序号不一致。zju-53、zju-18 这类混装机上，指定卡之前先设 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，启动后用 `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory` 核对自己的进程到底在哪张卡上。`scripts/stage1_gpu.sh` 已加这条。

## a800-1（独立存储，GitHub 不通）
- 登录：`ssh a800-1`（公网 `101.37.174.109:8082`，8× A800 80G）。家目录 `/data/home/hejingyu`，与 zju 的共享家目录**不是同一份**；系统盘 `/` 只剩约 4 GB，**所有东西放 `/data` 下**，装包前设 `TMPDIR`、`UV_CACHE_DIR`、`PIP_CACHE_DIR` 到 `/data/home/hejingyu/` 下。
- 没有 conda、没有 GitHub、没有 `/home/shenyl/hf/model`。清华/阿里云 PyPI 镜像和 hf-mirror 可用。
- 配方（共约 25 分钟，其中 miniconda 安装包走清华只有约 0.4 MB/s 占了十来分钟）：
  1. 清华下 Miniconda，装到 `/data/home/hejingyu/miniconda3`；`conda create -p .../envs/rift-gpu-train python=3.12 --override-channels -c <清华 main>`。
  2. `uv pip install vllm==0.19.1`，**用阿里云 PyPI**（`https://mirrors.aliyun.com/pypi/simple/`，大文件 33 MB/s）；第一次用清华时接收速度掉到 0，uv 空等，停掉换阿里云并设 `UV_HTTP_TIMEOUT=300` 后顺利完成。
  3. 与 zju 环境对齐的固定版本：transformers 5.10.4、ray 2.60.0、tensordict 0.10.0、accelerate 1.15.0、codetiming 1.4.0、datasets 5.1.0、dill 0.4.1、hydra-core 1.3.8、liger-kernel 0.8.5、omegaconf 2.3.1、peft 0.21.2、pybind11 3.1.0、pylatexenc 2.11、torchdata 0.11.0、wandb 0.31.0。
  4. VERL：把 zju 上 `/home/hejingyu/src/verl-v091-cuda`（26 MB）打包经本机中转，`uv pip install --no-deps -e`。
  5. 模型：`HF_ENDPOINT=https://hf-mirror.com`，`huggingface_hub.snapshot_download('Qwen/Qwen3-4B', local_dir=...)`，约 7.6 GB、半小时。
  6. 代码：本机 `git archive HEAD` 打包后 scp 过去，解压到 `projects/RIFT-RSI-<提交>`；没有 `git pull`。
  7. flash-attn：同一个第三方预编译轮子（见上），从 zju 缓存经本机中转（253 MB，直连跳板会断，用 `ServerAliveInterval`），核对 SHA256 后 `uv pip install --no-deps`。**A800（sm_80）上只验证了能导入，没有跑过端到端**。
- 启动示例：`CUDA_DEVICE_ORDER=PCI_BUS_ID RIFT_PLATFORM=cuda RIFT_WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=3,5 python -m rift.runtime.ascend_910b3 --service-model /data/home/hejingyu/models/Qwen/Qwen3-4B ...`。

## 用法
```bash
source /home/hejingyu/miniconda3/etc/profile.d/conda.sh && conda activate rift-gpu
W=/home/hejingyu/projects/RIFT-RSI-0cf9a8d          # 固定到某个提交的 worktree
export PYTHONPATH="$W/src:$W/third_party/usami" CUDA_VISIBLE_DEVICES=<空闲卡>
cd $W && python -m rift.designer.stage0 --run-root <新目录> --failures <failures.json> \
    --designer-model /home/shenyl/hf/model/Qwen/Qwen3-4B --samples 64 --tensor-parallel-size 1
```

## 实测耗时（单卡 Qwen3-4B，enforce_eager）
- 加载权重 211 秒（NFS 冷读；同一台机器第二次会快很多）。
- 第一条带 schema 的生成 98 秒，其中大部分是 xgrammar 第一次编译 schema，之后同一 schema 复用。
