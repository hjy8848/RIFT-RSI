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
- 训练：RIFT 的 VERL 运行时目前只有昇腾实现（`rift.hardware` 里 CUDA 平台 `runtime_ready=False`，调用会明确报 `PlatformNotReady`）。这个环境现在只用于阶段 0（Designer 出题 + CPU 认证）。

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
