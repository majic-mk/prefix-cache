# PaceKV R1a：RTX 5090 租机与环境预检

日期：2026-09-23。用户负责租机、开机和费用；本文件不授权自动下单。旧 A800 无卡快照与结果保留为历史工件，**不得在新卡上直接执行**。

## 选择与租机规格

- 首选单张独占 **GeForce RTX 5090 32GB**。RTX 4090 只有 24GB（标称十进制容量），而目前冻结的诊断入口要求加载前可用 GPU 显存至少 **24 GiB**，因此 4090 不满足现行任务预检。不能仅删 guard 就宣称 4090 等价；若要用 4090，须另注册缩小负载及其协议，不与 5090 结果混用。
- Linux x86_64，Ubuntu 22.04/24.04；建议至少 **64GB 主存**，GPU 模型加载时可观察的 host/cgroup 可用量至少 **32 GiB**。建议 8 个以上 CPU 核、NVMe、模型与依赖放置后仍有至少 **50GB** 数据盘余量（80GB 更稳妥）。2GiB cgroup 上限的旧无卡实例不合格。
- Linux NVIDIA 驱动选择供应商支持 RTX 5090 与 CUDA 12.8 的 **R570 或更新**版本，优先采用平台提供的更新稳定驱动。记录驱动、GPU UUID、PCIe 实际链路/宽度和完整 `nvidia-smi` 输出。`nvidia-smi` 报告的“CUDA Version”是驱动支持上限，不等于 Python 中 Torch wheel 的 CUDA 版本。
- 首批 R1a 仅限隔离的 Hugging Face 模型/KV 输运微诊断；不把它当成原生 vLLM 多请求吞吐测试。旧 vLLM0.4.1/CacheBlend-cu121 环境不可复用。

## 独立 Python 环境

选 Python 3.11，先建立新的虚拟环境；下面命令只供用户所租 **5090 Linux 实例**使用，不能在历史旧环境执行：

```bash
python3.11 -m venv /root/autodl-tmp/pacekv-env-cu128
source /root/autodl-tmp/pacekv-env-cu128/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==4.40.2 safetensors numpy==1.26.4 sentencepiece
```

预编译 Torch wheel 一般不要求另装完整 CUDA Toolkit；**驱动仍必需**。只有编译自定义 CUDA 扩展或从源码构建 vLLM 时才单独规划 Toolkit/NVCC。不要在此环境里顺手安装 vLLM 并让它替换已审计的 Torch；R1b 另建环境，按所选 vLLM 发布版支持的 CUDA/Torch 组合安装和绑定。

安装后先验证：

```bash
nvidia-smi
python - <<'PY'
import torch, transformers
print('torch', torch.__version__, 'torch CUDA', torch.version.cuda)
print('transformers', transformers.__version__)
print('GPU', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
print('BF16', torch.cuda.is_bf16_supported(), 'free', torch.cuda.mem_get_info()[0])
x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16)
print('SM120 kernel smoke:', (x @ x).float().abs().sum().item())
PY
```

必须观察到 `torch==2.7.1+cu128`、`torch.version.cuda==12.8`、`capability==(12,0)`、BF16 可用，以及真实 GPU kernel 成功；仅安装成功不算通过。实际 GPU 名称须为 `NVIDIA GeForce RTX 5090`，可用显存至少 24 GiB。模型资产 revision、tokenizer、代码 SHA 和 cgroup 内存仍由本批 CPU/GPU audit 检查。

## 阶段门

1. 在新实例上，使用**新代码快照、新唯一输出目录**运行 `scripts/pacekv/prepare_pilot.py`；旧 A800 `cpu_preflight.json`、`plan.json` 和旧代码摘要不能复用。代码尚未部署到新实例时，保持待验证。
2. 先做同环境的 CPU 小模型缓存/连续生成兼容性检查，再启动 7B 模型；Torch2.7.1/Transformers4.40.2 在 5090 上尚无实测，不把版本推断写成已通过。
3. 核对代码、模型、依赖、可用容量、GPU 是否独占后，用户明确给定时限，再运行显式 `--execute-gpu --max-seconds` 的 R1a；脚本仅停止任务，不关机、续租或付款。
4. 数值与原始事件通过后，才决定 R1b 现代原生 vLLM 后端；它目前**未实现**。不得从 R1a 推断生产系统收益。

参考：[5090 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)、[4090 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)、[PyTorch CUDA 12.8 wheel](https://pytorch.org/get-started/previous-versions/)、[CUDA 12.8 驱动/架构说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/)、[vLLM 安装和 CUDA 变体](https://docs.vllm.ai/en/v0.25.0/getting_started/installation/gpu/)。
