本次已依据用户真实回复“授权这 2 次 G2 验证并继续”执行第一项 off 正常模型资格验证。项目为 /root/autodl-tmp/prefix-io-v1-handoff/project，GPU UUID GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2，源锁 SHA 5e7cc04fe759411800b895169d669f75cade6b2f0adda34d3992ecf2a7dbd760，guard 限时 300 秒，累计 8 小时预算没有重置。

实际结果：off 退出 1，child_exit=1，没有超时。账本计入 65.7177117979154 秒，SID 6731，前后残留成员均为空，session_drained=true。G2 当前只运行 1 个 job，shadow01 未启动；失败的 off01 不在本 scope 下重试。原累计已用 16604.382256507408 秒，剩余 12195.617743492592 秒，active reservation 为 null。账本 339513 B，SHA 1e5b29fbc7fb66f84f3554149e5ac24f5a342b349f02700062d9a1bf53a0679e，events=224。

模型确实开始了 GPU 初始化：原作者 vLLM 识别 Qwen2ForCausalLM，4 份已有 safetensors 全部载入，日志记录权重加载 2.87 秒、模型加载约 14.29 GiB 显存。但 LLM 构造没有完成，phases=[]，尚未生成 cold 或 repeat 的 128 tokens，也未安装 worker 观测。不能把加载成功算作完整模型、Prefix 命中、P4 或方法收益通过。

失败链路由服务器实际源码与环境核验确认：
1. launch 环境 CUDA_HOME / CUDA_PATH / NVCC / CUDACXX 均未设置，PATH 未找到 nvcc。
2. FlashInfer cpp_ext.get_cuda_path 回退 /usr/local/cuda，该路径实际解析到 CUDA 12.8；它的 get_cuda_version 优先运行选中 nvcc --version。
3. 实际 CompilationContext 的 SM 12.x 规范化要求 CUDA >=12.9。当前选中 12.8，目标架构不能加入集合，日志出现两次 “SM 12.x requires CUDA >= 12.9”。
4. core.check_cuda_arch 在目标集合没有合格项时抛 “FlashInfer requires GPUs with sm75 or higher”。这条次级错误不表示 RTX 5090 硬件低于 SM75。
5. 原模型 AttentionBackend 是 TRITON_ATTN，但采样器仍选择 FlashInfer，因此只选择 Triton attention 不能回避此依赖。

可用修复条件已实际找到：虚拟环境已有 nvidia/cu13/bin/nvcc 13.0.88，PyTorch 源码 cuda='13.0'、版本 2.11.0+cu130。没有安装或下载 CUDA。其 bin/include/nvvm 三目录完整只读 inventory 共 1934 个文件、217111529 B；CUDA13_SOURCE_INVENTORY.json SHA 2ed079701c53d3e37005cf33432937e6c442f06cad16c5aded06efffcc7c36e0。现有 wheel 只有 lib/libcudart.so.13，没有 FlashInfer 默认链接期望的 lib64/libcudart.so，因此不能只设 CUDA_HOME 后就宣称可行。

已完成真实服务器 CPU 编译/链接检查：新增项目私有 SDK overlay 只链接现有 cu13 bin/include/nvvm，lib64 内建立指向已有 libcudart.so.13 的别名，lib64/stubs/libcuda.so 仅供链接指向已有 driver 二进制；没有改 /usr/local、.venv、驱动或系统配置，stub 路径没有加入运行时 LD_LIBRARY_PATH。nvcc --version、compute_120f/sm_120f 对象编译（0.724 秒）、c++ 共享对象主机链接（0.070 秒）、readelf 四条命令全部 exit=0。readelf 记录实际依赖 libcudart.so.13。没有加载该 .so，没有执行 GPU kernel，没有调用 torch/FlashInfer 运行时。

这批 CPU 检查证明已有工具链可以满足架构编译与链接的前置条件，不证明真实模型执行已修复。下一步准备最小项目私有 SDK/helper 候选与新 runner 标签 off02/shadow02，完成 CPU 回归、独审和新源锁后，提出下一轮具体 GPU 授权；两种模式使用相同环境修复，研究策略仍关闭。原执行器、attention、精度、采样、Prefix Cache 和系统/驱动均不替换。

授权文件的首轮 CPU preflight 曾因静态模板把温度 0.0 写成 int 0 而退出 78，尚未导入 GPU。保留 old scope 与失败回执，新增 V2 只纠正 JSON 类型为 float 0.0，数值、源锁、预算、模式、标签与真实用户回复不变；V2 实际 CPU preflight 通过后才进行了本次 GPU job。该格式修复不属于实验策略，也未消耗额外 GPU job。

运行命令：
```sh
cd /root/autodl-tmp/prefix-io-v1-handoff/project && .venv/bin/python -B artifacts/prefix_io_v1/server09-g2-normal-worker-20261001/run_g2_normal_model_lifecycle.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --mode off --name server09-g2-normal-off-01 --source-lock artifacts/prefix_io_v1/server09-g2-normal-worker-20261001/gpu-source-lock-candidate.json --scope-record artifacts/prefix_io_v1/server09-g2-normal-worker-20261001/GPU_STAGE_AUTHORIZATION_V2.json --launch
```

CPU 命令、工具链完整核验、编译与链接参数、stdout/stderr 均在 ACTUAL_CPU_CUDA_TOOLCHAIN_AUDIT.json、ACTUAL_CPU_CUDA13_LAYOUT_AND_SOURCE.json、ACTUAL_CPU_CUDA13_COMPILE_LINK_RECEIPT.json 和 cuda13-cpu/CPU_COMPILE_LINK_RESULT.json。实际 GPU 日志与结果位于 runs/off/process.log、runs/off/result.json、runs/off/details/normal-model-lifecycle-result.json；完整账本快照为 GPU_BUDGET_LEDGER_AFTER_OFF.json。

本记录是失败后的真实交付，不改写前一份 CPU 完成时“待授权”报告。真实 GPU 总数为 G1 两次 + G2 off 一次；本次 CPU 编译新 GPU 操作为 0。P4 和效果验证仍未完成。

