# 新服务器第三轮：实际命令与复现入口

工作目录 /root/autodl-tmp/prefix-io-v1-handoff/project。所有实际命令与输出见交付包 server-root/commands.jsonl、专项 commands.jsonl、launch.json 和 runs。下列解释已运行步骤及今后重放入口，不表示应再次消耗预算。每次重新运行必须使用新 label，保留旧证据；预算或活动预留门禁不能绕过。

## 已执行 CPU 检查

~~~bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv-prefix/bin/python -m pytest -q -ra tests/prefix_io_v1_download --junitxml=artifacts/prefix_io_v1/new-server-03/resume-review/first.xml
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/prefix_io_v1_native_prefix -q --junitxml=artifacts/prefix_io_v1/new-server-03/native-prefix-kv-dtype/cpu-tests.xml
CUDA_VISIBLE_DEVICES='' RUN_E2E_TESTS='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:third_party/work/py-kvcache-observer .venv-prefix/bin/python -m pytest artifacts/prefix_io_v1/new-server-03/delivery-redaction/test_redact_delivery.py -q --import-mode=importlib -W error::pytest.PytestUnhandledThreadExceptionWarning
~~~

结果分别为 109、61、15 passed。本节命令保留原证据输出路径，重跑时必须改成新的输出位置。历史 39／65 和 24／49／51 是上述 suite 的早期版本，不额外累计。短 IPC 实际 bind/close、header 实际审计另有 CPU 证据。

~~~bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python experiments/prefix_io_v1/scripts/audit_local_model_headers.py --model-dir models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444 --output artifacts/prefix_io_v1/new-server-03/model-header-audit.json
/root/miniconda3/bin/python -I -S artifacts/prefix_io_v1/new-server-03/platform-probe/io_uring_setup_probe.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m prefix_io_control.preflight --controller docs/prefix_io_v1/templates/controller_spec.yaml --permissions experiments/prefix_io_v1/configs/permissions.yaml --capabilities artifacts/prefix_io_v1/new-server-03/capability-report.json
~~~

header 审计通过；探针 exit2/EPERM；preflight exit2，io_uring 和 real handler 未通过。后两者是保留的真实阻塞，不是测试成功。

## 模型准备的实际命令

实际 argv、wrapper PID、日志、退出码分别保存在 new-server-03 的 download-launch.json、download-retry-launch.json、resume-launch.json、final-download-launch.json。

下载器入口：

~~~bash
PYTHONPATH=src timeout --signal=TERM --kill-after=15s 1800s .venv/bin/python experiments/prefix_io_v1/scripts/download_pinned_model.py --manifest artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json --model-dir models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444 --label qwen-ms-02
~~~

qwen-ms-01 先因未知 CDN 跳转被拒绝；核验官方跳转并通过 CPU 检查后，qwen-ms-02 下载前三分片，第四分片遭原 1800 秒上限终止，exit124。两个失败按完整预留保守结算，没有退款。

实际显式续传：

~~~bash
PYTHONPATH=src timeout --signal=TERM --kill-after=15s 900s .venv/bin/python experiments/prefix_io_v1/scripts/resume_pinned_model.py --manifest artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json --model-dir models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444 --label qwen-ms-resume-03 --failed-label qwen-ms-02 --file model-00004-of-00004.safetensors --partial models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444/model-00004-of-00004.safetensors.qwen-ms-02.partial --offset 3321888768 --partial-sha256 533b7c828ecc2fa7641c61c5063b4efb3037e0153d81d3a90204ad52d3976214
~~~

实际收到严格 206／Content-Range／Content-Length 对应的缺失 234,488,904 B，重组整文件 SHA 通过，exit0。最后以原下载器、新 label qwen-ms-final-04、900 秒上限复用验证已有文件，补 tokenizer.json 和 vocab.json，最终 VERIFIED_LOCAL_FILES／exit0。已经发布的文件不可再原样续传；脚本拒绝覆盖，勿重放历史失败命令。

## 原生 GPU 命令与环境

原生实现来自固定作者 vLLM 源码构建，不能换成上游 wheel。成功运行的原始 argv 和精确环境在 artifacts/prefix_io_v1/new-server-03/native-prefix-04-launch.json。wrapper 在独立会话启动预算 runner；真正模型子进程会话由 runner 创建和清理。作业 01／02／03 的不同冻结配置及失败保留，04 为最终通过配置。

下面是等价重放入口，已使用新 label 示例；只有原权限有效、目标设备空闲、剩余预算充足、无未结算预留时可运行：

~~~bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
export CUDA_HOME="$PWD/experiments/prefix_io_v1/cuda-toolkit"
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export NVCC_THREADS=1
export MAX_JOBS=4
export FLASHINFER_WORKSPACE_BASE="$PWD/experiments/prefix_io_v1/runtime-cache/flashinfer"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export FLASHINFER_NO_DOWNLOAD=1
.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label native-capacity-replay-01 --seconds 30 -- .venv/bin/python experiments/prefix_io_v1/scripts/check_selected_gpu_capacity.py
# 只有上一容量门禁退出0，才执行下面模型命令。
.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label native-prefix-replay-01 --seconds 900 -- .venv/bin/python experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py --model-dir models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444 --model-plan artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json --output-dir experiments/prefix_io_v1/runs/native-prefix-replay-01/details
~~~

这些 export 仅影响当前 shell 和子进程，不应写进系统配置。runner 绑定批准 UUID、设置 HF／Transformers offline、启动前预留、退出后按墙钟结算并清理整个子会话。其本身不提供网络隔离；FlashInfer 额外使用上述原生禁止下载开关。

smoke 本身固定项目 .p1tmp 和 VLLM_RPC_BASE_PATH、项目缓存、私有 run cwd、关闭 usage stats；为受信本地一次 KV metadata callable 显式使用作者序列化开关，并写入 frozen-config。没有启动 HTTP API 服务；不声称 native NCCL 初始化完全不使用本机 TCP。auto KV dtype 必须继承实际 torch.bfloat16，worker 再校验 28 层 CUDA BF16 tensor；无量化、无外部 KV connector。

## 补丁与交付

现有源码已含最终 smoke，不要重复 apply。若从旧版还原，按本轮最终 0001 provider、0002 cwd、0003 KV metadata、0004 IPC/prefill、0005 auto/BF16 顺序应用一次，忽略 before-env 历史变体；每阶段都有源码 SHA 与正反检查。作者引擎共同补丁保持第二轮版本，观察／研究补丁不激活。

交付只复制白名单源码、配置、锁和证据，不包含模型权重、partial、虚拟环境、SDK、运行时 JIT 二进制或完整第三方 checkout。用固定 SHA 获取第三方源码，用模型清单验证现有本地文件。复制时脱敏工具和 SOURCE→COPY 哈希映射位于 delivery-redaction；原始日志保留服务器。

下一允许阶段仍为 P1。当前不要原样启动作者 80k/32GiB staging E2E 默认配置、自动 wipe 驱动或会逃离预算会话的 server helper；本轮 native-integration-review 已记录具体限制。先由平台恢复 io_uring，再按真实成本准入与原生命周期验证 staging／SSD／生产 KV，不能以替代执行器解除门禁。
