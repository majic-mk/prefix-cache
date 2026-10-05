# 本轮交付草稿：新服务器 02

> 草稿快照，须由主线程补入本轮最终构建与资格检查结果后再用于正式交付。此文不修改正式阶段状态或锁，也不将运行中的构建计为成功。

服务器 connect.westc.seetacloud.com:24801；工作目录 /root/autodl-tmp/prefix-io-v1-handoff/project。本轮仍处于 P1 原版复现与共同基础准备，P2 仅保留既有 CPU 预备成果，P3–P7 未验收、未进入研究实验。io_uring_setup=EPERM 是当前固定作者执行路线的独立硬门禁。

## 权限与实际资源使用

授权来源见 artifacts/prefix_io_v1/new-server-02/authorization.json，修改前权限见 permissions-before.yaml。现行 experiments/prefix_io_v1/configs/permissions.yaml 允许首轮累计最多 8 GPU 小时、20 GiB 模型下载，唯一批准设备为 GPU-8b500efe-1a50-0e8e-b21e-716807eebedf；批准实验目录为项目内 experiments/prefix_io_v1/runs，依赖目录为本项目。未授权更改系统/驱动、另租服务器、付款或清理共享数据。

截至此草稿读取账本时，真实 GPU 作业仅 cuda-base-01，退出码 0，runner 记录墙钟时间 **3.268901824951172 秒**，约 0.00090803 GPU 小时；这不是 CUDA kernel 净执行时间。模型下载为 **0 字节 / 0 GiB**。软件依赖与构建源码下载不冒充模型下载。最终数字须以主线程交付时的 gpu-budget-ledger.json 为准。

cuda-base-01 实际使用 Torch 2.11.0+cu130、CUDA runtime 13.0、RTX 5090（compute capability 12.0），执行 1024 元素 CUDA 张量运算并检查精确结果。它证明基础 CUDA kernel 可运行；输出明确 vllm_verified=false、cache_verified=false。没有模型、KV 缓存、作者拷贝或 SSD 资格结论。

证据：experiments/prefix_io_v1/gpu-budget-ledger.json；experiments/prefix_io_v1/runs/cuda-base-01/{result.json,process.log}。

## 实际改动与验证

1. 在项目独立 .venv 内补充构建依赖并修正 CUDA 工具链。nvcc 从 13.4.92 固定为 13.0.88，CRT/NVVM 固定为 13.0.88，CCCL 固定为 13.0.85，与 Torch cu130 对齐；prepare_cuda_toolkit.py 在项目目录组织所需链接。未更换系统驱动或原 CPU .venv-prefix。最终依赖清单为 runtime-build-freeze.txt，runtime-build-pip-check.txt 报告 211 个已安装包依赖兼容；pip check 不代表扩展已编译或运行通过。
2. 作者 vLLM 保持锁定源码 817a7e3124f817cd6e549581d3e5483207a753a4，构建工作树为 third_party/work/vllm-author-build。build-source-lock.json 记录此次实际 FetchContent 源码与子模块 commit。构建脚本关闭预编译替代、隐藏 CUDA 设备，显式为 12.0 架构编译；本轮编译属于 CPU 构建，不计为 GPU 缓存验证。
3. run_gpu_stage.py 增加持久预算预留、并发互斥、异常/中断处理和整 session 清理；未结算预留会阻止下一次启动。修复了 Popen 启动窗口及子进程另建 process group 的清理缺口。最终 **21 项 CPU 测试通过**（session-final.txt/xml，5.18 秒），包括嵌套 group 和拒绝 TERM 的进程。测试仅启动 CPU 子进程并使用隔离账本；真实 GPU 账本、权限和基础 smoke 脚本哈希保持不变。主动 setsid() 逃离 session 不在可信命令合同内；不可捕获终止/宿主故障须人工核对持久预留，不能自动清零。
4. 已准备 qualify_author_copy.py：限定真实作者扩展路径与 schema，计划分别执行零长度 CPU ABI 检查及微量 GPU H2D/D2H 字节检查。当前只通过 AST/参数/已知字节区间、--help 及 UUID 不匹配拒绝测试，**没有加载扩展、没有执行真实 ABI op、没有执行作者 GPU copy**。
5. 已准备 qualify_author_imports.py：计划验证真实导入路径、Plan 类型/枚举身份、profiler、handler API 和作者编译扩展/注册符号。当前只执行 AST 与 compile-to-code-object 静态检查，**没有运行该脚本的真实 imports**。运行时 import 可能探测 NVML/CUDA，因此后续也由权限/预算 runner 包装。

本轮没有新增研究策略、普通传输额度、替代 I/O 队列、缓存引擎或模型执行器。既有共同修复与 observer 补丁仍分开；不把 runner 的 CPU 测试计作缓存后端/GPU 测试。

## 构建失败与重启记录

- author-build-01、author-build-02 失败：CMake/Caffe2 未能定位 CUDA libraries。完整日志和退出文件保留，不能删去后只报告成功尝试。
- 工具链版本与项目内链接随后修正，author-build-03 进入源码编译。
- 在确认 25 核 CPU 配额、90 GiB 内存和 359 项编译动作后，将并发从 4 调为 8，停止 03 并启动 04。第一次按 process group 检查不足以覆盖 timeout/ninja 的嵌套 group，随后按完整 session 清理并确认无剩余成员；纠正记录见 author-build-03-stopped.json。该重启不是构建通过。
- **author-build-04：RUNNING / PENDING。** 此草稿读取时尚无 author-build-04.exit；仅存在日志和中间构建证据。CMakeCache、build.ninja、CMakeConfigureLog 与 ninja-execution.log 保存在 author-build-04-detail/。不得据此宣布 wheel/install 成功。

## 已实际执行的关键命令

以下工作目录均为 /root/autodl-tmp/prefix-io-v1-handoff/project。详细执行记录位于上级 /root/autodl-tmp/prefix-io-v1-handoff/commands.jsonl，以及各 review/commands.jsonl、runner-review/session-commands.jsonl。以下为关键入口，不应重复运行以覆盖既有 label 或证据文件。

基础真实 GPU smoke：
```bash
.venv-prefix/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py \
  --label cuda-base-01 --seconds 45 -- \
  .venv/bin/python experiments/prefix_io_v1/scripts/cuda_base_smoke.py
```

项目内工具链修正（各自日志已保留）：
```bash
experiments/prefix_io_v1/tools/uv-bootstrap/bin/uv pip install \
  --python .venv/bin/python --index-url https://mirrors.aliyun.com/pypi/simple \
  nvidia-cuda-nvcc==13.0.88
experiments/prefix_io_v1/tools/uv-bootstrap/bin/uv pip install \
  --python .venv/bin/python --index-url https://mirrors.aliyun.com/pypi/simple \
  nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88 nvidia-cuda-cccl==13.0.85
```

构建通过 Python subprocess.Popen 启动以下脚本，并以 start_new_session=True 隔离；当前脚本使用 MAX_JOBS=8、NVCC_THREADS=1、5400 秒 timeout，结果写 author-build-04.log/.exit：
```bash
bash experiments/prefix_io_v1/scripts/build_author_vllm.sh
```

最终 runner CPU 验证：
```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
.venv-prefix/bin/python -m pytest -q -ra tests/prefix_io_v1_runner \
  --junitxml=artifacts/prefix_io_v1/new-server-02/runner-review/session-final.xml
```

缺陷复现先得到预期失败，修复后 21 项通过；before/final 证据均保留。真实 ABI/copy/import 的计划命令在 copy-review/CPU_SOURCE_REVIEW.md、import-review/README.md，未列为已执行。ABI 必须使用与构建匹配的 .venv Torch 和真实作者 .so 路径；示例占位路径不能原样执行。

## 待主线程填写的最终结果

| 项目 | 草稿状态 | 后续证据占位 |
|---|---|---|
| 作者 source build 04 / editable install | PENDING，读取时仍运行 | 最终 .exit、完整日志、真实安装路径 |
| 作者扩展 CPU ABI | PENDING，未执行 | 实际 .so 哈希、schema、零长度调用/拒绝检查 |
| 作者真实 GPU H2D/D2H | PENDING，未执行 | runner result、精确字节/guard、预算结算 |
| 作者真实 import/Plan/handler/copy 符号 | PENDING，未执行 | 真实 module origins 和 capability 结果 |
| 原生模型、GPU Prefix、CPU staging、SSD store/restore | BLOCKED / 未执行 | 必须先满足对应 P1 条件 |
| 本轮最终 GPU 小时/下载量 | 暂为 3.2689 秒 / 0 GiB | 主线程以最终账本补齐 |

## io_uring 门禁与下一允许阶段

现有新服务器实际证据是 raw io_uring_setup(depth=2, flags=0) 和作者 LiburingRing 都返回 EPERM；kernel.io_uring_disabled=0、进程 Seccomp=2，但没有读取过滤规则，不能断言唯一根因。O_DIRECT 4096 字节往返通过不能代替 io_uring。证据见 artifacts/prefix_io_v1/new-server-01/io-uring-diagnostic.json 与 odirect-probe.json；平台请求见 docs/prefix_io_v1/PLATFORM_IO_URING_REQUEST.md。

允许完成不依赖 SSD reactor 的作者构建及小规模 ABI/copy/import 独立资格，但即使全部通过，P1 仍不能验收。平台解除限制后先复测 LiburingRing 和原生 async-open 三项测试，再在预算内完成真实模型及 GPU/CPU staging/SSD 各层 smoke，并以真实 SSD read 字节或 I/O 事件证明确实走过 SSD。不得换成同步 I/O 或其他执行器绕过门禁。

P1 通过后才验收 P2 的真实 mandatory/shadow、owner/fence/generation 与端到端开销；普通额度和 P3–P7 继续依赖前序证据。此草稿没有吞吐、ITL、goodput、干扰、释放收益或论文结论。
