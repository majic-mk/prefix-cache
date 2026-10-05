# 服务器09 CUDA13 G2 off02 真实失败与 CPU 修复准备

本轮新的直接人类授权下，真实 GPU 作业只执行了一次 off02，计入原累计预算 66.00550774950534 秒；退出 1、无超时、无中断，原 guard 确认真正会话排空，active_reservation=null。shadow02 未执行。没有完成任何 cold/repeat 请求或正常 128-token 输出，G2、GPU kernel 和效果资格仍未通过。

## 实际执行和失败

GPU 命令（在原项目根执行，未使用额外 PYTHONPATH）：

```sh
cd '/root/autodl-tmp/prefix-io-v1-handoff/project' && .venv/bin/python -B artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-20261001/run_g2_normal_model_lifecycle.py --project '/root/autodl-tmp/prefix-io-v1-handoff/project' --mode off --name server09-g2-normal-off-02 --source-lock artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-20261001/gpu-source-lock-candidate.json --scope-record artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-20261001/GPU_STAGE_AUTHORIZATION.json --launch
```

对应原 guard 子命令及精确会话 8496、UUID、300 秒执行限时和原权限引用记录在 `runs/off/result.json`。新源锁 f3d21df…、独立 human record 和 19 字段 scope 由真实 runtime.scope_template 生成，temperature 保持 Python float0.0；CPU 门禁和独立审查通过后才启动 GPU。

实际日志确认全部 4 safetensors 分片已加载，模型内存约 14.29GiB。私有 CUDA13 SDK 六链接和六个进程环境字段真实出现在结果中，采样器构建文件实际采用 CUDA13 nvcc 和 compute_120f/sm_120f。上一轮 SM12.x/CUDA12.9 警告未再出现。随后 LLM 初始化进入 FlashInfer 原采样器 JIT 构建，报 `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`。

`phases=[]`，没有 frontend 输出文件。LLM 构造未成功赋值，`original_engine_shutdown_returned` 不存在；日志出现 NCCL 未显式 destroy 的警告。结果中的 `source_lock_unchanged_after_original_shutdown=True` 是无条件 finally 中的源核验，不能证明原 shutdown 曾调用或返回。这里唯一真实清理证据为原 guard 的 OS 会话排空。

实际源锁仍为 `f3d21df27310ce2c5199ab69f2a62f80b2111390f3ad160aba6ea843dd06f1d9`，scope 为 `cfced875bd91a92f949bc66cae85717f66b8742308166ceb441ba1f8f9111b4a`；这次失败没有修改冻结入口、原引擎或缓存实现。

## CPU 排查与真实编译证明

只读审计发现 Ninja 已安装：`.venv/bin/ninja` 是非链接、可执行 ELF，370448B，SHA-256 `08639e194fffa7f08b259fc4abfa4803aff66b64de52549cee42ec527d55cea6`。真实原 PATH 只有 CUDA13 SDK/bin 加系统目录，没有 .venv/bin；系统各候选位置无 Ninja。缺的是进程路径，不需要下载/安装或修改系统。

实际 CPU `ninja --version` 返回 `1.13.2.git.kitware.jobserver-pipe-1`；`ninja -t commands` 从 off02 原 build.ninja 读出 3 个 CUDA 编译与 1 个主机链接命令。该原文件 5049B，SHA `ef69f2b7ad3570fbc9ba42de3eed9e1c7def990915b077a5bb17ad5516e930ee`。全部 CUDA13 compiler、SDK、SM120f flags、include 和原 sampling.cu/renorm.cu/flashinfer_sampling_binding.cu 输入保持。

随后仅将绝对输出 build 目录改到新 CPU 目录 `artifacts/prefix_io_v1/server09-ninja-cpu-20261001/sampling-build`，实际执行：

```sh
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/ninja -f build.ninja -j 1 -v
/usr/bin/readelf -d sampling.so
```

三项 CUDA 编译、最终主机链接、readelf 全部退出 0，合计 96.23423069156706 秒，PASS_CPU_REAL_SAMPLER_NINJA_COMPILE_LINK_ONLY。真实 `sampling.so` 为 13184480B，SHA `89a017e07ff41d9e3afceb6b550f8e09eed94528319bf781f2165c2f0fadd858`；三项 .o、依赖列表、编译日志和原/新 Ninja 文件均保留并原字节镜像。

该 CPU 命令没有导入 framework，没有加载 sampling.so，没有执行 GPU kernel，也没有改写旧 GPU build 目录或预算。它验证了真实采样器的 CPU 构建依赖，比小探针完整；不证明动态库能成功加载、模型能完成初始化或 GPU 能输出。未来 GPU 仍走原 FlashInfer 每作业 JIT，不借用这份 CPU .so。

## 实际资源与证据

本次 GPU 作业 1 次、shadow 0 次。原累计用时 16670.387764256913 秒，剩余 12129.612235743087 秒（约 3.369 小时），225 个预算事件，无活动预留；账本 341046B，SHA `2d9dbb06280e10aa395f4d9eb1fba7606c401d1ff0c8d4d6a474c454f5c16276`。

off02 私有缓存真实只有 4 个常规文件、5797B 逻辑大小、12288B 唯一分配块和六个 SDK 链接；链接目标没有计作复制数据。CPU 实际编译产物另计，CPU 收口时 PRIMARY 仍可用 12948668416B，超过 8GiB 下限与 128MiB 预留。未下载、删除任何数据，未修改系统、驱动或已安装软件。

证据文件：

- `runs/off/result.json`、`process.log` 和 `details/normal-model-lifecycle-result.json`：真实 GPU 原文件及哈希。
- `GPU_BUDGET_LEDGER_AFTER_OFF.json`：完整原预算快照。
- `HUMAN_AUTHORIZATION_RECORD_G2.json`、`GPU_STAGE_AUTHORIZATION.json`、`ACTUAL_NEW_HUMAN_SCOPE_CREATION.json`：本次实际人类回复、范围与创建回执。
- `ACTUAL_AUTHORIZED_OFF_CPU_PREFLIGHT.json`、`ACTUAL_OFF_LAUNCH_RECEIPT.json`、`POST_OFF_READONLY_RESULT.json`：真实执行命令、退出码和只读收集。
- `INDEPENDENT_OFF02_FAILURE_REVIEW.json`：独立实际失败事实核验；成功资格为 false。
- `ACTUAL_CPU_BUILD_LAUNCHER_AUDIT.json`、`ACTUAL_CPU_NINJA_VERSION_AND_GENERATED_COMMANDS.json`：Ninja 与真实生成构建路径审计。
- `ninja-cpu/`、`ACTUAL_CPU_REAL_SAMPLER_BUILD_LAUNCH.json`、`ACTUAL_CPU_SAMPLER_PROOF_AND_BUDGET_CLOSURE.json`：真实 CPU 构建及账本/资源收口。
- `ACTUAL_CPU_COMPILER_EVIDENCE_REMOTE_MIRROR.json`：11 个真实 CPU 编译文件的服务器同字节镜像。

本机交付根为本文件所在目录；对应服务器 `artifacts/prefix_io_v1/server09-g2-cuda13-20261001`。原 CPU 编译目录、原 off02 目录和原 CUDA13 source 包均保留。

## 下一允许阶段

继续 CPU 开发与测试：新的独立 Ninja 入口只将固定已安装 Ninja 放入新作业私有 build-tools/bin 单链接，追加到 CUDA13 SDK/bin 之后的 PATH。SDK、模型、原 worker、Prefix、采样和执行器保持，off/shadow 共同使用。新候选、源锁、服务器 CPU 测试和独审须先完成。

off02 已使用且失败，旧 scope 未用的 shadow02 只有 off02 成功后才有效，不能借作 retry。任何 off03/shadow03 实跑需新的直接人类授权绑定新冻结源锁和固定名称，继续原预算；不会擅自重跑 off02 或启动 shadow02。

本轮无性能实验、成本/时钟映射/SSD 资格或方法优势结论。下一 GPU 仍为完整模型输出与生命周期资格验证，不能等同 P4 或效果对照完成。
