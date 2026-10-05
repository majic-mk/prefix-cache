# 当前环境原始成本校准入口只读审核

日期：2026-10-02。审核限定为本地已有源码/JSON 的读取、SHA-256 与字节数核验；没有运行脚本入口、框架导入、编译、RPC 或 GPU，没有产生测量、曲线或新执行器。G3 已冻结包和全部旧代码保持原字节。

结论：现有 `acquire_native_aio_costs.py` 是可以复用的真实 f/gSSD/gMem 采集主体；三个指定脚本分别是早期 CPU 候选计划、旧曲线长上下文集成消费者和已有真实样本的候选导出器。**当前没有一个可直接启动、同时满足当前导入环境、源码/授权/账本、真实 owner 尾部闭环和新曲线资格的完整入口。** 下一步需要新建薄启动/证据接线，继续调用原 LLM/原 generate/原 connector。不能将旧候选曲线改 UUID/标签，也不能将 G2 普通模型资格当 SSD 或成本资格。

## 源码核验范围

当前对照锁：`artifacts/prefix_io_v1_server09_candidates/g2_normal_worker_site_cache_v4_final/gpu-source-lock-candidate.json`，899396 bytes，SHA-256 `0325cb500f74051dedf6aefe6b51fd9d0b817f831ee81fe45e1c3942df13336b`，4050 项。逐项核验结果见同目录 `READONLY_SOURCE_CHECKS.json`。

指定三个脚本的本地 mirror 在 `artifacts/prefix_io_v1_server08_primary_qualification/contents/experiments/prefix_io_v1/scripts/`；它们及以下主体/分析器均与锁里的原路径、bytes、SHA 完全一致：

| 当前服务器原路径末段 | bytes | SHA-256 |
|---|---:|---|
| prepare_native_calibration_plan.py | 16365 | bbea9525726a3e88a6cd2e5e7489cefe77b391606213999ad70737a36a1567f0 |
| run_long_calibration_replay.py | 13833 | 8780a715cb48b65a836c34870e1a6d4367a584476f8e4660eacf1f3a88d8d8fb |
| export_uniform_calibration_candidate.py | 9035 | 45df16e11e56d4ec07b00c4a3850882a9283c3b55d22515061e8d8bc32b6a95e |
| acquire_native_aio_costs.py | 18995 | 68b2c045bcc9a7de84771d556b0e01180a592280002a2d4360d1c7500c9c856e |
| analyze_cached_references.py | 7783 | 2aaed52742c23750d068b16f4ad51e1caaa9427dc1ff39fdd09e3440460bbc1c |
| analyze_repeated_native_costs.py | 6695 | 21bf7fc7562dcf1711a0e3dbad9b740e122a7126304d7f048258f79b32dc773a |
| export_native_aio_costs.py | 6397 | 19f6911e78d59eacc148a36f48004f887946d9a655f1be53d2c3a40628787b49 |

本地 `concurrent_pilot_contract.py` **不匹配**：已有所有八份 mirror 为 3991 bytes / `1c203cd21b6b04e6d50449131da72cea8de6d8f60ffc15bf45f2714d4b843952`；当前锁是 3982 bytes / `645320e10a23795435f8a7eb77b6d7c24def8b58b1c6f82c662fa069482811f8`。其中 API 内容不能视作当前服务器已核验源码。primary mirror 的旧 `run_gpu_stage.py` 也不匹配当前 13013 bytes / `3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a`；新入口应复用当前锁定 guard，不能复制旧 guard。

审核中 root 另行完成服务器只读镜像补齐，保存于本报告目录 `source_readonly/<project-relative-path>`。本 agent 再次独立核验当前 `concurrent_pilot_contract.py`3982/645320…、`acquire_native_aio_costs_capacity_p316.py`20018/c4f50…、`vllm/v1/metrics/stats.py`18538/08eb6…均 exact 匹配0325；后续判断已基于这些当前字节，旧 primary 依赖差异保留作证据，不再列为未解决缺镜像。

## f、gSSD、gMem 实际如何产生

采集主体 `acquire_native_aio_costs.py:78` 的 CLI 已有 `--mode cold|populate|restore|paired|planned`、`--sizes`、`--reps`、`--domain 1024|4096|8192|16384`、`--max-num-seqs`、`--iodepth`、`--model-dir`、`--model-plan`、`--storage`、`--output-dir`。

轴是 N 个可复用 prefix token；每个实际请求含 **N+1 prompt token，输出一个 token**。`prompt(n,rep)` 用不同第一 token 区分每个 repetition。所有 f/g 的值取原 `RequestOutput.metrics.first_token_latency` 秒数（225 行），额外 `time.perf_counter` 包围原 `llm.generate` 的 `wall_seconds` 仅诊断；导出器没有拿 wall 时间替代 f/g。

| 路径 | 原执行流程与必须有的证据 |
|---|---|
| f | `cold` 不安装 KV connector；GPU-only reset 后原 generate；`num_cached_tokens==0`，输出 1，prompt N+1，原 TTFT>0。 |
| gMem | `populate` 先原 generate 存 prefix、原 sentinel engine 请求递送完成元数据，再 drain；GPU-only reset 不 reset connector 后加载已有 staging。要求 full N 命中、成功 load/H2D、实际 SSD 读 0、`src_cache>0`。 |
| gSSD | `restore` 或 `paired` 使用已发布目录，**新 engine 的 GPU/staging 从冷状态开始**；N 命中、成功 load/H2D，前台逻辑读+成功 preload 实际读合计 N×57344 bytes。 |
| paired gMem | gSSD 完成后只 reset GPU prefix，保留同一 engine staging，再原 generate；要求 N 命中、`src_cache==N//16`、H2D 成功、SSD 读 0，两个 cached 输出一致。 |

核心位置：generate/metrics 177–210 行；populate 216–220 行；GPU-only reset 221–222 行；f 225–226 行；SSD/staging 路径 244–258 行；original shutdown 268–270 行。复制、合并、预加载、共享 staging 和异步 I/O 仍来自原 connector/reactor，没有新 backend。

gSSD 与 gMem 都是**端到端首 token 成本**，包含原加载与剩余模型处理路径；它们不是孤立 SSD 带宽、GPU-only kernel 时间或 P4 decode 干扰成本。`gSSD-gMem` 不能无条件解释为纯 SSD 服务时间。原 profiler trace 证明 transfer 来源/覆盖，未证明 GPU-event 与 host-clock 映射，也没有完整 128-output decode trace 的成本资格。

这里的 one-token、显式 GPU-only reset、主动尾部等待仅限**原基础成本校准**，用来隔离冷计算/SSD/staging 路径；不能直接充当 P4 自然请求流、受控 decode 干扰或新方法性能对照。它不验证有限候选调度/干扰额度的有效提升，也不代替原正常128-output生命周期。之后若做 P4受控流/效果对照，必须另有该目的的完整正常trace/基线策略一致性证据。

补齐的当前 `stats.py:325–371` 已证明准确公式：`IterationStats.iteration_timestamp=time.time()`；prefilling output 分支将 `first_token_latency=iteration_timestamp-req_stats.arrival_time`。该 frontend arrival 字段注明 wall-clock（207–208），包含前端排队/传送/原采样返回所观察到的时间，不是 CUDA event 成本。engine-core queued/scheduled/first/last 字段注明 monotonic（210–214），其中 first_token_ts 另取传入 engine_core_timestamp（395–402），**不能拿 wall arrival 与 monotonic token timestamp 相减**。`output_processor.py` 原调用/arrival赋值源尚未在本地镜像逐字节审阅；完整接线时补该一份有限源码即可，无需更换时钟或重写统计器。

## 模型、布局与资源几何

原 smoke 基础 ENGINE 为 Qwen2.5-7B-Instruct、BF16、KV auto、无量化、tp=1、Uni、16-token block、TRITON_ATTN、enforce_eager、compile=0、精确 Prefix Cache 和 chunked prefill。采集器复用基础 ENGINE，但增加 SHA256 prefix hash、单输出和独立 domain 的 KV/staging 几何（143–159 行）：

| domain | max_model_len / max_num_batched_tokens | KV | staging |
|---:|---:|---:|---:|
| 1024 | 1040 | 256 MiB | 128 MiB |
| 4096 / 8192 | domain+16 | 1 GiB | 512 MiB |
| 16384 | 16400 | 2 GiB | 1 GiB |

原逻辑 token 密度是 57344 bytes，16-token 存储块为 917504 bytes；分析器同时检查 storage_block_bytes、I/O size 和 pinned staging。这里的常数不能代替新运行 actual KV tensor dtype/shape/layout、实际 allocated staging 和直接 I/O 对齐证据。`torch.cuda.mem_get_info()[0]>=24GiB` 是原采集器加载前的真实门槛，不是已有 CPU 元数据推出的可用显存。

采集器配置 `io_backend=linux_aio`，`sync_on_store=False`、preload on、lookahead1、preload_share_staging=True、staging_cache=lru、load_planner off（只有 planned on），iodepth 默认4。原源码没有显式固定 `async_scheduling=False`；未来薄启动必须明确同步模型调度，并保留原异步 I/O 流水线。本阶段不激活新候选调度/干扰额度 I/J、研究策略或成本准入。

新镜像的当前 `concurrent_pilot_contract.acquisition_delta` 确认 max_num_seqs=1返回空增量；只有 domain16384允许并发2。`acquisition_io_depth` 确认默认4可用于单序列，2/8只允许 domain16384且并发2。本次单点候选没有暗中采用 P316 并发/深度变体。

当前 P316 capacity 入口只接受 `cold|paired`、domain16384、max_num_seqs2、iodepth8，必须 `--capacity-domain` 和 heldout `--prompt-manifest`；按 `concurrent_capacity_contract_p316` 返回值设置 staging/预加载，并附 capacity_geometry+owner_snapshot。它保留原 TTFT/介质/数值 gate，不是小域采集替代品；capacity-domain具体集合/值尚未审阅对应6611 bytes / `49a33366092bf116a6a3b07ee2053c67602418c2f94c375fc58840c50f68934b` contract，不猜测它能使用的小预算。

## 三个指定脚本为何不能直接作为当前完整资格入口

1. `prepare_native_calibration_plan.py:197–244` 仅从原 Pareto AST 的 Job/build_job_plan 生成候选，`driver_ready=False`、f/g全部None、token-fit未证、budget未预留。nominal doc_size 是 words-derived 参数，和 exact-token 采集器不是同一轴。其 io_uring EPERM 是历史计划里的阻塞记录，不能当本机当前 Linux AIO 结果；gMem依赖SSD估计的说明也是这个早期计划路线，不能否定后来 paired 入口已存在真实 staging 测量。
2. `run_long_calibration_replay.py:54–190` 消费已有 curves+manifest+发布目录，配置原 planner on，固定长上下文 2GiB KV/1GiB staging、128 outputs，普通分支硬绑定 server07 storage3072×917504B。它可复用原 add_request/step/shutdown 结构，**不采集 f/g 新曲线**；不能换 PYTHONPATH 后把旧曲线/旧环境资格变为当前资格。
3. `export_uniform_calibration_candidate.py:24–120` 要求真正通过的独立 cached numerical reference、非 logprobs 的 cold/paired fit sessions、完全匹配 model/alias/GPU/config/sampling/domain、每点每路径至少4 measured samples和计划规定的独立 engine session数。rep0保留作 correctness/warmup但剔除计时。PCHIP 是原库，gSSD>=gMem检查与 break-even suffix gate 在测量范围内。结果明确 `candidate_only=True`、仅 same-budget calibration integration diagnostic、independent_content_validation=False；f(0)=0 / g-floor 是原插值约定，并非已测零 token 成本。

旧 `export_native_aio_costs.py` 还要求 cold/store/cached 四路完全相同输出，其覆盖没有 uniform 的冷/热 shape 分离 reference。应保留其历史语义，未来候选优先复用已有 uniform protocol，不能为了导出删掉真实冷/热差异。

## 原 LoadPlanner 的 raw calibration 门禁

当前 exact `py_kvcache/vllm.py:703–712` 即使 `load_planner=off`，仍先调用 `load_break_even(prefix_cache_break_even_path, model_name, dtype)`；只有 planner on 才 `_build_planner`。所以 raw cold/populate/paired 必须**不传 `--curves`**并保证 path=None，不仅将 planner置off。原 `break_even.py:123–158` 的无文件路径返回disabled thresholds，两个介质阈值0；若误给旧路径则原 worker scalar load gate 仍可生效、污染 raw 路径。保留 exact Prefix Cache / preload / shared staging，关闭成本选择即可采集原路径。

planner on 的已有门禁包括真正v2曲线、planned-defer API、preload/share-staging/positive lookahead、kv_bytes_per_token；原 reader核对model string，dtype差异仅warning，并**不检查 GPU UUID/source-lock/independent-content资格**。因此未来启用仍须外层真实当前环境/curve证据门禁（已有 G3 context audit可复用只读否决层），不能凭原 parser接受就授资格。原 `FileMapper` 使用实际 `model_config.model` 字符串建路径；采集器沿用 MODEL_ID alias，G2用local model绝对路径。薄启动应保留并冻结原采集 alias/model_name，不可仅因模型字节相同就混合两种模型名的storage/curve。

## 当前 CUDA13 / Ninja / site 修复可复用边界

当前 G2 runner 46746 bytes / `82cea015522596a89a89eb1ea0245f9ea111c942b74456cd87288507aec54422`、worker13090 / `096f2fffe91de7132f45997be51fecceec57d65fdf86caab6f8bbe5c71748f7b`、optional helper13805 / `48138660e3ac43e1ea7b342a2727772e8745ed0f59c355c50f10616584a956d3` 均 exact 匹配0325。

可复用该 runner 的 **common process initialization**（450–470、614–669 行）：新 run 内 cache env → 原已安装 CUDA13 overlay → 私有单一 Ninja link/PATH → original locked author Finder（P4 Python+逐个锁定 binary fallback）→ 原 import → exact optional-capability probe → actual GPU UUID/模型/worker source 验证。CUDA13/Ninja 只选择已安装字节，site helper只处理唯一事实缺失可选 deep_gemm 能力查询、保留原 cached query first 和必须已存在 canonical FileFinder；不修改系统、作者源码、precision、kernel、JIT/架构检查。finally 原 shutdown→probe detach→finder remove→post-sourcehash 也可复用。

原 acquirer 调用 `base.configure_runtime_environment()`，写共享 cache/.p1tmp；要求导入旧 `base.AUTHOR_ROOT=vllm-author-build`。这与当前严格 P4 Python+locked binary 的环境不同，原 script 不含新 SDK/Ninja/site 接线。**不能直接复制 G2 的 whole launcher/scope 当 calibration 授权**：G2 purpose是NORMAL_MODEL_FULL_OUTPUT_LIFECYCLE_QUALIFICATION_ONLY、native_io none、两请求各128输出和固定05 job；成本采集会启用 connector、不同 KV/staging、1输出和不同请求数。需要独立新 source/scope/argv/storage/budget gate，账本沿用真实累计值不 reset。

## 尚缺的最小 CPU 准备与下一有限 GPU 边界

没有已就绪的完整新 GPU calibration command。以下为下一 CPU 准备的具体依赖，不是本次实施或授权：

1. **有限源镜像**：concurrent、stats、capacity入口的3份当前源码已补齐并通过；仅完整timestamp接线需 `vllm/v1/engine/output_processor.py`31616/464c3…；若实际选择 P316 容量域，再补 `concurrent_capacity_contract_p316.py`6611/49a333…，按 exact lock row 读取，不靠未读源码猜容量。
2. **薄启动接线**：独立新增 acquirer launch adapter，复用当前 common environment/source/guard gates和原采集方法；新 scope明确 exact GPU、model/ref、domain、KV/staging、iodepth、timing、prompt/reps、output1、per-run源身份/失败存档，不改 author/executor/backend。CPU只读 preflight 不能 mkdir/import框架，actual startup 才进 cache/toolchain initialization。
3. **真实尾部/路径证据**：原 worker_control 主动 submit原queued stores并等待原Future/worker.wait，ring.snapshot检查accepted/completed/reaped；它不抢consume get_finished，但这不是 G3 release闭环的充分证据。新薄接线应把已冻结 source-bound owner observer 的 actual字段/正常original shutdown尾部证据附加到原采集输出，不能只凭 Future.done 或旧 `check_drains` 授 CUDA-copy/owner release 资格。CPU fixture/source-binding PASS仍不是实际GPU backing证据。
4. **只做可行性 point pilot，再决定拟合**：源码允许 `--domain 1024 --sizes 128 --reps 3 --max-num-seqs 1 --iodepth 4`；cold、populate、fresh-engine paired 共3原 engine 启动，3+9+6=18原请求（populate含store和sentinel），每条一个输出。rep0仍 warmup，f/g每路径仅2 measured samples，所以不能交 uniform 导出器，更不能形成有效曲线或效果结论。这是精确 CLI 可表达的最小路径/TTFT 可行性候选；**其256MiB KV/128MiB staging只适用于该固定域，不是16384域或P316容量的曲线**。若下一真实目标必须长域，要按原目标几何冻结采集而不能复用这个小点。实际 GPU 秒数要结合当前真实启动和存储预算新冻结，不据此预报完成时间。
5. **若目标是候选曲线**：必须另冻结至少2 fresh fit sessions与每点每路径至少4非warmup样本、独立原 native-hot/cached top5 numerical references、实际N-token layout、完整source/UUID和时钟来源。多点范围、统计与独立内容验证按原 uniform/heldout 协议，测量前冻结；不从单点/旧环境外推。

可复用的原 CLI 形状（**NOT READY / 不可执行授权**）：

```text
acquire_native_aio_costs.py --mode cold --domain 1024 --sizes 128 --reps 3 --max-num-seqs 1 --iodepth 4 --model-dir <locked-model> --model-plan <locked-plan> --storage <new-private-storage> --output-dir <new-cold-details>
acquire_native_aio_costs.py --mode populate [相同参数，new-storage，new-populate-details]
acquire_native_aio_costs.py --mode paired [相同参数，已发布同一storage，fresh-engine/new-paired-details]
export_uniform_calibration_candidate.py --plan <new-matching-reference-and-fit-plan> --out <new-candidate-dir>
```

最后一条只有完整原 reference+fit evidence 后才有意义，point pilot不满足。所有原入口最终都须走当前原 budget guard及新 scope封装；本审核未执行这些命令、测试或 GPU操作。此次实际命令仅 Get-Content/rg/Get-FileHash 和 stdlib 报告生成；JSON证据列出全部 source checks 和诊断性不匹配。下一允许阶段为 CPU 镜像/门禁/薄启动与 owner-tail 接线准备，GPU启动仍需新的明确范围与预算。
