# P0 patch map
Frozen sources remain clean. Modified py-kvcache: third_party/work/py-kvcache, based on 3abba7a502d553f6e7e2e58b92086487e3395d7e.

Common CPU repairs: transfer.py imports / ParsedKvLayout.allocate_staging_buffer retain CPU Torch without claiming GPU copy capability; staging.py budget helpers and reactor.py IoReactor.__init__ reject actual backing over-budget before allocation and record storage bytes. All experiment arms share these fixes. Project config/preflight rejects malformed/null active configs and fallback runtime capabilities.

Observer/progress preparation: a later handler.wait bridge must signal existing parents before futures_wait; reactor-owned immutable snapshots must distinguish failed Future from completed drain. No second I/O queue or ownership system. Do not activate quotas before mandatory progress passes.

Research policy remains gated by P1–P3. Reserve off/shadow/fixed/pressure/interference/dependency_only/joint config names, but refuse live activation without prerequisites. No speculative calibration values.

Unmodified: Prefix identity, LoadPlanner, allocator, attention, model/decode executor, liburing executor, D2H→SSD continuation, parent success protocol and swap_blocks_batch fusion. Common / observer / policy patches are separate. No Source/repair/Prefix-shadow code is imported.

## Continuation 01 — isolated CPU preparation

observer/0001-mandatory-progress-bridge.patch adds optional progress_run_id and native-queue Future markers in py-kvcache-progress. It changes handler.wait, constructor setup, intake and signal-reference cleanup only. Original common worktree stays separate; normal construction defaults to disabled. No new ordinary quota or policy is active. See CONTINUATION_01_REPORT.md and native-path-preservation.json for exact evidence.

## Continuation 02 — optional bounded observation

observer/0002-bounded-snapshot-hook.patch adds optional observation_sink to IoReactor and TransferCoordinator, then invokes it after the native pump. Default is None; ordinary dispatch/completion/fusion methods remain unchanged. Project publication.py supplies a rate-limited latest immutable snapshot with nonblocking publication. dependencies.py now requires native owner confirmation before counting immediate reusable bytes. No live GPU owner adapter or ordinary quota was enabled. See CONTINUATION_02_REPORT.md.

## New server 01 publisher review fix

Project-only correction moves owner/thread validation into the existing publisher invalidation guard. Three prior-failing tests now pass. The patch and originals are preserved in new-server-01/review-fix and review-fix-before. Author native sources and GPU lifecycle remain unchanged.

## New server 02 — source build and common UUID compatibility

Author vLLM is built from 817a7e3124f817cd6e549581d3e5483207a753a4 in third_party/work/vllm-author-build. Original author sources remain clean. No C++ kernel, model executor, Prefix identity, admission, preload, fusion or reactor implementation was replaced.

common/vllm-author/0001-full-gpu-uuid.patch modifies only NvmlCudaPlatform.device_id_to_physical_device_id in vllm/platforms/cuda.py: a complete GPU UUID is resolved through NVML before physical-index use. Numeric, empty and unset CUDA_VISIBLE_DEVICES follow the inherited path; invalid UUIDs fail instead of selecting another device. This is a common compatibility fix for every experiment arm, independent of research policy. CPU regression: 16 passed; real specified-GPU metadata/import checks passed. Forward/reverse apply and pinned Ruff checks passed. Evidence: artifacts/prefix_io_v1/new-server-02/platform-fix.

Project run_gpu_stage.py now reserves and accounts wall time before/after each launch and drains the complete child session, including nested process groups. Twenty-one CPU tests cover budget refusal, interruptions and descendant cleanup; uncertain cleanup/reservations fail closed. This runner does not enforce model download bytes or network isolation. Offline HF/Transformers settings apply to launched GPU jobs. Evidence: new-server-02/runner-review.

Build/qualification helpers under experiments/prefix_io_v1/scripts are experiment infrastructure. The small author copy test proves its explicitly recorded byte buffers only. Observer patches remain in separate inactive worktrees; policy_patches is empty.

## New server 03 — project experiment infrastructure

Added pinned official model download and explicit recorded-failure Range continuation, model/header validation, approved-UUID capacity gate and native Prefix evidence. Download and GPU execution share the ledger lock; failed conservative charges and unresolved-reservation gates remain. No cache or I/O executor added.

native_gpu_prefix_smoke.py keeps the author LLM. Final project-only patch sequence in artifacts/prefix_io_v1/new-server-03:0001 provider/private-runtime provenance;0002 per-run cwd;0003 bounded existing worker RPC and explicit trusted-local serialization;0004 short project IPC/native chunked-prefill configuration;0005 native auto KV with model/actual-tensor BF16 assertions. Historical before-env files are evidence, not additional patches. Use final script or apply final patches once in order. Author attention,sampler,block pool,LoadPlanner,reactor and parent completion unchanged.

The successful launch uses project .venv/bin Ninja,private NVCC13,explicit private FlashInfer workspace and FLASHINFER_NO_DOWNLOAD=1. Exact environment:native-prefix-04-launch.json. Run02 default external FlashInfer cache is disclosed/preserved. Existing engine common patches remain identical for every future arm, observer isolated/inactive, policy_patches empty. Final hashes/CPU/GPU scope:NEW_SERVER_03_REPORT.md.

## 新服务器第四轮新增位置（2026-09-27）

- experiments/prefix_io_v1/scripts/prepare_native_calibration_plan.py：纯CPU实验准备工具，复用原作者任务生成函数；历史来源/版本/作业/UUID校验，以及仅供审查的v2结构检查。
- tests/prefix_io_v1_calibration/test_prepare_plan.py：32项CPU合成边界测试；没有GPU或成本测量。
- docs/prefix_io_v1/NEW_SERVER_04_REPORT.md、REPRODUCE_NEW_SERVER_04.md 与 artifacts/prefix_io_v1/new-server-04/：命令、候选计划、审计和阻塞证据。

本轮无缓存引擎/模型执行器补丁，无观察桥接启用，无研究策略变更；policy保持off。


## 2026-09-29 AutoDL AIO CPU 补充
独立可选工作树 py-kvcache-aio-cpu 的 Linux AIO CPU 功能验证通过：310 passed、16 skipped。默认 io_uring 与活动运行目录不变；AIO 的 GPU 集成与性能结论未验证，小块适配开销仍明显。完整边界及补丁见 AUTODL_AIO_CPU_REPORT_20260929.md。


## 2026-09-29 AIO CPU 混合续验增量

可选工作区 py_kvcache/linux_aio.py:LinuxAioRing.submit_pending 仅在 pending 非空时唤醒；close 保留无条件唤醒。新增 aio_cpu_mixed_screen.py 与 test_mixed_lifecycle.py。完整兼容补丁已更新，本轮差异另存 artifacts/prefix_io_v1/autodl-aio-mixed-cpu-20260929/empty-submit-fix.patch。原活动工作区与共同修复未变，不含研究策略。


## 新服务器 07 验证驱动

新增 qualify_aio_gpu_reactor.py 和 qualify_native_aio_kv.py，复用原 handler、reactor、worker RPC 和作者模型执行器。只添加诊断编排，没有更改缓存运行时。permissions.yaml 根据用户新服务器指令替换 GPU UUID，预算、禁止项及累计账本保留。验收范围见 NEW_SERVER_07_GPU_REPORT.md。


## 2026-09-29 server07 原生标定与规划器验证

详见 SERVER07_CALIBRATION_REPORT.md 和 REPRODUCE_SERVER07_CALIBRATION.md。本轮完成有界 P1 AIO 原生路径、实测成本、原 LoadPlanner 调用与自然 GPU 复用。短前缀 SSD 成本不利；未改变原准入或缓存执行器，未启用研究策略。P2 mandatory/release witness 和观察开销仍待验证。

## Server07 P2 已落地位置（2026-09-29）

- 隔离 py-kvcache-p2-aio 继承原 AIO 公共修复。P1 基线 py-kvcache-aio-cpu 未混入观察策略。
- observer/0001-mandatory-progress-bridge.patch：原 handler.wait 与 reactor incoming/STOP/finish；只标记已接受 Future，不替代原队列或提前完成。
- observer/0002-bounded-snapshot-hook.patch：原 pump 后可选钩子；观察失败退出观察，不改变原作业。
- observer/0003-native-options-adapter.patch：PyKvCacheOffloadingSpec 初始化/get_handlers，传可选 off/shadow 配置。
- 项目层 native_options.py、native_owner.py；后者仅显式 GPU 诊断装卸，未接入普通模型请求默认路径。
- 新增驱动 qualify_p2_gpu_progress.py、qualify_p2_native_observer.py、analyze_p2_qualification.py；tests 中新增 options/owner 测试。
- 本轮未编辑作者 vLLM 执行器。完整 tracked diff、适配器 reverse-check、源 SHA 位于 artifacts/prefix_io_v1/server07-p2-01/。

## Server07 P3 实验增量（2026-09-29）

新增token_timeline.py、prepare_p3_pilot.py、run_p3_native_pilot.py、analyze_p3_pilot.py与对应CPU测试/冻结manifest。扩展acquire_native_aio_costs.py的显式采集domain与GPU-only冷/热诊断，原默认参数保留。实验脚本diff在server07-p3-01/acquire-costs-change.patch；未修改原成本导出门禁或原运行时表。P2锁定29文件保持不变，无新增执行器/模型补丁。


## 2026-09-29 P3 复测增量

详见 [SERVER07_P3_RECHECK_REPORT.md](SERVER07_P3_RECHECK_REPORT.md)。新增仅限采集诊断、离线校验及 CPU 测试；29 个 P2 锁定文件未变。4K 原生冷/热候选概率排序/并列值变化已观察；16K 三组复测均观察到原有 SSD 恢复获益，但存在明显延迟波动。未导出长域运行时曲线，P3 尚未完成。


## P3 统一预算与长请求集成增量

详见 [SERVER07_P3_UNIFORM_REPORT.md](SERVER07_P3_UNIFORM_REPORT.md)。24 个缓存参考比较零容差通过；生成单独的标定候选表并完成两次原始准入下的长请求集成诊断，每次 6 请求/768 token。实际 CUDA 复制与计算有约 1.016 ms 重叠，pending flush 等待仍为零。候选仅用于标定集成，独立数据与强简单基线未完成；P3 继续，joint 关闭。

## P3 辅助存储与独立验证增量（2026-09-29）

config.py 增加向后兼容的可选辅助存储权限对象；新增 experiment_storage/probe_auxiliary_storage/copy_calibration_to_aux 及边界测试。采集与离线分析器支持显式 run_details/storage_path。新增 prepare_long_heldout_validation、long_validation_contract、run_long_native_reference、qualify_long_validation、analyze_long_validation；run_long_calibration_replay 增加 opt-in 资格检查，默认标定分支保持。新增内容为实验设施，没有缓存引擎/模型/reactor 新补丁。29 个 P2 锁定文件仅 config.py 改变；旧候选和失败结果哈希不变。精确差异、新增源文件及 SHA 见 server07-p3-05/existing-files.patch、source-lock.json。


## Server07 P3 两并发与 I/O 深度验证（2026-09-29）

本轮仅扩展 acquire_native_aio_costs.py、validate_heldout_costs.py 的显式并发/I/O 深度实验分支，并新增受限 contract、manifest 准备、资格检查、重放、原生参考、离线分析和 2 个 CPU 测试文件。旧默认路径和既有资格结果已回归。既有 90 个源锁条目仅 2 个实验脚本变化；无新缓存引擎、模型执行器、reactor 或生命周期补丁。 详见 [本轮报告](SERVER07_P3_CONCURRENT_REPORT.md)。


## P310 incremental delivery boundary

GPU runtime sources are unchanged from their P309 locks. Added offline analyze_blocked_store_chains.py, analyze_mixed_start_budget.py and verify_private_cache_archive.py under experiments/prefix_io_v1/scripts, with two diagnostic test files. Changed only consolidate_private_cache_copies.py and its path-scope tests to support explicitly authorized auxiliary audit receipts. Each cache mutation still requires a matching manifest SHA and explicit grant; permissions.yaml is unchanged. Two approved immutable archive consolidations are experimental maintenance, separate from common compatibility fixes, passive observation and the isolated P3 start-budget code. See SERVER07_P3_MIXED_RECURRENCE_REPORT.md.


## P311 增量记录

P311: 新增 dispatch_budget.py 为未接入 native 的 CPU 合同；store_readiness.py 与实验 worker/driver/analyzer 为可关闭观测。公共修复、author reactor/vLLM 未改。CPU 合同与 observer patch 分开，见 SERVER07_P3_DISPATCH_READINESS_REPORT.md。
