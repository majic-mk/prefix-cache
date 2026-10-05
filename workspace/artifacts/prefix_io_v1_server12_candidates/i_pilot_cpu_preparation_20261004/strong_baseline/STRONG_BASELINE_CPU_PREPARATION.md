# 原 strong U / I 请求入口的 CPU 准备

已新增实际可执行配置桥、作者参数桥和原函数 CPU 闭环验收，没有修改底座函数或替换缓存/模型执行器。

`strong_baseline_config.build_runtime_pair(engine_common, sampling_common, connector_common, storage_paths, run_ids, i_policy, max_accepted_parents)` 生成两侧原 `OffloadingConnector → PyKvCacheOffloadingSpec` 配置。U 为 P4 off，I 为 interference；除 P4 策略、隔离 storage namespace 和 run identity 外，原 engine、sampling、parent cap、planner、preload、shared staging、I/O depth 和所有原参数完全一致。验证器拒绝关闭 Prefix Cache、chunked prefill、原 LoadPlanner、预加载、共享 staging、fusion/pipeline，拒绝两侧配置漂移及复用同一 namespace。

原 parser 的实际接口是 `SharedFileConfig.from_extra_config`，并无本任务需要的 `Config.from_mapping`。CPU 验收执行完整原 parser 模块，原 `PyKvCacheOffloadingSpec._build_planner` 完整 AST、原 `CostTables` / `Pchip` / `LoadPlanner` 算法及原 staging 容量公式；函数/类 AST 未改写。需要绕开 Linux/CUDA 初始化的 import 只在明确的 CPU 值 namespace 中绑定原纯函数/常量，不能据此声明安装包或 GPU 能力存在。CPU 曲线、PLAN_API_AVAILABLE 标记和微小 step budget 都是注明的分支 fixture，未存为实际 GPU 配置或成本表。

作者输入来自当前服务器 SHA manifest：`../source_inputs/STRONG_BASELINE_SOURCE_INPUTS.json`、`AUTHOR_BENCHMARK_SOURCE_INPUTS.json`、`STRONG_COMMON_ENGINE_SOURCE_INPUT.json`。作者 kvcache-experiments commit 为 `0e023a84a21246b9bbc06266fa8070397eccbdc9`。复用了原 `prefix_cache_common`、原 `shared_storage_trace_replay` 的 parser、trace 解析和 global schedule；不执行其多实例 cluster launcher，不导入 OpenAI client，不下载 dataset。`author_trace_arguments` 显式保留至少 128 输出 tokens，禁止前缀注入及 wipe-shared-storage。已有 P3 runner 与 prepare 的原 strong 配置、native common engine 常量也来自实际服务器文件，未将旧本机副本当当前来源。

原 config parser 支持 planner=on；原 builder 检查 v2 curves、planned-defer API、preload、shared staging、lookahead、KV bytes/token 和 model length。验收确认两侧原 planner 生成相同表和容量，mid-write 仍 DEFER，缺 v2 curves/接口正确拒绝；原 P4 parser 将 U 解析为 off/no active config，将 I 解析为 interference/no fixed dispatch override。

## 执行和证据

本机最终命令：

```text
Python 3.12 -B -I -S verify_strong_cpu.py --output LOCAL_STRONG_CPU_RESULT.json
```

实际 exit=0：15/15 CPU tests，0 skip，53 个真实 source inputs before/after SHA 相同，GPU runs=0。完整命令、stdout/stderr、原方法行号和 AST SHA 保存于 `LOCAL_STRONG_CPU_RESULT.json`。初版测试曾因未解析原 literal `128/1024` 造成 14 tests 中 4 errors/1 failure；修复仅限测试原参数读取器的 literal arithmetic，底座及配置准入保持不变。之后 14 项通过，再加入原 off/interference options parser 后最终 15 项通过。

可执行配置生成命令：

```text
Python 3.12 -B -I -S strong_baseline_config.py --input <explicit-common-config.json> --output <new-cpu-config.json>
```

输入包含 engine_common、sampling_common、connector_common、storage_paths、run_ids、i_policy、max_accepted_parents。输出使用 xb；它只生成 CPU 配置，`gpu_effect_qualified=false`、`native_execution_verified=false`、`cost_receipt=null`，不能凭此启动真实效果比较。

## 明确边界

旧 single-file calibration 的 planner=off 是机制诊断；其普通成本 receipt **不覆盖这份 planner=on 的 strong U/I 请求流域**。旧反例保留，不能拿旧 receipt 或重新放宽阈值使强 baseline 资格“自动通过”。真实效果还需独立开发 deadline、正常冻结 trace、strong 域标定/留出、on 生命周期和原累计预算 guard。

原 builder 在 GPU Prefix Cache 启用时会发出“recompute overpriced”警告，本轮真实 CPU 输出保留。它要求对开发域的成本准入重新验证；没有通过关闭 Prefix Cache 或改写 LoadPlanner 来消除警告。

当前已完成配置桥和原函数 CPU 验收，未声称完整 effect GPU runner 已合格，未产生有效 GPU receipt、真实 GPU token事件、normal workload 结果或性能提升。独立 deadline/正常 stream 缺失时保持机器 BLOCK。
