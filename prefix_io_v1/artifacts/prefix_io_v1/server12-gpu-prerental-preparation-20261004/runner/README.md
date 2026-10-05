本目录是原 py-kvcache/作者 vLLM 的薄驱动与紧凑观测，未另建缓存底座、模型 executor、资源 owner 或完成队列。本轮仅 CPU 准备；没有 GPU 实验、成本资格或提升结论。

实际修改：`strong_trace_runner.py`/`native_runtime.py` 消费冻结原 P3 12×512-token、原 arrival/concurrency 和完整 128 输出，调用原 LLMEngine.add_request/step，记录每个实际输出 token 和全部请求失败分母，保留原模型 shutdown、native drain、原外层 GPU guard。强 U 使用原 LoadPlanner on、原共享 staging/preload/fusion/pipeline，旧不同 GPU 的 curves-v2 仅结构输入，仍未校准当前强域成本。

`bounded_native_full_step_collector.py` 只将原128-frame完整 CUDA observer 参数化到4096总step、128 pending Events，原 prepare/live tuple/CUDA query/wait/detach AST 保留。query-only、没有新增 synchronize。所有 U/I 使用同源同限观测；观测失效保留模型的真实输出和原生命周期，成本/干扰资格转 UNKNOWN。原 NativeWindowJournal/原 SparseOwnerSink 保存 bounded 4096 native events，不把未观测ordinary-ready当作零机会。

`finite_startup.py` 仅在同包 privately issued exact CostTable 下，在原 coordinator 构造前提供原 typed bridge；原 owner passive callback 在真实排空后、第一请求前挂 `finite_current_binding.py`。普通 ReadyFd 只增一处原 If 的有限资格口，使用原 sequence/open_start/hash/preload_info，unknown/stale/changed cell 保留 native U，mandatory/max_wait 优先仍由原路径决定。初版只有限普通 issue/deferral，原 production batch gate 未启用。

`activation_request.py` 分开有资格 development shadow 与 effect。development 需要真实 finite raw 校准/heldout闭包及独立预声明 full deadline，同 controller 原计算实际执行，但每个普通提案返回 native fallback；effect 必须重放 `verify_existing_native_reserve` 的真实完整 guard/source/SDK/128 CUDA/output/host four-category 证据，并从独立 deadline 减去实测 reserve。CPU数学、公开布尔值、table=None、旧planner-off/context144 receipt均不能开启 effect。当前这些真实新域输入缺失，门禁保持关闭。

`host_boundary_binding.py` 仅观察原 Scheduler.schedule、InputBatch.refresh_metadata、OutputProcessor.process_outputs、公共前端 bookkeeping 和原 owner 构造的同 finite controller metadata 路径。用 actual `_profile_step` 关联原 ordinal；不观察整个GPU execute/sample/step，不用 wall-GPU 相减。ordinal异常记录未知，原方法仍调用恰好一次并保留原返回/异常。64 bounded intervals/step，溢出未知。没有真实 eligible preview 时不能声称已有 I reserve。

所有自然发生的有限 ordinary preview 都记录连续 attempt_id，与原 host controller interval 一一匹配。development 仅观察真实 CostTable 的全部已发证 cell，保持原 U 发出选择；effect 必须消费独立 read-only reserve replay 返回的 covered_cell_signatures，仅实际覆盖的子集可用减去 reserve 后的额度，其他 cell 仍回原 U。CPU 合法值/区间探针不授予 GPU 资格。

`strong_native_cost_runner.py` 复用原 `calibration_v2/run_native_cost_experiment.py`（SHA 22f49f361c94406e2929d6c6472909f1b21440ed420a2bf3ac8cebe2d97d19e9），原六个 fresh process、AB/BA/AB、2 calibration+1独立holdout、原SSD-only preload、原128-frame CUDA/raw/native journal/原 shutdown，薄参数仅强配置、512 prompt、原块规则 cached496、context527、父元数据。原 estimator 数学未修改，holdout不参与fit。限900+20秒、512MiB预留/8GiB floor、原8小时总额内；仅实际强 off成功且原 guard/session自然排空后可用。

服务器 CPU 命令（D 为该交付目录，ROOT 为项目绝对路径）：

```text
.venv/bin/python -B -I -S D/runner/verify_runner_cpu.py --output D/runner/SERVER_CPU_RESULT.json
.venv/bin/python -B -I -S D/runner/build_private_pair_cpu.py --output D/runner/PRIVATE_U_I_CONFIG.json --run-u server12-strong-u-qual-off01 --run-i server12-strong-i-effect-preview01
.venv/bin/python -B -I -S D/runner/strong_native_cost_runner.py --project ROOT --prepare-geometry --output D/runner/kv_geometry.json
.venv/bin/python -B -I -S D/runner/strong_native_cost_runner.py --project ROOT --prepare-plan --source-lock D/PRERENT_SOURCE_LOCK.json --pair D/runner/PRIVATE_U_I_CONFIG.json --input-manifest ORIGINAL_NATIVE_INPUT_MANIFEST --geometry D/runner/kv_geometry.json --output D/runner/STRONG_RAW_CPU_PLAN_TEMPLATE.json
```

CPU geometry仅由真实离线model/config.json推导，必须canonical compact无newline并先纳入 full source lock；不能称读过GPU KV layout。CPU raw模板gpu_uuid=null/cpu_preparation_only=true，不能执行或被issuer接受。root若已生成geometry/PRIVATE pair，不覆盖这些append-only文件。

后续代码接口已准备：`bind_live_plan(root, template_ref, source_lock_ref, source_proof_ref, off_qualification_ref, off_guard_ref, permissions_ref, gpu_uuid, output)` 检查 actual /dev/nvidia0、actual UUID、原 permission parser/原剩余8h及闭合off。`make_actual_configuration` 生成严格 raw config 后真实 full CPU preflight。`close_actual_raw(root,pending_ref,plan_ref,before_ref,after_ref,guard_ref,output)` 只加入独立完整 before/after sources 与原 completed guard，仍由真实issuer作最终capture/IO/数学/heldout资格验证。CPU模板、pending child和编译均不冒充GPU receipt。

本机最终验证：`LOCAL_CPU_RESULT_FINAL.json` 实际 47 项 CPU PASS（25+10+12），源码前后相同，真实 GPU 运行 0 次。服务器须执行同一 `verify_runner_cpu.py --output D/runner/SERVER_CPU_RESULT.json` 并用完整真实 source lock/model metadata 执行上面的 `--prepare-plan`；help 或 AST 回放不能代替服务器实际模板闭环。

目前自然请求数据缺失，资格流仍明确 controlled_original_P3_mechanism/fit_or_evaluation_input_allowed=false；独立deadline与新强域费用、真实development reserve、I lifecycle/effect均尚无GPU证据。下一允许GPU是 root 有界原guard off01资格300+20秒，之后以真实证据决定成本/开发shadow/on，不保证性能提升或论文结果。
