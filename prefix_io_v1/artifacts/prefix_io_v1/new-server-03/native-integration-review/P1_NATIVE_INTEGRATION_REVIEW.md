# P1 原作者 staging / SSD 集成入口审计

状态：**BLOCKED，未执行 GPU / io_uring / 模型加载；未激活 P2**。本次读取源码、计算资源公式并保存审计，未修改缓存、I/O 或模型执行器。

版本：kvcache-experiments 0e023a84a21246b9bbc06266fa8070397eccbdc9；py-kvcache 3abba7a502d553f6e7e2e58b92086487e3395d7e（现有 common 工作树）；作者 vLLM 817a7e3124f817cd6e549581d3e5483207a753a4。下列源码路径相对项目根。

## 可复用入口及限制

| 入口 | 实际内容 | 本阶段适用性 |
|---|---|---|
| third_party/work/py-kvcache/tests/test_e2e_kvcache.py:85 | 真 vLLM API server + OffloadingConnector + PyKvCacheOffloadingSpec，cold store 后同 prompt 请求；检查文件及 completion | 最接近真实模型 P1 的参考。默认 Llama、约80k输入、92k上下文、32GiB staging、GPU Prefix关闭；sleep(3)不证明排空，文件存在和输出相同不证明SSD读取。不能原样执行。 |
| third_party/upstream/kvcache-experiments/bench.py + common/vllm_server.py | 真 server 生命周期、构造CLI、运行请求、保存profile | 可复用命令构造和请求驱动。bench.py:873自动wipe存储；start_server:83–102另建session并派生tee。不可直接放入当前budget runner后宣称其session清理能涵盖服务器。 |
| prefix_cache_benchmark.py + common/prefix_cache_common.py | 原作者真实HTTP请求驱动、X-Request-Id关联trace | 可复用send_request/请求标识协议。当前build_prompt按词而非精确token构造；reuse还插入后缀；send_request不保留生成文本/token IDs或最终usage。因此需极小P1结果采集适配，不能直接拿TTFT CSV作路径/输出一致性证据。 |
| scripts/kv_connector_scenarios.py + common/kv_connector_harness.py | 真scheduler/connector，但无模型forward；手造ModelRunnerOutput，sampled_token_ids=[0]（:739–764），dummy ForwardContext | 仅connector机制工具。MODEL_DTYPE=float16，默认131072长度，并有skip_gpu_copy/metadata_only分支。不能称Qwen BF16真实模型/生产KV资格，也不以它替代执行器。 |
| test_staging_cache.py、test_py_kvcache_shared_preload.py、test_py_kvcache_preload_handler.py、test_py_kvcache_copy_batch.py | policy、refcount、handler future、copy batching CPU/fake测试 | 生命周期参考及已有CPU回归。不能冒充真实staging/SSD/GPU资格。 |

PyKvCacheOffloadingSpec是真实入口；SharedStorageOffloadingSpec/PyKvcacheOffloadingSpec是同类别名（py_kvcache/vllm.py:621–780）。不能因NoopSharedStorageOffloadingHandler名字含Noop就将它当stub：当前真实handler会把transfer交给TransferCoordinator。

## 最小建议，尚未实现或执行

复用已准备的native_gpu_prefix_smoke的固定本地模型验证、同一作者LLM、私有run目录、离线和预算条件，只在独立P1驱动添加原生KVTransferConfig：

- kv_connector=OffloadingConnector，kv_role=kv_both。
- spec_name=PyKvCacheOffloadingSpec，spec_module_path=py_kvcache.vllm。
- 保留原LoadPlanner唯一准入：load_planner=on，真实v2 prefix_cache_break_even_path；enable_preload=true、preload_share_staging=true、preload_lookahead_requests>0。
- staging_cache使用原有lru；保留copy fusion、异步store、原parent/fence回调。不得注入“必定准入”曲线或覆盖planner决策。
- 独立cold store阶段保留新私有storage目录；在明确的一次性GPU-only置冷诊断中复用原LLM.reset_prefix_cache(reset_running_requests=False, reset_connector=False)，校验返回值并记录置冷操作；不在正式请求流逐请求reset。
- 一次性置冷后的同token请求须由原生成功transfer/source-cache + H2D完成 + staging_cache.hit证明CPU staging路径；若原planner不准入或trace不满足就报告失败/未覆盖。
- 保留disk目录，结束首引擎并由原native shutdown完成收尾；另起同配置引擎做SSD restore，GPU与CPU staging自然为空。必须由真实read完成字节和成功H2D/父任务证明路径，而非仅cached_tokens或速度。
- 保存每次请求的真实完整token输入、生成token IDs、输出一致性及所有失败。独立BF16重算有形状数值差异时保留失败；不能写成生产KV逐字节相同。

此方案只增加实验编排/取证。尚缺“原生请求结束之后store完成及profile完整性”的可验收条件绑定；不能以固定sleep替代证据。任何待执行驱动都必须通过原生完成/flush路径，不从外层提前complete_store或自行释放资源。

当前无需为P1引入新的CQE队列或观测桥接。优先解析已有完整原生trace；只有现成证据确实不足才另行说明极小取证缺口，不借机启P2。

## 真实读取字节如何取证

py_kvcache/liburing_file.py:277返回真实CQE(user_data,res)；reactor.py:882以op_kind路由read/write，避免把open返回fd或close返回0当字节。

1. Foreground：reactor.py:955–970先检查res>=0且res==file_store.io_size，才追加file_samples；transfer.py:325–341发py_kvcache.file_read。这里num_bytes是逻辑storage_block_bytes，不是原始res。只有完整且成功的父任务，其成功file_read事件数×已冻结实际io_size，才能严格推出这一成功子集的实际CQE完成读取字节。
2. Preload/shared preload：reactor.py:978–1008、:1056–1084的py_kvcache.preload.file_read记录success与nbytes=result_nbytes。共享读取只计一次；不能按等待者数乘字节。place_from_read与transfer src_preload可关联具体load。
3. Staging：reactor.py:1298–1315的staging_cache.hit与成功transfer src_cache/H2D组成命中证据；相应请求的SSD读计数不得混入别的请求或预加载。
4. py_kvcache.transfer的num_bytes是逻辑transfer_size；connector_load_bytes或cached_tokens也不是SSD读取量。所有逻辑KV字节与实际I/O完成字节分别保存。
5. 失败父任务可能在其他child drain前emit，其trace不保证包含所有失败/短读实际流量；上述推导不能扩展成全失败流量统计。
6. profiler必须真实active且完整flush；fallback/noop trace不能通过。原EngineCore启动profiler使用相对cwd的results_engine_core_0.json、merge.json及registry（vllm/v1/engine/core.py:1166）。每阶段使用新的私有工作目录，保留原始trace和解析摘要，不复用陈旧registry。

当前native LiburingRing(2)返回EPERM（new-server-03/io-uring-current.json）。因此不存在本轮真实staging/SSD通过证据，也不启动GPU来重复撞此已知阻塞。

## 生命周期约束

- H2D：reactor.py:802–838只有CUDA end_event完成后才settle_load_slot/file_terminal。cache slot DMA期间pin；shared slot在cached或copies_inflight>0时保留（:495–498、:1280–1295）。
- D2H：:842–870只有copy end event完成后才提交磁盘write；staging保留到write CQE精确完整完成、finish_write及原release/cache逻辑（:1150–1170）。
- **源码与旧注释不一致**：fs_config.py:80说store buffers不缓存，但实际_on_write_complete:1167–1169写成功后调用_release_or_cache，原lru开启时可由cold store暖CPU staging。以当前真实代码为准，不据旧注释设计假前提。
- 父任务成功future需全部done_files==total（:2016–2033）；失败future可能早于其他inflight结束，future.done(error)不能证明物理释放；_finish_jobs等inflight_files==0（:2035–2044）。
- vLLM offloading/scheduler.py:318–335为每GPU块记录全部pending父job的集合；restore/compute复用触发jobs_to_flush（:874–882、:933–945）；全部worker完成后才complete_store并移除对应job fence（:1133–1179）。
- request_finished:1190–1215允许请求所有权释放后store继续，full-attention块受pending-job fence保护。block_pool.py:419–432的refcount0/free queue不等于可立即覆盖。不能把保护叙述成“块始终不进free pool”，也不能把D2H完成说成允许移除全部父任务fence。
- worker.py:392–397先提交store再wait jobs_to_flush；:468–487对transfer_result.success有assert，当前失败路径不是完整可恢复协议。
- sync_on_store=false的完成是文件发布/可见，不是掉电持久化；liburing_file.py:391–418只有开启sync_on_store才fsync文件/目录。该参数必须冻结，不能混用两种承诺。

## 必须冻结的资源与准入参数

固定模型provider/revision/manifest/local path、author/common patch hashes、weights/KV BF16、single GPU UUID、TP/PP/DP=1；真实单KV group、attention backend/kernel block/storage block factor、logical storage bytes与aligned io_size；实际GPU KV分配及容量、max_model_len、每个prompt/output长度、并发；staging nominal预算、slot count、backing storage nbytes、pin状态及额外allocator/ring开销；iodepth、open_lookahead、copy headroom、preload lookahead、shared-staging与cache policy；sync_on_store、root目录、initial GPU/CPU/disk状态、原LoadPlanner曲线及阈值/容忍度/deadline；store/restore/flush/失败处理及末尾drain；profiler开关、工作目录、请求ID、outer budget含启动/收尾。

理论小预算候选（尚未冻结为实际运行配置）：Qwen28层×4KV头×128维、BF16，KV/token=57,344 B；storage block=64 token时payload=io_size=3,670,016 B（4096对齐）。若staging=128 MiB、iodepth=2，原修正公式min_slots=2+max(4,1)=6，slot_count=floor((134,217,728-4095)/3,670,016)=36，backing=132,124,671 B，max_preload_slots=34。这只是源码/模型结构公式，非实际分配；请求长度还必须满足真实LoadPlanner准入、GPU池和上下文，不能为命中而放宽成本门槛。

尚无本机Qwen2.5-7B BF16对应v2曲线。原_build_planner要求curves、kv_bytes_per_token、预加载与共享staging；库存break-even只有其他模型/硬件，不能套用。使用本地模型路径时原break_even校验model_name与model_config.model精确相等，冻结曲线还须绑定本地路径+provider/revision，不能只改旧曲线模型名冒充校准。

## 待执行入口命令与阻塞

以下为已核实的原作者入口语法参考，**BLOCKED，不是当前可执行批准命令**：

```bash
# 原E2E入口：其默认尺寸、禁GPU Prefix、sleep和清理均尚未适配，禁止原样执行。
RUN_E2E_TESTS=1 E2E_MODEL="$VERIFIED_LOCAL_MODEL_DIR" \
  .venv/bin/python -m pytest third_party/work/py-kvcache/tests/test_e2e_kvcache.py::KvCacheE2eTest -v -s

# 真server入口，必须由经审查的同session、项目内、带预算的P1编排启动；
# FROZEN_NATIVE_KV_CONFIG_JSON 尚未形成可运行配置。
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$VERIFIED_LOCAL_MODEL_DIR" --host 127.0.0.1 --port "$FROZEN_PORT" \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --enable-prefix-caching \
  --kv-transfer-config "$FROZEN_NATIVE_KV_CONFIG_JSON"

# 原作者HTTP请求入口，不单独提供路径/完成/输出一致性资格：
cd third_party/upstream/kvcache-experiments
"$PROJECT_ROOT/.venv/bin/python" prefix_cache_benchmark.py \
  --host 127.0.0.1 --port "$FROZEN_PORT" --num-requests 3 \
  --doc-size "$FROZEN_PROMPT_LENGTH" --prefix-reuse-pct 1 --prefix-size 1 \
  --max-concurrency 1 --output-len 16 --pre-warmup-requests 0 --completions
```

缺项：io_uring在现容器获准并真实可用；实际模型/Prefix基本资格；本机合法v2成本曲线；真实layout/预算冻结；最小driver的token/trace/lifecycle证据采集；全程预算与session覆盖。不会以fake ring、同步替代I/O、强制planner准入、隐藏reset或dummy model completion解除这些阻塞。

## 当前 native LLM 的 session 覆盖（定向源码结论）

当前smoke固定distributed_executor_backend=uni、单卡，未发现二次setsid/start_new_session。EngineCoreClient.make_client（core_client.py:82–104）在multiprocessing模式选SyncMPClient；utils.py:104–195的CoreEngineProcManager用get_mp_context().Process(...).start()，没有创建新session。system_utils.py:168–183返回标准multiprocessing fork/spawn上下文；实际CPython3.12.3的fork使用os.fork，spawn走multiprocessing.util.spawnv_passfds，其_posixsubprocess.fork_exec参数start_new_session=False。UniProcExecutor（uniproc_executor.py:51–72）直接在EngineCore内初始化driver_worker。NUMA可选wrapper最终exec numactl/python，不调用setsid。

全vLLM源码中检出的start_new_session仅benchmarks/sweep/server.py:50；entrypoints/openai/dp_supervisor.py:232只有setpgrp，且非当前离线uni路径。正常EngineCore/worker因此仍在budget runner的session内，现有session枚举清理涵盖它们和合法nested PGID。该结论仅限当前审核路径，不扩张为任意第三方插件/未来外部launcher都不会escape；SIGKILL/宿主故障等既有reservation限制不变。

与之相反，kvcache-experiments/common/vllm_server.py:83–102明确start_new_session=True，原bench/server wrapper不得未经适配直接作为runner子命令。可只复用命令构造和请求函数，或者选现有native LLM入口；无需改变缓存或模型执行器。
