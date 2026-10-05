# P0 capability matrix
Source paths are relative to third_party/upstream. Exact symbol lines and SHA256: artifacts/prefix_io_v1/p0/source-symbols.json.

| Capability | Status / evidence |
|---|---|
| Exact Prefix identity / GPU allocator | Existing vllm-author/vllm/v1/core/block_pool.py; unchanged |
| External mapping / dedup | Existing py-kvcache file_mapper.py, SharedStorageOffloadingManager; CPU tests |
| Cost admission | Existing LoadPlanner; selected author on_preload_candidates calls manager.plan_candidates; original preload candidate lacks API |
| Shared staging / preload / clean eviction | Existing reactor _shared_decref, _maybe_release_shared, _reserve_foreground_slot; CPU/mock evidence |
| Fusion / async pipeline | Existing _pump_once, _flush_copy_batch; real overlap unmeasured |
| Parent success | _file_terminal waits total_files; failure may resolve before physical drain |
| Independent source-safe ACK | Not exposed; do not release at single D2H completion |
| Mandatory signal to quota controller | Needs bridge: handle_preemptions → worker.wait → handler.wait → futures_wait |
| Actual staging budget | Needs common guard: iodepth + max(4, iodepth//2) slot floor and 4095 backing alignment bytes |
| Single KV group | Explicit transfer.py canonical-layout and first_group_block_index checks |
| Runtime Plan/Torch/vLLM handler | Not qualified; CPU fallback cannot pass runtime acceptance |
| Profiler | py-kvcache can noop; vLLM directly requires simple_profiler |
| io_uring | BLOCKED: setup EPERM; no policy/system changes |
| O_DIRECT | Separate bounded CPU probe in capability-report.json; not proof of io_uring |
| Durable file publication | fsync only when sync_on_store=true; cache-visible is separate |
| Disk GC/eviction | Absent upstream, out of scope |
| Compact observer / release witness | New project preparation only; live GPU generation evidence missing |
| Finite policy / interference | P3/P4 gated; no calibration or pilot evidence |
| Layerwise consumption | Unverified; no new model executor |

## New server 01 update

Migration hashes and source locks match. RTX 5090 device 6 is exposed; compute availability/free VRAM are unprobed. io_uring still fails EPERM, while actual 4 KiB O_DIRECT passes. CPU regression after the publisher invalidation fix: 388 passed, 8 skipped. GPU/Plan/handler runtime qualification remains false; newly installed Torch 2.11.0+cu130 dependencies are not a built author vLLM runtime.

## New server 02 — current qualification overrides earlier runtime status

| Capability | Current evidence and limit |
|---|---|
| Built fixed author vLLM | Source build04 exit 0; _C SHA256 e30616d12f903169493f73c28e707540be17916e89794a9214b141d6e4769b94 |
| Plan, handler, profiler API identities | All 22 import/symbol capability checks true; no handler/ring construction or full call-chain execution |
| Author swap_blocks_batch | CPU ABI and real GPU exact guarded byte-copy passed: 334 B H2D and 1358 B D2H; not production KV |
| Full approved GPU UUID | Common UUID patch; platform-02 reports expected RTX 5090, capability 12.0, 34190917632 memory bytes |
| GPU base arithmetic | Passed under single UUID, recorded in budget ledger |
| io_uring / real SSD | Author LiburingRing(2) still EPERM; real restore/store blocked |
| Mandatory progress / release witness | CPU/mock preparation only; no live owner adapter or GPU acceptance |
| Research policy | Off; P1 incomplete, P2–P7 acceptance closed |

Latest fail-closed preflight has two blockers: io_uring unavailable and real handler unverified. No CPU fallback is promoted to these GPU qualifications; source origins and registered native dispatch were checked explicitly. See new-server-02/capability-report.json and runs/author-import-02/qualification.json.

## New server 03 — current native GPU-only qualification

| Capability | Evidence and boundary |
|---|---|
| Fixed local model | Official ModelScope Qwen revision/manifest;11 files hash-verified,339 BF16 tensors; no asserted HF commit equivalence |
| Real native model / GPU Prefix | native-prefix-04 exit0;cached0→112,equal16-token outputs for identical128-token input;same engine,no reset |
| Actual KV storage |28 unique CUDA BF16 storages,66,977,792B,73 blocks;64MiB config;not allocator-reserved or release witness |
| Local native JIT | Installed FlashInfer sampling source/private NVCC;sampler unchanged;no-download flag |
| Real handler / staging / SSD | Unverified: io_uring setup EPERM;complete P1 blocked |
| Native cost admission prerequisites | Local Qwen7B BF16/RTX5090 v2 curve absent;other-model curves cannot be substituted |
| Release / policy | Live release adapter absent;observer inactive;policy off;P2–P7 closed |

Current artifacts/prefix_io_v1/new-server-03/capability-report.json and preflight.json (exit2) supersede earlier model-unavailable status without changing historical results.

## 新服务器第四轮 CPU 复核（2026-09-27）

原生 io_uring 本轮探针仍 EPERM；preflight BLOCKED。新增纯CPU候选标定准备器，固定源码SHA后复用作者 Job/build_job_plan，历史身份与清理状态交叉验证。32项CPU测试通过；计划未执行，曲线为空，不升级P1或GPU能力。原作者导出器为v1，当前planner需要v2；详情见 NEW_SERVER_04_REPORT.md 及 artifacts/prefix_io_v1/new-server-04/calibration-format-audit.md。

## 第五轮环境可行性裁定（2026-09-27）

NO_GO_CURRENT_CONTAINER：标准C与固定作者LiburingRing均在setup阶段EPERM；当前可访问SSH环境没有发现平台支持的容器管理入口。已记录用户有限io_uring授权，未进行系统变更。该结论仅限当前容器配置/管理权限，不否定架构在其他环境的可行性。P1仍未验收，后续阶段关闭。本轮GPU/下载为0，预算账本未变；详见NEW_SERVER_05_FEASIBILITY_REPORT.md。


## 2026-09-29 AutoDL AIO CPU 补充
独立可选工作树 py-kvcache-aio-cpu 的 Linux AIO CPU 功能验证通过：310 passed、16 skipped。默认 io_uring 与活动运行目录不变；AIO 的 GPU 集成与性能结论未验证，小块适配开销仍明显。完整边界及补丁见 AUTODL_AIO_CPU_REPORT_20260929.md。


## 2026-09-29 AIO CPU 混合负载续验

候选 896KiB/3.5MiB/14MiB 文件真实 CPU 生命周期通过；元数据线程阻塞时数据读写可推进；空提交不再唤醒后台线程，关闭排空仍通过。最终 320 passed、16 skipped。36 组混合测量仅初筛，不宣称稳定收益。真实 GPU staging/KV/mandatory drain 仍未验证；P1 不标完成。证据：AUTODL_AIO_MIXED_CPU_REPORT_20260929.md。


## 2026-09-29 新服务器 07 GPU 验证

新指定 RTX5090 上作者真实拷贝、原 handler/reactor + AIO SSD、shared preload、pinned staging、真实模型 KV 字节往返及正常 wait/shutdown 通过。最终证据为 runs/server07-native-aio-kv-02，成功后不重写 CPU 备份，Prefix 0/112 与输出 token 一致。当前模型 scheduler 未接入外部 connector，AIO 成本曲线与 GPU 源块释放见证尚待验证；P1 仍 partial，不进入策略验收。完整范围、预算、命令见 NEW_SERVER_07_GPU_REPORT.md。


## 2026-09-29 server07 原生标定与规划器验证

详见 SERVER07_CALIBRATION_REPORT.md 和 REPRODUCE_SERVER07_CALIBRATION.md。本轮完成有界 P1 AIO 原生路径、实测成本、原 LoadPlanner 调用与自然 GPU 复用。短前缀 SSD 成本不利；未改变原准入或缓存执行器，未启用研究策略。P2 mandatory/release witness 和观察开销仍待验证。

## Server07 P2 实际能力更新（2026-09-29）

| 能力 | 状态 | 证据/限制 |
|---|---|---|
| 原生 off/shadow 配置适配 | 已新增并真机验证 | native_options.py + observer/0003；不是作者原有 API |
| mandatory wait/shutdown | CUDA/AIO 机制验证通过 | progress-05；test-only 零额度，尚无生产普通额度 |
| 有界 reactor 快照 | 真机验证通过 | shadow-01/02；GPU 释放字段未知 |
| 真实 owner 释放 witness | 有界同线程诊断通过 | owner-02：13 次安全重分配；不是生产 controller 接线 |
| off/shadow 对应输出 | 5 臂各 32 请求一致 | 冷/热族 3 差异在 off 中也复现，原因未归因 |
| 观测开销 | CPU/短 E2E 已测 | 合并调用耗时 +1.156%；2% goodput 未证明 |
| 固定/压力/联合策略 | 未接入生产执行器 | P3/P4 门禁仍适用 |

## Server07 P3 开发筛查更新（2026-09-29）

- 原生连续输出事件：84条请求、10752个逐token时刻；完整cohort/drain、发送迟到已记录，尚无SLO goodput。
- 短流资源压力：原生强制flush等待观测0；已安装槽位计数的三轮申请失败0；free_slots=0不能等同依赖阻塞。
- 2K/4K/8K/16K成本路径已实际运行；16K出现原缓存SSD复用机会，但仅2个测量样本/点，未发布新运行时表。
- 4K冷/热差异已在无外部connector的原生GPU路径复现；数值机制未确认，严格导出检查未放宽。
- fixed/pressure/joint尚未接入；P3未完成。证据见PILOT_PROBLEM_REPORT.md与server07-p3-01/qualification-summary.json。


## 2026-09-29 P3 复测增量

详见 [SERVER07_P3_RECHECK_REPORT.md](SERVER07_P3_RECHECK_REPORT.md)。新增仅限采集诊断、离线校验及 CPU 测试；29 个 P2 锁定文件未变。4K 原生冷/热候选概率排序/并列值变化已观察；16K 三组复测均观察到原有 SSD 恢复获益，但存在明显延迟波动。未导出长域运行时曲线，P3 尚未完成。


## P3 统一预算与长请求集成增量

详见 [SERVER07_P3_UNIFORM_REPORT.md](SERVER07_P3_UNIFORM_REPORT.md)。24 个缓存参考比较零容差通过；生成单独的标定候选表并完成两次原始准入下的长请求集成诊断，每次 6 请求/768 token。实际 CUDA 复制与计算有约 1.016 ms 重叠，pending flush 等待仍为零。候选仅用于标定集成，独立数据与强简单基线未完成；P3 继续，joint 关闭。

## P3 辅助存储与独立长前缀验证（2026-09-29）

授权辅助根通过真实 O_DIRECT AIO 读回与额度检查。新存储 24 项标定缓存参考、独立长内容 6 项缓存参考、12 个完整 128-token 服务输出均通过严格比较。16256-token 独立成本预测在原定 25% 内通过；不覆盖旧 2K 失败、不授予全域资格。普通策略仍未接线，max_num_seqs=1，P3 未完成。详见 [本轮报告](SERVER07_P3_AUXILIARY_REPORT.md)。
