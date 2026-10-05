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
