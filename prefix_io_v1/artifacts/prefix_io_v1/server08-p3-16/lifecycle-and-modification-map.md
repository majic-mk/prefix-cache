# P316 生命周期与修改边界静态复核

当前共同实现对已接纳的 copy/SSD 所有权、父任务失败发布和 STOP 保留边界作了增量修复；P3 fixed/pressure 通过默认关闭的薄控制桥接入。本文只复核冻结小型源码和已有证据，不判定 P3 已完成，也不宣称性能提升。

源码身份基于 [execution-lock-08.json](/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-p3-16/execution-lock-08.json)，锁文件 SHA-256：`afd322a2795d045934e206334a3920fc38edf9dd2957bcaae87cabc9c81f00fd`。下表 9 个文件的实际 SHA 与该锁一致；此选择性核对不替代 root 的完整源快照。

## 实际生命周期

1. **接纳与父任务退休。** 共同 accepted-parent 计数覆盖 incoming、active、失败后排空中的任务；default None 不开启上界。同步 Condition 背压复用原 incoming 控制项，不创建第二作业队列、不使用 transfer_async=False 假延期。整父真实排空后、Future callback 之前退休；owner 满额重入只能拒绝尚未接纳候选。 父计数不是 GPU free-capacity 或 GPU block reusable ACK。

   位置：[IoReactor.submit_job](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:550)、[IoReactor.parent_admission_snapshot](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:502)、[IoReactor._retire_accepted_parent](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:529)。
2. **正常完成与 staging 归还。** 原真实 end_event.query 清算 CUDA copy。D2H staging 继续归原 SSD write/CQE 持有；写完后可入缓存，因此不必然增加 free pool。缓存/shared H2D 最后 copy pin 消失后才可由原生路径回收。 无新增 fsync 持久性承诺；局部 D2H 完成不提供全局 GPU owner 立即释放接口。

   位置：[IoReactor._drain_cuda_copies](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:1402)、[IoReactor._on_write_complete](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:1768)、[IoReactor._release_or_cache](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:1882)、[IoReactor._settle_load_slot](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:1898)。
3. **异常接纳与父失败发布。** backend 正常返回后、end_event.record 前记录真实 accepted。可能 partial 的异常不补造 accepted=0；record 失败保留已知接受事实、accounting sticky invalid。实际 stream synchronize 成功后才释放。单文件失败不能在同父其他已接纳 DMA/SSD 排空前发布失败 Future；整父排空后一次性失败，先退休再 callback。 StageAccounting/策略 accepted 是 API 接受观测；unknown/invalid 不能包装为 valid+empty drain。

   位置：[IoReactor._flush_copy_batch](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:2921)、[IoReactor._launch_swap_blocks](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:3010)、[IoReactor._sync_copy_failure](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:1052)、[IoReactor._file_terminal](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:3092)、[IoReactor._finish_jobs](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:3113)。
4. **drain 未证实的 fail-closed。** fatal 清理前须真实同步原 copy streams、排空原 AIO；缺少证明则 NativeDrainUnknown、停止服务和接纳，保留原 owners/Futures/slots/FD，父总量不能假清零。handler fatal 检查阻止未知排空成为 scheduler 可消费的 TransferResult。 需要外层进程/context teardown，没有透明恢复器或新排空队列。

   位置：[IoReactor._freeze_native_drain_unknown](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:481)、[IoReactor._drain_native_before_fatal_failure](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:1066)、[IoReactor._cleanup_after_verified_fatal_drain](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:1082)、[IoReactor._fail_everything](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:1127)、[NoopSharedStorageOffloadingHandler._check_native_fatal](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/vllm.py:466)、[NoopSharedStorageOffloadingHandler.get_finished](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/vllm.py:521)、[NoopSharedStorageOffloadingHandler.wait](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/vllm.py:551)、[NoopSharedStorageOffloadingHandler.shutdown](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/vllm.py:574)、[LinuxAioRing.close](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/linux_aio.py:388)。
5. **STOP 仍持有 busy 对象。** shared STOP 取消缓存身份，但 busy slot 由原 ReadyCopy/PendingCopy 保留；仅 cached=False 且 copies_inflight=0 时释放。cache drain_unpinned 保留 pinned 原对象与策略键；原工作全排空才正常终清理，残余 positive pin fail-closed。已接纳 preload open/read 计数保留到真实清算；STOP 后完成 slot 直接释放、不再缓存。 无新 holding queue/buffer/Future 集合，同一 slot 原 pool exactly-once 归还。

   位置：[IoReactor._intake](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:1198)、[IoReactor._shared_uncache](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:923)、[IoReactor._maybe_release_shared](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:932)、[StagingDataCache.drain_unpinned](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/staging_cache.py:297)、[IoReactor._cleanup_retained_cache_after_stop](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:1026)、[IoReactor._run](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:1005)。
6. **紧凑前台容量观测。** distinct physical pinned-slot scalar 在 0→1/1→0 转移；使用原 preload 总量，shared 不重复累计。free + cache_clean + ordinary_preload_clean + shared_clean 乘原 io_size，得到原 foreground 可回收字节；无需扫描大 registry。typed scalar/owner/identity/invalidation 错误 sticky None，元数据错误不影响原 refcount/eviction。 不是 speculative preload 新资格；原 reservation/eligibility 独立保留，不从 Future.done 推导容量，不制造 slot 或 GPU release credit。

   位置：[StagingDataCache.pin](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/staging_cache.py:274)、[StagingDataCache.unpin](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/staging_cache.py:278)、[IoReactor._shared_pin](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:909)、[IoReactor._shared_unpin](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:913)、[IoReactor._prefix_capacity_components](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:1967)、[IoReactor._prefix_capacity_snapshot](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:2015)。

backend accepted、局部 CUDA copy 完成、write CQE 完成、缓存可见、staging 可回收、整父 transfer 完成、持久存储和 GPU owner 可复用是不同状态。D2H event 仅能证明这次 copy 不再读取源；没有新增 whole GPU owner 立即释放 ACK。原作者 worker 的保护仍是 GPU 复用边界：[OffloadingConnectorWorker._submit_store_jobs](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/vllm-author-build/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:334)、[OffloadingConnectorWorker.handle_preemptions](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/vllm-author-build/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:392)、[OffloadingConnectorWorker.get_finished](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/vllm-author-build/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:457)、[OffloadingConnectorWorker.shutdown](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/vllm-author-build/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:553)。

## 共同修复、研究控制与观测的边界

| 类别 | 包含内容与关闭行为 |
| --- | --- |
| 共同正确性 | accepted-parent 接纳/真实退休；异常真实同步/AIO drain 后清理与 unknown 保留；真实 accepted accounting；handler fatal；STOP pinned/shared/preload 所有权。研究关闭后这些修复继续存在。 |
| 共同紧凑观测 | O(1) clean/pinned scalar、typed validity、原 incoming 上的 owner snapshot；观测错误不改变原 refcount/eviction。没有 GPU 释放信用。 |
| 薄控制桥 | 原 reactor 的 stage decide/settle 钩子；属于通用控制适配，不是独立正确性或已证实性能贡献。controller=None 在策略 clock/state/求和前返回。 |
| P3 研究策略 | simple_stage_policy 的 fixed/pressure、四 stage 与共享额度、epoch/age、有界 keys≤64/records≤96；continuation/mandatory-support 按事实 override 并计真实物理量。无 P4 joint、新缓存/模型引擎、第二作业队列或策略资源持有。 |
| 混合 factory glue | simple_stage_options 与 native vllm.py 同时含共同 parent-admission/StageAccounting 和可选 stage-policy 构造，不能整文件归为研究贡献。off 构造 None，回到保留共同修复的原调度。 |
| observer / CPU probe | 可选 metadata/controller/sink hooks 的 thread_time_ns，嵌套只测外层，故障 sticky invalid、原异常优先；包含 instrumentation 开销，不是整体 CPU 分解或 GPU/backend/model 时间。 |
| 资格 facade | HeldEvent 只用于资格脚本，委托真实 record/query，再施加 host gate；不进入生产执行器。 |

控制、factory 和 probe 的精确位置分别为：

- [IoReactor._prefix_stage_decide](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:2048)、[IoReactor._prefix_stage](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:1936)、[IoReactor._prefix_stage_copy](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:1958)、[IoReactor._prefix_stage_unknown](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py:1948)。
- [make_dispatch_controller](/root/autodl-tmp/prefix-io-v1-handoff/project/src/prefix_io_control/simple_stage_policy.py:110)、[DispatchController.decide](/root/autodl-tmp/prefix-io-v1-handoff/project/src/prefix_io_control/simple_stage_policy.py:292)、[DispatchController._settle](/root/autodl-tmp/prefix-io-v1-handoff/project/src/prefix_io_control/simple_stage_policy.py:339)。
- [parse_simple_options](/root/autodl-tmp/prefix-io-v1-handoff/project/src/prefix_io_control/simple_stage_options.py:25)、[build_simple_kwargs](/root/autodl-tmp/prefix-io-v1-handoff/project/src/prefix_io_control/simple_stage_options.py:77)、[PyKvCacheOffloadingSpec.__init__](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/vllm.py:639)、[PyKvCacheOffloadingSpec.get_handlers](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/vllm.py:771)。
- [MetadataCpu.install](/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/scripts/simple_stage_worker_probe_p316.py:198)、[MetadataCpu.export](/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/scripts/simple_stage_worker_probe_p316.py:229)、[MetadataCpu.restore](/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/scripts/simple_stage_worker_probe_p316.py:253)、[OptionalSampling.suppress](/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/scripts/simple_stage_worker_probe_p316.py:107)、[OptionalSampling.restore_sinks](/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/scripts/simple_stage_worker_probe_p316.py:115)。

研究 OFF 不等于逐字节恢复旧的不安全异常路径；共同修复与 scalar bookkeeping 保留。patch roundtrip 的 common-only/research-OFF 检查是 CPU composition 与源字节可逆证据，不证明所有 GPU 运行时回退。

## 已有资格证据的范围

- P315 native-fault 的 4 个 case 使用真实 copy、人工 partial/record 异常和真实 synchronize，不能称为 genuine driver fault、未知同步 real-GPU 资格或本次 P316 重跑。
- P316 simple primitive 为 16 个有界 case，不能覆盖所有配置、异常或 fallback。
- P316 compact STOP 中 cache/shared 均有 80 个 registry 成员、1 个 distinct busy slot，分别 1/2 个 copy consumer。host-held 期间真实 query 已 ready 的次数分别为 108/139；它验证了软件不提前归还及最终 exactly-once 释放，**不证明真实 DMA 在 gate 期间忙或有额外 DMA 时延**。位置：[qualify.HeldEvent.record](/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/scripts/qualify_compact_staging_gpu_p316.py:226)、[qualify.HeldEvent.query](/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/scripts/qualify_compact_staging_gpu_p316.py:228)。
- immediate GPU owner reuse/free-capacity 仍为 **UNKNOWN**，不能填 0；parent/StageAccounting 为零或 session drained 不能替代 owner 可复用证明。
- roundtrip 没有 CUDA stream/event/tensor/AIO/model 执行。其重复 component 检查不计入新的唯一 CPU 测试数量；旧冻结源的历史 fixture 检查不是新生命周期证明。

| 已有证据 | 原状态 | SHA-256 |
| --- | --- | --- |
| [server08-p3-15/native-fault-summary.json](/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-p3-15/native-fault-summary.json) | PASS_REAL_GPU_NATIVE_FAULT_DRAIN | `08107d64fe902f3b29a7def953f44b9afacb65c9f5562d360fc84847d6dbd2cf` |
| [server08-p3-15/patch-boundary-plan.json](/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-p3-15/patch-boundary-plan.json) | STATIC_PATCH_SPLIT_PLAN_NOT_APPLIED | `edd031173dc6ea9e1736d23ed2a46d38a4dbb9096515c8656142d277755db799` |
| [server08-p3-16/simple-primitive-summary.json](/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-p3-16/simple-primitive-summary.json) | PASS_REAL_GPU_SIMPLE_STAGE | `4fafba0530e2837d29ff533655a0cb4b7cc32899d49f26d59ca567cc8ed1d5ab` |
| [server08-p3-16/compact-capacity-summary.json](/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-p3-16/compact-capacity-summary.json) | PASS_REAL_GPU_P316_COMPACT_STAGING | `3fbe07b51df016d323057beab79264a891a9386ac772b333013e12f999c556b5` |
| [server08-p3-16/patch-boundary-proposal.json](/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-p3-16/patch-boundary-proposal.json) | STATIC_COMMON_NATIVE_INCREMENT_PREPARED_NOT_ROUNDTRIP_TESTED | `01b620adc0cc1cfc085fe8851c70e3e3c8e8414bbb8b840e8e5cc4884ae0f565` |
| [patch-roundtrip-01/result.json](/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-p3-16/patch-roundtrip-01/result.json) | PASS_PATCH_ROUNDTRIP | `6c2e3308d93e205b0671266ef23f7006b45fb1f8298a08e25bed6bab266ba640` |
| [server08-p3-16/cpu-qualification.json](/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-p3-16/cpu-qualification.json) | PASS_CPU_SIMPLE_STAGE | `71cbab3a097ddcf230805d5f37332482cd6ecba400ce445319eff9ed6d7f8b08` |
| [server08-p3-16/additional-closeout-cpu-qualification.json](/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-p3-16/additional-closeout-cpu-qualification.json) | PASS_CPU_ADDITIONAL_P3_CLOSEOUT | `b7d5cf441e903c7b7568e29f4e41091888b2b57cd6f3b0f5e2335736be494fa8` |

早期 STATIC patch plan/proposal 是历史记录；后续 roundtrip 是独立 CPU receipt，原记录未回写。完整实验结论和下一阶段资格由 root 的交付及 stage gate 决定。

## 冻结源身份

| 选定源文件 | SHA-256 |
| --- | --- |
| [third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py) | `5c5129bd75313d86a1012794cfda8096115818e4e806775427ee1f7bed9c6ca2` |
| [third_party/work/py-kvcache-p3-16-cpu/py_kvcache/staging_cache.py](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/staging_cache.py) | `f505da0a942dfded2019fd5b292a986ddb7a29e18bb47c568f9a614eb2a0d928` |
| [third_party/work/py-kvcache-p3-16-cpu/py_kvcache/vllm.py](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/vllm.py) | `e88732f3165f8421263f6c3bc78448e225cccb8ee3a7cdef690d78dee0a08f3d` |
| [third_party/work/py-kvcache-p3-16-cpu/py_kvcache/linux_aio.py](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-16-cpu/py_kvcache/linux_aio.py) | `0a987479722c7d6b520b99c9b283febb55c5145fc17c5623b8b1151c4f45a6d7` |
| [third_party/work/vllm-author-build/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py](/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/vllm-author-build/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py) | `92a967d9cd87daf719d77d1fd2df71c9b90f81a76a7258537129865621318e92` |
| [src/prefix_io_control/simple_stage_policy.py](/root/autodl-tmp/prefix-io-v1-handoff/project/src/prefix_io_control/simple_stage_policy.py) | `8a2dc9586f5e1fe5257f988e44466284d10e0e46a8cee46f876a8febe5caf1a5` |
| [src/prefix_io_control/simple_stage_options.py](/root/autodl-tmp/prefix-io-v1-handoff/project/src/prefix_io_control/simple_stage_options.py) | `63fe6a27087ec604e82f47bd15a4e72eb3f475304203006fd9b5718f5a90a8dd` |
| [experiments/prefix_io_v1/scripts/simple_stage_worker_probe_p316.py](/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/scripts/simple_stage_worker_probe_p316.py) | `d800267ec66d6bcee17988798f47f91f56f8b5d809e7cefa20dc533713be317f` |
| [experiments/prefix_io_v1/scripts/qualify_compact_staging_gpu_p316.py](/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/scripts/qualify_compact_staging_gpu_p316.py) | `6470836066efb4d1ce640f2febedc91ecf0cf2cabc84efa9463ecc3f3c7433b1` |

本次只以 open-x 新建 JSON/Markdown，并核对明确列出的小型文本 SHA；GPU 操作 0、测试 0、项目模块导入 0、模型/缓存 payload 读取 0、递归哈希 0、冻结源修改 0。JSON 同时保存函数起止行、分类和限制。
