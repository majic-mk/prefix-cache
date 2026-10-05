# 有界原生释放观测的 CPU 准备

本目录新增了可执行只读 adapter 和 CPU 验收，复用现有 `NativeFlushProbe` 的 opt-in 安装、原 owner 线程检查、原方法返回后回调和卸载。没有改动任何既有冻结源码，没有分配或释放原生资源、增加工作队列、调用 `complete_store`、替换模型执行或签发 release/dispatch permit。

`owner_release_adapter.py` 的 `make_observer_class(NativeFlushProbe)` 生成最小子类。只有显式安装的后续诊断作业可以使用它。普通 off 路径无需导入这个模块。原 `immediately_reusable_bytes=None` 保持原值，新增 export 的 `production_release_capability` 永远为 false。本 CPU 准备没有将它接入当前 I-only 策略候选，也没有将依赖排序 D 作为 I 首轮验证的先决条件。

## 输入实际来源

`../source_inputs/SOURCE_INPUTS.json` 和 `KV_OFFLOAD_SOURCE_INPUT.json` 记录本轮从当前服务器下载的原 fork 路径、字节数和 SHA。`verify_owner_preparation_cpu.py` 按这些 SHA 读取六份真实服务器源码，并核对以下三份已冻结观测/handler/reactor 源；任一已存在版本 SHA 不符就硬失败，不能换用另一个“能通过”的副本。

| 输入 | 原有来源和处理 |
|---|---|
| run / block identity | 原诊断作业 run，`BlockPool.blocks[bid].block_id`；不同 run 不合并 |
| generation | 原 `NativeFlushProbe.allocated` 在 `BlockPool.get_new_blocks` 原方法成功返回后记录的单调 allocation sequence；没有观察到历史 allocation 就 unknown |
| active refs | `KVCacheBlock.ref_cnt`；bool、负值和未知不能当作零引用 |
| native free-list state | 原 `prev_free_block`/`next_free_block` 双向邻接与 `is_null`，依赖原同 owner 的 free-list 不变量；不扫描或移动队列 |
| store source membership | `TransferJobStatus.non_sliding_window_block_ids` 的原 first/last 样本；省略的中间 block 不推测父成员关系。当前 base 未发布 sliding source，不能用本扩展声称已覆盖它 |
| protectors | 原 scheduler `_block_id_to_pending_jobs`；运行请求由 refs 保护，非滑窗 stores 在 `request_finished` 后注册，滑窗由原 scheduler 提前注册 |
| native parent retirement | 原 `update_connector_output` 先扣完 `pending_count`，再 `manager.complete_store`、撤 pending protector 和 `_jobs`；原 probe 只在该方法返回后、真实 worker completed_jobs 正值且 job 已不存在时记录 retirement |
| native wait cause | 原 `_observe_prefix_flush` 的 `new_allocation` / `restore_destination` 和真实冲突 block/parents；all-finished、preemption、cache reset 本身不算自然资源等待 |
| worker completion chain | 原 worker `get_finished` 成功结果→`mark_completed`，`OffloadingWorker` 转交原 handler；handler 的 Future 只有在确切 reactor DMA/CQE 全父收尾链已核对时才能用来理解原 worker ack。adapter 不接受任意 CPU `future_done` 事件 |

三份固定输入 SHA：

- native_flush_probe.py：`b24902ca76aa998c564218738ec02cfc3c96f5b12523a2df80765a489c04ebda`
- py_kvcache/vllm.py：`901cc5b9a20245e60d2200ef190979b80adb77ca5a049624c6bc4ec171a712d1`
- py_kvcache/reactor.py：`a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47`

实际字节、方法行号和每个完整 AST 的 SHA 保存于 CPU_RESULT 的 source_bindings。未导入上述 engine/backend 模块；原 scheduler 的两个完整原 AST 方法只在独立 CPU namespace 中执行。该 namespace 使用明确标记的 CPU 值 fixture，不是原生 allocator、CUDA Event、SDK 或模型执行。

## 有界和拒绝条件

新增 adapter 最多 8 个资源、每个最多 32 个父、96 条诊断记录；原 NativeFlushProbe 的独立固定域与截断字段继续保留。超限只丢弃可选观测，不丢弃原工作。所有输入 snapshot 只读，adapter 没有 owner refcount 或 pending map 的写权限。

只有同一观察到的 source generation、完整且一致的真实保护父闭包、全部父原 native retirement、refs=0、无原 pending protector、非 null 且原 free-list 邻接存在，才可能记录 allocator availability。默认 `worker_completion_verified=False`，在 worker/fence 语义未满足时仍返回 unknown；这个布尔值不是 GPU Event，也不提供原生权限。CPU fixture 中显式 true 只验证分支，不能记入 GPU 结果。

新的 allocation 导致 source generation≠current generation，或未采到父 source、出现未观察 protector、父截断、任意 CPU Future完成、清单错误，都不给立即可复用字节。仅观察到 pending 父时的 potential_bytes 是依赖预测，不是已释放显存或性能收益。

现有 base 的 original safe() 包含观察异常与 owner 线程变化；子类继承它，观察错误使诊断 faulted，原有方法结果仍由原路径返回。不会为了收集证据重置 refs、撤保护或重试原作业。

## 实际 CPU 验收

本机命令：

```text
Python 3.12 -B -I -S verify_owner_preparation_cpu.py --output LOCAL_OWNER_PREPARATION_CPU_RESULT_V2.json
```

实际 exit=0，22/22 tests、0 skip，9 份输入 before/after SHA 全部相同，GPU jobs=0。初版 19 项验收记录 `LOCAL_OWNER_PREPARATION_CPU_RESULT.json` 保留；新增候选界限、只读性和父代际不一致拒绝测试后，V2 为最终本机证据。

测试涵盖：全部父 AND 闭包、两 worker 部分 ack 保留原保护、第二 ack 才执行原 complete_store 并撤保护；原 probe callbacks 与这个实际方法回放的顺序；generation变化、active reference、free-list缺失、未知 protector、采样遗漏、截断、CPU Future、自然等待分类、bool身份、候选界限、输入只读、未知 fence、原观察异常 containment。没有导入 Torch、vLLM、py-kvcache 或 SDK。

服务器验收命令由根代理执行并保存新文件，不覆盖本机证据：

```text
.venv/bin/python -B -I -S <new_candidate>/release_observer/verify_owner_preparation_cpu.py --output <new_candidate>/release_observer/SERVER_OWNER_PREPARATION_CPU_RESULT.json
```

## 未来真实等待见证与当前限制

后续 D 资格至少需要正常 workload 的原资源冲突原因、run/source/current generation 一致、完整原保护父闭包、原 worker waiting/fence 与 parent retirement 对应、真实 allocator 原状态更新，以及独立请求/token时间测量。仅有 flush metadata、Future done、CPU fixture 的 availability、或者无关 shutdown drain 都不满足效果见证。

本次准备证明 adapter 能执行、字段来自真实源、拒绝边界和原 CPU 合同得到验收。它尚未产生真实 GPU release、没有生产 NativePublication GPU-release capability，没有证明当前固定请求存在可受益的资源阻塞，也没有验证 D/joint 或论文性能收益。I-only 的首次 pilot 可以在其独立成本、生命周期、预算和 workload 条件通过后先进行；不应为了等本 adapter 的 D 资格继续重复无关 off 实验。
