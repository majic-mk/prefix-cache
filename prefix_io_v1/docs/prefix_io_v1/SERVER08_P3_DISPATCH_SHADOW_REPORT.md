# Server08 P313：迁移资格与四阶段接受观测

日期：2026-09-30。阶段仍为 P3；本轮不是 P3 全量验收或研究效果评估。

## 结论与本轮边界

新指定 AutoDL 实例可以继续开展本路线的受限实验。真实 CUDA/Linux AIO、原生 7B 完整输出和新四阶段观测桥接均有成功证据。本轮不更改原 LoadPlanner、Prefix identity、attention、模型执行或完成协议；没有安装/修改驱动，没有新增云租用、下载、删除、缓存合并或远端推送。

新增 DispatchShadow 只审计真实接受，不实施普通延期、硬容量限制、parent admission、progress override 或研究排序。默认 dispatch_shadow=None 回到原生路径；模块 off/shadow 是提交观测模式，未新增 Prefix shadow 功能。fixed/pressure 的旧实现与证据保持原边界，interference/dependency_only/joint 尚未启用。收益仍未成立，也不能据本轮功能通过推断不可行。

## P0 迁移审计与版本锁

用户指定 SSH 为 connect.westd.seetacloud.com:20739。工程根目录为 /root/autodl-tmp/prefix-io-v1-handoff/project。主机名 autodl-container-v07skcjs40-bfc26e02；RTX 5090，UUID GPU-83ac724c-6d18-6f08-ab96-fb871fb1f4a8，驱动 595.71.05；nvidia-smi 显示 CUDA 13.2。克隆环境的 Python 3.12.3、Torch 2.11.0+cu130 与项目 CUDA toolkit 保持原安装，不把驱动显示版本当作 toolkit 版本。依赖清单通过 importlib.metadata 冻结 214 项；隔离 venv 没有 pip 模块，原 pip freeze 失败与替代清单均留档，没有为此安装软件。

工程 HEAD cc7898b1ba59d89ce7fdbb186ded21880f1adf08；py-kvcache 作者 HEAD 3abba7a502d553f6e7e2e58b92086487e3395d7e；作者 vLLM HEAD 817a7e3124f817cd6e549581d3e5483207a753a4。原分支、未提交与未跟踪内容保留。新增隔离工作树 third_party/work/py-kvcache-p3-shadow-cpu，分支 codex/prefix-p3-dispatch-shadow。31 个 Python 文件先逐字节复制旧 P312 seed，之后只增量修改 reactor.py；vLLM 及旧工作树未修改。

开始时核验 P312 的 266 个源码锁输入和 464 个交付证据 SHA 全部一致；模型 11 文件、15,242,788,168 字节完整 SHA 一致；原始缓存 3,048 文件、2,796,552,192 字节运行前后完整 SHA 一致。不是仅检查大小或能够导入。

到达时 GPU 利用率连续三次显示 100%，显存 2 MiB，无可见计算进程。仅运行获授权的 CUDA/AIO 正确性检查后，连续两次显示 0%，后续每次性能无关资格运行均重新检查 UUID、无进程、利用率 0%。原因没有被证明；不能宣称已修复驱动或已排除云端干扰。末次读数 0%、2 MiB、无计算进程。

更新 permissions.yaml 的目标 GPU UUID及辅助存储授权引用，新增 authorizations/new_server_08.json、system_disk_validation_server08_20260930.json，继承原用户授权与预算，不重置账本。一般 driver/system 修改许可仍为 false，旧有范围特定例外不扩张。旧缓存合并清单均已闭合，不覆盖本轮新文件。

新实例主数据盘为 100 GiB，首检可用 62,287,171,584 字节；原授权主实验目录的 3 GiB 下一轮预留已通过。辅助根目录仍受 20 GiB 占用上限、8 GiB 余量约束，不能将新 bulk 模型缓存放回已接近上限的系统盘。新 CPU/GPU 输出与交付 ZIP 都使用已授权的主实验目录。

## 实际增量与接口位置

- src/prefix_io_control/dispatch_shadow.py：有界 96 条记录，四方向实际 API 接受总量、累计额度审计、过期与未知字段、接受不确定/完成未知分类。复用已有 DispatchBudget/Amount 值对象，未修改 DispatchLedger。
- tests/prefix_io_v1_dispatch_shadow/test_contract.py 与 test_native_bridge.py：34+33 个参数展开案例。
- 隔离 reactor.py：IoReactor/TransferCoordinator 新增可选 dispatch_shadow；四个实际提交位置为 _submit_read_from_ready、_on_store_copy_done、_launch_swap_blocks、_flush_copy_batch。begin/settle 均是旁路记录，不参与分支许可或队列排序。
- run_shadow_cpu_qualification.py：新旧工作树的合同测试分进程执行，禁止 CUDA 初始化、存储预检、180 秒/进程上限与会话清理。
- qualify_dispatch_shadow_gpu.py：相同原生路径的 off/shadow 六场景；qualification-only ZeroAudit 在 owner 发布零累计/在途额度，证明它只审计、不会阻挡任务。
- package_dispatch_shadow_delivery.py：仅打包代码、锁、补丁和原始日志，模型与 .bin 缓存留在服务器。

本轮没有新增共同基础/生命周期修复或研究策略。源码新增、观测补丁、实验设施与授权记录分开。observer/0007-native-dispatch-shadow.patch 只针对已验证 P312 seed；不要直接应用到裸作者 HEAD。项目新文件另在 observer/0008-dispatch-shadow-experiment.patch。两份补丁共 7 文件 apply/逐字节 roundtrip 全部通过，最终源码锁为 317 个输入；结果单独留档。

## 生命周期与未知能力

SSD 接受边界为 queue_read/queue_write 正常返回，可能仍在后端待提交；CUDA 接受边界为 swap_blocks_batch 正常返回，发生在 end_event.record 之前。API 返回不等于 DMA/CQE 完成、cache-visible、durable 或 GPU 可回收。

D2H 完成后必要 SSD 写入、已取得 FD/slot 的 read、read 到 H2D、共享 copies、原生 fusion、close/CQE、父任务完整回调都继续走原路径。源 GPU 块仍受原父任务/保护者协议限制，没有提前 complete_store 或释放 credit。

四方向 in-flight 来自 owner 上有效 StageAccounting；_data_inflight 不冒充四方向总量。可回收 staging 与全体已接受 parent 数目前不可完整取得，字段明确 None；不把 len(_active)、free_count 或过期快照当作已验证容量。native_issue_safe=True 仅表示本次已走过原来的 slot/FD/mapping/stream 依赖路径，不表示事件已完成。

后端抛异常可能已部分接受，桥接保守记 uncertain。已接受 CUDA 调用后 end-event 失败记 completion_unknown，不退款。CPU 注入测试检查分类与原异常保留，未进行真实 GPU 部分 DMA 故障；原有该类异常 cleanup 的物理安全与无损在线恢复仍未被证明。正常 GPU 资格通过另外检查原生 worker/ring/在途资源实际排空，不能用 audit snapshot 的完整标志代替排空。

## CPU 命令、结果与保留失败

所有项目测试在服务器执行，本地仅用于 SSH 传输与 ZIP 哈希验证。

迁移原矩阵命令：
```text
.venv/bin/python experiments/prefix_io_v1/scripts/run_order_cpu_qualification.py --label server08-p3-13-migration-matrix
```
实际环境见 migration-cpu-command.json。860 passed、16 skipped、0 failed；cuda_initialized=false，gpu_workloads_run=0。

新矩阵最终命令：
```sh
PYTHONPATH=src:experiments/prefix_io_v1/scripts PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python experiments/prefix_io_v1/scripts/run_shadow_cpu_qualification.py --label server08-p3-13-shadow-matrix-02
```
新工作树 924 passed、16 skipped、0 failed；旧 order 工作树的三项模型 fixture 3 passed。两个独立进程/环境/原始 XML/guard 留存；去重合计 927 passed、16 skipped、0 failed，新增 67 项。16 跳过为真实 vLLM 环境不适用的 13 fallback 项与 io_uring EPERM 的 3 项，不称 io_uring 可用。

保留两个初始失败：一次启动未设 PYTHONPATH，import prefix_io_control 失败，未执行测试；首次新矩阵 943 项中 924 passed、16 skipped、3 failed，均是旧 P312 store_order_worker_probe 的严格实际模块路径门禁。最终 wrapper 只将这三个完整 nodeid 路由到其原锁定工作树；旧生产门禁与旧测试未改，不冒称旧模型许可已经允许新 shadow runtime。

新测试包括同 epoch 不补充、跨到期接受不退款、未知/陈旧能力、接受不确定、四方向计数、融合实际字节、共享读一次、mandatory 与续接、short CQE、重复 completion、audit fault 回退、off 不读新 clock/sizes、严格构造拒绝、移除审计 hooks 恢复全部 seed AST。

## 真实 GPU 命令与结果

每次通过原 run_gpu_stage.py，实际环境、冻结问题、源码锁、fresh 空闲/存储/预算检查见本轮三个 plan 与 receipt。已完成 label 不能重复执行。以下是本轮实际 wrapper 命令摘要，完整参数为对应 plan.json 中 command 数组：

```text
run_gpu_stage.py --label server08-p3-13-order-migration-01 --seconds 90 -- ... qualify_store_order_gpu.py ...
run_gpu_stage.py --label server08-p3-13-native-fullref-01 --seconds 240 -- ... run_concurrent_native_reference.py ...
run_gpu_stage.py --label server08-p3-13-dispatch-shadow-01 --seconds 90 -- ... qualify_dispatch_shadow_gpu.py ...
```

三次均 exit=0，无 timeout，session_drained=true：
1. 旧原生 store-order 迁移：15.1198483468 秒，6 个 CUDA/Linux AIO 场景；off 与 order 仍按已验证的 [1,2]/[2,1] parent 完成，CUDA 测试张量逐字节一致。
2. 新 GPU 的原生 Qwen2.5-7B BF16 全输出参考：80.1733633569 秒，5 家族 × cold/gpu_hot，共 10 输出、1,280 token，与旧相同类型参考全部逐 token 一致；GPU 热命中为 16,256 token，冷命中为 0。原驱动每家族建立 cold/hot 参考，共 5 次 GPU Prefix 重置；这是正确性参考，不是连续真实 cohort。该测试没有 SSD/offload、新策略、逐 token ITL 或性能收益结论。
3. 新四方向观测：15.1003695042 秒，off/shadow × store/restore/shared，全部 GPU 字节一致；每份 65 文件 × 131,072 字节。单臂 store SSD_write/D2H 各 8,519,680 B；restore SSD_read/H2D 各 8,519,680 B；shared SSD_read 8,519,680 B，H2D 17,039,360 B。共享读一次、两目的真实复制总字节正确；融合 H2D 为 65 次 API，而非误记 130 次。

shadow 三场景各 130 次接受，共 390 次零审计额度超额，逐方向 op/byte 与原 StageAccounting 完全一致；无 audit fault、接受不确定或完成未知。每场景 96 条记录、34 条覆盖，截断明确可见。off 记录数与超额为零，store parent 顺序仍 [1,2]。实际 staging 16,650,239 B，小于 16 MiB 批准预算；native worker、AIO 与在途资源均真实排空。这里只证明功能和接受观测，不证明硬额度执行或端到端开销达标。

## 预算、证据与下一允许动作

本轮真实 GPU 增量 110.3935812078 秒；继承总量 10,266.4771967158 秒 / 28,800 秒，剩余 5.1482007787 小时。新增模型下载 0 字节；累计 19,422,798,722 字节 / 20 GiB。最终 active_reservation=null。禁止因迁移重置统计。

证据根：artifacts/prefix_io_v1/server08-p3-13。migration-before、版本锁、授权、模型/源码/原始缓存哈希、CPU XML/log/guard、所有 GPU 原始 result/process.log、plan、source-lock-before-*、后验 source verification、补丁与交付 manifest 均收入 ZIP。模型权重与缓存 payload 不打包。3,048 份原始缓存后验 SHA 全部保持。

下一允许阶段仍为 P3：在新 UUID/驱动上重新冻结并测量成本与持续 decode 干扰；旧 server07 曲线、qualification、GPU provenance 保持原样，不改 UUID 冒充新标定。继续实现/验证共同 pre-accept admission 与完整各方向 cumulative/in-flight/共享额度桥接，先完成 owner/mandatory/续接与故障协议的 CPU 验证，再做预算内限定 GPU 资格。只有这些门禁与强简单基线评估满足后，才能启用 P4；P4–P7 继续 gated。

未完成：普通预算执行、完整已接受 parent 容量入口、真实部分 DMA 故障安全、新服务器成本标定、模型级 shadow overhead、release witness 因果闭包、干扰/依赖/joint、正式 SLO/goodput、稳定收益。原 P312 四次 ABBA 的约 2.88% 描述性均值差仍正反不稳定，不因本轮迁移和功能成功变为有效提升。
