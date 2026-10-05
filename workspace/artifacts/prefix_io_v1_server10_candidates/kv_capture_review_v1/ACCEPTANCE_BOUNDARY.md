# 真实生产 KV 捕获的最小验收边界

旧 `g3_kv_byte_evidence.py` **不能直接接收 live capture JSON 或授予 GPU 资格**。它只接受精确 `CPUFixtureBundle` 类型，绑定旧 `0325cb…` 源锁，输出固定为 CPU 内容/结构一致、`GPU_verified=false`、`real_byte_qualification=false`。它最多作为兼容子集的附加 CPU 比较器，不能改写其资格标志。

旧接口还限制每 bundle 至多 8 文件、单 store/load job pair、一个 run_id、单组 int8 二维连续 canonical 页、4096 整除且无外部 padding；所有 consumer 页 capture 必须先于任一 consumer compute。三个 rep、两进程、多 job 的真实记录必须保留原身份，不得为迁就旧 schema 合成假的 job 或同步顺序。

| 原生阶段 | 必须保存的来源及验收条件 |
| --- | --- |
| producer canonical GPU 页 | 原模型 canonical tensor 的设备 UUID、进程/会话、tensor ordinal、shape/stride/dtype/page 大小、storage factor、实际 block IDs；在原 store 源状态下真实读回完整 GPU 页，保留不可变 payload 或独立文件及 SHA。不得用 staging 或 BF16 重新计算代替。 |
| D2H 完成 | 原 copy/owner/job/offload key 的关联、原 CUDA event 完成证据；其后复制对应 staging 的精确区间，逐页与 producer capture 比较。`Future.done()` 或 `Future.result()` 本身不足以证明 GPU 页来源。 |
| write 完成及文件 payload | 原 AIO 完成结果必须等于请求字节、无短写；绑定 FileMapper 实际路径、offload key、offset、逻辑/物理大小；原 publish 完成后独立读取文件内容并比较。write 完成不额外宣称断电持久性。 |
| read 完成 | 原 read 的真实完成结果、请求字节与 staging 区间；逐字节等于已发布的同一文件。若复用 shared preload，保存实际共享 owner/read 的祖先引用，不伪造本消费者的新 SSD read。 |
| H2D 完成及 consumer GPU 页 | 原 H2D event 完成后，从真实目标 CUDA tensor 的映射 block IDs 再读回；与 producer canonical 页、D2H、文件及 read payload 逐字节一致。CPU staging 相等不能替代目标 GPU capture。 |
| consumer compute 前 | 在原模型计算入口检查本请求所需全部页已完成上述 capture；缺一页即拒绝资格。必须保存实际 hook/原方法来源、请求与 job 关联及先后证据，不能事后用完成时间或返回 token 猜测。 |

最小账本按 `(experiment_id, producer_pid/sid, producer_job_id, consumer_pid/sid, consumer_job_id, offload_key, file_index, rep)` 关联；每阶段记录局部单调序号、实际 owner、block 映射、payload 大小/SHA/文件引用、源锁和捕获安装器引用。跨进程顺序由 populate 的真实 publish/正常 shutdown/OS 排空及 fresh paired 的启动记录连接；不能直接排序不共享时钟的数值，也不能改写为伪造单进程序号。

新存储必须独占、初始为空；paired 只读本轮 populate 的冻结 publication，不能混入旧缓存。同一 prefix hash 只是关联键，不是 KV 内容 SHA。验证覆盖全部预期文件、层、块和请求，检查映射无重复、字节数无遗漏；missing、捕获异常、短 I/O、来源漂移、跳过或 unsupported 一律不产生 PASS。若捕获预算不足，应失败并保留不完整收据。

诊断新增 CPU 拷贝、读回及同步会改变时序。两作业固定原 `LoadPlanner=off`、size 128/reps 3；其时延不得进入成本拟合或性能结论，`latency_fit_allowed=false`、`performance_claim=false`。通过最多证明本次选定生产 KV 页在该完整链条上一致，不自动推广到全部模型/长度/布局、物理释放或策略收益。

本说明只读审查旧校验器及现有布局源代码，未运行 GPU、RPC 或捕获；真实资格仍需现场捕获和独立完整账本验收。
