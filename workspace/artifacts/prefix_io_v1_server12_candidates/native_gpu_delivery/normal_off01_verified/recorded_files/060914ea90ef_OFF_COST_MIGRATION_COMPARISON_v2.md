# off01 成本迁移对照（真实失败保留）

原模型、I/O、128 个完整输出与原 shutdown 均已通过；冻结成本上界未通过 normal 迁移。normal 的选中整步为 16.893473 ms，超过原 16.238752 ms 上界 0.654721 ms（约 4.03%）。下一模式不获准。

| 实际窗口 | 首 token / seed | 整步 GPU ms | normal 差额 ms | 原 host frame ms | SSD CQE 观测延迟 ms |
|---|---|---:|---:|---:|---:|
| pair-0 B | 18100 / 1829 | 16.000256 | 0.893217 | 15.940820 | 1.552621 |
| pair-1 B | 19100 / 1830 | 16.238752 | 0.654721 | 16.181185 | 1.443930 |
| pair-2 B | 20100 / 1831 | 16.097504 | 0.795969 | 16.041057 | 1.635198 |
| normal off | 28100 / 2829 | 16.893473 | — | 16.835257 | 1.517552 |

normal 是预注册第四种 workload：首 token 28100、seed 2829；原三种为 18100/1829、19100/1830、20100/1831。完整 prompt 和前缀 key 不同，128 个输出 token IDs 恰好全部相同。normal 与 B0 的后台文件相同；其余 B 使用不同文件但保持 917,504 bytes / 1 op 物理形状。

实际 reactor/collector、原 _run、parent 上限 8、bridge=None、模型/UUID、非 storage 引擎配置、144 context 的单 decode 形状均一致。没有发现实际来源或模型执行器接线漂移。off 通知没有安装（installed=false、get_calls=0），原队列保持相同并恢复。

normal 多出完整请求范围的 gross CPU start/stop 计量和通知附件审计；源码显示它们不是选中 step 的逐帧 profiler，但没有隔离对照测量，不能把 0.654721 ms 确认归因于这些代码。SSD 的实际 CQE 观测延迟 1.517552 ms 落在三次 B 的 1.443930–1.635198 ms 区间，也不能据此宣布因果或 GPU/host 时钟重叠。

选中 frame 的 host 时长同时增加。end-record→end-query 约 1.45–1.47 s 是原 128-step 完整 capture 后的延迟 query 观测；它不是选中整步 GPU 耗时。CUDA/host 绝对时钟未映射。

当前结论限于：旧三 pair heldout 通过仍保留，但冻结上界在这次 normal workload 迁移上失败。不能推广为数学估计器在所有情况下无效，不能扩大上界、改变 A-only 预算、重拟合、挑种子复跑或推进 shadow/on。

JSON 保留全部实际文件的 bytes/SHA-256、source identity、事件跨度与限制；本机纯 stdlib 只读对照，没有远端/GPU 操作。


## v2 来源字段澄清

原报告 `collector_source_ref_equal` 比较的是 collector 持有的 scalar adapter/reactor 引用，已在 v2 更名为 `collector_scalar_adapter_reactor_ref_equal`。真正 collector 文件独立检查通过：calibration plan、实际 calibration SITE_SOURCE_LOCK、normal 的 notification source_refs.collector，以及实际本机归档文件均是同一 24,553 bytes / SHA-256 `bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf`。normal 的 combined metadata pin 亦完全相同。

本机复核校验实际小型证据和此 collector 字节；没有把服务器已通过的完整 4,791 实体来源审计冒充为本机重新哈希了全部权重。normal COMMON_SOURCE_LOCK 的实际引用保留在 v2，由主进程真实服务器审计支持。原报告保留，不覆盖。无模型、GPU、远端调用或新增测试。
