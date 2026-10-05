# 本轮模型下载会计说明（只读审查）

快照时间：2026-09-26 14:55:38 UTC 附近。此时 qwen-ms-02 仍在下载首个权重，不能把本快照当作最终结算。审查没有网络/GPU操作，没有写共享账本或主状态/锁，也未因等待增加测试。

模型来源为 ModelScope 官方 Qwen/Qwen2.5-7B-Instruct 仓库，固定 **ModelScope revision 16c174980d8a1492910551634b4969e69cdc2444**。实际清单为 modelscope-source/modelscope-download-plan.json（11文件，15,242,788,168 B；SHA256 9b1c79986aa1fd62eb9064ecb8b4e2a03fc235a34ac2d69c33a4fcb531a91017）。Hugging Face 访问失败记录仍保留；不能将此 provider commit 写成已验证的 Hugging Face revision。

## 三个量必须分开

- model_payload_received_bytes：**已结算请求实际读取的 HTTP response payload**，包括metadata、HTML/错误body及文件body；不是模型目录体积，也不是物理网络链路流量。
- model_download_bytes：**已结算的预算扣额**；成功按实际payload，失败按完整预留保守扣额，可能高于实际收到的payload。
- active_reservation.reserved_bytes：**当前尚未结算的预留**；它减少可再分配额度，但不能直接写成已实际收到或已最终扣除。

下载器逐文件完成后才更新累计payload。因此下载进行中，账本“已结算实际payload”不能当成实时已接收总量；在途文件可能已收到大量字节，其额度由active预留覆盖。

本快照：

| 指标 | 字节 |
|---|---:|
| 总授权额度（20 GiB） | 21,474,836,480 |
| 已结算预算扣额 | 3,947,229,451 |
| 已结算实际payload | 1,788,010 |
| 当前首权重预留 | 3,945,441,441 |
| 减去已扣额和当前预留后可再分配额度 | 13,582,165,588 |

已扣额与5个download_events的charged_bytes之和一致；已结算实际payload与其actual_payload_bytes之和一致。当前预留不在上述已扣额中，不能重复相加为“已下载量”。仅减已扣额所得17,527,607,029 B包含当前已承诺的预留，不能全部作为另一个并发任务的自由余额。

## 已结算事件的准确含义

1. 官方metadata/HTML/错误body：实际115,928 B，扣额115,928 B。此前64 MiB预留67,108,864 B已结算，剩余66,992,936 B已释放。证据明确最早几次已完成请求被纳入后续有界审计汇总；不得倒称所有这些历史请求都在发送前建立了预留。
2. generation_config.json：实际243 B，扣额243 B；merges.txt：实际1,671,839 B，扣额同值。成功后各自用于超长探测的额外1 B预留未消费。
3. 首次权重请求 qwen-ms-01：官方响应跳向当时未批准的CDN，执行器未读取body，**实际payload为0，但保守扣额3,945,441,441 B**（权重3,945,441,440 B加1 B探测）。这不是实际下载了约3.67 GiB，更不是成功权重；后续修复或重试不退还这笔已扣额度。
4. redirect/header-only审查：8次请求已停止，redirect body均未读，CDN仅HEAD。实际payload为0、扣额0；64 KiB预留65,536 B已全部释放。另记录HTTP响应header 9,680 B，仅作诊断；当前预算口径不把header/TLS/TCP字节混进payload。
5. qwen-ms-02：本快照仍为RUNNING，当前预留3,945,441,441 B。它不是已消费的第二笔失败费用，也不能提前报告全部权重已下载。后续成功按实际payload结算，失败继续按完整预留保守扣额。

来源证据：experiments/prefix_io_v1/gpu-budget-ledger.json；new-server-03/modelscope-source/{metadata-accounting,budget-settlement}.json；redirect-review/budget-settlement.json；downloads/qwen-ms-01/result.json、qwen-ms-02/result.json。

## 最终交付的推荐口径

待任务结束后重新读取账本，按下列顺序报告：

“已验证本地模型文件 N 个、逻辑总大小 S B；累计实际HTTP payload P B（含metadata与失败时实际读到的body）；累计预算扣额 C B（包含失败保守扣额3,945,441,441 B）；未结算预留 R B；20 GiB预算可再分配余额为21,474,836,480 − C − R B。最终下载状态为……。”

只有result为VERIFIED_LOCAL_FILES且逐文件size/hash全部通过时，才能报告固定清单已完整下载。S、P、C通常不同，不能互换。若还有active预留，明确列出并保持运行中/未结算，不填最终数值。GPU墙钟独立核算；本快照GPU累计仍为45.273048002272844秒，下载耗时不计为新增GPU作业。
