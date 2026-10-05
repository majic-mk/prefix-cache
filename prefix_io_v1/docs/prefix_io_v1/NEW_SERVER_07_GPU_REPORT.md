# 新服务器 07：真实 GPU / AIO / 生产 KV 验证

结论：**这台 AutoDL 服务器上的 AIO 兼容路线已通过 GPU 功能验证，可以继续实验。P1 尚未完成，策略性能与研究收益尚未验证。**

## 环境、授权与同步审计

- connect.westd.seetacloud.com:38819；主机 autodl-container-282b4b870f-825707a9。
- RTX 5090，32,607 MiB；UUID GPU-f8744916-1693-fa6a-93b6-7f503c03459c；驱动 580.105.08。
- Torch 2.11.0+cu130；作者 py-kvcache 3abba7a502d553f6e7e2e58b92086487e3395d7e；作者 vLLM 817a7e3124f817cd6e549581d3e5483207a753a4。
- 已核对上轮 CPU 冻结输入 SHA256，与克隆内容一致；本轮结束后再次核对，原缓存/执行器实现没有变化。
- 用户明确指定新服务器继续剩余验证，记录于 configs/authorizations/new_server_07.json；permissions.yaml 只更新指定 GPU UUID 与来源注释。8 GPU 小时、20 GiB 模型下载上限及所有禁止项保持不变，累计账本未重置。
- 连接初期 GPU 利用率显示 100% 而显存为 0；短资格测试可正常完成，退出后监控恢复为 0%。保留读数，不将原因猜作已确认的外部干扰；本轮不做性能结论。

## 实际改动

新增两个项目内验证驱动：
1. qualify_aio_gpu_reactor.py：调用原 CanonicalKVCaches、ParsedKvLayout、NoopSharedStorageOffloadingHandler、TransferCoordinator、IoReactor；验证真实 pinned staging、CUDA copy、AIO SSD 往返、共享预加载、wait 和 shutdown。
2. qualify_native_aio_kv.py：复用已有 native_gpu_prefix_smoke 的本地模型哈希校验、离线环境、作者 LLM 与固定引擎配置；在无请求执行时，通过原 worker RPC 对其真实 KV 存储做诊断往返，然后继续原生 Prefix 推理。

这些是有界验证驱动；没有新缓存引擎或模型执行器，没有改变 Prefix 身份、LoadPlanner 算法、共享 staging、预加载、复制合并或研究策略。
诊断文件使用独立目录和诊断 key，不冒充模型调度器的 Prefix key；正式 scheduler→connector 的路径仍需另验。
原活动安装目录未切换。本轮 GPU 命令通过 PYTHONPATH 显式选取可选 AIO 工作区，默认 io_backend 仍为 io_uring。

## 实际执行与结果

全部 GPU 相关命令经过现有 run_gpu_stage.py：启动前持久预留，绑定明确 UUID，限时执行，退出后清理整个子会话并累计结算。

| 作业 | 结果 | 证据目录 |
|---|---|---|
| server07-author-copy-01 | 作者真实 H2D/D2H 算子字节、边界保护、ABI 检查通过 | runs/server07-author-copy-01 |
| server07-imports-01 | 固定作者模块来源、真实 API/类身份、拷贝符号通过 | runs/server07-imports-01 |
| server07-aio-reactor-01 | 896 KiB、3.5 MiB 文件真实 GPU＋SSD 往返、两个共享预加载需求、wait、关闭排空通过 | runs/server07-aio-reactor-01 |
| server07-native-aio-kv-01 | 真实模型 KV 字节检查通过；后续输出验证存在下面说明的备份回写干扰 | runs/server07-native-aio-kv-01 |
| server07-native-aio-kv-02 | 去除成功后的备份回写；真实 KV 字节检查与后续 Prefix 输出验证均通过，作为最终证据 | runs/server07-native-aio-kv-02 |

各目录相对于 experiments/prefix_io_v1。所有作业 exit=0，均无超时，session_drained=true。

### 最终真实模型证据

Qwen2.5-7B-Instruct，固定本地 ModelScope revision 16c174980d8a1492910551634b4969e69cdc2444；启动前重新核对全部本地模型文件哈希，无下载。
BF16、单 GPU、TRITON_ATTN、eager；实际 KV 存储 66,977,792 字节，28 层，73 个已分配块。覆盖整个分配存储，因此包括已用和未用页；不声称所有 73 块都包含有效 prompt KV。

冷请求生成 16 个 token 后，在无并发 forward 时：
1. 原 handler 将这份真实 GPU KV 经原 pinned staging 写入 73 个 SSD 文件；
2. 等待原 Future 完成后将目标 GPU 存储清零；
3. 原 handler 从 SSD 读取并经原 CUDA 拷贝恢复；
4. 对全部选定字节精确比较通过；
5. 再提交原 store 后立即调用原 shutdown，已接受工作全部排空；
6. 不通过 CPU 备份再次覆盖成功恢复的 KV，继续原模型生成。

随后冷/重复请求的 num_cached_tokens 分别为 0 / 112，生成的全部 16 个 token IDs 相同。
实际 staging backing 33,034,239 字节，小于 33,554,432 字节预算，且 is_pinned=true。
最终 AIO accepted/completed/reaped 均为 365；outstanding=0、closed=true、drained=true。原 reactor 没有在途文件或 CUDA copy 遗留；引擎正常关闭，预算 runner 确认子会话清理完成。

首次驱动在 finally 中无条件回写 CPU 原始快照，虽然发生在字节精确比较之后，仍使“后续模型直接使用 SSD 恢复结果”的证据不够独立。已修正为仅在失败时恢复备份；第二轮最终结果明确记录 cpu_snapshot_reapplied_on_success=false。保留第一轮，不把它冒充最终证明。

### CPU 回归

在新服务器重新运行原作者与 AIO 套件：**320 passed，16 skipped，0 failed**。
GPU 初始化被 CPU runner 明确禁止。13 项为已安装 vLLM 时不适用的 fallback 测试；3 项 io_uring 因 EPERM 跳过。GPU/model E2E 测试没有混入此 CPU 计数；真实 GPU 证据在上表独立记录。
本轮没有修改缓存运行时代码，因此未引入额外缓存算法回归范围。

## 命令与证据

完整复现入口见 REPRODUCE_NEW_SERVER_07.md。
artifacts/prefix_io_v1/new-server-07/ 包含：
- clone-audit.json、授权记录和 permissions 前后副本；
- GPU launcher 的准确 argv、stdout/stderr、退出码；
- qualification-summary.json、CPU JUnit、跳过原因；
- GPU 驱动源码锁、执行过的初版源码、最终 GPU 环境锁；
- 300 个诊断 SSD 文件的大小与 SHA256 清单（不在交付 ZIP 中重复携带大块 KV 文件）；
- ledger、execution_state 前后副本。

各 runs 目录保留 process.log、预算结果、frozen-config 与详细诊断结果。模型权重、环境、运行时缓存和 SSD payload 不放入交付 ZIP。

## 真实预算消耗

- 本轮 5 个预算作业合计 **176.534314 秒**，包括资格检查、模型加载和进程退出时间，按 runner 墙钟保守结算。
- 历史累计 **461.781797 秒 / 28,800 秒**；剩余约 **7.8717 GPU 小时**。
- 本轮模型下载 **0 字节**；历史累计 19,422,798,722 字节，未清零或扩大限额。
- 驱动、系统 CUDA、安全策略、内核修改均为 0；没有新租赁、付费、远程推送或共享数据删除。
- 所有本轮 reservation 已结算；没有残留 GPU 子会话。

## 边界与下一允许阶段

已验证的是 AIO 兼容基础上的真实 GPU 功能可行性。原 io_uring 路线依旧受 EPERM 限制，不能把 AIO 结果称作原 io_uring 复现或性能证据。

这次模型诊断使用真实 KV 和原 handler，但**没有把外部 OffloadingConnector 接入模型调度器**。真实 LoadPlanner 的成本决策、scheduler 的源 GPU 块释放见证、正常请求流中的 CPU staging / SSD 选择、缺少普通额度时的强制进展，以及 TTFT/ITL/goodput 均未完成验收。
普通 wait/shutdown 在 policy 未接入条件下通过，不等价于“未来限流策略下也不会死锁”。

下一允许阶段是按已批准 AIO 路线重新进行该模型/设备/布局的真实成本标定，再恢复原 LoadPlanner 与 scheduler 的 P1 集成。不得把本轮 GPU smoke 墙钟或 CPU 磁盘微基准当作 f/g 曲线，也不能用旧 io_uring 曲线或“必然准入”的虚构曲线推进。
P1 保持 partial；P2–P7 尚不验收，不宣称整个研究计划已经完成或有性能收益。
