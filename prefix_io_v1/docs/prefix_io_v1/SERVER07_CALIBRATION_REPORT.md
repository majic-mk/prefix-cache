# Server07：P1 原生 AIO 成本标定与 LoadPlanner 接入

## 结论
当前 AutoDL 服务器可以继续本路线，无需换服务器、镜像或修改驱动。本轮完成有界 P1 原生模型基线验证：真实 OffloadingConnector、生产 KV 的 SSD / staging 回读、实测成本曲线、原 LoadPlanner 调用及自然 GPU Prefix 复用均有证据。可进入 P2 的观测和 mandatory 进展验证。完整 P0–P7 计划尚未完成，研究策略仍未启用。

这里只确认 Linux AIO 兼容路线；默认 io_uring 路线此前的 EPERM 不因本次成功而消失。没有切换全局安装，没有更改缓存引擎、模型执行器、准入算法或 Prefix 身份算法。

## 实际修改
- 新增 experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py：复用固定作者 LLM、OffloadingConnector、handler、profiler 和结束接口。采集原生 RequestStateStats.first_token_latency；保留输入、输出、内部请求 ID、实际读字节、H2D、原始 trace、失败和收尾。
- 新增 experiments/prefix_io_v1/scripts/export_native_aio_costs.py：核验模型/设备/配置/输出一致性、介质证据及样本后，导出作者 v2 格式；插值和 golden 检查复用原 PCHIP/cost_model。
- 新增 4 个 trace 身份关联测试、8 个成本数据拒绝测试。更新执行状态、锁和复现材料。
- 使用项目内相对模型别名指向已验证的离线模型，避免 FileMapper 禁止绝对模型名的限制；安全路径检查未放宽。
- 使用作者支持的 PYTHONHASHSEED=0 与 sha256 配置，保证重启后 Prefix 根哈希一致。未改哈希函数或放宽精确匹配。

共同兼容补丁仍来自前一轮。本轮没有新增研究策略，也没有对 py-kvcache / vLLM 运行时文件作新修改。

## 冻结条件
服务器 westd:38819；GPU UUID GPU-f8744916-1693-fa6a-93b6-7f503c03459c；RTX 5090；Qwen2.5-7B-Instruct BF16；作者 vLLM 817a7e3124f817cd6e549581d3e5483207a753a4；py-kvcache 3abba7a502d553f6e7e2e58b92086487e3395d7e 加此前共同/AIO 补丁。

GPU KV 256 MiB；staging 上限 128 MiB，实际 pinned backing 133,959,679 字节，小于 134,217,728 字节；storage block=16 token，payload=aligned I/O=917,504 字节；iodepth=4；原 LRU、共享预加载、lookahead=1、复制合并、异步 store 保留。

标定轴 N 为可复用前缀 token 数。完全相同的请求含 N+1 个输入 token，并生成 1 个输出 token；三路径均支付同一个额外查询 token。该单 token、受控 token-ID 标定不是主服务流性能实验。每档 1 次预热保留但不入中位数，另有 5 次测量；不删除慢样本。采样域 16–1024；标定上下文 1040 仅容纳末端查询，集成上下文 1024，不向更大范围外推。

标定使用原未定价路径采集成本，明确 load_planner=off；集成使用原 load_planner=on 和实测曲线。GPU-only reset 仅用于标定；最终 36 请求的规划器集成流 reset=0。

## 实测首 token 延迟
以下为毫秒中位数，f 是独立同请求冷重算，g 是同引擎的 SSD/staging 配对测量。它们包含相应原生路径与观测开销，不是硬件带宽峰值。

| 前缀 token | 冷重算 f | SSD g_ssd | staging g_mem |
|---:|---:|---:|---:|
| 16 | 13.44 | 27.68 | 26.14 |
| 64 | 14.11 | 29.98 | 25.87 |
| 128 | 16.65 | 40.67 | 34.77 |
| 256 | 24.35 | 53.89 | 40.09 |
| 512 | 40.31 | 86.16 | 44.21 |
| 1024 | 73.28 | 127.43 | 58.37 |

本采样域 SSD 均不划算；1024 token 的 staging 路径较快。不能把这组结果说成整体系统/研究策略加速。曲线来自有界单机诊断，尚未完成独立服务流预测误差与观测开销验证。

最终曲线：artifacts/prefix_io_v1/server07-calibration-01/curves-paired/curves-v2.json。
SSD 阈值 1040 表示在 <=1024 的域中没有持续盈利区间，不表示 1040 及以上已经实测盈利。staging 的 608 阈值是 PCHIP 插值推导，不是 608 token 的独立实测结论。零 token floor 沿用原工具约定；golden 仅为数值一致性检查。

## 路径与原规划器证据
- 36 组相同输入的冷重算、store、staging、SSD 输出 token 均一致；72 次配对加载具备成功 parent、H2D、正确介质和字节证据。
- 最终配对 SSD 组成功完成读取 688,128,000 字节；staging 配对组读取 0 字节。前景 read 以成功父任务、冻结 io_size 和完整 CQE 约束推导；preload read 记录实际返回字节。逻辑搬运字节与磁盘字节分列。
- 最终规划器测试 server07-cal-planned-03 共 36 个真实模型请求，原 LoadPlanner 调用 36 次，记录 36 次 break_even 拒绝。12 个目标重复请求自然 GPU 命中、无外部加载，输出一致。调用次数不等于独立请求数。
- 512 token 例：原 planner 预测加载 0.086158 s、重算 0.041325 s，自行拒绝；没有覆盖 decision、伪造 always-admit 曲线或改已启动 H2D 为重算。
- 原 scheduler 只对等待候选调用 planner。单独到达的请求可能先走原 worker 阈值路径；该原有边界已在失败 trace 中记录。队首冷请求占用原单序列执行槽，使后续已有前缀的两个请求成为真实等待候选。未修改调度器以制造调用。
- 原 manager 当前 predicted_dram_resident=False；未伪造驻留提示。更广范围内的规划器收益/CPU 命中选择仍不能由本次诊断推出。

## 生命周期
诊断 RPC 只使用原 _submit_store_jobs、Future.result 与 worker.wait 排空已经接受的 store；不读取并吞掉 handler.get_finished，不提前 complete_store。后续正常引擎 step 传递完成元数据，保留原 scheduler fence 协议。单独的生产 KV 字节一致性与 shutdown drain 证据沿用 server07-native-aio-kv-02。

正常 drain 成功不等于普通额度为零时仍可释放 GPU 块的 witness。后者属于 P2，仍未完成；joint 不能启用。

## 测试、失败与资源
CPU：36 项标定/采集器测试通过；8 项拒绝非法成本数据测试通过。4 个新增采集器测试又在禁止 CUDA 初始化的 guard 下通过，属于重复验证，不另加到 44 的唯一测试数中。本轮没有重跑未改变的 320 项缓存/AIO 回归，其前轮结果仍单独保留。

初次将整组纯 Python 测试放进会导入 torch 的 guard，导致其中“禁止导入 torch”的旧测试失败（35 pass / 1 fail）；随后用原纯 Python 方式整组 36 pass，并单独 guard 新 GPU 无关测试。失败日志保留。

GPU 实际执行 10 个预算任务，6 成功、4 失败，均安全排空。失败包括内部请求 ID 关联、未固定 Prefix 根哈希、单请求与两个 async-load 请求未形成规划候选。初始独立运行聚合在 128 token 出现 g_mem > g_ssd；该旧曲线目录明确排除，不用于集成。最终采用同引擎配对结果，未覆盖负数据。

本轮新增 GPU runner 墙钟 707.561 秒（11.79 分钟）；累计 1169.343 秒；8 小时预算剩余 7.675 小时。新增模型下载 0；累计下载账本 19,422,798,722 字节不重置。无系统/驱动修改，无后台 GPU 任务遗留，最终显存 0 MiB、利用率 0%，磁盘约剩余 18 GiB。日志中的 Python resource_tracker semaphore 警告保留；session 清理确认没有存活的 GPU 子进程，不宣称从未出现告警。

## 下一允许阶段
P2：仅先做 off/shadow 观测、真实 GPU 源块释放条件与 mandatory 排空验证，再测观察开销。尚未允许跳过这些条件启用普通干扰额度或 joint。P3–P7、持续 decode、正式请求流、独立评估和论文收益均未完成。
