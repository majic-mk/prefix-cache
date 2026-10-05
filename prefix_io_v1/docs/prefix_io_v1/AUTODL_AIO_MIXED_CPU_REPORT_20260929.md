# AutoDL AIO CPU 混合负载续验（2026-09-29）

结论：CPU_MIXED_FUNCTIONAL_PASS；PERFORMANCE_SCREEN_ONLY；GPU_UNVERIFIED。仍处于 P1 的 CPU 准备部分，P1 未完成，P2–P7 不开启。

## 实际环境与修改

所有开发、磁盘测试与测量在 connect.westd.seetacloud.com:40168 的原项目目录执行。主机名 autodl-container-mz3u56hqpn-f855e672。
作者 py-kvcache 锁定提交 3abba7a502d553f6e7e2e58b92086487e3395d7e；继续使用独立工作区 third_party/work/py-kvcache-aio-cpu。

本轮唯一运行时代码改动是 LinuxAioRing.submit_pending：没有待提交请求时，不再写 eventfd 唤醒后台线程。原 reactor 每轮泵循环都调用此方法，原实现导致空唤醒。关闭仍无条件唤醒，已接受的请求、完成事件和元数据完成保留各自通知，不改变队列容量、数据所有权、线程数、准入或研究策略。

新增：
- experiments/prefix_io_v1/scripts/aio_cpu_mixed_screen.py：调用作者 DirectIoFileStore 的真实 O_DIRECT 与元数据混合验证。
- tests/prefix_io_v1_aio/test_mixed_lifecycle.py：10 项 CPU 检查。
- 更新共同兼容补丁 patches/prefix_io_v1/compatibility/0001-optional-linux-aio-cpu.patch；本轮最小增量另存 empty-submit-fix.patch。

原活动目录 third_party/work/py-kvcache 的 Git diff 前后完全相同；未切换活动运行路径，默认仍为 io_uring。兼容补丁相对于已有共同修复生成，正向和反向 git apply --check 均通过。

## 实际块大小审计

按历史 native-prefix-04 的真实 KV 张量元数据计算：28 层，每层 2×16×4×128 个 BF16 元素；合计每个 16-token GPU 块为 917,504 字节（896 KiB），每 token 57,344 字节。

作者 ParsedKvLayout.storage_block_bytes 乘以存储合并因子；DirectIoFileStore 再按 4096 字节对齐。CPU 覆盖：
- 因子 1：16 token，896 KiB；
- 因子 4：64 token，3.5 MiB；
- 因子 16：256 token，14 MiB（正确性测试）。

这些是有来源的候选尺寸；当前 SSD 运行配置尚未冻结，历史元数据也不代表本机 GPU 已通过。原 reactor 实际给后端的 ring depth 为 max(iodepth + open_lookahead + 16, 16)，不能把裸适配器 depth 与 reactor 的数据并发直接等同。审计记录与原始证据 SHA256 位于 layout-audit.json。

## 测试结果

最终冻结联合运行：320 passed、16 skipped、0 failed。
- AIO 检查 72 项通过（上轮 62 项，本轮新增 10 项）。
- 作者原有检查 248 项通过、16 项跳过；13 项依赖无 vLLM fallback 的环境条件，3 项 io_uring 因 EPERM 跳过。
- GPU/模型端到端测试显式排除。
- 新增测试覆盖目标尺寸真实混合读写、同步发布、完整字节比较、FD/线程无泄漏、两个元数据线程均被阻塞时数据读写仍完成，以及容量满后回收与重试。
- 空提交测试先失败：1,000 次空调用产生 1,000 次唤醒；修复后为 0。关闭空闲后端仍完成唤醒与安全排空。
- CUDA 初始化被测试保护禁止，结果 cuda_initialized=false、gpu_workloads_run=0。

精确命令、stdout/stderr、JUnit 与 GPU guard JSON 均保留。失败的修复前记录也保留，未计入最终通过数。汇总见 qualification-summary.json；此处不把不同轮次通过数相加。

## 混合负载测量与限制

修复前、后各 18 组，共 36 组。尺寸 4 KiB、896 KiB、3.5 MiB；并发 1/4/16；每组合各两次，第二轮反转组合顺序。后端元数据线程数为 2。
用交替 load/store 作业调用作者的打开、读写、发布和关闭函数；读种子文件预写并 fsync，写目标是新文件。保留作者同步 open_temp_write/finish_write 行为，没有改造发布策略。

所有组的字节检查、FD/线程检查通过，已接受/完成/回收计数一致。代表性原始吞吐（MiB/s，包含 Python 泵循环、完整读字节校验和元数据耗时）：

| 尺寸 / 并发 | 修复前两次 | 修复后两次 |
|---|---|---|
| 4 KiB / 1 | 3.420 / 3.577 | 4.682 / 5.059 |
| 4 KiB / 4 | 5.139 / 13.708 | 4.951 / 11.668 |
| 896 KiB / 4 | 233.193 / 228.549 | 499.341 / 511.047 |
| 3.5 MiB / 16 | 193.486 / 245.129 | 236.915 / 275.595 |

这些短样本波动明显，前后顺序没有交叉平衡，不足以把差异因果归于此次修改，更不构成稳定加速结论。4 KiB / 并发4还有较低的修复后样本，未删去。
当前可确认的是消除了空唤醒，并通过功能回归；不能据此声称小块开销已解决、GPU 干扰下降或算法收益。empty_submit_calls 统计上层空调用次数，修复后上层仍可空调用，只是后端不再因此唤醒。

本轮测量没有调用完整 IoReactor、pinned staging、CUDA copy、真实 KV 或模型执行。使用原文件存储接口与真实磁盘，但内容为合成字节；不能替代 mandatory drain 的 GPU 生命周期验证，也不是原 io_uring 性能对照或 SSD 带宽标定。

范围文字勘误：两轮原始 JSON 的 scope 写“50/50”，实际为交替作业，奇数总数时多一次 load；各行 operation_counts 是准确计数。原记录不覆盖，scope-erratum.json 明示修正。交付脚本只修正此文字，不改变测量逻辑；measured-script.py 保存两轮实际执行版本，原始结果内 SHA256 与之匹配。

## 资源与预算

两轮微基准计时内逻辑 I/O 共 930,086,912 字节；另有预写 385,695,744 字节、计时外文件验证读取 454,033,408 字节。正确性与回归测试的少量 I/O 未混入上述微基准计数。
GPU 工作负载 0、模型加载/下载 0、驱动/系统修改 0。权限与 GPU 预算账本 SHA256 前后不变，历史预算没有重置。当前设备身份未重新探测。

## 下一允许动作与未解决项

CPU 混合功能验证与这项有证据的小修复已完成，可准备 GPU 接入前核验。
进入 GPU 运行前必须核对本机实际 UUID、permissions.yaml 的明确设备授权、剩余预算与驱动可用性；本轮未执行该阶段。当前权限仍是历史设备 UUID，不应把本轮“继续”解释为替换授权设备或修改驱动。

下一项决定系统是否可行的关键证据是真实 pinned staging、CUDA D2H/H2D、生产 KV 字节往返及 mandatory 排空。没有这些证据，无法宣布系统或研究策略有效。io_uring 原路径的环境阻塞也没有解除。
