# Server07 P3 数值诊断与 16K 复测交付

本轮在 connect.westd.seetacloud.com:38819 的既有项目内执行。阶段仍为 P3，未进入 P4。现有 AutoDL + 显式 linux_aio 路线具备继续实验的功能条件；尚未证明新增研究策略有收益，也未完成完整 P3。

## 实际修改及边界

- `experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py`：增加仅供原生 GPU 冷/热诊断的 `--diagnostic-logprobs`，通过现成 SamplingParams 收集输出 token 的 top-5 log-prob。要求同时启用 `--native-hot-diagnostic`，因此外部 connector 关闭。默认采样与成本采集不变。
- `experiments/prefix_io_v1/scripts/analyze_repeated_native_costs.py`：新增离线复测校验，核验同模型、同 GPU、同采样、同引擎及 connector 预算；保留热身样本做正确性检查，检查完整 SSD 字节、staging 命中与物理 I/O 排空。只输出单点结果，不导出或安装运行时曲线。
- `tests/prefix_io_v1_pilot/test_cost_domain_guard.py`：补充诊断开关非法组合拒绝测试。
- `tests/prefix_io_v1_pilot/test_repeated_cost_checks.py`：增加 12 项 CPU 合成证据校验测试，覆盖热身输出差异、未排空 I/O、错误介质字节、corruption、重复行、热身归类和配置混用。

P2 的 29 个锁定文件逐项哈希相同；作者 py-kvcache 和 vLLM 运行时未修改。原严格成本导出器未修改。原成本准入、共享 staging、预加载、复制合并和异步流水线保持原样。fixed / pressure / dependency / joint 均未在生产路径启用。

版本：py-kvcache `3abba7a502d553f6e7e2e58b92086487e3395d7e`，作者 vLLM `817a7e3124f817cd6e549581d3e5483207a753a4`。GPU UUID 为 permissions.yaml 中已授权的 `GPU-f8744916-1693-fa6a-93b6-7f503c03459c`。

## 4K 输出差异的新增证据

真实 GPU 运行 `server07-p3-native-logprobs-01`，外部 connector 为 null。2K、4K 各三个前缀，分别冷重算与自然 GPU 热命中；诊断数据不参与延迟拟合。

4K rep0 冷路径输出 token 1095；其 log-prob 为 -1.8005958796，token 70 为 -1.8630958796，差值为 0.0625。原生 GPU 热路径输出 token 70，70 与 1095 的记录值均为 -1.8418247700。其余五对选词相同。

这直接证明此次选词差异也存在于不经 SSD/AIO 的原生 GPU 缓存路径，且伴随 top 候选排序/并列值变化。尚未定位到具体算子，未记录完整 logits，不能断言全部差异都是某一种 BF16 舍入，更不能以此忽略真实拷贝错误。

严格导出中的 token 相等条件没有放宽；4K 旧数据仍不合格。T06 数值容差仍需在独立验证前预注册；不能用本例事后选择一个刚好通过的容差。同一生产 KV 字节往返的一致性与独立重算数值差异继续分开处理。

证据：`artifacts/prefix_io_v1/server07-p3-02/numeric-diagnosis.json`，原始 top-5 值位于对应 run 的 `details/result.json`。

## 16K 重复测量

先冻结 `repeat-plan.json`，随后执行三个独立冷/SSD进程对：两组用于汇总，一组用于时间复验，第二组交换执行顺序。复用已有私有 `server07-p3-16k-storage-01`，未创建新的大体积缓存，也未删除数据。

各进程：Qwen2.5-7B BF16、单序列、max_model_len=max_num_batched_tokens=16400、配置 GPU KV 2 GiB；外部路径 staging 预算 1 GiB，实际固定分配 1,073,483,775 字节且 pinned，iodepth=4、LRU、共享 staging 和预加载开启。这里 GPU KV 是配置预算，不能冒充本轮重新检查了实际 CUDA tensor storage。

每组每条路径保留 1 个热身样本、2 个测量样本。三组共核验 9 个前缀比较组（只有 3 个不同内容家族，其中 2 个计入延迟），冷/原始 store/SSD/staging 的 token 输出全部相等，包括热身。9 次 SSD 恢复合计实读 8,455,716,864 字节；另有 9 次 staging 恢复。已接受、完成和回收 I/O 数一致，未排空计数为零。

| 进程对 | 冷重算中位数 | SSD 中位数 | staging 中位数 | SSD 相对冷重算延迟降低 |
|---|---:|---:|---:|---:|
| 02，汇总 | 1.671354 s | 1.331837 s | 0.181519 s | 20.31% |
| 03，汇总 | 1.668872 s | 0.805315 s | 0.129889 s | 51.74% |
| 04，时间复验 | 1.672660 s | 1.311919 s | 0.180659 s | 21.57% |

三组都观察到作者原有 SSD 恢复比冷重算快，但 SSD 延迟存在明显波动，不能只报告第二组的最大收益。前两组会话中位数汇总预测为 f=1.670113 s、gSSD=1.068576 s、gMEM=0.155704 s。时间复验相对预测误差分别约 +0.15%、+22.77%、+16.03%。

这是既有缓存的单 token 成本机会，**不是新增策略收益、持续 decode 性能或 goodput**。时间复验使用相同 prompt/KV 家族，不能称为独立内容 held-out。单点结果没有导出为运行时曲线，不能外推到 2K/4K/8K，也不能混合此前不同上下文/预算的测点。

证据：`repeated-point.json` 保存全部测量值、配置/结果哈希；旧 discovery 样本不计入本轮汇总。

## 命令、测试及 GPU 账本

准确 argv、退出码见 `artifacts/prefix_io_v1/server07-p3-02/commands.json`；环境与命令说明见 `REPRODUCE_SERVER07_P3_RECHECK.md`。CPU 测试命令：

```text
.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py -q --import-mode=importlib tests/prefix_io_v1_pilot tests/prefix_io_v1_native_costs --junitxml artifacts/prefix_io_v1/server07-p3-02/cpu-tests.xml
```

本轮 CPU **37 passed，0 failed，0 skipped**；包含原有相关用例，不与以前轮次相加。P2 整套测试未重跑，29 文件哈希核验是保持不变的证据，不能代替新增 GPU 资格。

真实 GPU 作业 **7 次，均退出码 0**：一次数值诊断，六次 16K 采集。数值诊断成功采到差异，不等于 4K 输出相等测试通过。本轮 GPU 预算保护器记账 511.800790 秒；累计 **3176.327435 秒（52.94 分钟 / 8 小时）**，剩余约 **7.118 小时**。模型下载本轮 0，累计保守记账 19,422,798,722 字节，未重置预算。

结束时 GPU 0 MiB / 0%，无计算进程，所有作业会话已排空，active_reservation=null，/dev/shm 无残留命名 semaphore。磁盘剩余 9,899,057,152 字节，高于既有 8 GiB 采集保护线。本轮配对进程日志包含 resource_tracker semaphore 退出告警，原始日志保留；以作业排空与最终健康检查为实际收尾证据。无驱动/系统修改、无租机付费、无远端推送、无数据删除。

## 未解决问题及下一允许动作

仍在 P3。下一步是同资源预算的多长度标定及独立内容验证，在预注册的数值比较协议下区分原生冷/热差异与存储恢复错误；通过后才可将新成本表用于长前缀服务流。随后收集真实 SSD 读写与持续 decode 重叠、实际资源等待，并实施与验证 fixed/pressure 薄桥接。

此前短请求流未观察到 pending flush 阻塞，本轮单 token 成本测量没有推翻这一负结果。没有真实释放等待证据时不能推进 joint。若正常长负载仍无可重复目标问题，应按 P3 停止条件交付负结论，不能人为延迟 I/O 制造收益。P4–P7 尚未通过阶段门槛。
