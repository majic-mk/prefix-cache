# Server07 P3：原生 flush 与资源所有权复测

本轮结论：**工程路径继续通过，研究收益尚未成立；P3 未完成，P4–P7 未开启。** 新增可关闭的紧凑诊断，实际完成 4 次 GPU 运行。40/40 个请求的 128-token 完整输出与已冻结的原生参考精确一致；两次混合读写均没有重现 pending flush。不能把“诊断运行通过”写成“已验证 flush 原因”或“方法有提升”。

## 实际修改与边界

- 作者 vLLM `offloading/scheduler.py`：原生集合更新之后增加 5 个观察点（恢复目标块、新分配块、抢占、全部请求结束、reset）；默认探针为 None。原集合、调用顺序、complete_store、模型执行、worker.wait 不变。
- 新增 `src/prefix_io_control/native_flush_probe.py`：同一所有者线程观察实际 BlockPool 分配代次、父任务及部分源块状态。最多 4096 个块代次、64 个父任务、每父任务 8 个源块、128 条记录；缺失、截断和错误显式报告。不会给未知释放量记账。
- 新增实验安装器 `flush_cause_worker_probe.py`；`run_concurrent_pilot.py` 增加显式 `--flush-cause-probe`，省略参数不安装新探针。没有新增配置键、元数据协议或第二套队列。
- 新增分析脚本与 2 个测试文件，验证真实父任务 ID 匹配，歧义不推测。
- 共同兼容修复与新观察补丁分开：本轮观察补丁见 `patches/prefix_io_v1/observation/0003-native-flush-diagnostic.patch`。py-kvcache reactor、LoadPlanner、预加载、共享 staging、融合复制及异步执行均未修改。fixed/pressure/dependency/interference/joint 原生额度仍未安装。

生命周期仍是：原生父任务完成 → worker 回报 → scheduler 汇总所有 worker 的完成确认 → 原 complete_store 与 fence 清理。**写盘完成不是“GPU 源块立即空闲”**；活动引用、代次和 fence 必须独立核验。本轮只观察，不提前确认或释放。

## 实际执行与结果

最终 CPU 命令（实际环境见 execution-environment.json）：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/third_party/work/py-kvcache-p2-aio:$PWD/src:$PWD/experiments/prefix_io_v1/scripts" \
.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py \
  -q --import-mode=importlib tests/prefix_io_v1/test_config.py \
  tests/prefix_io_v1_pilot tests/prefix_io_v1_native_costs tests/prefix_io_v1_observer \
  --junitxml artifacts/prefix_io_v1/server07-p3-07/final-cpu.xml
```

**260 通过、0 失败、0 跳过；CUDA 初始化被禁止。** 包括去除观察钩子后原生调度 AST 与修改前一致、关闭路径、观察故障不改变原调用、长分配、代次变化、多父确认及歧义匹配。前置 253 项测试属于本轮早期版本，不与最终 260 项相加。

GPU 使用原预算包装器：

```bash
.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py \
  --label <冻结标签> --seconds 240 -- \
  .venv/bin/python experiments/prefix_io_v1/scripts/run_concurrent_pilot.py <冻结参数>
```

四条完整实际命令、参数与环境分别保存在 `run-plan.json`、`*-launch.json`、`execution-environment.json`。不是上述占位符命令直接运行；本次具体标签如下：

| 标签（均以 server07-p3-flush- 开头） | 深度 | 新观察 | 含末尾 drain 的 cohort 秒 | ITL p95 毫秒 | pending flush 秒 |
|---|---:|---|---:|---:|---:|
| allhit-off-01 | 4 | 关闭 | 20.4090 | 33.75 | 0 |
| allhit-on-01 | 4 | 开启 | 20.0213 | 31.32 | 0 |
| d8-mixed-on-01 | 8 | 开启 | 26.4581 | 68.04 | 0 |
| d8-mixed-on-02 | 8 | 开启 | 24.3897 | 59.86 | 0 |

四轮均为真实 RTX 5090、Qwen2.5-7B BF16、作者 vLLM + py-kvcache linux_aio；实际 GPU KV 2,146,959,360 B、staging 1,073,483,775 B，max_num_seqs=2。冻结的模型、成本表、到达序列、初始 SSD、准入和输出长度保持相同。没有新增模型下载、人工延迟或逐请求 reset。所有 AIO 请求均完成并回收，源 SSD 3048 个文件每轮校验不变，engine 与进程会话全部排空。

新观察开启的三轮记录了真实分配与 98 个父任务退出（22+38+38）；147 次采样源块状态均为同代次且仍有活动引用。这是有界样本，不是所有源块的统计，也不证明稍后不能释放。没有给这些状态计入可立即释放字节。

## 对方法优势的影响

1. 本轮两次深度 8 混合负载没有 flush 事件，**GPU flush 原因分支未被覆盖**；五个分支只具备 CPU/结构验证。本轮无法给上轮两次等待补写触发原因。
2. 两轮最大 token 间隔为 1.3786、1.3330 秒，都与请求 c2-4 的冷预填充窗口重叠。非预填充窗口也有 I/O 与较长 token 间隔同时出现；这只是时间关联，不能排除计算、批形状、调度和缓存路径等混杂，更不能作为干扰额度的因果收益。
3. 关闭/开启观察的单次耗时差不能作为策略收益，也不足以证明 <2% 开销。此次开启观察的混合结果不直接并入上轮未开启观察的性能基线。
4. 上轮深度 8 相对深度 4 的描述性改善仍属于原参数调节；深度 2 的 SSD 成本误差 +39.37% 和早先 2K 成本失败继续保留，未放宽 25% 门槛。
5. 当前负载下，依赖排序的稳定投入理由变弱；干扰额度仍是待验证方向。尚无“有效提升”“论文收益”或普遍不可行的结论。

## 预算与版本证据

- 本轮 GPU 包装器计费时间 **364.2995 秒（约 6.07 分钟）**；累计 **8560.2893 秒（2.378 小时）/ 8 小时**，余 **5.622 小时**。4/4 正常退出、无超时、无悬挂 reservation。
- 辅助目录已用约 **18.71 GiB / 20 GiB**，文件系统空闲约 **10.48 GiB**。下一次混合回放需保守预留 3 GiB，既超过目录额度，也会低于 8 GiB 空闲底线；未启动新一轮、未删除数据、未扩大授权。
- 原 99 项源锁中 98 项未变，改动已有实验 driver；另增加作者 scheduler 修改和 5 个新文件，最终源锁 **105 项**。
- 审计限制：最初 102 项预运行锁沿用旧选择列表，遗漏新增修改的 scheduler。预运行 before 副本与观察 patch 已保留，重建后与 GPU 运行结束保存的 scheduler 逐字节一致，见 runtime-source-reconstruction.json；最终锁已补齐。
- GPU 后唯一作者代码调整是把原类文档字符串恢复到类首行，所有函数 AST 相同；GPU 使用的精确源码另存 gpu-tested-scheduler.py。最终 CPU 测试在此调整后通过，不虚称最终逐字节源码又跑了一次 GPU。

主要证据：`artifacts/prefix_io_v1/server07-p3-07/` 中的 qualification-summary、qualified-analysis、remaining-latency-evidence、next-mixed-storage-gate、final-cpu.xml、GPU ledger、source-lock、version-lock 与各 launch 文件。完整原始 GPU 结果和时间线位于 `/root/prefix-io-v1-validation/runs/server07-p3-flush-*/details/`；交付包保留 JSON/trace/log，不复制模型和 KV 数据。

## 下一允许阶段

继续 **P3 的 CPU 接口开发和现有时间线归因**，准备固定额度/压力基线的薄桥接，并保留 mandatory 独立推进及关闭回原路径。仍不能越过 P3 宣称进入 P4。相同混合负载的额外 GPU 重复受存储合同阻塞；无需换服务器或改驱动，也不应靠人为慢 I/O 制造研究优势。后续若扩展 GPU 对照，先具体解决存储预算/归档授权，再冻结正常负载与停止规则。
