# Server07 P3：辅助存储与独立长前缀验证（2026-09-29）

本轮在用户指定的 AutoDL westd:38819 上执行。结论：已授权的 linux_aio 路径可以继续真实 GPU 实验；独立 16256-token 前缀的缓存数值、成本预测和持续解码通过了本轮有界检查。P3 尚未全部完成，尚无研究策略收益结论。

## 授权、环境与版本

用户“允许，继续”授权系统盘专用目录 `/root/prefix-io-v1-validation`：上限 20 GiB、至少保留 8 GiB 空闲；记录在 `experiments/prefix_io_v1/configs/authorizations/system_disk_validation_20260929.json`。原 GPU 8 小时和模型下载 20 GiB 总预算沿用，未清零。未删除旧数据、下载新模型、修改驱动/系统或产生新租用。实际文件系统报告为 overlay；没有把它认定为特定物理 NVMe。

GPU UUID 为 GPU-f8744916-1693-fa6a-93b6-7f503c03459c，RTX 5090，驱动 580.105.08；Python 使用项目 .venv，Torch 2.11.0+cu130。Qwen2.5-7B-Instruct 的 ModelScope revision 为 16c174980d8a1492910551634b4969e69cdc2444。

| 依赖 | 固定提交 |
|---|---|
| py-kvcache | 3abba7a502d553f6e7e2e58b92086487e3395d7e |
| 作者 vLLM | 817a7e3124f817cd6e549581d3e5483207a753a4 |
| kvcache-experiments | 0e023a84a21246b9bbc06266fa8070397eccbdc9 |
| profiler | ec0d563bf68856df83c5824ac579700ec076b9e2 |

运行工作树仍为 py-kvcache-p2-aio 和 vllm-author-build。原 io_uring 默认值与其历史 EPERM 结果保持记录；linux_aio 仍须显式选择，没有自动回退。

## 实际改动

- `src/prefix_io_control/config.py`：向权限合同增加严格校验的可选辅助存储对象，旧配置兼容。P2 锁定的 29 个文件中仅该文件变化；其余 28 个、原缓存/reactor/观测桥及作者模型执行器未变。
- 新增 `experiment_storage.py`、`probe_auxiliary_storage.py`、`copy_calibration_to_aux.py`：授权路径、唯一 inode 空间统计、写入前后额度/剩余空间检查，真实 O_DIRECT AIO 探针和保留原件的带哈希复制。模型符号链接不计入新存储；硬链接只计一次。
- `acquire_native_aio_costs.py` 与参考/导出/独立成本分析器支持显式辅助结果路径，保留旧默认路径和数值门槛。
- 新增独立长文档准备、128-token 原生 GPU 缓存参考、限定验证许可及结果分析器；长回放脚本仅增加显式选择的验证分支。许可绑定曲线、内容、成本证据哈希与相同预算，不能作为生产/全域资格。
- 新增 25 个 CPU 测试，分别覆盖存储边界 15 项、长请求合同 10 项。实际差异见 `existing-files.patch`，新增源文件与哈希见 `source-lock.json`。

以上新增属于实验与证据设施。精确 Prefix Cache、原 LoadPlanner 准入、共享 staging、预加载、复制合并和原异步流水线保持。普通额度、fixed/pressure/interference/dependency_only/joint 尚未接入生产路径；off/shadow 边界未改。

## CPU 与实际文件 I/O

最终命令为：

```bash
AIO_CPU_GPU_GUARD_PATH=artifacts/prefix_io_v1/server07-p3-05/cpu-gpu-guard.json \
.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py \
-q --import-mode=importlib tests/prefix_io_v1/test_config.py \
tests/prefix_io_v1_pilot tests/prefix_io_v1_native_costs \
--junitxml artifacts/prefix_io_v1/server07-p3-05/final-cpu.xml
```

最终 **143 passed，0 failed，0 skipped**；中间轮次不重复累加，CUDA 初始化受 CPU guard 禁止。真实 CPU AIO 探针完成 3670016 字节写入与读回，逐字节一致，accepted/completed/reaped 均为 8，关闭后无在途任务。该探针是文件 I/O 测试，不能替代 GPU 缓存验证。

复制了 5760 个已有标定文件，共 5284823040 字节，源和目标 SHA256 匹配；原文件全部保留。

## 实际 GPU 结果

23 次预算封装启动：22 次真实模型运行成功，1 次在模型初始化之前失败。四个原标定点的 24 项缓存参考比较全部零容差通过；独立长前缀另有 6 项 top-5 ID/logprob/选定 token 比较通过。跨 batch 的 BF16 冷重算与 GPU 热缓存差异单列：4K 的历史选词差异再次出现，不放宽缓存等价性检查。

在新存储上重新取得 2K/4K/8K/16K 标定成本，正/反顺序各一轮、每路径两个新引擎，warmup 不进入统计。候选曲线 SHA256：

`13ea532044c2dcb5cb213ba17e885c274177704f71ac8b9f5ef134a5381f0e86`

SSD 插值交点 4240 只是稀疏曲线插值，不能当成实测稳定边界。旧曲线与旧 2K staging +26.27% 的未通过结果未改；本轮通过不能抹去该失败。

独立内容为新编写的天文台、温室、铁路文档，每族前缀 16256 token、请求 16257 token；一族 warmup、两族计量。内容 manifest SHA256 为 `a8a5f10a7f4596a2cb57fff9ba373c4f6a6c20a9398f44d6fed1ccddb76c0195`。各族前 16 token 与标定和先前 2K 内容不同。到达日程复用原作者种子 1703，不宣称独立到达轨迹。

独立成本门槛预先固定为绝对相对误差不超过 25%，未根据结果调宽：

| 路径 | 预测秒 | 实测中位秒 | 相对误差 |
|---|---:|---:|---:|
| 冷重算 f | 1.653444 | 1.643716 | -0.59% |
| SSD g_ssd | 1.270024 | 1.130644 | -10.97% |
| staging g_mem | 0.164055 | 0.171623 | +4.61% |

每路径 4 个计量值，原始样本全部保留；只支持该长度/内容/预算的有限检查，不是全域校准或置信区间结论。

随后运行两次连续请求：每轮 6 请求 × 128 输出 token，无逐请求缓存重置或人为暂停，原准入开启，观察为 shadow。**12 个完整 128-token 输出均与对应原生 GPU 热缓存参考完全相同**，两轮输出也一致，记录共 1536 个真实 token 时刻。参考采集另有诊断重置，不计入服务性能。

| 指标 | replay-02 | replay-03 |
|---|---:|---:|
| 含末尾 drain 的 cohort 秒 | 23.080844 | 19.053704 |
| 末尾 drain 秒 | 0.001958 | 0.002580 |
| 实际 SSD 读字节 | 4777443328 | 4777443328 |
| 实际 SSD 写字节 | 19267584 | 19267584 |
| 强制 flush 等待次数 | 0 | 0 |
| 前台 staging 申请失败 | 0 | 0 |

每轮保留 3048 个来源文件哈希；末尾 accepted=completed=reaped，outstanding/pending/ready/unreaped 均为零，fatal 为空，engine 关闭且进程会话排空。实际 GPU KV 2146959360 字节、staging 1073483775 字节；单 KV 组、max_num_seqs=1、上下文/批 token 上限 16400。

cohort 波动如实保留；这些数字不是两种策略对照、SLO goodput 或加速收益。本轮没有采集新 CUDA 时间线，不能以 I/O 字节或无等待推断真实 CUDA 重叠。并发上限为 1，也不能外推更高并发无阻塞。引擎有非目标 warmup，目标前缀在 cohort 前不在 GPU/staging。

## 失败与纠正

第一次 long-replay-01 因命令捕获文件名与资格 JSON 同名，资格文件被捕获结果覆盖，触发 `KeyError: curves_sha256`。失败发生于模型初始化前，13.295677 秒仍记入 GPU 预算。保留失败文件与日志；将捕获文件改为独立的 `*-command.json` 且独占创建，重发 `long-validation-permit.json`。后续使用新标签 02/03，未改变工作负载、预算、误差门槛或样本。原/有效计划和 `launch-failure.json` 均保留。

## 证据与预算

项目根为 `/root/autodl-tmp/prefix-io-v1-handoff/project`。本轮主证据在 `artifacts/prefix_io_v1/server07-p3-05/`；GPU 封装日志在 `experiments/prefix_io_v1/runs/server07-p3-aux-*/`；模型运行细节在辅助根 `runs/<label>/details/`。交付压缩包把这些辅助细节映射为 `auxiliary/runs/`，清单保留原服务器路径。未包含模型权重和批量 KV 文件。

关键证据：`commands.json`、`final-cpu.xml`、`aux-reference-result.json`、`long-reference-result.json`、`long-cost-result.json`、`long-service-result.json`、`final-health.json` 和 `workspace-audit.json`。全部实际命令及环境复现说明见配套 REPRODUCE 文档。

本轮 GPU 预算计费 1724.032804 秒（28.73 分钟），累计 6615.838822 秒（110.26 分钟），8 小时中剩余 6.162267 小时。新增模型下载 0 字节。结束检查 GPU 0 MiB、0% 利用率、无计算进程、无活动预算预留、无 sem.* 残留。打包前辅助根唯一文件占用 8267911168 字节（约 7.70 GiB），文件系统空闲约 21.49 GiB；项目数据盘空闲约 8.08 GiB，后续大结果须留在辅助根并检查额度。

## 下一允许阶段

继续 P3：事先冻结更高并发与正常 hit/miss 混合开发负载，按新预算配置重新校验数值/成本适用性，再检查持续 decode、写入、恢复、后续 miss、末尾 drain 和真实释放依赖；补齐 fixed/pressure 强简单基线前，须先完成安全的普通额度及 owner/freshness 接线验证。继续保留旧 2K 预测失败与波动，不用新样本反调旧验证门槛。

当前 AutoDL 环境的有界缓存流水线功能可行；研究问题是否在正常负载下成立、策略是否带来收益、P3 全部验收以及 P4–P7 仍未完成。
