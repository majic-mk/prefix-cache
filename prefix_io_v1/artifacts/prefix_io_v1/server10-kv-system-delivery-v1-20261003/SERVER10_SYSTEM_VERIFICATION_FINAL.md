# 新服务器有限系统验证交付

2026-10-03，`connect.westd.seetacloud.com:12351`。**这台服务器可以继续 GPU 实验，现有引擎的有限正确性与生命周期验证通过，无需再换服务器或镜像。**这不代表 P4 全部完成，也不代表研究策略已有性能收益。

实际环境：RTX 5090，驱动 580.95.05，Python 3.12.3、Torch 2.11.0、Triton 3.6.0、Transformers 4.57.6，使用现有 Qwen2.5-7B-Instruct 和 CUDA 13 私有 SDK 接线。服务器项目仍为 `/root/autodl-tmp/prefix-io-v1-handoff/project`。完整版本、能力归属及前六轮证据见同级上层 `system_validation_v1/` 交付；服务器对应 `artifacts/prefix_io_v1/server10-system-validation-v1-20261003/`。

## 实际 GPU 结果

| 作业 | 实际结果 | 原 guard 秒数 |
| --- | --- | ---: |
| `server10-g3-reference-native-01` | 原生冷/热参考通过 | 126.536 |
| `server10-g3-reference-paired-01` | SSD/staging 参考通过；父进程校验问题经 CPU 后处理修复，原失败保留 | 126.828 |
| `server10-p4-native-off-01` | 合成 KV 的原生 CUDA/AIO 四阶段通过 | 48.550 |
| `server10-p4-native-shadow-01` | 同四阶段及只观测路径通过 | 48.930 |
| `server10-g2-normal-off-01` | 冷/重复各完整生成 128 tokens | 142.536 |
| `server10-g2-normal-shadow-01` | 冷/重复各完整生成 128 tokens，采集 256 帧 | 141.863 |
| `server10-kv-byte-populate-01` | **整体失败，exit 78**：字节检查成功，但最终内外请求 ID 对账错误 | 128.298 |
| `server10-kv-byte-populate-02` | 修复后完整通过，发布凭据、源检查和关闭均完成 | 127.235 |
| `server10-kv-byte-paired-02` | 新进程真实 SSD 恢复及 staging 命中完整通过 | 129.979 |

合计 **9 次真实 GPU 作业，8 次 guard/child 成功，1 次失败**，无超时，9 次 OS 会话均自然排空。未执行 `paired-01`。失败耗时正常计入预算，未改写为成功。

本机累计原 guard 时间 **1,020.754 秒（17.01 分钟）**；跨机器原 8 小时预算累计 **18,829.894 秒**，剩余 **9,970.106 秒（约 2.77 小时）**。该口径包含受 guard 管理的进程时间，不是 CUDA kernel 时间，也不等于 AutoDL 开机账单。最终检查无 active reservation、无 GPU 计算进程；归档前 PRIMARY 可用 **10,664,869,888 B**，高于原 8 GiB 底线。

## 已得到的证据及边界

| 能力 | 本次实际证据 | 适用边界 |
| --- | --- | --- |
| 现有模型与执行器可用 | G2 共 4 请求、512 生成 tokens；重复前缀命中 112 tokens | 完整输出资格，尚非持续负载性能实验 |
| 缓存参考输出一致 | 6 个 cached 对照的返回 token、top-5 及 logprob 完全一致，最大差 0 | 冷重算与 GPU-hot 的 top-5 差异另行保留，不声称两者逐 bit 一致 |
| 真实模型 KV 字节往返 | 3 个固定前缀、每个 128 个缓存 tokens，24 文件，共 22,020,096 B | 单布局、单卡、固定请求；不能外推所有模型和并发 |
| 写入与计算前恢复 | populate-02：24 次 D2H、24 次 SSD 发布、24 次 H2D，以及 3 次计算前校验通过 | 在原 CUDA event 完成、原 CQE 和原模型计算边界观测 |
| 新进程 SSD 与 staging 恢复 | paired-02：24 次完整 payload 读取、48 次 H2D，以及 6 次计算前校验通过 | 实际 g_ssd 经 shared 路径，随后 g_mem 经 cache 路径，均绑定原请求 |
| 正常生命周期 | 两轮原 shutdown 返回；AIO accepted/completed/reaped 分别为 24/24/24、72/72/72，均 closed/drained；所有在途计数为 0 | 不据此推断提前物理释放信用 |
| 紧凑观测 | G2 shadow 有真实帧；字节诊断记录两轮各 78 条 | 原报告的完整 GPU collector/时钟资格仍未闭合 |
| 主动研究策略收益 | **未验证** | 本次诊断 LoadPlanner=off；生产成本、释放信用、策略效果及诊断时延可用标志均为 false |

字节诊断在原完成点额外读回 GPU 数据，并比较完整 bytes，不仅比较哈希或 Future 状态。它带来额外同步，**这两轮耗时不得用于成本拟合、干扰模型或性能收益结论**。观察器不增加原引擎的完成队列消费或资源释放操作，不替换缓存引擎或模型执行器；退出后恢复私有绑定。

## 改动与修复

本轮共同兼容修改包括新机 GPU UUID/SDK/权限接线，以及 Python 3.12 源码校验的 CPU 后处理适配。增量均保存在 `artifacts/prefix_io_v1/server10-*`，作者源码、系统、驱动和安装包未改；未下载模型、未删除数据。

新增 `server10-kv-capture-v1-20261003/production_kv_capture.py` 在原数据通路增加有限诊断观察。`server10-kv-diagnostic-v1-20261003/` 保存第一次失败版本，`server10-kv-diagnostic-v2-20261003/` 只修正对账并使用全新作业/缓存目录。

错误原因：作者 vLLM 内部使用 `external_id + '-' + 8 位随机十六进制字符`，外部输出仍使用原 ID。v1 错把两者直接比较。v2 对冻结作者源码作 SHA 校验，严格全字符串匹配，检查双向唯一、完整 129-token 输入 SHA、原生 job、每请求 8 个文件及实际 SSD/cache 角色；相同输入也不能交换两条路径。没有关闭请求号随机化、放宽字节容差或改写失败凭据。由于失败发生于原外层发布凭据生成前，修复后完整重验了两次。

服务器相关 CPU 检查累计 **81/81 通过**：基础适配 48 项、KV 接线 v1 8 项、字节观察 14 项、ID 修复 v2 11 项。v2 在真实失败 JSON 上重现旧错误并验证修复，反例覆盖错误前缀/后缀、重复与外来 ID、缺失观察、完整输入不符和路径错配。最终源码锁完整核验 **4,112 项**；SHA-256 为 `a064a9c17ebfba6b3eed459af7313af097569da7ed288b59c3851be7ef58f794`。

## 执行命令与原始证据

实际 argv、环境覆盖、stdout/stderr 和退出码分别在交付目录的 `*_COMMAND.json`、`*_STDOUT.log`、`*_STDERR.log`、`*_RESULT.json`；GPU 原 guard 命令和结果在每作业根目录 `result.json`。修复版核心命令如下，已消费作业名，不能直接重放：

```sh
cd /root/autodl-tmp/prefix-io-v1-handoff/project
.venv/bin/python -B -I -S artifacts/prefix_io_v1/server10-kv-diagnostic-v2-20261003/run_server10_kv_diagnostic_v2.py --mode populate --preflight
.venv/bin/python -B artifacts/prefix_io_v1/server10-kv-diagnostic-v2-20261003/run_server10_kv_diagnostic_v2.py --mode populate --launch
.venv/bin/python -B artifacts/prefix_io_v1/server10-kv-diagnostic-v2-20261003/run_server10_kv_diagnostic_v2.py --mode paired --launch
```

本目录 `ACTUAL_NINE_GPU_RUN_SYSTEM_SUMMARY.json` 汇总全部九个 guard 引用、实际 KV 记录计数、请求映射、关闭计数和预算。原始交付包 `KV_EVIDENCE.tar` 与 `KV_EVIDENCE_MANIFEST.json` 包含三次 KV 作业、失败现场、源锁/范围、实际日志及两组各 24 份生产者字节文件；本地接收后逐文件验 SHA。前六轮单独保存在 `system_validation_v1/SERVER10_SIX_JOB_EVIDENCE.tar.gz`，共 164 份原始证据。

## 下一允许阶段

历史 P3 有限 pilot 已闭合，其 **U 优于 F16/P4/F8/P512** 的结果继续保留，本次新机资格通过不会把旧负结果改成提升，也不把 P3 全部归零。

下一项应在当前机器、同一执行器上，恢复不带字节诊断同步的正常路径，以完整 128-token decode 做无 I/O 与合法 SSD-only preload 的 ABBA 小配对，先核验完整 step、时钟、实际负载和最终排空，再取得该条件成本。随后才进入共同固定额度及独立 U 的策略对照。新作业须继续采用有限范围、原累计预算与空间底线；不能复用本次已经消费的作业名。

P4 仍需生产成本/ETA、真实资源释放依赖、零普通额度下 mandatory 进展、主动 D/I/J 及关闭回退的真机证据。缺少释放证据时不计提前释放信用。**当前结论是系统具备继续实验的条件，方法有效性尚未得到证明。**
