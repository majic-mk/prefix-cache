# Server12：真实 GPU 标定与成本准入交付

2026-10-04，connect.westd.seetacloud.com:24828。
服务器项目：/root/autodl-tmp/prefix-io-v1-handoff/project。
状态：真实原生执行和有限条件成本验证通过；当前单文件候选超预算，应延后。
这不是 P4 正常 on 在位验证或 P5 性能通过。

## 实际 GPU 结果

唯一作业 server12-c5-native-common-cost-gpu01 使用 RTX 5090，
UUID GPU-a67d2f38-8e1b-29c3-7e5b-375e1410f8ac，驱动 580.95.05。
原 guard 内执行 A0/B0/B1/A1/A2/B2 六个 fresh 原模型进程；
各窗口完整生成 128 tokens，合计 768 tokens 和 768 个真实完整 CUDA 步记录。
三个配对的完整模型输出一致；六个进程正常 shutdown。
guard exit=0、child_exit=0、timed_out=false、session_drained=true；
结束前后及最终复查 OS 会话成员均为空。

原累计 8 小时 GPU 实验预算中，本次实际计入 764.6123794279993 秒，
不是 1200 秒限时，也不是 controller 的 17.60 秒派发时间。
累计 23894.049682236742 秒，剩余 4905.950317763258 秒。
本次只有一个实际 GPU reservation：bbe0d5e7cc084147985d4d413301e2ce。
没有重试、模型下载、数据删除、系统/驱动/已安装包改动。
云实例租赁墙钟费用与实验账本不同；CPU 后处理不消耗实验 GPU 账本，
但 GPU 模式的实例仍由平台计费。当前 GPU 利用率 0、空闲显存 32110 MiB、
无计算进程；数据盘剩余 53714223104 bytes（约 50.03 GiB），无需本轮扩盘。

## 成本、预算与真实边界

预登记有限条件保持不变：BF16 eager/Triton、batch1、active decode1、
prefill0、context144、offset16、SSD 单文件 917504 bytes、1op、existing I/O=0。
两个 calibration 配对决定上界；第三个 heldout 配对只验证，没有参与预算或重拟合。

| 配对 | A 基线完整 GPU 步 / ms | B 带原生动作完整 GPU 步 / ms | B-A / ms |
|---|---:|---:|---:|
| pair-0 calibration | 13.171328 | 16.000256 | 2.828928 |
| pair-1 calibration | 13.054688 | 16.238752 | 3.184064 |
| pair-2 heldout | 12.964064 | 16.097504 | 3.133440 |

原冻结公式得到：
baseline 13.113008 ms + 配对增量 3.006496 ms + 标定残差 0.119248 ms
= 整步联合成本上界 16.238752 ms。
实际 A-only 工程步预算为 13.171328 ms，差额 3.067424 ms；
fits_calibration_a_only_budget=false。
16.238752 ms 包含基线整步，不能描述成纯 SSD 或新增复制耗时。
本次没有隔离证明 SSD、CPU 竞争或观测本身各自贡献了多少。

heldout B 实测 16.097504 ms；上界余量 0.141248 ms，欠预测为 0。
native_execution_verified、conditional_cost_cell_qualified、heldout_covered 均为 true。
这些只证明当前有限条件的原生执行和一次留出覆盖，不能外推 SLO、
未知文件大小/负载、其他 GPU、概率覆盖率或论文收益。

实际公开 Q.load_verified_single_file 从真实源、原始录制、guard、
前后源锁和累计账本重放成功，返回 ExactSingleFileReceipt。
REAL_CANONICAL_RECEIPT_SUMMARY.json 只是摘要，不能代替 typed receipt。
另一次 CPU 合同投影调用原 P4Policy.issue_preview，真实返回 defer；
execution=cpu_mock 被原 API 拒绝为 native_fallback。
投影明确不是 live owner capability、GPU on 安装或已影响实际 dispatch。
正常 runtime binding、full-runtime cost、actual on、production、
strategy effect 和 performance benefit 的资格仍为 false。

## 实际改动

保留服务器和本机所有旧源码与原始数据；原 G/common 64 个 Python 文件
和原公共模型/缓存引擎路径未变。当前源完整 4756 项前后核验通过。
迁移准备的新 site_sdk_binding 显式绑定已有 580 SDK，与实际 SDK helper
和 Ninja 接线；源闭包、授权、现场 GPU 信息及作业路径按新域冻结。
数值公式、A-only 预算、未知条件拒绝、缓存规则、原模型执行、
preload/staging/异步流水线和 shutdown 均未重写。

本次 GPU 后处理只新增独立 CPU 辅助脚本：
issue_real_native_binding_v2.py、audit_final_runtime_state.py、
diagnose_real_cost_budget.py，以及实际命令、证据和交付文件。
v2 只把 tuple 在原 CLI 写入 JSON 后成为数组的结果，按完整严格 JSON
序列化比较；没有删字段、修改数值、放宽条件或重跑 GPU。
本机 14 个此前已修改 tracked 文件的字节复查全部保持不变。
正常请求的新卡接线在独立 normal_native_cpu 目录继续 CPU 准备，
不属于本次 GPU 作业通过范围。

## 实际命令与测试

以下命令在服务器项目根目录执行；完整 argv、stdout、stderr、返回码
保存在本目录 *_COMMAND.json/*_STDOUT.log/*_STDERR.log/*_RESULT.json。

| 命令记录 | 实际结果 |
|---|---|
| 01_BIND_FRESH_USER_SCOPE | CPU 绑定新卡单次六窗口有限授权通过，29.223 秒 |
| 02_PREPARE_REAL_NATIVE_PLAN | CPU 失败：-S 隐藏已有 PyYAML；没有 GPU 作业 |
| 03_PREPARE_PLAN_WITH_EXISTING_SITE_PACKAGES | 去除 -S、使用已有包，通过；没有安装包，4.130 秒 |
| 04_FREEZE_SINGLE6_SCOPE | 六窗口范围和原 guard 参数冻结通过 |
| 05_ORIGINAL_GUARD_LAUNCH | 仅派发一次原 guard；GPU 结果以上述 result.json 为准 |
| 06_COMPLETE_GUARD_AND_FULL_SOURCE_AFTER | 完整源 4756 项、唯一结束事件和 guard 审核通过，18.299 秒 |
| 07_REAL_RAW_SERIALIZE_AND_NATIVE_COST_VERIFY | 全量真实 raw 重序列化与原成本公式审核通过，18.914 秒 |
| 08_ISSUE_REAL_PUBLIC_CANONICAL_RECEIPT | CPU 失败：tuple/list 的 RAM 对象比较；原失败日志保留 |
| 09_PUBLIC_CANONICAL_RECEIPT_WIRE_COMPARISON_V2 | 严格完整 JSON 比较与公开 typed receipt 重放通过，38.260 秒 |
| 10_FINAL_REAL_GUARD_LEDGER_SOURCE_RESOURCE_AUDIT | 账本、OS 会话、22 项直接冻结来源和现场资源复查通过，0.354 秒 |
| 11_REAL_RECORDED_COST_BUDGET_POLICY_DIAGNOSIS | 真实证据算术及原策略 defer CPU 合同回放通过，18.972 秒 |

CPU 迁移的 121/121 项针对性测试、4779 项旧来源迁移审核见
migration_audit/FINAL_MIGRATION_CPU_REPORT.md；旧 normal 100 项 CPU
历史结果未冒称本次重跑。以上两个后处理/计划失败均为 CPU 辅助层，
不是模型/GPU 失败。一次观测请求因返回过大的 JSON 超过工具输出上限，
仅导致本地显示解析失败，未修改原始记录或运行进程；之后改用紧凑观测。

主要实际 GPU 命令：
```bash
.venv/bin/python -B experiments/prefix_io_v1/scripts/run_gpu_stage.py \
  --permissions-path artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004/EFFECTIVE_GPU_PERMISSION.json \
  --label server12-c5-native-common-cost-gpu01 --seconds 1200 -- \
  .venv/bin/python -B artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004/run_native_cost_experiment.py \
  --execute --config artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004/NATIVE_COST_CONFIG.json
```
这是已用完的历史命令，不能作为新运行或重试授权。
CPU posthoc 命令：
```bash
.venv/bin/python -B -I artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004/issue_real_native_binding_v2.py --project /root/autodl-tmp/prefix-io-v1-handoff/project
.venv/bin/python -B -I artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004/diagnose_real_cost_budget.py --project /root/autodl-tmp/prefix-io-v1-handoff/project
```
输出均追加且拒绝覆盖，不应盲目重跑已经成功的写证据命令。

## 证据定位与下一阶段

新标定目录：
artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004。
原始 GPU 目录：
experiments/prefix_io_v1/runs/server12-c5-native-common-cost-gpu01。
本目录保留 CPU 后处理命令、真实账本快照和最终现场审计。
GPU_EVIDENCE.tar.gz 与 GPU_DELIVERY_MANIFEST.json 将归档原始 JSON/log、
新域源码和锁、权限/授权、验证/诊断及必要共同源码；
不打包大型模型、SDK 二进制、私有 JIT/缓存副本，这些全部保留在服务器。
本机下载后需以逐文件 SHA 清单复核，下载核验另保存独立证据。

关键 SHA-256：
- SITE_SOURCE_LOCK.json：1c0e7ccaa24aef878ea513df6744ff9037c00f02329b95793801b8a05b3ac5c0。
- result.json：250194dda58bb7f47c10e49c566e7630a682aba6d2889c0c2815b5a6ae9f6cad。
- NATIVE_COST_VERIFICATION.json：e5260588b846f7d90b163ecc7d37ed773511407a20d6eec1f20de3c48c57e277。
- REAL_COST_BUDGET_DIAGNOSIS.json：a6412bdc1fa3d52a628d02c1b42b63cf5926ee4ac3fcac28d51e108daa899932。
- 完成后账本快照：f7b749aff3744b65f146d7c7fa44cee813d2a7b9bd1b7c24968e150aa7216e77。

现在允许继续 CPU 正常接线、来源冻结和资格准备；GPU 已可切回无卡模式。
当前有限条件应 defer，而不是调宽步预算来取得 issue。
后续正常 off/shadow/on 可检验延期策略、off 回退、真实观测成本和生命周期；
其 GPU 授权必须另定范围，本次单次 single6 授权已用完。
即使正常资格通过，也需真实请求流比较才可讨论性能提升。
当前没有证明可达到论文收益要求，也没有证明整个方法在所有负载上不可行。
