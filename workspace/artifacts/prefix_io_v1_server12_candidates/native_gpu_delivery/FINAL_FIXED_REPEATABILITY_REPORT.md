# 固定三次 off 诊断实际交付（2026-10-04）

已按预登记完成全部三次真实 GPU 运行，未重试。原模型执行和缓存读取均完成，输出一致，原 shutdown 返回，三个 OS 会话均排空。但是三次中两次超过原固定成本上界，当前正常成本资格失败，未开启 shadow/on，未验证 P4 策略收益，不能据此宣称论文所需性能提升，也不能推出整个方法永久不可行。

服务器：root@connect.westd.seetacloud.com:24828；GPU RTX 5090，UUID GPU-a67d2f38-8e1b-29c3-7e5b-375e1410f8ac。工作区：`/root/autodl-tmp/prefix-io-v1-handoff/project`。Git HEAD：cc7898b1ba59d89ce7fdbb186ded21880f1adf08。本交付不包含登录密码。

## 真实结果及其范围

固定上界为 16,238,752 ns，预算为 13,171,328 ns。二者均为同一 full_decode_step / cuda_event_elapsed / nanoseconds 域的完整 GPU 步成本，不是 SSD 独立延迟、整请求时间或启动时间。

| 实际作业 | selected 完整 GPU 步 | 相对固定上界 | guard 墙钟 | 输出及排空 |
|---|---:|---:|---:|---|
| server12-c5-native-repeat-off01 | 16.266945 ms | 超出 0.028193 ms | 173.421246 s | 128 tokens，成功 |
| server12-c5-native-repeat-off02 | 15.458848 ms | 低于 0.779904 ms | 175.386692 s | 128 tokens，成功 |
| server12-c5-native-repeat-off03 | 19.185921 ms | 超出 2.947169 ms | 174.917104 s | 128 tokens，成功 |

三次均 exit=0、child_exit=0，无超时、无 guard 错误。全部 384 个实测输出 tokens、每次 128 个完整采样帧和事件见证均通过原校验；三次完整输出 ID 数组完全相同。每次真实 SSD read 917,504 bytes、1 个操作，原 shutdown 返回且会话排空。CPU I/O completion 不充当 GPU 非重叠或真实资源释放额度的证明；本轮资源释放 credit 为 false。

汇总分类：OBSERVED_THRESHOLD_CROSSING_VARIABILITY，n=3，覆盖 1 次、超界 2 次；这是小样本描述，不能推导总体尾界或因果结论。此前正常 off 的 16.893473 ms 超界反例继续保留；由于本轮启动观测/入口版本不同，不把旧记录混入 n=3。冻结阈值、种子、提示词、偏移、KV 布局和校准公式均未调整，heldout 未回填拟合。

仅在当前 GPU、模型、KV 布局、单文件候选及冻结负载条件下，完整步成本上界 16.238752 ms 大于 A-only 工程预算 13.171328 ms，差额 3.067424 ms（预算的 23.29%）。正确等价比较为增量加不确定性 3,125,744 ns <= 剩余额度 58,320 ns，结果为 false。不能将约 3 ms 的增量直接与约 13 ms 的完整步预算比较。该预算不是独立服务 SLO 或总体置信界。当前候选普通准入应 defer；原 native_progress 和最大等待期限推进分支未改变，不能把 defer 解释为永久无法执行或死锁。

## 实际改动与 CPU 验证

增量改动限于本轮私有实验入口、观测与证据交付：

- C/R 入口增加明确的进程内验证上下文及阶段计时。真实运行每个进程仅调用公开校准加载器 1 次，内部执行 7 次元数据/身份重检；完整 4,816 项源/资产字节核验仍在 before/launch/after 执行。公开 API 的默认完整重放保留。
- 新增固定三槽 protocol、prepare、supervisor 和 verifier。真实结果逐槽保存、无重试、无阈值重拟合，成本超界不丢弃；语义、生命周期、源锁错误仍中止。
- 用户持续 GPU 授权按原文记录，保留原 8 小时累计预算，无需逐轮重新请求授权。新增报告、只读成本复算及有界字节归档/本地核验工具。
- 原缓存引擎、模型执行器、精确 Prefix Cache、共享 staging、预加载、复制合并和异步流水线保留；未新增候选/策略。原观测器、计量器及核心数学函数不变，共同入口修复和研究策略分开。

服务器最终 CPU 测试实际通过 79/79、0 skip。AST 审计确认 22 个 runtime helpers 和 18 个 CPU routines 的原主体一致；execute_window 主体仅允许上下文元数据规范化。成本/记账/捕获数学函数不变，公开收据及资格验证沿用原校验。完整冻结源锁包含 4,816 项，保留旧 4,791 项原字节。源锁 SHA-256：
`4aab66888592a8fdf6544513003e9b7382f2194f3da2407b8991073a8e86ffef`。

本轮保存了两类冻结前 CPU 修复证据：测试中的旧 Windows 路径，以及旧 raw frame 的空 elapsed 占位字段需通过同 ordinal 的真实 event witness 读取。失败日志、修复说明和原文件在服务器保留；没有因这两项 CPU 修复产生额外 GPU 运行。新 verifier 增加真实旧 raw 回归检查。未修改旧资格结论、原公式或成本上界。

归档器本地 14 项 primitive 检查通过；本地字节核验器 22/22 CPU 测试通过。这些工具测试不是 GPU 资格证明。实际服务器归档和实际下载字节核验的最终回执另存 FINAL_FIXED3_DELIVERY_VERIFIED.json；本报告生成时尚未执行最终归档，不预先声称该项通过。

## 执行命令与实际日志

下面命令的工作目录均为 `/root/autodl-tmp/prefix-io-v1-handoff/project`；P 是 `.venv/bin/python`，D 是 `/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-c5-normal-repeatability-v1-20261004`，A 是 `/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004`。CPU 命令通过 run_logged 记录 COMMAND/STDOUT/STDERR/RESULT，并设置 CUDA_VISIBLE_DEVICES=''。正常 site 的 YAML/权限流程使用 -I 而非 -S。

```text
P -B -I -S -m unittest discover -s D -p test_*.py -v
P -B -I -S D/audit_process_entry_source.py
P -B -I D/prepare_repeatability_cpu.py --project-root ROOT --expected-ledger-sha256 7015729867cac4d285bf6277433fd395e8d721bcd7dc71d92e92bf3cd416637f
P -B -I A/repeat_runtime_authority.py --root ROOT --controller-sha256 823e73c4ac0b22544a20861d6a3afa04e689d92b8b3573ac9206bc5001be6252 --controller-bytes 44414
P -B -I -S A/prepare_fixed_scopes_parallel.py
P -B -I -S A/execute_fixed_repeatability.py --execute
```

CPU 对应实际日志为 D/CPU_FULL_TEST_04_*、CPU_ENTRY_SOURCE_AUDIT_01_*、CPU_FREEZE_CONFIG_02_*、CPU_ACTUAL_AUTHORITY_ALL_01_*、CPU_ACTUAL_SCOPES_ALL_01_*，均 exit=0。冻结执行 63.170145 s，授权核验 105.789523 s，并行独立槽证明 55.851550 s；这些 CPU 工作未增加原 GPU ledger。

实际 GPU wrapper 去掉 CPU 外层的 CUDA 屏蔽，调用：
```text
P -B -I D/run_fixed_repeatability.py --project-root ROOT --execute
P -B experiments/prefix_io_v1/scripts/run_gpu_stage.py --permissions-path D/EFFECTIVE_GPU_PERMISSION_offNN.json --label server12-c5-native-repeat-offNN --seconds 300 -- P -B D/run_p4_single_file_experiment.py --execute --config D/CONFIG_offNN.json
```
NN 依次为 01、02、03。原 guard 负责真实预约、300 s 执行限时、收尾及累计记账。D/GPU_SUPERVISOR_COMMAND.json、STDOUT/STDERR/RESULT 和三个 run 的 process/guard/details 保存了实际完整参数和输出；本节 P/D/A/ROOT 是阅读缩写，实际 COMMAND 文件保留绝对路径和参数数组。

成本准入复算命令为本机 Python 3.12 的 -B -I - 标准输入脚本，读取保存的真实校准数据，重用未修改的 analyze_paired、原估计器和 _calibration_a_budget；10 项来源前后 SHA 一致，无模型/共享库导入、无 GPU 操作。细节见 A/CURRENT_BINDING_ADMISSION_CPU_ANALYSIS.json/.md；其生成时只收到本轮第一条结果，最终三条以 D/REPEATABILITY_SUMMARY.json 和 ACTUAL_THREE_REPEAT_OBSERVATIONS.json 为准。

## 慢的原因与真实计时

本轮 guard 墙钟 173.4–175.4 s，相对此前正常 off 的 261.1 s 描述性减少约 33%。入口公开重放计时为 19.49–19.79 s，包含在完整进程验证的 24.07–24.47 s 内，不能相加；SDK 环境准备 1.58–1.65 s，LLM 初始化约 85.07–85.43 s，是已单独计时的最大阶段。日志有实际 Triton JIT 提示，但尚未将整个初始化阶段的每一部分隔离归因。实测整请求约 1.69–1.70 s，selected GPU 步约 15.46–19.19 ms。

入口减少重复全量重放已经在调用计数中得到真实验证，但单次旧/新墙钟差额不能全部因果归于该改动，也不能作为策略吞吐、TTFT 或尾延迟收益。CPU 源核验与 guard 外工作仍占总控制器墙钟。

## 预算、存储和下一允许阶段

三次原 guard 累计 523.725042 s（约 8.73 分钟）。原 8 小时 ledger 已累计 24,678.895239 s，剩余 4,121.104761 s（约 68.69 分钟），未重置预算。supervisor 包含 CPU 核验的总墙钟 1,189.626203 s（约 19.83 分钟），与 ledger 的 GPU guard 记账范围不同；AutoDL 实例计费可能还包括 GPU 开机期间其他时间，不能把 ledger 时间当作完整账单。

最终只读状态：无活动预约、无 compute applications，显存 0 MiB、利用率 0%；PRIMARY 可用约 49.76 GiB，无需为本轮证据扩大数据盘。没有下载模型、修改系统/驱动/已安装包、删除原数据或作者文件。模型/SDK/原实验数据保留在服务器；小型交付归档不重复打包模型、SDK、runtime-cache/private-storage 或旧归档。

下一允许阶段为 CPU 分析：审查固定成本覆盖失败和既有有限候选的准入/推进条件，核对工作负载条件及启动观测，保留全部反例。不得用挑选通过样本、扩大成本上界、任意放宽预算或新候选掩盖失败。若需要后续独立校准，应先形成明确、预登记的新验证设计；本轮不启动额外 GPU 试验。正常 shadow/on 和 P4 策略收益对照仍阻塞，本报告不宣称 P4 完成。目前没有待运行 GPU 作业，证据备份完成后可切回无卡模式以减少支出。

## 证据位置

服务器 D 保存固定 protocol、4,816 项 COMMON_SOURCE_LOCK、配置/授权、三条 DIAGNOSTIC_RESULT、REPEATABILITY_SUMMARY、ACTUAL_THREE_REPEAT_OBSERVATIONS 及所有 CPU/控制器日志。
三个真实 run 位于 ROOT/experiments/prefix_io_v1/runs/server12-c5-native-repeat-offNN，完整 raw 为 details/p4-single-file-runtime-result.json。
A 保存持续授权、旧校准及旧正常失败归档、FINAL_FIXED3_GPU_LEDGER_SNAPSHOT.json、FINAL_FIXED_REPEATABILITY_STATE.json、成本准入复算和本报告。

本地本报告：`C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server12_candidates/native_gpu_delivery/FINAL_FIXED_REPEATABILITY_REPORT.md`。
本轮计划交付 A/REPEAT_REPEAT_DELIVERY_fixed3-final.tar.gz 及 MANIFEST/RESULT；最终真实回执、归档 SHA 和本地扁平路径映射见本地 FINAL_FIXED3_DELIVERY_VERIFIED.json 和 repeat_fixed3_verified/LOCAL_REPEAT_DELIVERY_BYTE_VERIFICATION.json。原服务器路径在 manifest 和映射中保留。完整 4,816 项源/资产字节校验在服务器实际执行；本地只核验归档包含的源和证据，不声称本地重新读取未打包的模型/SDK。

