**U 与 F8 + D_R 的真实模型对照交付，2026-10-05**

相邻 U04 → F8 + D_R01 单对真实 GPU 实验已完成。组合耗时增加 16.98598%，D_R 实际 kernel 改序 0 次。最终类别为 4：方法组件接入并实际执行依赖检查，但当前负载没有形成可执行 D_R 排序机会（NOT_EXERCISED）。不能证明 D_R 加速，也不能把组合退化归因于一次实际依赖改序。无反向重复、I/J 或完整数据集。

| 指标 | U04 | F8 + D_R01 |
|---|---:|---:|
| 首个预定到达到全部完整输出，秒 | 28.295357870 | 33.101601666 |
| 到原生 I/O 排空，秒 | 28.296033146 | 33.104073992 |
| 末尾排空，毫秒 | 0.672843307 | 2.470403910 |
| 输出吞吐，tokens/s | 45.237102351 | 38.668823730 |
| 完整守护作业墙钟，秒 | 137.935962940 | 142.687758802 |
| 平均 TTFT，秒 | 11.183555546 | 13.584381824 |
| 平均请求完成延迟，秒 | 15.511046888 | 18.223070172 |
| 各请求 ITL P95 的平均，毫秒 | 61.314219721 | 93.367538620 |
| 真实 SSD 读字节 | 6,251,872,256 | 6,252,789,760 |
| 真实 SSD 写字节 | 1,896,480,768 | 1,896,480,768 |
| 原生实际接受 H2D 字节 | 5,593,104,384 | 5,593,104,384 |
| 完整输出 | 10 × 128 tokens | 10 × 128 tokens，逐 ID 相同 |
| native/AIO/OS | 全部闭合 | 全部闭合 |
| 实际 SSD 恢复改序 | 0 | 0 |

耗时降低比例 (T_U - T_D_R) / T_U = -0.169859798842，即 −16.98598%。完整作业墙钟另报告约增加 3.444929%。自身观测、预测加载和调度开销均计入，没有事后扣除。独立标定费用另列，无人为摊销。只有单对描述结果，无统计显著性和 SLO 声称。ITL 列是各请求 P95 的平均，不是全局 P95。

**机制和覆盖缺口**

真实最终 journal 有 1,314 个策略窗口：753 次 no_observed_blocked_target、560 次 no_proved_legal_closure:no_witness、1 次 native_progress_before_policy。这是窗口数量，不是请求数。纯 SSD 排序窗口 0、H2D 窗口 1、未知恢复 ETA 窗口 561；父任务 SSD 队列接受收据 0，实际 before/after 改序集合和 kernel 确认逆转集合均为空。

实际依赖示例：窗口 2 依赖父任务 27，完整就绪闭包和已知 ETA 父任务集合为空；工作 ID [26,59,d2h] 前后相同，按 no_witness 回退。窗口 3 的 H2D [27,401..404,h2d] 原生可推进，保留原进展。没有把 D2H/H2D 的事件当 SSD 改序。

自然预加载确实产生约 6.25 GB SSD 读，但没有形成当前 D_R 接线所处理的合法、具备预测的恢复父任务 SSD 候选。共享 staging 中已就绪的恢复表现为 H2D，mandatory/continuation 保留原路。SSD 读取、pending flush 或 enabled=true 都不是 D_R 激活证据。

独立 CAL07 的完整单家族开发输入正常完成 10×128 tokens 并排空，但只有 1 条 H2D 闭包样本（copy_ready=2），没有算法要求的 4 条同几何、同条件的纯 SSD 测量。严格 loader 核验真实设备、源码、数据和自然关闭后进入限定 coverage-missing 回退。方法 forecast_disabled=true、calibration_present=false；forecast_calls=0、forecast_fallbacks=3。没有空成本表、CPU mock、假预测、资格布尔值绕过或效果结果拟合。

F8 和 D_R 自身依赖检查开销真实运行。组合变慢可能包含固定额度、检查开销和波动，没有 F8 消融，不能归因于依赖排序本身。系统在该长负载上功能及生命周期可行；本轮没有验证研究组件的性能优势，不能代表完整 D/I/J 或论文级提升。

**冻结实现、负载和初态**

- 已有 Qwen2.5-7B-Instruct/BF16，模型资产目录 models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444；没有下载新模型。
- 作者执行器 third_party/work/vllm-author-p4-02-cpu，保留原模型执行实现和异步流水线；加载路径与全部源码 SHA 在 source lock/adapter 证据中。
- 原 P316 runner SHA-256 248c901e6bfd2e7fbea3e210623f7556f9fef9c99d66c39cf50daf6ef5b0f26e；正式 mixed_readwrite SHA-256 1128f0b4c5481837f87c689e29f1b0f92fc80d29d8d13573fe08714bef58c197。
- 原 10 条请求，最终 input_ids 每条 16257，直接传 token IDs，无额外模板/截断；每条完整输出 128。家族顺序 [0,3,1,0,4,2,3,1,4,2]，外部预定到达时序逐值相同，实际发送阻塞计入排队。
- GPU KV 2 GiB、CPU staging 1 GiB、最大并发 2、I/O depth 8；上下文和批 token 上限 16400；sampling temperature0、min/max128、ignore_eos、seed0。
- 同一前三家族 SSD 预存、后两家族空；3048 源文件共 2,796,552,192 字节。每次用原安全硬链接初始化隔离输出缓存，新进程初始化 GPU/staging；模型仅 symlink，不重复复制。
- 保留精确 Prefix Cache、原成本准入曲线、共享 staging、预加载、复制合并和异步 I/O。沿用历史 P3 原曲线，不冒称在本 GPU 上全域重新标定。两臂使用相同共同修复。
- 共同 runtime/resource 域 SHA-256 e75c3adff856be6dd9993dc1990a4709b5ea00971a35557bdf5731ee377fcd75，70 个共同 runtime refs 相同；U lock05、方法 lock07只增加实际独立标定证据引用。

**实际改动**

研究组件仅私有恢复闭包观测/有限预测/排序：p4_restore_forecast、p4_restore_order_journal、p4_policy、p4_bridge 及 native reactor 接线。原生成功 kernel 接受序列在两臂共同记录；不增加工作队列、不持有源块 owner、不授予 GPU 释放 credit。

共同 V5 薄适配器调用原 P316 请求循环；接入已有私有 CUDA13/Ninja/作者包，修复同进程 transfer metrics dataclass→dict 兼容、guard 新目录与旧 runner mkdir 兼容，以及 shutdown 深层标量证据的有限投影。完整恢复历史单独保留，原 native/AIO 生命周期验证仍执行。源块释放协议和执行器未重写。

native V3 唯一新共同优化：普通无绑定 job 的预加载不再采集立即被丢弃的依赖视图；实际 job、单文件预加载和 mandatory/support/continuation 采集保留。CPU 检查证明其余模块 AST 相同、必要采集和控制输出相同。两臂都用 V3，V2 的旧 U01 不参与主配对。

配置错误真实保留：CAL05 runner_ref 误指 V4，CAL06改为实际 V5；U02 off 配置误加 run_id，U03/U04恢复原严格 off 映射。两类均为本轮准备错误，不列为成功性能样本。

用户本次明确取消累计固定 GPU 上限；直接授权文件绑定当前 GPU 和本轮 U/F8+D_R 范围，私有 standing guard 在验证授权后允许 max_gpu_hours=null。原权限 YAML、原 guard、驱动/系统/作者文件未改，账本历史未归零；保留单次300秒+20秒收尾及数据/内存生命周期保护。

**逐请求完整配对**

| ID | U TTFT s | 方法 TTFT s | U 完成 s | 方法完成 s | U ITL P95 ms | 方法 ITL P95 ms | 输出 |
|---|---:|---:|---:|---:|---:|---:|---|
| c2-0 | 3.031804 | 3.318217 | 8.386250 | 9.150670 | 103.356526 | 172.928492 | 128 相同 |
| c2-1 | 1.853508 | 2.026097 | 8.028838 | 8.851505 | 103.356526 | 232.044498 | 128 相同 |
| c2-2 | 8.015084 | 8.770472 | 11.835837 | 12.644952 | 17.791086 | 20.815732 | 128 相同 |
| c2-3 | 7.856998 | 8.633153 | 9.603251 | 10.392210 | 17.598510 | 19.053463 | 128 相同 |
| c2-4 | 11.607312 | 12.370978 | 17.669698 | 19.917198 | 110.666319 | 218.799230 | 128 相同 |
| c2-5 | 9.263357 | 15.056984 | 15.812753 | 18.963692 | 114.909412 | 77.624649 | 128 相同 |
| c2-6 | 15.255497 | 17.960569 | 19.816921 | 23.781007 | 35.456958 | 61.859123 | 128 相同 |
| c2-7 | 20.056920 | 24.565644 | 23.352659 | 27.809031 | 57.391297 | 46.355327 | 128 相同 |
| c2-8 | 14.987448 | 18.412933 | 18.996923 | 24.306854 | 35.564736 | 68.465940 | 128 相同 |
| c2-9 | 19.907628 | 24.728773 | 21.607340 | 26.413584 | 17.050828 | 15.728932 | 128 相同 |

**全部本轮守护作业**

| 实际标签 | exit | 守护墙钟 s | 用途/失败 |
|---|---:|---:|---|
| server14-dr-cal01 | 1 | 1.091609009 | details目录兼容错误，模型未运行 |
| server14-dr-cal02 | 1 | 110.708478509 | 真实加载模型，warmup metrics 类型错误 |
| server14-dr-cal03 | 1 | 0.135891361 | 配置误用绝对路径，模型未运行 |
| server14-dr-cal04 | 1 | 297.535380939 | 旧三家族标定 cohort 超时、深层 tail 解析失败；trace保留，未冒认native闭合 |
| server14-dr-long-u01 | 0 | 141.211916991 | V2 U描述性31.111856537s，不在当前域配对 |
| server14-dr-cal05 | 1 | 1.038026506 | runner_ref误指V4，模型未运行 |
| server14-dr-cal06 | 0 | 120.848959494 | 独立单家族标定闭合，覆盖不足 |
| server14-dr-long-u02 | 1 | 101.973568471 | off误加run_id，真实启动解析失败 |
| server14-dr-long-u03 | 0 | 135.144461632 | 当前域额外U描述性26.152582656s，不替代主配对 |
| server14-dr-cal07 | 0 | 122.029900748 | 同输入刷新过期标定，13.128706949s，覆盖仍不足 |
| server14-dr-long-u04 | 0 | 137.935962940 | 主配对U |
| server14-dr-long-dr01 | 0 | 142.687758802 | 主配对F8+D_R，排序未触发 |

上述12个守护作业累计墙钟 1312.341915401 秒，包含失败启动/初始化/请求/排空，不等于 kernel 活跃时间。历史账本累计 28947.970493132 秒，按直接授权无新累计限额，没有归零。

CAL06在配置纠错期间超过现有10分钟有效期，CAL07同输入刷新后连续执行 CAL07→U04→DR01。未改效果负载、未拟合效果数据；遇到无合法 D_R 动作后停止，未跑反向pair。额外U和失败结果不参与挑选收益。

**实际命令与验证**

真实完整绝对路径 argv 在各 run/result.json 和 LAUNCH 文件；服务器执行如下冻结命令：

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
.venv/bin/python -B /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server14-dr-long-20261005/run_gpu_stage_standing_v1.py --permissions-path artifacts/prefix_io_v1/server14-dr-long-20261005/EFFECTIVE_UNCAPPED_D_R_GPU_PERMISSION_02.json --label server14-dr-long-u04 --seconds 300 -- .venv/bin/python -B /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server14-dr-long-20261005/dr_p316_original_adapter_v5.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --config artifacts/prefix_io_v1/server14-dr-long-20261005/CONFIG_U07.json --execute
.venv/bin/python -B /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server14-dr-long-20261005/run_gpu_stage_standing_v1.py --permissions-path artifacts/prefix_io_v1/server14-dr-long-20261005/EFFECTIVE_UNCAPPED_D_R_GPU_PERMISSION_02.json --label server14-dr-long-dr01 --seconds 300 -- .venv/bin/python -B /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server14-dr-long-20261005/dr_p316_original_adapter_v5.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --config artifacts/prefix_io_v1/server14-dr-long-20261005/CONFIG_DR01_02.json --execute
```

具体顺序脚本 /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server14-dr-long-20261005/execute_actual_cal07_u04_dr01_v1.py；任一失败或非自然排空就停止。全部stdout/stderr保留。

已有针对性CPU检查：恢复预测/journal/提交9/9；V5 adapter13/13；standing guard9/9；独立输入祖先5/5；普通预加载无用采集3/3；结果分析6/6；原P316 CPU合同exit0。没有用这些CPU通过充当性能结论。主pair模型/输出/初态/源码/生命周期核验全通过；主pair与CAL07的21个原始文件已与远端逐字节SHA核验。

实际服务器CPU命令/日志保存于 actual_server14/cpu_checks，各命令如下：

```bash
# DR_RESTORE_CPU_01: /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S -c '
import os,runpy,sys
root="/root/autodl-tmp/prefix-io-v1-handoff/project";d=root+"/artifacts/prefix_io_v1/server14-dr-long-20261005"
base=root+"/artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate/source/third_party/work"
os.environ.update(DR_CPU_ORIGINAL_CONTROL=base+"/prefix-io-p4-02-cpu/src/prefix_io_control",DR_CPU_ORIGINAL_NATIVE=base+"/py-kvcache-p4-02-cpu/py_kvcache/reactor.py",DR_CPU_PATCH_CONTROL=d+"/runtime_source_v2/control/prefix_io_control",DR_CPU_PATCH_NATIVE=d+"/runtime_source_v2/native/py_kvcache/reactor.py")
sys.argv=[d+"/test_restore_development_cpu.py"]
runpy.run_path(sys.argv[0],run_name="__main__")
'
# DR_ADAPTER_CPU_05: /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server14-dr-long-20261005/test_dr_p316_original_adapter_v5.py
# DR_STANDING_GUARD_CPU_01: /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server14-dr-long-20261005/test_run_gpu_stage_standing_v1.py
# DR_CALIBRATION_INPUT_CPU_02: /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server14-dr-long-20261005/test_dr_p316_calibration_input_v1.py
# DR_UNUSED_PRELOAD_COLLECT_CPU_01: /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server14-dr-long-20261005/unused_preload_collect_candidate/test_unused_preload_collect_cpu.py
# DR_ORIGINAL_CONTRACT_CPU_01: /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -c '
from pathlib import Path
import json,sys
root=Path("/root/autodl-tmp/prefix-io-v1-handoff/project");d=root/"artifacts/prefix_io_v1/server14-dr-long-20261005"
sys.path[:0]=[str(d/"runtime_source_v2/control"),str(d/"runtime_source_v2/native"),str(root/"experiments/prefix_io_v1/scripts")]
from qualify_concurrent_pilot import verify_gate
from concurrent_pilot_contract_p316 import validate
from storage_source_registration import validate_registration
c=json.loads((d/"CONFIG_CAL01.json").read_bytes())
manifest=json.loads((root/c["manifest_ref"]["path"]).read_bytes());curve=json.loads((root/c["curves_ref"]["path"]).read_bytes())
historical=c["device_migration"]["historical_gpu_uuid"]
permit=verify_gate(root/c["qualification_ref"]["path"]);engine=validate(manifest,curve,historical)
reg=validate_registration(root,root/c["storage_registration_ref"]["path"],root/c["storage_source"],root/"experiments/prefix_io_v1/runs/server14-dr-cal-cpu-check/details",historical)
print(json.dumps({"historical_gate_verified":permit["status"],"historical_metadata_gpu":historical,"current_prediction_qualification":False,"engine":engine,"source_registration_verified":reg,"GPU_operations":0}))
'
```

本地只读实际分析调用 analyze_dr_long_results.analyze_runs，索引 ACTUAL_MAIN_PAIR_RUN_INDEX_01.json，实际设备/共同域、退出、原始SHA及源码见执行证据。补充审计用最终全量journal，不只看末尾published空summary。

**证据与停止状态**

- [主配对完整分析](<C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server14_candidates/dr_long_20261005/ACTUAL_MAIN_PAIR_ANALYSIS_01.json>)
- [真实机制与逐请求审计](<C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server14_candidates/dr_long_20261005/ACTUAL_MAIN_PAIR_MECHANISM_LATENCY_SUPPLEMENT_01.json>)
- [实际分析命令/退出/源码](<C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server14_candidates/dr_long_20261005/ACTUAL_MAIN_PAIR_CPU_EXECUTION_EVIDENCE_01.json>)
- [U04原始模型结果](<C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server14_candidates/dr_long_20261005/actual_server14/runs/server14-dr-long-u04/details/result.json>)
- [DR01原始模型结果](<C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server14_candidates/dr_long_20261005/actual_server14/runs/server14-dr-long-dr01/details/result.json>)
- [U lock05](<C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server14_candidates/dr_long_20261005/actual_server14/DR_SOURCE_LOCK_05.json>)
- [方法lock07](<C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server14_candidates/dr_long_20261005/actual_server14/DR_SOURCE_LOCK_07.json>)
- [U配置](<C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server14_candidates/dr_long_20261005/actual_server14/CONFIG_U07.json>)
- [方法配置](<C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server14_candidates/dr_long_20261005/actual_server14/CONFIG_DR01_02.json>)
- [实际资源与远端原始SHA](<C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server14_candidates/dr_long_20261005/actual_server14/POST_PAIR_RESOURCE_AND_EVIDENCE_AUDIT_01.json>)
- [21文件下载核验](<C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server14_candidates/dr_long_20261005/ACTUAL_MAIN_PAIR_DOWNLOAD_VERIFICATION_01.json>)

服务器交付目录 /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server14-dr-long-20261005；原始作业为 /root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runs/server14-dr-long-u04、server14-dr-long-dr01，各目录保留guard、adapter、完整冻结配置、原runner报告、shutdown证据。历史模型、缓存、失败均未删除。

最终GPU RTX5090/32607MiB，显存占用2MiB，计算进程为空、active_reservation=null。数据盘剩余 38.056GiB，高于8GiB保留。无自动续租、删除或关机；GPU已无本任务作业，可切回无卡。

本轮按无合法 D_R 动作停止。后续要证明排序作用，需要另行明确真实预加载→恢复依赖与当前排序入口的覆盖缺口，以及必要成本覆盖；这不是单纯增加重复次数。没有自动进入该开发、完整D/I/J、通用CostTable、SLO或数据集准备。

历史受限I的181次真实推迟及约7.279%耗时退化仍保留，不能归零为完全未测；它不属于本次D_R配对。现有证据没有证明研究方法性能优势。
