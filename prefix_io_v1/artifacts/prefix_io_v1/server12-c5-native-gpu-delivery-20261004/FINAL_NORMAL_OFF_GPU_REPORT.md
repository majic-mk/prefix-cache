# Server12 正常 off 真实 GPU 与持续授权交付

当前用户指示“授权gpu，只要有gpu就用，以后无需授权”已写入
PROJECT_GPU_AUTHORIZATION_AMENDMENT.json，并实际用于本轮 off01。
后续项目 GPU 操作不再逐轮请求授权；原 8 小时累计预算及阶段先决条件继续生效。
GPU_AUTHORIZATION_CURRENT.md 说明当前指示与旧报告的关系。
没有修改原 permissions.yaml、系统、驱动、已安装包或作者源码，没有删除已有实验数据。

## 实际 GPU 结果与可证明范围

在 root@connect.westd.seetacloud.com:24828 的 RTX 5090 /
GPU-a67d2f38-8e1b-29c3-7e5b-375e1410f8ac / driver 580.95.05 上：
此前 single6 校准真实完成六个窗口，768 个测量输出 tokens，
原 guard 实际耗时 764.612379428 秒。其原成本公式和三 pair 原始数据均通过公开重放；
冻结上界 16,238,752 ns，原 A-only 预算 13,171,328 ns，故该候选应 defer。
该校准证据已经在 GPU_EVIDENCE.tar.gz 单独归档并逐文件 SHA 验证。

随后本轮 server12-c5-native-normal-off01 真正运行原模型和 I/O：
一个预注册 B 窗口完整生成 128 tokens，128 个完整 frames/witness，
1 个 917,504 bytes SSD 操作和真实 CQE，
原 engine.shutdown 返回，OS 会话已排空，无遗留 native/aio worker。
原 guard exit/child_exit=0，未超时，实际墙时 261.120514885 秒。
真实 GPU 命令由冻结 controller 启动原 run_gpu_stage.py，
不是 CPU mock、探针、摘要或 launch receipt 代替完成结果。

完整 raw、源、guard、账本、生命周期和输出审核通过，但成本迁移失败：
选中整步 GPU 实测 16,893,473 ns，高于冻结上界 16,238,752 ns
654,721 ns（约 4.03%）。原 verify_p4_single_file.py 返回 2，
写出 native_execution_verified=true /
frozen_cost_migration_pass=false /
runtime_condition_qualified=false / permits_next_mode=null。
不能把 raw 的 PASS_REQUIRES_VALIDATION 当作阶段资格通过。
shadow/on 没有运行，P4/P5、策略收益、完整控制成本、生产资格仍未完成。

normal 使用预注册第四种 workload（prompt 首 token 28100 / seed 2829）；
calibration/heldout 的三种为 18100/1829、19100/1830、20100/1831。
完整 prompt/prefix key 不同，128 个输出 IDs 恰相同；
normal 的 SSD 文件与 B0 相同，B1/B2 是同尺寸不同文件。
模型、UUID、非 storage 引擎配置、共同 reactor/collector、
parent 上限 8、bridge=None、单 decode / context144 形状均一致。
off 的通知 adapter 未安装，get_calls=0。实测 SSD CQE 1.517552 ms
在三次 B 的 1.443930–1.635198 ms 区间；这不能单独证明因果。
没有发现作者模型或缓存源码接错，也没有证据确认超界的具体原因。
有限上界在本次第四种 workload 上欠预测，不证明整个方法普遍不可行，
更不证明性能提升或论文收益。v2 对照报告保留全部证据与来源字段澄清。

两轮新卡 guard 累计实际耗时 1,025.732894313 秒；
原总账本已用 24,155.170197121333 秒，剩余
4,644.829802878667 秒（约 77 分 25 秒）。
账本空闲、实际原会话为空；现场 PRIMARY 剩余约 50 GiB。
这里只核算实际原 guard，未把授权预留或 CPU 后处理墙时充作 GPU 实验时长。

## 实际改动和 CPU 检查

独立 normal_native_cpu 源绑定现有 CUDA13 私有 SDK、Ninja、当前驱动资产、
真实完成的 single6 校准和当前 append-only 账本。
固定工程路线、原精确 Prefix Cache、LoadPlanner 成本准入、共享 staging、
preload、copy merging、异步流水线、模型执行器及原 cache engine 均保留。
改动限于现场来源/历史账本/SDK 接线、完整引用元数据规范化与验证，
没有调整 cost formula、A-only 预算、未知条件拒绝、原等待期限或研究策略。

服务器实际 normal CPU 检查 107/107 通过（42 controller + 40 runtime + 25 history），
0 fail/error/skip，模型/GPU import 尝试为 0。
45 个 runtime/verifier 函数 AST 来源审核通过。
输入源闭包 4,774 项，正常 COMMON_SOURCE_LOCK 4,791 项；
GPU 后再次逐个完整 byteSHA 核验全部 4,791 项通过，
原 guard 完整字节一致，真实 guard 事件唯一、账本前缀与实际墙时完全相符。
本机原有 14 个 tracked 修改文件 SHA 均保持不变。

另新增独立 CPU 元数据复用原型 entry_receipt_session.py。
它只处理已完成 off01，首次调用原公开 canonical loader 产生真实 ExactSingleFileReceipt；
两个 get 返回同一个公开类型和对象，并分别重查完整 4,791 个源、
config/authority/前后证明、原校准和 normal guard、
live ledger/原 SDK 全资产/current driver、当前 PID/SID/process group。
本机 20/20 边界测试及服务器 20/20 实际边界测试通过；
服务器真实 completed CPU 正向调用通过、公开 loader 只调用一次，
GPU 操作为 0，原账本前后 SHA 完全相同。
公开 loader 实测 19.075308678 秒，两次完整 get
19.978091017 / 20.023207193 秒，整个 CPU harness 84.741359510 秒。
这未对比原七次路径，不宣称加速；原型尚未接入 active GPU 启动，
active guard 入口固定拒绝，也不改变失败的 off 资格。

启动慢的一项真实源码原因已定位：main 和 execute_window 的嵌套配置/guard 检查
合计调用公开 receipt loader 七次，均在正式测量前。
完整校准/source/history/SDK 重放反复执行；
只有一次 loader 已实测计时，其余六次旧运行没有分段计时，
不能精确把 261 秒或选中步超界归因于它。
CPU 原型只验证复用可行性和拒绝边界，不是已经完成 GPU 启动修复。

## 服务器实际命令及结果

项目根 ROOT=/root/autodl-tmp/prefix-io-v1-handoff/project。
完整 argv/stdout/stderr/result 保存在本 delivery 目录中。

| 记录 | 结果 |
|---|---|
| 14_FREEZE_REAL_NORMAL_CPU_INPUTS | 4,774 项实际源冻结，CPU 通过 |
| 15_NORMAL_RUNTIME_ANCESTRY_AST_AUDIT | 45 个函数来源审核通过 |
| 16_FROZEN_NORMAL_SERVER_CPU_TESTS | 107/107 真实服务器 CPU 检查通过 |
| 17_FREEZE_ACTUAL_CURRENT_NORMAL_BINDING | 4,791 项 COMMON / 真实公开 receipt 重放通过 |
| 18_PREPARE_UNAUTHORIZED_NORMAL_OFF_CONFIG | 只准备配置，没有 GPU 启动 |
| 19_REAL_PREPARED_NORMAL_CLI_REJECTS_ABSENT_AUTHORITY | 当时缺授权的实际入口在 GPU import 前拒绝；历史证据保留 |
| 20_PERSISTENT_AUTHORITY_OFF_LIVE_BINDING | 用当前用户持续 GPU 指示完成现场 metadata 绑定 |
| 21_NORMAL_OFF_SCOPE / 22_NORMAL_OFF_FULL_SOURCE_BEFORE | 原 guard 范围及完整来源前查通过 |
| 23_NORMAL_OFF_ORIGINAL_GUARD_LAUNCH | 实际派发一次原 guard；完成以 run/result.json 为准 |
| 24_NORMAL_OFF_AFTER | 完整来源后查 / 原完成事件与会话排空通过，54.697 秒 CPU |
| 25_NORMAL_OFF_REAL_QUALIFICATION | CPU 命令路径参数误用绝对路径，0.070 秒拒绝，未运行 GPU |
| 25B_NORMAL_OFF_REAL_QUALIFICATION | 正确项目相对路径、真实完整资格审核 58.316 秒；成本失败，退出 2 |
| 26_NORMAL_OFF_FINAL_REAL_STATE_AUDIT | 全部 4,791 项、原 source/guard/ledger/会话/资源后审通过，13.707 秒 CPU |
| 27_SERVER_COMPLETED_RECEIPT_REUSE_CPU_TESTS | 本机布局路径不适用于远端，CPU 0.077 秒拒绝 |
| 27B_SERVER_COMPLETED_RECEIPT_REUSE_CPU_TESTS | 只适配测试源路径，服务器 20/20 通过，0.133 秒 CPU |
| 28_SERVER_ACTUAL_PUBLIC_RECEIPT_COMPLETED_CPU_REUSE | 真实公开 receipt 两次 completed CPU 复用通过，84.810 秒 CPU |

失败的 25/27 只是辅助 CLI 路径错误，不是模型/GPU 执行失败；
原日志保留，修正不改变冻结控制器或数值验证。
原 README 中独立 prototype 目录为最初建议；
真实上传和执行位置为本目录 normal_entry_dedup_cpu/，以 COMMAND.json 的实际 argv 为准。

主要实际命令：
```sh
.venv/bin/python -B -I artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004/control_p4_single_file.py --root "$ROOT" launch --mode off
.venv/bin/python -B -I artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004/control_p4_single_file.py --root "$ROOT" after --mode off
.venv/bin/python -B -I artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004/verify_p4_single_file.py \
 --root "$ROOT" \
 --config artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004/CONFIG_off.json \
 --result experiments/prefix_io_v1/runs/server12-c5-native-normal-off01/details/p4-single-file-runtime-result.json \
 --guard experiments/prefix_io_v1/runs/server12-c5-native-normal-off01/result.json \
 --before artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004/SOURCE_off_BEFORE.json \
 --after artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004/SOURCE_off_AFTER.json \
 --output artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004/QUALIFICATION_off.json
.venv/bin/python -B -I artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004/normal_entry_dedup_cpu/entry_receipt_session.py \
 --root "$ROOT" --requests 2 \
 --output artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004/normal_entry_dedup_cpu/COMPLETED_CPU_REUSE_RESULT.json
```
这些是已完成的历史执行记录；不要重跑 append-only 输出命令覆盖证据。
后续任何真实 GPU 作业可按持续授权安排，但必须新建准确来源、
现场上下文、原预算 guard 和独立作业名称，不复用一次性已运行 job metadata。

## 证据、结论与下一允许阶段

主要源码/配置/COMMON/binding/qualification：
artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004。
原 GPU raw/guard/process.log：
experiments/prefix_io_v1/runs/server12-c5-native-normal-off01。
独立 cost 对照：
normal_native_cpu/evidence/OFF_COST_MIGRATION_COMPARISON_v2.json / .md。
持续授权、完整来源后审、所有实际命令与 CPU 原型：
artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004。

关键 SHA：
- COMMON_SOURCE_LOCK：aaa67309fab4c6dc7fe07853a31c3a99cdef727408da9fab604201bf95416d4d。
- NORMAL_RUNTIME_BINDING：fb2d54f3fbac769051cea5ed1c05a3b3c7901ea4fb6738ea0cee367559cb091e。
- off guard：4ac9a83500300c6ad14c530b7a3b81bd8987fe7328dc0f7b8a89bce3393ebc39。
- off raw：2853d4ae1c427d549cf2d4a046d5afd11c8c06f91913ff6e74ca453f96d7d3a9。
- off qualification：b28ee3a35aafb69a879350ee8534c838bab8b93713d2ad83970bf632d9b61501。
- 当前 ledger：7015729867cac4d285bf6277433fd395e8d721bcd7dc71d92e92bf3cd416637f。

当前 normal 第四 workload 成本迁移不合格，停止这一 off→shadow→on 链；
不是 GPU 权限不足、没有卡、驱动调用被禁或 GPU 作业一直未执行。
下一允许阶段是有限成本稳定性与迁移诊断，并在 CPU 完成启动校验去重的 active-entry
设计/拒绝检查与新的源码冻结。若需要新数据，应先独立预注册训练/留出与停止规则，
保留本次失败，不把旧失败样本事后回填训练以证明同一上界通过，
不挑 seed 重复直到通过、不改变步预算、不扩大负载或研究策略。
现有证据足以证明当前引擎/I/O/输出/生命周期可真实运行；
尚不足证明已验证 on、性能提升或论文收益。
当前没有 GPU 作业运行，可按用户需要切回无卡；本任务未操作实例计费/电源。

正常交付归档将包含真实源、全部 normal authority/proof/raw/guard/log、
107 项及 20 项实际 CPU 记录、实际计时和持续授权； prior single6 archive 作为完整 SHA
绑定的独立依赖。不打包大型模型/SDK二进制/private-cache/KV，原文件全部留在服务器。
归档和本机逐文件核验的实际结果另见 NORMAL_NORMAL_DELIVERY_off01-final_RESULT.json
和对应 LOCAL_BYTE_VERIFICATION.json，不用预计尺寸代替实际完成。

