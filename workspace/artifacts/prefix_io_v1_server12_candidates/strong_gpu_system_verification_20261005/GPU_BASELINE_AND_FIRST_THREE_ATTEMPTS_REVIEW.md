# 基线与前三次真实 GPU 尝试的记录复核

本复核直接读取当前目录的 OFF、CAL01–CAL03 及 GPU_STRONG_RAW_01–03 原始记录。它只解释这些已结束作业，不覆盖根代理仍在进行的 CAL04。复核自身没有 RPC、GPU 或模型导入；这不表示全局没有真实 GPU 运行。原记录与冻结代码均未修改。

## 原 U 基线已经运行了什么

12 个请求全部完成，每个 prompt 为 512 tokens、输出为 128 tokens，共 1,536 个输出 tokens；错误或未提交请求为 0。记录的 original engine steps 为 1,047。9 个请求的 frontend cached tokens 为 448，3 个为 0。这是该受控请求流的实际命中报告，不能单独当作策略加速或自然流量收益。[原请求输出](OFF_ACTUAL_REQUEST_OUTPUTS.json)、[原 child](OFF_ACTUAL_CHILD_RESULT.json)

原 planner 实际存在并与 manager 共用，`load_planner=on`、preload 与共享 staging 开启；bridge 为 None、实际 I 策略未激活，原 U 路径保留。[原 child 的 original_planner_identity / strategy_runtime](OFF_ACTUAL_CHILD_RESULT.json)

有效 native journal 中，D2H 和 SSD write 分别有 216 个 accepted 与 216 个 completed 操作，每阶段各 198,180,864 bytes；记录的 SSD read/H2D 事件数为 0。各阶段只按事件计数一次，没有反复累加快照，也没有把 D2H 与 SSD write 加作两份独立数据。这些计数证明已记录的写入和完成，不证明所有可能的读取均为零，也不证明已发现普通策略候选。[原 child 的 actual_native_stage_journal](OFF_ACTUAL_CHILD_RESULT.json)

child 保存了原 shutdown 返回与 native tail 排空；child 当时尚不能观察整个 OS 会话，因此其 `os_session_drained=false` 保留原样。作业结束后的原 guard 及独立 parent join 则确认会话排空、剩余成员为空、账本活跃 reservation 为 None。[原 guard](OFF_ORIGINAL_GUARD_RESULT.json)、[parent join](OFF_GUARD_CLOSED_QUALIFICATION.json)、[入口计时与账本](OFF_ACTUAL_ENTRY_RESULT.json)

基线保存的 bounded capture 为 invalid，52 个 frame、0 个 CUDA witness，原 failures 字符串仍保留。基线的完整请求输出和生命周期通过，不等于已取得合格 kernel 成本或策略收益；`full_step_evidence_qualified`、`cost_qualified`、`formal_effect_qualified`、`formal_goodput_allowed`、`strategy_improvement_proved` 均为 false。负载明确是 `controlled_original_P3_mechanism`，`natural_trace_bound=false`。[原 child](OFF_ACTUAL_CHILD_RESULT.json)、[parent join](OFF_GUARD_CLOSED_QUALIFICATION.json)

## 三次校准在哪一步停止

| 作业 | guard 计入 GPU 账本的秒数 | guard exit / 首 child exit | 真实停止证据 | 完成测量窗口 |
| --- | ---: | --- | --- | ---: |
| CAL01 | 15.607374024 | 1 / 2 | 子入口拒绝 `project relative path`，传入的 config 是绝对路径；没有保存模型 child runtime/capture | 0 |
| CAL02 | 144.246099111 | 1 / 1 | hot-warmup 实际 cached=512，但包装器要求 496 | 0 |
| CAL03 | 146.638762113 | 1 / 1 | collector 检查拒绝 `all 128 actual frames use the same correctly bound scalar adapter`；测量 capture/window 未落盘 | 0 |

CAL01 来源：[原 guard](CAL01_result.json)、[首 child 命令与退出](CAL01_details_windows_00_CHILD_RESULT.json)、[实际拒绝日志](CAL01_details_windows_00_STDOUT.log)、[父 raw](CAL01_details_native-cost-runtime-result.json)、[账本计时](GPU_STRONG_RAW_20261005_01_RESULT.json)。父记录的 `original_model_subprocesses_started=1` 只证明启动了子入口进程，不能用它声称已加载模型或运行 kernel；首 child 日志的 `GPU_job_started=false` 也不取消已实际开始并计时的外层 guard 尝试。

CAL02 来源：[真实 child](CAL02_details_windows_00_details_native-cost-runtime-result.json)、[父 raw](CAL02_details_native-cost-runtime-result.json)、[原 guard](CAL02_result.json)、[账本计时](GPU_STRONG_RAW_20261005_02_RESULT.json)。primer 有一个 completion、1 个输出 token；hot-warmup 有一个 completion、128 个输出 tokens，cached=512。`output_token_ids` 是嵌套列表，外层长度 1 是 completion 数，不是 token 数。child 保存了原错误，`windows=[]`、`warmups=[]`；独立 hot-warmup 观测不是六窗校准完成证据。

CAL03 来源：[真实 child](CAL03_details_windows_00_details_native-cost-runtime-result.json)、[父 raw](CAL03_details_native-cost-runtime-result.json)、[原 guard](CAL03_result.json)、[账本计时](GPU_STRONG_RAW_20261005_03_RESULT.json)。hot-warmup cached=512、完整 128 tokens，并保存了 1 个 warmup record；保存的 collector binding 含实际 reactor SHA。随后验证失败，没有保存 measured window 或完整 capture。因此现有证据无法定位具体坏帧、确认首帧 prepared 值，或重算真实 CUDA 成本；不能根据错误文字推断“第几帧错了”或“所有帧都坏了”。

CAL02/CAL03 的 child 原 shutdown 返回均为 true，handler shutdown 为 true、worker/AIO worker 不存活、reactor closed；实际 native tail 各计数为零，AIO accepted/completed/reaped 均为 39，outstanding/pending/ready/unreaped 均为零。两次 journal D2H/SSD write 每阶段均为 39 个完成操作、35,782,656 bytes。[CAL02 child](CAL02_details_windows_00_details_native-cost-runtime-result.json)、[CAL03 child](CAL03_details_windows_00_details_native-cost-runtime-result.json)

三次 guard 均未超时，退出后 `session_drained=true`、剩余 session members 为空；三个 GPU_STRONG_RAW 账本报告均 `active_reservation_after=null`。父六窗记录的 shutdown-success flag 仍为 false，不能将其改写为六窗成功；CAL02/CAL03 的真实 child shutdown 证据应与父六窗未完成状态分别阅读。[CAL01 guard](CAL01_result.json)、[CAL02 guard](CAL02_result.json)、[CAL03 guard](CAL03_result.json)、[账本 01](GPU_STRONG_RAW_20261005_01_RESULT.json)、[账本 02](GPU_STRONG_RAW_20261005_02_RESULT.json)、[账本 03](GPU_STRONG_RAW_20261005_03_RESULT.json)

## 时间、费用口径和目前能证明的边界

三次校准合计计入原 GPU 账本 **306.492235247 秒**；加上原 U 基线的 **143.513610497 秒**，这四个作业合计 **450.005845744 秒**。入口壁钟耗时分别为 OFF 162.125881001 秒、CAL01 15.720416222 秒、CAL02 144.361409057 秒、CAL03 146.766348012 秒；它们不是 kernel 时间，也不等于账单金额。[OFF 入口](OFF_ACTUAL_ENTRY_RESULT.json)、[账本 01](GPU_STRONG_RAW_20261005_01_RESULT.json)、[账本 02](GPU_STRONG_RAW_20261005_02_RESULT.json)、[账本 03](GPU_STRONG_RAW_20261005_03_RESULT.json)

CAL03 结束时记录的累计 used 为 25,128.901084650 秒，原 8 小时额度当时剩余 3,671.098915350 秒。根代理已继续后续作业，本复核不把这个历史快照称为当前剩余额度。原文件没有供应商账单与实际计费单价，人民币费用保持 unknown，不用 GPU guard 秒数自行估价。[账本 03](GPU_STRONG_RAW_20261005_03_RESULT.json)

现有记录证明当前卡上的原模型/缓存路径能够完成这条 12×128 受控基线，并保存实际命中、写入和正常退出证据。前三次成本采集均未完成六窗，不能发行资格成本表；这些记录没有独立 SLO 资格，也没有方法性能收益证据。下一阶段应保留所有失败记录，检查当前新修复版本的实际采集结果，再按真实完整 capture/guard/source/heldout 回放决定成本资格，不能补写 CAL03 缺失帧或将基线效果当作 I 的效果。

独立统计：[GPU_BASELINE_AND_FIRST_THREE_ATTEMPTS_REVIEW.json](GPU_BASELINE_AND_FIRST_THREE_ATTEMPTS_REVIEW.json)。本地执行使用 Python `-B -I -c` 与标准库 JSON/hash 解析，共 13 项真实记录连接与计时一致性检查通过，24 个输入文件在复核前后 SHA 相同；没有重放 issuer、发行能力或再次运行 GPU。
