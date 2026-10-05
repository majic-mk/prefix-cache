# 当前服务器真实 GPU 系统功能与单位成本交付
本轮实际连接 connect.westb.seetacloud.com:36526，所有实验在 /root/autodl-tmp/prefix-io-v1-handoff/project 执行。RTX 5090，32 GiB，GPU UUID GPU-bff07e52-6cfb-5828-bb80-9aa47a3a4a9b，驱动 595.71.05。
结论：此 AutoDL 环境可执行作者 vLLM 的正常模型生成，以及 py-kvcache 的原写出、SSD 读取、pinned staging、H2D 精确恢复和双消费者共享预加载；这是一轮功能与原始单位成本验证，完整 P4 有限候选策略资格和方法性能提升尚未成立。无需换平台、系统驱动或镜像来继续已验证的路径。

## 实际结果
|验收|真实结果|范围|
|---|---|---|
|原模型 U/off 基线 uoff03|5 条开发输入各生成128 tokens，共640；485 模型步骤/485 CUDA见证，其中153个异质步骤；guard退出0，198.500624秒|关闭新策略后正常模型及全流观测可用|
|同自然输入单位 A02/B02|185-token原校准输入，两边各128输出tokens且完全一致、各128步见证；guard均0，104.450091/105.415493秒|一对受控 SSD-only 操作，不是普通 I 或策略对照|
|实际 SSD 读|A为0；B接受/完成1次917504字节读取，原因果重叠校验通过，native tail排空|offset16、context200；不把主机包含关系伪造为GPU因果|
|被测步骤 CUDA elapsed|A=13.428096 ms，B=16.180511 ms，差值+2.752415 ms，B/A=1.204974|仅1对描述性成本；不推断总体、统计显著或策略收益|
|原 G1 off / shadow|两个当前GPU原guard均0，8.339450/8.456137秒；各store、age_store、restore、shared四项原断言通过|真实 CUDA 与 Linux-AIO，无模型加载|
|原恢复与共享|每臂restore SSD读/H2D各8650752字节，内容完全一致；shared SSD读8650752，H2D17301504，两个目的地内容完全一致|共享读取一次、向两个消费者复制，原任务/账本/AIO排空|
|资源范围|accepted parents回到0，各stage失败/inflight为0，全部native worker/tasks/copies退出；staging16650239<=16777216字节|不推断GPU allocator释放，gpu_release_credit仍None；没有正常模型冷SSD恢复结果|

## 实际改动
新增独立 U/off 观测消费者、V3异质帧适配及原步骤返回后的query-only CUDA事件回收；保持128 pending和4096帧上限，不新增同步、不改变原模型执行、缓存准入、复制或I/O选择。关闭新策略路径已实际通过。
新增私有SDK迁移绑定，将克隆的历史580编译证明与当前595真实驱动文件字节区分核验；没有修改驱动、系统、已安装包或作者源码。历史编译证明未被冒充为595下的新编译证明。
新增原单位窗口薄消费者和CPU后处理。V2只补回原execute_parent的真实关闭元数据join，不改变原execute_window/collector/validator/cleanup。单位模型身份明确继承刚完成的uoff03真实全权重核验，并检查当前文件stat/大小；不是再次全量散列。
原G1资格脚本29692字节、SHA5d9604fc07d80495a49febfcd488cb2601f74a2bef107261c8e623a4d0a1c77d原样使用；只新增当前UUID/持续授权/源码冻结记录及纯CPU闭合工具。当前G1只核验1903个原必要源码/二进制叶，共72761169字节，没有重哈希模型。
本轮未重新实现缓存引擎或模型执行器；没有下载模型、删除数据或增加研究策略种类。

## 命令与验证
实际命令完整argv、环境、stdout、stderr、退出值和耗时在同目录 *_COMMAND.json、*_RESULT.json、*_STDOUT.log、*_STDERR.log。GPU作业的原result.json与process.log保存在下表对应的 experiments/prefix_io_v1/runs/ 子目录；这些原记录全部收入交付清单。
- GPU基线：.venv/bin/python -B experiments/prefix_io_v1/scripts/run_gpu_stage.py --permissions-path artifacts/prefix_io_v1/server13-public-development-gpu-20261005/EFFECTIVE_STANDING_GPU_PERMISSION_01.json --label server13-public-development-uoff03 --seconds 300 -- .venv/bin/python -B artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/strong_trace_runner_v7.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --config artifacts/prefix_io_v1/server13-public-development-gpu-20261005/MIGRATED_UOFF_RUN_CONFIG_03.json --execute
- GPU原单位A/B：相同原run_gpu_stage guard，--label server13-natural-unit-A02 / server13-natural-unit-B02，--seconds300；子入口 raw_unit_cost/natural_unit_raw_runner_v2.py --project PROJECT --config raw-unit-cost-02/NATURAL_RAW_UNIT_RUN_A02.json / B02.json --execute；精确相对路径见 ACTUAL_GPU_RAW_A02_COMMAND.json / B02。
- GPU原G1：.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --permissions-path artifacts/prefix_io_v1/server13-public-development-gpu-20261005/EFFECTIVE_STANDING_GPU_PERMISSION_G1_01.yaml --label server13-current-g1-off01 / shadow01 --seconds180 -- .venv/bin/python experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py --mode off / shadow --name同label --execute --scope-record artifacts/prefix_io_v1/server13-public-development-gpu-20261005/CURRENT_G1_STANDING_SCOPE_01.json --source-lock artifacts/prefix_io_v1/server13-public-development-gpu-20261005/CURRENT_G1_SOURCE_LOCK_01.json --permissions-path同上。
服务器本轮新增针对性CPU检查45项通过；旧大套件未重跑。新GPU基线预检约2.1秒，单位A/B metadata实际两臂预检5.316/修复版5.327秒，单位闭合2.232秒，G1必要源与作用域预检1.454秒、闭合0.589秒。没有把CPU回放当GPU实验。
真实GPU共8次guard尝试，其中5次成功、3次失败；本轮合计731.570865秒，累计26954.199067秒，原8小时预算剩1845.800933秒（30.76分钟）。所有GPU作业已结束，active reservation为空，GPU进程为空，实际显存查询仅2 MiB，利用率0%。

## 失败与更正完整保留
|作业|原guard退出|秒|真实原因|
|---|---:|---:|---|
|server13-public-development-uoff01|1|2.497542|克隆的旧580驱动路径为空；框架导入前停止。|
|server13-public-development-uoff02|1|199.561756|640输出和shutdown完成，但同类观测器遇到混合上下文失效，不能作为完整成本证据。|
|server13-public-development-uoff03|0|198.500624|通过。|
|server13-natural-unit-A01|1|104.349772|原窗口和shutdown完成；我新增入口漏接原parent关闭元数据，strict校验KeyError。CPU回放只确认修复，不改判原guard。|
|server13-natural-unit-A02|0|104.450091|通过。|
|server13-natural-unit-B02|0|105.415493|通过。|
|server13-current-g1-off01|0|8.339450|通过。|
|server13-current-g1-shadow01|0|8.456137|通过。|

更正早期阶段报告：原运行时在uoff02/uoff03各完整散列11个模型文件15242788168字节，共30485576336字节；“未重做16GiB全源审计”不能写成“无模型重哈希”。BASELINE_REPORT_CORRECTION_01.json已更正，旧报告/zip保留。之后原单位作业明确继承真实模型身份，current_whole_model_rehash_performed=false。

## 证据与下一允许阶段
ACTUAL_UOFF03_COLLECTION_CLOSED_02.json、ACTUAL_UOFF03_OBSERVATION_ANALYSIS_01.json、ACTUAL_NATURAL_RAW_AB_CLOSED_01.json、ACTUAL_CURRENT_G1_CLOSED_01.json、ACTUAL_FINAL_SERVER_STATUS_01.json是本轮主要证据。各冻结source-lock/source-proof、实际raw captures/frontend/journal和原guard/失败日志均保留在服务器，并收入阶段zip；模型权重、编译缓存和私有KV文件保留原路径，不删除，不在阶段zip重复打包。
可继续的下一阶段是补齐独立校准前缀族、覆盖实际有限候选的条件成本，并预注册能够触发真实模型SSD恢复的相同输入U/I对照。当前冻结30条calibration全部同一prefix family，不满足正式fit/holdout独立来源；本轮仅1对A/B，不能发行当前GPU正式CostTable，不能把这组SSD-only注入当作普通I或性能提升。原development/evaluation分区保持独立，未借来拟合成本。
GPU持续授权有效，后续仍受剩余8小时预算约束；不需要每次再问授权。用户选择先完成功能和成本验证、未设逐token SLO，因此无formal goodput或服务目标结论。本轮没有否定方法，也没有证明论文所需提升。G1通过原native H2D，正常模型冷SSD恢复及有限候选策略在真模型中的资格仍待完成。
当前数据盘空闲51693871104字节（48.14 GiB），无需为本轮扩盘。全部当前作业已停止，下一步是CPU输入/成本资格准备，可在无卡模式进行。
