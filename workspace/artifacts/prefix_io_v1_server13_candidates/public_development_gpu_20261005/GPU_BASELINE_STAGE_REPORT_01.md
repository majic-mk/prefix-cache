# 新服务器 GPU 原路径与完整观测阶段交付
实际服务器：connect.westb.seetacloud.com:36526；项目 /root/autodl-tmp/prefix-io-v1-handoff/project。
GPU：RTX 5090 32 GiB，GPU-bff07e52-6cfb-5828-bb80-9aa47a3a4a9b；实际驱动 595.71.05；沿用已克隆的 CUDA 13 私有 SDK、Qwen2.5-7B 和作者 vLLM。没有修改系统、驱动、安装包或作者执行器，没有下载模型、删除数据或重新全量哈希模型。
## 实际结果
原 guard 的 uoff03 作业退出 0，无超时，198.500624 秒；原 shutdown 返回、原生资源和 OS 会话均排空。CPU 原始证据闭合已通过。
冻结的开发分区全部 5 条真实自然输入各生成 128 tokens，共 640；485 原始模型步骤对应 485 CUDA 事件见证，保留全部 153 个异质步骤。输入与 uoff02 完全一致，640 个输出 token 也完全一致。
D2H 和 SSD 写各完成 48 次、44,040,192 字节；SSD 读和 H2D 为零。后四请求各复用 128 个缓存 tokens。读取、预加载与冷恢复尚未在本次五请求输入里实际验证。
全步 CUDA elapsed 的 p50=13.243328 ms，p95=18.456703 ms，仅为本次描述性观测。每帧 existing_io/new_io 未提供，未以阶段计数或 host 时间包含关系补造因果成本，不声称干扰或策略收益。
## 实际修改与验证
新增独立 U/off 观测入口，以真实新 GPU 身份执行，不借用旧 GPU 的私有成本表。
新增 SDK 迁移绑定：历史 580 编译证明与当前 595 驱动字节分别核验，保留原私有 SDK 布局；不冒称已在 595 下完成历史编译。
新增 V3 全流观测：保留同类帧原校验，观测不同上下文/混合步骤，并仅在原有步骤返回后查询、回收 CUDA 事件。128 待完成事件、4096 全流帧的边界保持；不增加同步、不改模型执行或 IO 选择。
新增只读闭合与分析工具。原 shutdown/finally、native tail、I/effect 资格门保留。新 U 观测尚未作为两臂共同观测资格。
服务器新增针对检查共 37 项通过：V5 接线16、SDK迁移9、混合观测12。旧大套件未重跑；新真实资产预检约2.1秒。当前源锁有5065项，完整继承V14的5014项，实际新核验51项及26个原REQUIRED引用；不声称新主机进行了16GB全量核验。
## 实际命令
命令完整 argv、stdout、stderr、exit 和耗时保存在本目录对应 *_COMMAND.json、*_RESULT.json、*_STDOUT.log、*_STDERR.log。
GPU入口：.venv/bin/python -B experiments/prefix_io_v1/scripts/run_gpu_stage.py --permissions-path artifacts/prefix_io_v1/server13-public-development-gpu-20261005/EFFECTIVE_STANDING_GPU_PERMISSION_01.json --label server13-public-development-uoff03 --seconds 300 -- .venv/bin/python -B artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/strong_trace_runner_v7.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --config artifacts/prefix_io_v1/server13-public-development-gpu-20261005/MIGRATED_UOFF_RUN_CONFIG_03.json --execute
CPU闭合：CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S artifacts/prefix_io_v1/server13-public-development-gpu-20261005/u_collection_closure_cpu_v3.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --config artifacts/prefix_io_v1/server13-public-development-gpu-20261005/MIGRATED_UOFF_RUN_CONFIG_03.json --output artifacts/prefix_io_v1/server13-public-development-gpu-20261005/ACTUAL_UOFF03_COLLECTION_CLOSED_02.json
## 保留的失败
uoff01：2.497542秒，旧580驱动路径为空，框架导入前停止，模型输出0。
uoff02：199.561756秒，真实640 tokens及关闭排空完成，但旧同类观测器遇到第二请求混合批次而失效；GPU作业仍按失败保存，不作为成本/性能对照。
uoff03：成功。首次CPU闭合工具漏传实际collection gate，保留其退出1日志；仅修CPU工具后的第二次闭合成功，没有再跑GPU。
## 预算与下一允许阶段
以上三次原guard作业合计扣时400.559922秒；原8小时预算累计26623.188124秒，阶段结束剩2176.811876秒（约36.28分钟），没有活动预约或本任务GPU进程。空闲数据盘51,938,328,576字节，约48.37 GiB。
已证明当前AutoDL能运行原模型、原写出路径和完整GPU观测；尚未证明新策略性能提升，未签发当前GPU成本表，未推断服务SLO。
继续在同一固定路线内准备实际校准分区的原生单位IO受控A/B成本测量，验证SSD读/H2D/真实释放。当前校准30条属于同一prefix family，不满足正式成本表所需独立fit/holdout来源；不能偷用开发或评估分区、不能把旧527条件或本次混合流升级为I表。单family测量只能交付raw功能/单位成本观察。GPU持续授权仍有效，下一作业继续受剩余预算与原guard限制。
证据：ACTUAL_UOFF03_COLLECTION_CLOSED_02.json、ACTUAL_UOFF03_OBSERVATION_ANALYSIS_01.json/.md、MIGRATED_U_SOURCE_LOCK_03.json、ACTUAL_POST_GPU_STATUS_01.json、原 runs/server13-public-development-uoff01/02/03 全部日志和原始记录。冻结输入、代码与旧失败均保留。
