本轮已在 server09 的 RTX 5090 上真实运行 2 次受限模型作业：cold-01 成功，populate-01 在首批缓存成功写入后因作者统计接口的 dataclass / dict 不一致失败。paired-01 未运行。GPU 作业已经结束，原引擎 shutdown 均正常返回，两个 OS 会话均排空，没有活动 GPU 预算预约。

实际硬件为 GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2，RTX 5090，驱动 595.71.05，开卡核验时总显存 32607 MiB、空闲 32110 MiB。驱动库字节/SHA 与冻结资产一致。一次只读硬件元数据查询及这两次模型作业均实际执行；之后的修复验证没有运行 GPU。原预算守卫检查权限、预算、子会话；实际设备 UUID 由受限运行时读取实际 torch 属性核对。

| 实际作业 | 结果 | 原守卫计费秒数 | 已取得证据 |
| --- | --- | ---: | --- |
| cold-01 | PASS，父/子/守卫退出 0 | 120.49416207242757 | 3 个完整 129-token prompt，各输出 token [398]，cached 0；rep0 为 warmup |
| populate-01 | FAIL，父/子/守卫退出 1 | 120.24441478867084 | 第一次 store 完整输出 [398]，随后 sentinel generate 在原 metrics.py:154 断言失败 |
| paired-01 | 未运行 | 0 | 原账本事件数 0，结果路径不存在 |

cold 的两个非 warmup 原始 TTFT 为 0.019702434539794922 和 0.015488147735595703 秒，来自原 metrics.first_token_latency 的前端 wall-clock 时间，不能视为 GPU event 成本或策略提升。populate 原始 trace 包含一次成功 GPU→storage 7340032 B 传输、8 次各 917504 B 的 file_write、8 个 D2H staging 事件。8 个已写缓存文件及失败日志保留。正常 shutdown 后 AIO accepted/completed/reaped 均为 8、pending/active 均为 0，closed/drained 为 true。尚未发布可供 paired 使用的完整存储清单。

失败根因有实际源码和 CPU 回放支持：原 OffloadingConnectorStats.record_transfer 产生 OffloadingOperationMetrics dataclass；原 reduce 和 Prometheus observe 要求普通 dict。当前同步进程路径保留 dataclass，原先依赖的 IPC 序列化转换未发生。首批 SSD 写入成功，因此这次失败不是 SSD/AIO 写入失败。

实际共同修复是新增 compat_offloading_stats_dict.py，在受限进程内按原源码 6195 B / SHA 5ab3c65af3c0f78b0696bcc60c68ecfe51332c6d5f6684bb2cfa0e0ea9677d73 绑定原模块。包装原 producer，每次仍调用原方法一次，只把新追加的准确 dataclass 转成含 op_size/op_time 的字典，保持原数值、原消费者和日志功能。finally 撤销包装并验证 witness.installed 为 false；撤销失败如实记录，保留原始异常。未修改作者文件、原采集器、缓存引擎、模型执行器或已安装包。

新增入口 run_g3_calibration_pilot_v3.py、新运行时 g3_calibration_runtime_metrics_v2.py 和新计划 g3_calibration_plan_metrics_v2.py 绑定上述共同修复，并把候选作业/私有存储改为 02。候选继续使用作者 py-kvcache/vLLM，策略和 load_planner 均 off，成本曲线为 null；保留精确 prefix、共享 staging、预加载和原异步 I/O。既往结果与存储发布必须具备修复安装/恢复证据。所有历史源码与 01 实验保留。

服务器实际 CPU 核验为 77/77 通过：计划 31、运行时合同 18、入口 14、真实 finally AST 清理合同 2、原统计 producer/consumer 回放 12。部分合同使用明确的 CPU stub，这些测试不代表真实模型/GPU 验证。服务器完整 preflight 通过 4075 项冻结引用及严格资产核验，未导入真实框架或初始化 GPU；CPU 核验前后原 GPU 账本 SHA 完全相同。独立审阅确认实际两次 GPU 原始证据、根因及两项 finally 清理合同。

新源锁 gpu-source-lock-metrics-v3-final.json 为 910009 B，4075 项引用，SHA-256：
b8ad831c294d978e36a0b1a6343118516af5c55a6e0a9c4380ff6644ec91092b
其中此前 4065 项逐项保持不变，新增 10 项。候选模板 FINAL_G3_METRICS_V3_SCOPE_TEMPLATE_NOT_AUTHORIZED.json 的 allow_gpu_runs/allow_gpu_initialization 均为 false，human_authorization_record 为 null；未把旧授权或“已开卡”记成修复版新作业的授权。

实际命令以结构化 JSON 保留：gpu-actual-20261002/G3_COLD_ACTUAL_LAUNCH_COMMAND.json、G3_POPULATE_ACTUAL_LAUNCH_COMMAND.json，以及 metrics-common-fix-cpu/METRICS_V3_SERVER_CPU_77_COMMAND.json。实际模型命令使用旧 run_g3_calibration_pilot_v2.py --mode cold/populate --source-lock gpu-source-lock-storage-v2.json --scope-record G3_AUTHORIZED_SCOPE_20261002.json --launch；原 run_gpu_stage.py 执行每次 300 秒限时及 20 秒收尾预约。CPU 测试和 preflight 均用 .venv/bin/python -B -I -S，CUDA_VISIBLE_DEVICES 为空。SERVER_CPU_77_AND_PREFLIGHT_RESULT.json 包含 5 个实际测试命令、逐项退出码/时长/日志 SHA 和完整 preflight 命令。

原始 GPU 证据位于服务器 /root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runs/server09-g3-calibration-cold-01 与 server09-g3-calibration-populate-01。完整证据副本、源码审计、CPU 收据及本报告位于服务器 /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002，且已镜像到本地同名候选目录。新交付清单 G3_GPU_ACTUAL_METRICS_V3_DELIVERY_MANIFEST.json 与两侧核验收据单独追加，历史两份交付清单仍保留。

原累计 8 小时预算实际已用 17447.08221173985 秒，剩余 11352.91778826015 秒（约 3.15 小时），本轮两作业合计 240.7385768610984 秒。账本 SHA 为 8bf9876c6afb543f878228baa8f2f9eff8b8efe4c8565bd400de20f8daaf7884。服务器 CPU preflight 的实际 statvfs 空闲为 12714713088 B，3 GiB 预约加 8 GiB 保留空间可容纳；真正启动前仍会读取最新空间。未下载、删除数据或修改系统/驱动。

下一可提请授权阶段是修复版 G3 cold-02 → populate-02 → paired-02，最多 3 次、各 300 秒执行加 20 秒收尾，总预约 960 秒（16 分钟）；现有 Qwen2.5-7B、固定 18 个原请求、各 1 输出 token，3 GiB 预留/8 GiB 空间底线，沿用原剩余预算。任一次失败停止，不重试、不运行后继。原授权人类记录明确规定“任一次失败即停止，不重试”，且绑定 01 冻结代码，所以需要这次新源锁与 02 作业的限定授权。修复版尚未运行 GPU。

当前没有完整 gMem/gSSD 采样、恢复内容逐字节验证、生产阶段核算或成本资格，更没有 P4 策略真机通过及性能提升结论。本轮 G3 完成后仍需依据交接包检查成本/路径证据，再进入相应 P4 真机资格工作；不能把本轮 1-token 原路径 pilot 当作性能实验。
