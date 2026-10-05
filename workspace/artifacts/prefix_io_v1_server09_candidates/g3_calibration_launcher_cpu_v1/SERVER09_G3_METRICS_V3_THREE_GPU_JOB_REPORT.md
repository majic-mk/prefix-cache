本轮已完成用户“继续”确认的修复版 G3 三作业包。server09 当前 AutoDL / RTX 5090 上，cold02、populate02、paired02 的父进程、原预算守卫和模型子进程均退出 0；原核验器对三作业联合重算得到 PASS_BOUNDED_NATIVE_PATH_POINT_PILOT_ONLY。18 个请求各完整输出 1 token，三个模型进程已经正常结束，OS 会话排空，活动预算预约为 null。

| 真实作业 | 请求数 | PID / SID | 原守卫预算秒数 | 结果 |
| --- | ---: | --- | ---: | --- |
| cold02 | 3 | 4379 / 4379 | 119.94951784517616 | PASS |
| populate02 | 9 | 5211 / 5211 | 121.05925491731614 | PASS |
| paired02 | 6 | 6063 / 6063 | 121.04938312992454 | PASS |

三个 fresh 进程依次结束后才启动后继。本轮计入原 GPU 预算 362.05815589241683 秒，约 6.03 分钟；原 8 小时累计已用 17809.140367632266 秒，剩余 10990.859632367734 秒，约 3.05 小时。未追加第四次作业、重试、性能四臂或扩大请求流。

GPU 为 GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2 / RTX 5090，驱动 595.71.05；启动核验时总显存 32607 MiB、空闲 32110 MiB。本轮前置共实际执行 2 次只读 GPU 元数据查询，第一次通过硬件检查后因授权记录类型错误停止在 CPU gate，未启动模型；第二次保留完整硬件收据。授权记录中的 temperature 0 被修复为冻结计划要求的 float 0.0：只追加 HUMAN_AUTHORIZATION_RECORD_G3_METRICS_V3_CONTINUE_V2_20261002.json 与 G3_METRICS_V3_AUTHORIZED_SCOPE_V2_20261002.json，保留失败版原字节、用户题目及原话“继续”。实验源码、请求、预算和范围均未改变，没有为记录格式修复索取新授权。最终 scope SHA 为 74bb817fbc34db38064ed6b0b7de7065f2bea101f3faa9a8f16c300fb821eb3c。

实际实施内容是启用上一轮已完成 CPU 核验的原统计接口共同修复。三作业均安装并在 finally 中成功撤销 compat_offloading_stats_dict.py，witness.installed 为 false，原 producer 每次仍调用一次、原消费者和统计日志功能保持，observer_fault_types 为空。上次 populate01 的 dataclass/dict 断言未再发生。本轮没有修改作者 py-kvcache/vLLM 文件、原采集器、模型执行器、已安装包或 frozen 运行时。load_planner/新策略仍 off，成本曲线仍为 null，精确 prefix、共享 staging、预加载、复制合并和作者异步 I/O 路径保留。

populate02 返回全部 9 个请求，其中 6 个主请求输出 [398]，3 个 sentinel 分别输出 [82]、[1478]、[486]。正常关闭后发布 24 个文件，共 22020096 B（21 MiB），发布清单 SHA 为 1707c51847e41fce2e5a46b4d84ee41490ad65fd84997c2b9077ae833c099efc。paired02 启动前核验该发布清单，后续在独立新进程内恢复缓存。

paired02 的每个 gSSD 样本均由预加载路径实际从文件读取 8 × 917504 B = 7340032 B，src_preload=8 / src_cache=0，缓存 token 数为 128；紧随其后的 gMem 样本 src_cache=8 / SSD 读字节为 0。6 个 paired 主请求均输出 [398]，cold3 也均输出 [398]。这证明本点确实经历 SSD→staging→GPU 恢复以及 staging 命中，而不是用已有 GPU KV 冒充 SSD 读取。没有进行 top5/logits 数值 reference 或生产 KV 全字节比较，因此单 token 相同不能替代内容一致性资格。

正常 shutdown 后，populate 的 AIO accepted/completed/reaped 为 24/24/24，paired 为 72/72/72；两者 outstanding/pending/ready/unreaped 与 native 活跃 parent、ring_ops、pending_copies 等均为 0，closed/drained 为 true。每个原引擎 shutdown 正常返回一次，每个 OS 会话的关闭前后残留成员均为空。stage_accounting 仍为 UNKNOWN_NOT_ENABLED，physical_release_credit 与生产释放/KV byte identity 等资格仍为 false；这里只证明原作业关闭与尾部回收，不将其解释成具体物理资源释放对等待的收益。

128-token prefix 单点的非 warmup 原始计时如下，单位 ms：

| 原路径 | rep1 | rep2 |
| --- | ---: | ---: |
| f：重算 | 19.657 | 22.061 |
| gSSD：SSD 预加载/恢复 | 247.179 | 384.790 |
| gMem：staging 命中 | 41.398 | 35.643 |

计时来自原 metrics.first_token_latency 的前端 wall-clock TTFT，并非 isolated I/O 或 CUDA event 成本。rep0 的预热计时排除。这个短 prefix 点的 SSD/staging 实测 TTFT 高于重算，当前没有速度收益；每路只有 2 个非 warmup 样本，不做统计或方法优势结论。新研究策略关闭，本试验也未比较 C00/C01/C10/C11，不能把这些结果当作方法有效/无效的完整判定。后续应通过成本准入校准判断哪些长度与负载值得恢复缓存。

实际执行命令保存在 gpu-metrics-v3-actual-20261002 下的 COLD02_LAUNCH_COMMAND.json、POPULATE02_LAUNCH_COMMAND.json、PAIRED02_LAUNCH_COMMAND.json。三个入口均为 .venv/bin/python -B run_g3_calibration_pilot_v3.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --mode cold/populate/paired --source-lock artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002/gpu-source-lock-metrics-v3-final.json --scope-record artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002/G3_METRICS_V3_AUTHORIZED_SCOPE_V2_20261002.json --launch。入口仍调用原 run_gpu_stage.py，每作业 300 秒执行限时并预约 20 秒收尾，最大三次/960 秒。

联合 CPU 命令和退出结果保存在 THREE_JOB_JOINT_CPU_RECHECK_COMMAND.json / THREE_JOB_JOINT_CPU_RECHECK_RPC_RESULT.json：CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S，直接使用冻结原函数 prior_results、validate_job_result、verify_point_pilot 重算，未创建新核验器。逐次重新绑定原 guard/ledger、scope、原 argv、完整源关闭收据、fresh PID/SID、统计修复撤销及原配置。联合结果保存为 ACTUAL_THREE_GPU_JOBS_AND_JOINT_CPU_RECHECK.json（27077 B，SHA c86f523f1dbb19478e305aa133694151862e1802b7465a772e90197c2f0965ab）。

此前服务器 77/77 CPU 合同测试已通过，代码未变所以本轮没有重复这些测试。实际三次 GPU 验证与联合 CPU 核验均通过。每个作业父进程关闭后以及联合 CPU 核验实际核验 4075 个冻结引用；源锁始终为 b8ad831c294d978e36a0b1a6343118516af5c55a6e0a9c4380ff6644ec91092b。66 个选定原始 GPU 文件共 896584 B，下载后逐字节/SHA 均匹配，含原生成输出、trace、配置、guard、runtime、源码关闭收据、发布清单和预算账本。私有模型/SDK/编译缓存及所有旧实验仍留在原路径，没有删除数据。

实际原始结果位于服务器 /root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runs/server09-g3-calibration-cold-02、server09-g3-calibration-populate-02、server09-g3-calibration-paired-02。镜像证据位于服务器 /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002/gpu-metrics-v3-actual-20261002，且镜像到本地同名目录。新交付清单及两侧核验单独追加，历史 97/24/90 文件交付清单保持原字节。最终 GPU 账本 SHA 为 be14eb7e34ccde501481777340ec3baf62e1f8511436614324ff976585ebf0f3，联合核验时磁盘空闲 12602847232 B，仍高于 8 GiB 空间底线。未下载、修改系统/驱动或删除数据。

当前授权的三作业包已经结束，后续 GPU 需要新的明确范围；现在可继续 CPU 分析与下一包准备，无需运行 GPU。本轮能确认当前 AutoDL 环境的原生缓存单点路径可运行；不能宣称整个 P3/P4 完成或新策略性能有效。

下一最小包的只读接口审计已保存为 gpu-metrics-v3-actual-20261002/NEXT_MINIMAL_GPU_PACKAGE_CPU_AUDIT.json：先复用原 acquire_native_aio_costs.py，cold 加 --native-hot-diagnostic --diagnostic-logprobs，paired 加 --cached-reference-logprobs，共 12 个单 token 请求，保持 domain1024/size128/nseq1/io4 和当前共同修复，通过原 analyze_cached_references.py 核对 GPU-hot 与 SSD/staging 输出及 top5 数值参考。参考计时不进入成本拟合。当前固定启动参数/结果门禁不能运行这个参考包，必须先在 CPU 完成薄参数适配、数值门禁、输入缓存清单和两作业计划冻结；目前没有新参考源码、新 GPU 计划或新授权，不能直接启动。

之后成本曲线仍至少缺两个测量点、每点每路至少四份非 warmup 样本及独立会话；旧 heldout 生成器的 2048/16256-token 范围不符合当前 1024 domain，需要先做 CPU 范围与内容分离准备。生产 KV 逐字节一致、真实释放依赖/干扰计时和持续 decode 的 P4 四臂比较尚未通过。
