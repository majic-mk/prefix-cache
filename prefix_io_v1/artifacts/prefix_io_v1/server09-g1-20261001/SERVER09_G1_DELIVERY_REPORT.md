新服务器 G1 传输资格已经通过。off、shadow 各实际运行一次，共通过 8 个原生传输案例；账本新增 GPU 消耗 18.101658 秒，没有超出本轮两次的限定授权。当前无需为这些传输接口继续换服务器。尚未加载模型、运行真实 decode 或证明方法收益，P4 未完成。

这是迁移后新增的交付记录。先前 server09-migration-20261001 中 GPU_runs=0 的报告保留原字节，描述的是本轮 G1 之前的状态。

| 实际运行 | 结果 | 资格程序耗时 | 原累计账本计费 |
|---|---|---:|---:|
| server09-p4-native-off-01 | PASS_REAL_GPU_P4_NATIVE_OFF | 6.337169 s | 8.752716 s |
| server09-p4-native-shadow-01 | PASS_REAL_GPU_P4_NATIVE_SHADOW | 6.625214 s | 9.348942 s |

服务端工作区：/root/autodl-tmp/prefix-io-v1-handoff/project。
GPU：GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2；迁移 G0 文件层审计为 RTX 5090、驱动 595.71.05。服务器现有驱动和系统未修改。G1 只证明本轮合成 INT8 page/tensor 传输所需代码路径可用，不能据此认定模型算子和完整 vLLM 执行兼容。

实际修改只有两个已审查的运行脚本：experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py 和 run_gpu_stage.py。新增可选 --permissions-path，使实际授权文件的 path/bytes/SHA 与 scope、source lock、子进程 argv 绑定；默认旧权限路径保持原行为。资格主体、原累计预算账本、进程 session 清理与 native drain 复用既有实现。没有重写缓存引擎或模型执行器，没有修改 attention、精度、模型权重或研究调度策略。两份旧脚本原字节保存在 historical-before/，HISTORICAL_SOURCE_PATH_MAP.json 记录版本映射。

新权限文件 experiments/prefix_io_v1/configs/permissions.server09.g1.yaml 仅绑定本机 GPU 的 G1 off/shadow；原 permissions.yaml 原字节保留。有效文件明确禁止模型下载、驱动/系统修改、删除/合并既有数据、租用、付款及 remote push；没有沿用旧 GPU 的 AUX 授权。原 8 小时总预算沿用，没有重置。

新源锁 gpu-source-lock.json 重新冻结 2,052 项实际当前内容。两份新脚本 SHA 为：
- qualify_p4_native_gpu.py：5d9604fc07d80495a49febfcd488cb2601f74a2bef107261c8e623a4d0a1c77d
- run_gpu_stage.py：3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a
- 当前源锁：5b438ebc1ec014c22d1d501915600b4cd9bc777ad9f05da1f7ac9b1a18cc1181
- G1 权限文件：eebc8ff9b3ca09c844514847e615099cf96c80d0361c7a493e0d5fdfc5a10e20

旧 P3/P4 锁未改写；它们是历史版本记录。两个 canonical 脚本升级后，不能把旧 SHA 声称为当前路径 SHA。POST_CPU_SOURCE_AND_BUDGET_CLOSURE.json 已实际再次验证新源锁全部 2,052 项、旧/新权限、账本以及 context v3 的 3 候选和 5 原始源引用。

版本保持：工作区 HEAD cc7898b1ba59d89ce7fdbb186ded21880f1adf08；py-kvcache 3abba7a502d553f6e7e2e58b92086487e3395d7e；作者 vLLM 817a7e3124f817cd6e549581d3e5483207a753a4。项目 CPython 3.12.3；metadata 为 py-kvcache 0.1.0、torch 2.11.0、vLLM 0.1.dev1+g817a7e312、transformers 4.57.6、pytest 9.1.1。完整 metadata 源引用保留在上一交付 SERVER09_VERSION_LOCK.json，本轮版本演进以新实际源锁为准。

| 能力 | 本轮真实证据 | 尚未授予的资格 |
|---|---|---|
| store 与 age-store | 真实 D2H 和 Linux AIO SSD 写入；精确内容通过 | GPU allocator 释放收益 |
| restore | 真实 SSD read 与 H2D；精确内容通过 | 模型前缀命中性能 |
| shared staging / copy fusion | 双消费者只读一份 SSD；H2D 为两份实际拷贝字节 | 并发模型收益 |
| accepted I/O 正常生命周期 | accepted/completed/requested/transferred 一致，failed/inflight 为 0，AIO reaped/closed/drained，父任务退出 | 故障注入与 zero-ordinary-quota 研究路径本轮未运行 |
| off / shadow | 原路径完成；shadow 记录为 valid、fault=null | 没有准备成本表、生产 ETA、已资格 owner/load 或 release credit |
| 完整输出的冷 prefill CPU 校验 | 新独立 schema3 候选 27/27 服务器测试通过 | 实际模型 trace、GPU timer 真实性与 paired 成本资格 |

每个模式四项物理字节：

| 案例 | SSD read | SSD write | H2D | D2H |
|---|---:|---:|---:|---:|
| store 66 块 | 0 | 8,650,752 | 0 | 8,650,752 |
| age-store | 0 | 131,072 | 0 | 131,072 |
| restore | 8,650,752 | 0 | 8,650,752 | 0 |
| shared 两消费者 | 8,650,752 | 0 | 17,301,504 | 0 |

实际 staging 为 16,650,239 B，低于 16 MiB 的 16,777,216 B 上限；accepted parent 峰值至多 2。store 出现一次原有 backpressure wait，随后正常 drain；没有把未接纳任务当作完成。AIO 的 198 个 accepted 操作包含 open/read/close，不能写成 198 次数据读。

生命周期继续区分 source-safe、buffer-reusable、cache-visible 和 durable。D2H 完成不等于完整父任务完成，Future.done 或单队列为空不能授予资源释放。正常验证以每阶段原生计数、已接纳 I/O 全量完成、父任务退出和 owner 原关闭状态收口；没有推断真实 allocator 已释放。shadow 的 gpu_release_credit=null、gpu_qualified=false、production_eta_qualified=false、成本 NO_PREPARED_TABLE，继续回退原生 U。ETA 不持有资源 owner，也不建立新工作队列。

本轮 GPU 实际执行命令如下，均使用外层 --launch；由原 guard 在已预留 session 内执行子程序：

```sh
cd /root/autodl-tmp/prefix-io-v1-handoff/project
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='third_party/work/prefix-io-p4-02-cpu/src:experiments/prefix_io_v1/scripts' .venv/bin/python experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py --mode off --name server09-p4-native-off-01 --launch --scope-record artifacts/prefix_io_v1/server09-g1-20261001/GPU_STAGE_AUTHORIZATION.json --source-lock artifacts/prefix_io_v1/server09-g1-20261001/gpu-source-lock.json --permissions-path experiments/prefix_io_v1/configs/permissions.server09.g1.yaml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='third_party/work/prefix-io-p4-02-cpu/src:experiments/prefix_io_v1/scripts' .venv/bin/python experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py --mode shadow --name server09-p4-native-shadow-01 --launch --scope-record artifacts/prefix_io_v1/server09-g1-20261001/GPU_STAGE_AUTHORIZATION.json --source-lock artifacts/prefix_io_v1/server09-g1-20261001/gpu-source-lock.json --permissions-path experiments/prefix_io_v1/configs/permissions.server09.g1.yaml
```

两次 exit 和 child_exit 均为 0，未超时、未中断、无错误；session 3341、3754 清理前后活成员均为空，session_drained=true。独立只读审查确认 process.log 内结果 JSON 与 result.json 语义一致；账本历史 221 个 events 保持，仅追加 2 个。唯一日志警告是固定 Transformers v4 的弃用提示。

CPU 验证分开记录，不用 CPU fixture 冒充真实 GPU：
- 迁移阶段项目 CPU 回归实际 1,956 passed、16 skipped、0 failed；skip 为 13 个无 vLLM fallback、3 个平台受限 io_uring，Linux AIO CPU 路径通过。未重算原来排除的 12 个历史 P3 fixture。
- 权限适配候选的服务器 CPU 测试 21 passed、0 skipped、0 failed；与部署后两个脚本的字节一致。不是在旧 source lock 下重新运行整体迁移回归。
- 独立完整轨迹 schema3 候选实际服务器 27 passed、0 skipped、0 failed，1.167 s；本机既有 Python3.12.14 的 27 项通过也保留，先前 Python3.8 的不兼容和 fixture 修正记录均保留。
- schema3 仅允许显式冷 prefill 的真实 prior-context=0；不把 0 改为 1、不把 computed+scheduled 冒充成本 context、不删首帧和输出。旧 v2 源/globals、正 context decode selection、paired 成本 loader 均未改；真实 GPU、production、effect 资格仍为 false。

实际服务器 schema3 命令：

```sh
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S artifacts/prefix_io_v1/server09-context-v3-20261001/test_p4_complete_trace_context_v3.py --control-source third_party/work/prefix-io-p4-02-cpu/src
```

账本最终本轮消耗 18.101658063 s，累计 16,538.664544709 s，8 小时余 12,261.335455291 s，约 3.406 h；active_reservation=null。两次 G1 作业名均已用完，计划 400 s 未花满不授权第三次。账本末态 SHA：32770dbca6f2e37d274f5007d151692e0a4c910a688d352abec79d60318ffaae。CPU 工作未新增 GPU 消耗，未下载模型、删除/合并既有缓存或清空账本。PRIMARY 空闲约 12.1 GiB，AUX 不扩额；G1 小空间方案不等于批准新 3 GiB AUX 轮次。

服务端证据目录：
- artifacts/prefix_io_v1/server09-g1-20261001/：授权、scope、计划、2,052 项源锁、部署历史、实际 G1 汇总、独审、收口与范围使用记录。
- experiments/prefix_io_v1/runs/server09-p4-native-off-01/ 和 server09-p4-native-shadow-01/：原始 details/result.json、process.log。
- artifacts/prefix_io_v1/server09-context-v3-20261001/：独立候选、冻结源引用、本机测试历史、实际服务器测试与独审。
- artifacts/prefix_io_v1/server09-migration-20261001/：迁移审计、受保护数据和旧源验证、原 CPU 回归、完整前期命令。旧结果未覆盖。

本轮结论是新服务器的限定 native 传输可行。模型执行与收益尚不能下结论。下一允许阶段继续 CPU 的薄运行时观测连接、实际计时/I/O 来源绑定、schema3 paired 成本与完整输出投影准备；均需保持原调用、关闭后原路径、mandatory progress 和历史证据。下一轮 G2 真机采集应先形成可复查的 runner、全部源锁、精确新授权、存储/累计时间门禁再进入 GPU。完整模型 step/output/drain、真实 load/时钟、合法单阶段成本、生产 ETA 和实际 owner/release witness 通过前，不启用 I/D/J 或启动 P5–P7；SLO 仍 null。
