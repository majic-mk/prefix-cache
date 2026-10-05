# 租用 GPU 前置 CPU 交付

本轮可在无卡阶段执行的实现、服务器 CPU 测试、完整源码冻结和原数据保护核验已完成。下一允许步骤是单次强基线 U/off 资格验证。CPU 完成不等于系统 GPU 可行性、P4 策略效果或论文收益已经成立。

完成时间 UTC：2026-10-04T16:44:47.038064+00:00。20261004 是沿用的本轮固定目录命名。
服务器 connect.westd.seetacloud.com:24828；项目 /root/autodl-tmp/prefix-io-v1-handoff/project。密码未写入交付材料。

## 实际改动

runner/ 沿用作者 LLMEngine、py-kvcache、LoadPlanner、精确 Prefix Cache、共享 staging、预加载、复制合并及原异步流水线；只增加薄运行接线、完整输出/生命周期检查和有限候选绑定。普通 preload 保留原 ready、owner 和资源路径，仅加一次有限决策；原批量复制合并保留。

activation/ 为独立私有成本资格接线，CPU 表、公开字典、布尔 GPU 标记均不能开启策略；当前模型/内核/条件不匹配或不在真实覆盖集合时回到原路径。activation/control_observation/ 记录 scheduler、sampling、output、controller 四类原方法的紧凑 host 区间和实际 preview 尝试。host wall 可含原等待，禁止拿 host wall 减 CUDA 时长编造纯 CPU 控制成本。reserve 只能由真实 development shadow、独立 deadline、已结束的原 guard 和实际 eligible 窗口连接后发行。

protocol/ 使用原 P3 的 12 个请求作资格负载，原 prompt token IDs、到达顺序和原 0.5 请求/秒保持；每个请求要求完整 128 输出 tokens。它是受控机制数据，不能作为自然业务 heldout 或正式收益证据。

entry_control_v3.py、freeze_prerental_sources_v3.py、audit_project_sdk_cpu.py 提供源/资产检查和明确限额入口；复用人的持续 GPU 授权和原 8 小时账本，实际出现设备后才绑定 UUID。review/ 独立复核实际原 API、拒绝条件、回退及成本来源。共同接线修复与研究策略分开；关闭新策略保留原执行路径，共同观测仍有开销，不能称零开销回退。

## 服务器实际测试与命令

| 最终套件 | 通过 |
| --- | ---: |
| 协议和原受控资格负载 | 30 |
| 有限成本资格拒绝/原估计器数学 | 19 |
| host 观测/reserve 接口 | 29 |
| 原执行器/有限当前条件/激活接口 | 47 |
| 根入口 v3 | 10 |
| 七组独立复核 | 36 |
| 合计 | 171 |

最终套件 **171 通过，0 失败，0 错误，0 跳过**。旧版本和重复测试没有加入分母。CPU fixture 不能签发真实 GPU 能力。每条命令完整 argv、cwd、CUDA_VISIBLE_DEVICES、stdout、stderr、退出码和耗时保存在对应 CPU_PRERENT_*_COMMAND.json、_STDOUT.log、_STDERR.log、_RESULT.json。

实际最终命令如下（均只在服务器 CPU 执行）：

~~~sh
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/protocol/verify_prerental_cpu.py --source-dir /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/source_inputs --controlled-source-dir /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/source_inputs --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/SERVER_PROTOCOL_CPU_RESULT.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/activation/verify_activation_cpu.py --original-control-root /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/i_bridge/frozen/control --pair-config /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/PRIVATE_U_I_CONFIG.json --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/SERVER_ACTIVATION_CPU_RESULT.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/activation/control_observation/verify_host_control_cpu.py --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/SERVER_HOST_CONTROL_CPU_RESULT.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/review/test_prerental_rejection_review.py --source-dir /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/source_inputs --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/SERVER_PROTOCOL_REJECTION_REVIEW_CPU_RESULT.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/review/test_original_request_id_review.py --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/SERVER_ORIGINAL_ID_REVIEW_CPU_RESULT.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/review/test_finite_capability_review.py --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/SERVER_FINITE_CAPABILITY_REVIEW_CPU_RESULT.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/review/test_observer_capacity_review.py --observer-source /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/g2_worker_observation.py --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/SERVER_OBSERVER_CAPACITY_REVIEW_CPU_RESULT.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/review/test_raw_request_id_review.py --runner-source /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/strong_native_cost_runner.py --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/CPU_PRERENT_RAW_ID_REVIEW_01_EVIDENCE.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/review/test_host_control_refusal_review.py --observer-source /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/activation/control_observation/host_control_observer.py --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/CPU_PRERENT_HOST_REFUSAL_REVIEW_01_EVIDENCE.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/review/test_host_boundary_fallback_review.py --binding-source /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/host_boundary_binding.py --observer-source /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/activation/control_observation/host_control_observer.py --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/CPU_PRERENT_HOST_FALLBACK_REVIEW_01_EVIDENCE.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/verify_runner_cpu.py --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/SERVER_CPU_RESULT.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/test_entry_control_cpu_v3.py
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/freeze_prerental_sources_v3.py --project-root /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/strong_native_cost_runner.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --prepare-plan --source-lock artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/PRERENT_SOURCE_LOCK_V3.json --pair artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/PRIVATE_U_I_CONFIG.json --input-manifest artifacts/prefix_io_v1/server11-native-cost-v1-20261003/SSD_INPUT_MANIFEST.json --geometry artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/kv_geometry.json --output artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/STRONG_RAW_CPU_PLAN_TEMPLATE.json --job-id server12-strong-exact-cal01
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/entry_control_v3.py preflight --project-root /root/autodl-tmp/prefix-io-v1-handoff/project --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/SERVER_PRERENT_ENTRY_PREFLIGHT.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/audit_project_sdk_cpu.py --project-root /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/run_cpu_source_protection.py --project-root /root/autodl-tmp/prefix-io-v1-handoff/project --phase AFTER
~~~

完整冻结实际读取 **4,851 文件 / 16,047,190,949 字节**，源锁 SHA-256：59b1531e178cb71ec0d26a86c6123d9edca96a950ab27065b63d5eeab182d052。原保护清单 **4,816 文件**、历史真实 GPU 交付归档和原账本 BEFORE/AFTER 字节相同；本地原有 **14 用户文件** SHA 相同。没有删除已有实验数据或缓存。

实际 prepare-plan 前两轮拒绝日志保留：01 为调用方传绝对路径；02 为源锁漏列已有强基线配置。v3 补齐真实原文件，并由 thin runner.REQUIRED 检查必需执行/生命周期闭包；03 实际生成成功。模板 gpu_uuid=null、cpu_preparation_only=true、不可执行，不能充当 GPU 实验或真实成本资格。

## 真实 GPU 情况、资源与下一允许阶段

本轮 **GPU 作业 0、编译器运行 0、模型/共享库加载 0、数据删除 0、驱动/系统修改 0**；NVIDIA 设备节点 []。没有租赁、支付或下载模型。

PRIMARY 当前剩 **49.74 GiB**，首轮只预留 128 MiB 并保留 8 GiB 底线，目前无需扩盘。原 8 小时累计 GPU 预算剩 **4121.105 秒，约 68.69 分钟**，账本 SHA 未变。

项目内 SDK 字节已核验；无卡模式的宿主驱动为零字节占位，完整 SDK runtime 未资格通过。切 GPU 模式后必须核验真实驱动、实际 UUID/空闲资源、完整源码、存储和剩余预算，不匹配即停，不修改系统/驱动、不使用 stub 库冒充驱动。

切当前实例到 GPU 模式后，下一入口为：

~~~sh
.venv/bin/python -B artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/entry_control_v3.py launch --project-root /root/autodl-tmp/prefix-io-v1-handoff/project
~~~

此 launch **本轮未执行**。入口复用已有授权，实时核验后仅执行一次 server12-strong-u-qual-off01，强 U/off 最多 300 秒，加 20 秒收尾。真实完整输出、原 shutdown 返回、native/OS 会话排空及原 guard 闭合全部通过后，才允许单有限成本单元采集（最多 900 秒加 20 秒收尾，存储预留 512 MiB）。后续仍受原累计预算限制，不保证现有余量可以完成全部阶段。

真实强域成本、有限策略动作、controller reserve、性能收益均未验证。正式效果阶段还需要独立预声明 deadline/SLO、实际 development shadow 和自然 heldout 数据；不从 U 的慢步骤反推有利预算，不把旧单文件诊断当当前强基线成本表。零 scheduled metadata/空输出 prefill 等不支持形状保留 UNKNOWN，不能丢弃后宣称全负载通过。现阶段不承诺提升或论文结论。

## 证据

本报告与 FINAL_PRERENT_CPU_DECISION.json 位于 /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004。同目录保留全部本轮代码、配置、实际 CPU 原始日志、源锁/证明、保护清单和独立复核。交付归档只包含小型代码/报告/配置/日志，不嵌入模型或 SDK；实际归档及本地逐文件 SHA 核验另见 PRERENT_CPU_DELIVERY_RESULT.json 和 LOCAL_CPU_AUDIT_BYTE_VERIFICATION.json。
