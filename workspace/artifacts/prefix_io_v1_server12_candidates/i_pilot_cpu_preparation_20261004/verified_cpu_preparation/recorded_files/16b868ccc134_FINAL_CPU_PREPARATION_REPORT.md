# 服务器 CPU 准备交付 — 2026-10-04

本轮有界 CPU 组件和下一次机制标定入口准备完成，服务器终版六组测试共 **209 项通过，0 失败、0 跳过**。这不代表完整 P4 效果实验已经准备完成：正式强基线 U/I 的 GPU runner 尚未绑定，独立开发预算、自然请求流、对应成本覆盖及真实 on 生命周期资格仍缺失。**本轮实际 GPU 运行 0 次、GPU 时间 0 秒，没有性能提升结论。**

服务器：`root@connect.westd.seetacloud.com:24828`，容器 `autodl-container-mrp23kaegw-72cc8dbc`。工作区：`/root/autodl-tmp/prefix-io-v1-handoff/project`，Git HEAD `cc7898b1ba59d89ce7fdbb186ded21880f1adf08`。本轮交付目录为该工作区下 `artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/`。所有实际服务器命令、标准输出、标准错误及退出结果均保留为 `CPU_*_COMMAND.json / STDOUT.log / STDERR.log / RESULT.json`。

## 实际新增内容

| 位置 | 改动与目的 | 验证边界 |
|---|---|---|
| `calibration_v2/` | 迁移原生标定入口；使用新 prompt/seed 家族；更新回执的源引用、序列化名称及私有 SDK 路径元数据；将用户持续 GPU 授权接入原限时、资源和累计预算检查 | 原生 runner 顶层函数与成本计算函数 AST 保留；未运行模型，未生成真实成本表或设备资格 |
| `i_bridge/` | 冻结已有控制模块与 reactor 源码，执行原函数链的 CPU 回放，覆盖 off/shadow/interference、容量、延期、STOP、故障和排空 | CPU 资源替身；未提交原生 I/O 或 GPU 任务；旧普通候选仍因成本超预算而拒绝 |
| `release_observer_v2/` | 在原 NativeFlushProbe 上增加只读、有限容量的 owner/依赖观测适配与原 scheduler ACK 生命周期回放 | 不取得所有权，不创建第二套释放协议；未知分配、generation、保护者或 CUDA fence 保持未知；生产释放资格始终为 false |
| `strong_baseline/` | 实现强 U/I 共同配置桥，回放作者配置解析、完整原 `_build_planner`、原 P4 参数解析及原 trace 解析/调度函数 | 两侧保留 LoadPlanner、精确 Prefix Cache、预加载、共享 staging、复制合并和异步流水线；只允许存储命名空间、run identity 与 off/interference 策略区别；未绑定真实效果 runner |
| `protocol/` | 冻结新标定、开发、评估家族，规定 AB/BA 顺序、逐 token 计时、失败/超时/排空计费、训练与评估隔离及门禁 | 独立 deadline、reserve、budget 与评估 SLO 保持 null；不能从开发 A 最大值或事后阈值获得 on 资格 |
| 根目录汇总及交付工具 | 原始源保护、真实服务器回执联接、末态记录和逐字节归档核验 | 不加载模型、GPU 库或缓存引擎；CPU 回执不能升级为性能证据 |

缓存引擎、模型执行器、作者源码及原累计 GPU 账本没有改动。共同配置、诊断入口与研究策略分开。off 仍走原路径；本轮 CPU 回放检验其不收集可选策略观测。单文件路径的 `dispatch_controller=None` 是已有设计，未为通过测试强行接入新调度器。

## 已执行命令与服务器结果

以下命令都在上述服务器工作区实际执行，环境 `CUDA_VISIBLE_DEVICES=''`。为便于阅读，以 `D=artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004` 表示目录；逐个原始 argv 见对应 COMMAND.json。

| 证据标签 | 实际命令（省略相同工作区前缀） | 终版结果 |
|---|---|---|
| `CPU_CALIBRATION_UNIT_02` | `.venv/bin/python -B -I -S -m unittest discover -s D/calibration_v2 -p 'test_*.py' -v` | 109/109 |
| `CPU_I_BRIDGE_01` | `.venv/bin/python -B -I -S D/i_bridge/test_i_bridge_cpu.py --output D/SERVER_I_BRIDGE_CPU_RESULT.json` | 20/20 |
| `CPU_OWNER_ADAPTER_02` | `.venv/bin/python -B -I -S D/release_observer_v2/verify_owner_preparation_cpu.py --output D/SERVER_OWNER_PREPARATION_CPU_RESULT_V2.json` | 22/22 |
| `CPU_CALIBRATION_AUTH_REVIEW_02` | `.venv/bin/python -B -I -S D/calibration_review/test_standing_calibration_review_cpu.py --output D/SERVER_STANDING_REVIEW_CPU_RESULT.json` | 21/21 |
| `CPU_PROTOCOL_TESTS_01` | `.venv/bin/python -B -I -S D/protocol/test_i_pilot_protocol.py -v` | 22/22 |
| `CPU_STRONG_BASELINE_01` | `.venv/bin/python -B -I -S D/strong_baseline/verify_strong_cpu.py --output D/SERVER_STRONG_CPU_RESULT.json` | 15/15，53 项输入 SHA 前后不变 |
| `CPU_CALIBRATION_SOURCE_FREEZE_01` | `.venv/bin/python -B -I -S D/calibration_v2/control_native_cost_job.py freeze` | 4,749 项新标定源/资产字节核验通过，约 116.8 秒 |
| `CPU_SOURCE_PROTECTION_BEFORE_01 / AFTER_01` | `.venv/bin/python -B -I -S D/run_cpu_source_protection.py` 的对应阶段 argv | 各核验旧 4,816 项，共 16,064,147,630 字节；前后完全一致，约 88.9/85.6 秒 |
| `CPU_CALIBRATION_PREFLIGHT_01 / BYTE_JOIN_01` | 原始 argv 见各 COMMAND.json；执行 CPU 预检及协议验证器 | 原始实际回执联接通过，状态 `CPU_READY_FOR_GPU_CALIBRATION`；效果仍 blocked |
| `CPU_FINAL_EVIDENCE_JOIN_02` | `.venv/bin/python -B -I -S D/summarize_cpu_preparation_v2.py` | 六组实际测试联接 209/209；未绑定强效果 runner |

209 为六组终版测试的合计，未把重复运行或源哈希数量算成测试。标定入口原型曾因旧测试目录和授权 fixture 不匹配失败；授权审查初次 CLI 缺 `--output` 而退出；第一次汇总脚本读取了外层日志，未读取嵌套测试日志，修正为 v2 后通过。早期日志和源码保留，不能将这些失败当成 GPU 实验结果。旧 19 项 observer 验收被终版 22 项替代，不重复计数。

CPU 回放执行了原有函数和固定源码，但 GPU 资源、原生 worker 和数据对象使用明确标记的 CPU fixture。因此这些测试证明配置、分支和生命周期约束的 CPU 实现行为，不能证明真实 CUDA completion、真实缓存命中率、吞吐或时延收益。原 `_build_planner` 的既有 prefix-cache/recompute 提示也保留；配置检查不等于成本模型已适配新强基线。

## 源锁、数据保护与证据

- `calibration_v2/REVISION_SOURCE_LOCK.json`：4,749 项，SHA-256 `20fbabe41c5fbef7a3e9a99ed48cea11b36afe5dd43e6c7861637ea7032f51e5`。`CALIBRATION_CPU_SOURCE_PROOF.json` 绑定实际 freeze 命令和结果；后续联接复用这份实际证明，没有宣称重复哈希了全部模型。
- 旧受保护源锁 SHA-256 `4aab66888592a8fdf6544513003e9b7382f2194f3da2407b8991073a8e86ffef`。`SOURCE_PROTECTION_BEFORE.json` 与 `SOURCE_PROTECTION_AFTER.json` 逐项一致，包含旧 GPU 证据归档。只新增本轮小文件，未删除实验数据。
- 原累计账本 SHA-256 `60368b94c494d15227eb528273e7b4a1a20119c6f371767ffde5cc96aa8f52f9`，与启动快照逐字节一致，`active_reservation=null`。原 8 小时预算剩余 **4,121.104761 秒（约 68.69 分钟）**，本轮未预留新 GPU 作业。
- 服务器 tracked Git 状态无改动；本机用户已有的 14 项未提交文件 SHA 前后完全一致，见 `LOCAL_USER_WORK_BEFORE.json / AFTER.json`。
- `CPU_PREPARATION_POST_STATE.json`：末次只读检查 NVIDIA 设备节点为空，数据盘可用 53,413,924,864 字节（约 49.75 GiB）。本轮 GPU 0 次，不安装包、不修改系统或驱动、不下载模型、不清理缓存数据。
- `FINAL_CPU_STAGE_DECISION.json` 是本轮完整机器汇总；`CPU_NEXT_PHASE_DECISION.json` 是标定入口的较窄判定；`SERVER_STRONG_CPU_RESULT.json` 独立记录强基线共同配置，不能认为它已自动纳入新标定源锁。

服务器归档使用 `pack_cpu_preparation_bytes.py pack --project-root /root/autodl-tmp/prefix-io-v1-handoff/project`；随后将小型源码/日志归档取回本机，使用 `verify --archive ... --manifest ... --result ... --result-sha256 <服务器实际结果 SHA> --output-dir ...` 逐成员核验。最终归档数量、SHA 与两侧核验结果由 `CPU_PREPARATION_FINAL_DELIVERY_VERIFIED.json` 记录。归档字节 PASS 不赋予模型或策略运行资格，也不宣称在本机重验全部 4,816 个模型/SDK 资产。

## 对下一阶段的实际含义

1. 有 GPU 后，先沿用持续授权和原累计预算做实时设备、UUID、源锁、存储及原 guard 复查。下一份已准备入口是 `server12-i-pilot-cal01` 机制诊断，执行上限 1,200 秒并预留 20 秒收尾，尚未预约或运行。它包含 3 对 A/B 的 6 个新进程窗口；不用本轮 CPU 数据替代 GPU 数据。
2. 这份诊断及既有单文件实验使用 `load_planner='off'`，**不是正式 U 的强基线**。其回执不能覆盖保留 LoadPlanner 的强 U/I 配置。完成正式收益验证仍需在真实强配置上完成成本覆盖和 runner 绑定；不能仅换名字复用旧回执。
3. 独立开发 deadline、reserve、预算与评估 SLO 尚无合法冻结值，自然服务 trace 也未绑定。它们与新的强配置成本覆盖到齐后，才允许 ordinary shadow 候选和真实 on 生命周期验收，之后执行完整 U/I 配对评估。不能为获取候选而扩大旧成本门槛，不能重置延期年龄或跳过容量/事件 fence。
4. 协议的 9 个候选作业最多 3,780 秒是条件规划，未实际预留，且不含新增强 planner-on 资格成本。**不能承诺全部效果实验一定能在剩余约 69 分钟内完成。**

已有真实 GPU 的三个 off 记录仅说明对应输出和正常生命周期可运行；它们未开启新策略，不能证明收益。冻结成本上界 16.238752 ms 高于原 A-only 预算 13.171328 ms，普通候选当前为空，缺口约 3.067424 ms。本轮没有放宽这个门槛。若合法新标定和自然负载下仍没有可调度候选，或简单原始方案取得同等结果，应如实停止收益主张，不能凭 CPU 通过断言可发论文或方法永久不可行。
