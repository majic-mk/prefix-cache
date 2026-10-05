# Server08 P3 强简单基线与稀疏干扰资格报告（结构草稿）

> 草稿状态：P3 进行中。此文件仅冻结已核对事实和最终交付结构，不是验收完成声明。计划最终保存为 docs/prefix_io_v1/SERVER08_P3_SIMPLE_BASELINE_REPORT.md；现阶段不修改已有 docs、计划、权限或实验状态。
>
> 核对截点：已完成 primitive、off/shadow 模型资格、normal off 两次复现、四个真实复制后的合成异常用例和独立 CPU 微基准；decode calibration 01 正在运行。其余结果仍为 pending。

## 结论与尚未完成事项

原作者 py-kvcache 与配套 vLLM 继续承担缓存和模型执行。本轮已验证共同父任务准入、实际阶段记账、有限 fixed/pressure 接线及原生 drain 协议的 CPU/小型 GPU 资格。四次完整模型运行全部通过 10 请求 × 128 输出 token 的精确参考与资源排空审计；normal off 两次均出现原生 pending flush 等待，尚未证明 fixed/pressure 能改善该问题。

P3 仍未完成。两次稀疏校准、独立验证、U/F/P/P/F/U 主对照、选定最强简单基线后的 all-hit U/B/B/U 负对照、最终预算/输入保全与汇总均为 pending。不能把已通过的正确性资格、原有 SSD Prefix 复用优势或单次 off/shadow 时间差表述为新增策略收益。

## 路线、版本与改动范围

- 固定路线保持作者 py-kvcache + 作者 vLLM 增量修改，不新增缓存引擎、模型执行器或第二套活动传输队列。
- 当前作者增量 worktree：third_party/work/py-kvcache-p3-15-cpu。实际 GPU Python 搜索顺序为该 worktree → src → experiments/prefix_io_v1/scripts。
- 共同修复包括 accepted parent 的 incoming/active/failed-draining 生命周期计数、有限背压、失败复制先真实同步再终止 Future/释放资源，以及原有 mandatory/下游续接进度。
- off/shadow/fixed/pressure 共享 max_accepted_parents=64 的共同协议。off 不启用研究控制器；shadow 只审计；fixed/pressure 只限制有限性能额度。mandatory 不能越过原生容量、保护事件或真实 drain。
- 阶段观测覆盖 SSD read/write、H2D、D2H、共享 SSD 与共享 copy 额度；实际字节来自原后端成功接受及原完成事件，不以计划字节或队列估计冒充实际传输。
- 原模型路径使用可选 --simple-stage-config；关闭该选项保留原默认路径。来源注册、成本准入、精确 Prefix、共享 staging、预加载、复制合并与异步流水线继续沿用原接口。
- 本轮新增的稀疏校准和 stdlib 离线汇总是独立入口。production 模型共同配置与校准条件域分开；dependency_only/interference/joint 与 P4–P7 未启用。
- 源版本与补丁证明以 execution-lock-01/02/03.json、driver-change.json、native-factory-change.json、native-worktree-seed.json 为准；最终交付需追加最终 SHA 清单和保全证据，不能回写旧锁。

## CPU 与原生协议资格

最终 CPU 主资格矩阵为 **1194 passed、16 skipped、0 failed**，每个子进程 CUDA 初始化均为 false。1194 的构成为当前 native 矩阵 1183 + 历史故障用例 8 + 历史模型 guard 3。历史 11 项保持其冻结 fixture/worktree，单独执行并明确归属；不能把不同 worktree 历史通过数声称为当前 native 故障资格。之前的子集/重跑计数不重复累加。

证据：artifacts/prefix_io_v1/server08-p3-15/cpu-qualification.json 与 full-native-cpu-02-command.json；原始子进程结果位于 experiments/prefix_io_v1/runs/server08-p3-15-cpu-02/cpu-evidence/。

真实 GPU primitive 为 16 cases，通过 off/shadow/fixed/pressure；精确内容正确、全部 AIO drained、最终 parent=0、阶段 accounting valid。其范围是小型真实原生传输与阶段协议，不是完整模型策略收益。

证据：primitive-summary.json、primitive-plan.json、primitive-01-command.json；原始结果目录 experiments/prefix_io_v1/runs/server08-p3-15-simple-primitive-01/。

## 实际 GPU 合成故障资格

四个用例 d2h-partial、h2d-partial、d2h-end_event、h2d-end_event 全部通过；每个用例记录 1 次真实 stream sync、1 次 Future callback。它们先执行作者真实 CUDA 复制，再注入 Python 合成异常，验证实际 drain/资源保护/单次终止协议。

这是 **真实 GPU 复制后的合成异常验证**。没有测试真实驱动故障、设备掉线、真实 Event 运行故障或 CUDA context 崩溃。synthetic_failure_only=true、genuine_driver_fault_tested=false 必须保留。内层用例耗时 16.498603 秒，预算 wrapper 计费 17.920781 秒，不能混用。

证据：native-fault-summary.json、native-fault-plan.json、fault-review-qualification.json 与 experiments/prefix_io_v1/runs/server08-p3-15-native-fault-01/。旧 SHA 的启动 guard 拒绝发生在 GPU 前，没有 GPU 尝试；该错误记录保留，不算实验成功或新增 GPU 用例。

## 已完成的真实完整模型资格与普通阻塞复现

以下四个独立运行各 10 请求、1280 输出 token；共 40 请求/5120 token 均完整精确匹配缓存/原生参考，逐 token 时间证据完整，engine shutdown 和原生物理 drain 通过，3048 个登记源文件保全。共同 parent limit=64，最终 parent=0、accounting outstanding=0；峰值 parent 分别为 9/7/7/7，未达到容量上限。

| 运行 | 完整 cohort 含 drain（秒） | pending flush calls | pending flush 等待（秒） | 实际 SSD read bytes | 实际 SSD write bytes |
|---|---:|---:|---:|---:|---:|
| model-off-01 | 31.411437 | 1 | 1.004606 | 3869114368 | 1896480768 |
| model-shadow-01 | 30.989434 | 1 | 0.662071 | 6525288448 | 1896480768 |
| normal-off-01 | 31.483485 | 1 | 0.795329 | 5736235008 | 1896480768 |
| normal-off-02 | 30.212632 | 1 | 0.992608 | 6525288448 | 1896480768 |

两次 normal off 在既定普通混合长前缀负载中均出现实际 pending flush，提供继续检验简单阶段基线的目标证据。它们不包含人工 I/O 限速。单个 off/shadow 是接线/输出资格，未形成性能资格；实际 SSD read 量和时序存在差异，不能用 31.411437 与 30.989434 的差值声称新策略提升。

推断范围为先前见过的 P3 development families；TTFT/ITL 来自原生引擎输出，未设置客户端 SLO、formal goodput、SSE 测量或未见过的 P5 evaluation。token 间隔与同 run 内请求有关联，独立采样单元是 run。

证据：model-off-01-analysis.json、model-shadow-01-analysis.json、normal-off-01-analysis.json、normal-off-02-analysis.json；原始 result/frozen-config/probe 位于对应 server08-p3-15-* 的 details/。

## 独立 Python 控制开销微基准

每种 mode 3 批、每批 3333 次，实际每 mode 9999 次；表中的范围是三个批均值的最小/最大，不是单次操作分布或置信区间。

| mode | median CPU us/operation | 三批均值范围（us/operation） |
|---|---:|---:|
| off | 1.762 | 1.756–1.784 |
| direct_accounting | 6.670 | 6.620–6.721 |
| shadow | 17.825 | 17.003–18.040 |
| fixed | 17.969 | 17.660–18.044 |
| pressure | 18.097 | 17.746–18.223 |

CPU 时钟为 process_time_ns，统计整个进程 CPU，而非单线程 CPU。测试使用同一完整合成状态、4096 个合成字节/operation、模拟后端成功；没有 GPU、真实 native I/O 或模型执行。结果包含 Python 快照构造、时钟、记账和控制器公开接口调用成本。

这些数据只量化隔离的 Python metadata/controller 微基准，**不是实时 observer CPU 开销、model CPU 分解、GPU 利用率、真实资源释放证明或完整模型端到端开销**。shadow/fixed/pressure 相对 off 的 median 差为 16.063/16.207/16.335 us/operation，相对 direct_accounting 为 11.156/11.300/11.428 us/operation；该差不能直接乘模型 token 数形成端到端性能结论。实时 observer CPU 开销仍未单独量化。

证据：isolated-cpu-overhead.json、isolated-cpu-overhead-command.json。source metadata、输出和文件 I/O 位于计时循环之外；三批只作描述性汇总。

## 稀疏持续 decode 干扰：pending

两次 calibration + 一次 independent validation；每次 7 cells × ABBA=28 windows，单活动请求、16257 prompt +128 outputs=16385。max_num_seqs=2 配置保留。所有 anchors 预分配相同 owned resources，使用作者原始 LLMEngine.step 与异步 I/O handler。

Owned BF16 synthetic KV 为 29360128 bytes；model KV 配置 2118123520 bytes，两者合计不超过 2GiB，独立 handler staging 不超过 1GiB。因此这是一张 **条件于该校准 engine/单请求/合成原生链的表**，与 production 模型 2GiB KV 域有约 28MiB 分配差异；不能复用 d8/C2 成本许可声称同资源域，也不能替代 production 资源释放 witness。

Anchor 包括 no-new-I/O、warm H2D（setup 原 SSD load 后 retained lru，measurement SSD=0）、D2H→write、cold SSD→H2D、joint（读/写 hash 和 source/target storage 不重叠）。units 为冻结的 1/8 稀疏边界；decode step16 起每8步一次预定机会、至多12，只有上一批 native parents 和真实 native resources drained 后才能发下一批，没有补发队列或人为限速。

每窗口必须审计完整 128 输出、hot cached tokens=16256、127 个严格正 ITL、对应 actual stage acceptance/completion bytes、synthetic round-trip 与实际 closed/drained。非 none B 的 native accepted→complete 占用跨度必须与原 worker decode wall 跨度相交；这不是瞬时 DMA/模型 kernel 重叠证明。

三个预冻结条件均为 25%：
1. validation loaded ITL 相对预测误差：abs(predicted-observed)/observed。
2. 两 calibration run loaded ITL 相对 spread：abs(C1-C2)/median(C1,C2)。
3. 每 run A 首尾 drift：abs(Afirst-Alast)/median(Afirst,Alast)。

每 run 先取两个同 role window 的 ITL median 的 median；两个 calibration run 等权 median 预测 validation run。七个 cell 全支持才叫完整 conditional table；失败 cell 为 None，域外/production 状态为 None 并回原有路径。delta 只描述，不做接近零 delta 的相对预测或置信区间，不声称策略收益。P4 不连接该表。

当前状态：calibration-01 running；calibration-02/validation/result audit/table support 均 pending。最终需填写三次 wrapper 实际退出码、预算秒数、完整窗口数、每 cell 误差/drift/spread、unsupported 原因及输入 SHA；未完成不得写 PASS。

证据：decode-calibration-01/02-spec.json、对应 plan/preflight/wrapper；interference-table-criterion.json（SHA 46fd189fc585915256c9c6d410538b0163681a22cf786a71feee2f434d98db9c）；NEW analyze_decode_interference_calibration.py 与最终输出 summary（pending）。

## U/F/P/P/F/U 与 all-hit 负对照：pending

U=研究额度 off，F=fixed，P=pressure；共同修复、父任务准入、作者模型/缓存路径和原 iodepth=8 成本资格相同。按冻结 U/F/P/P/F/U 六次运行，分别报告每 run 完整输出、实际字节、等待、ITL/TTFT、cohort duration 和末尾 drain。每 arm 两个独立 run 等权 median，F/P median 完整 cohort 时间较小者为最强简单基线，完全相同时选 F；不事后调参或挑样本。

之后以该冻结基线 B 做 all-hit U/B/B/U。必须从实际配置证明 B 确为选定基线，GPU Prefix 热状态/工作量一致且测量 SSD read/write=0；median 完整 cohort 时长相对 U 退化目标为不超过2%。满足此目标只表示本轮描述性 negative control target met，不表示 CI、统计显著性、客户端 SLO 或 observer 开销已单独测得。

当前六次开发对照、基线选择和四次负对照均 pending。最终填写原始 run-level 值及最小/最大，不把 token/window 数当独立重复数。

## 命令、GPU 预算与存储

实际完整命令以本轮 *-plan.json、*-command.json 和 wrapper result.json 为准；入口摘要不能替代复现参数。已执行的 CPU 主入口为：

.venv/bin/python experiments/prefix_io_v1/scripts/run_simple_stage_cpu_qualification.py --label server08-p3-15-cpu-02

CPU 子进程设置 CUDA_VISIBLE_DEVICES=""、PYTHONPATH=src:experiments/prefix_io_v1/scripts，并保留 CUDA guard。所有 GPU 经 run_gpu_stage.py 包装，独立唯一 label/output；模型四次每次上限240秒，primitive/fault每次上限240秒；稀疏校准每次上限600秒。没有绕过 permissions.yaml、扩大 GPU 8小时总预算或重用旧缓存合并授权。

GPU UUID：GPU-83ac724c-6d18-6f08-ab96-fb871fb1f4a8。六次已完成 wrapper GPU 计账秒数分别为 17.416658、98.290963、101.116461、99.779255、97.179152、17.920781，合计431.703270秒，全部 child_exit=0、exit=0、session_drained=true。本轮起点历史 GPU wall=11404.191354秒；截点已完成累计11835.894623秒（3.287749小时），剩余4.712251小时，**未包含正在运行的 calibration-01**。最终预算必须从追加 ledger 和最新 active reservation 重读，不能沿用此截点值。

模型 replay 预留3GiB、单校准预留1536MiB、primitive/fault预留128MiB，全部 PRIMARY 新 run 目录，最低保留8GiB。AUX 20GiB限制未扩大，原登记来源/旧曲线/旧许可不改；旧 SHA 缓存合并授权不覆盖本轮新缓存。最终追加实际占用/free/storage-device/source/model/input preservation 与执行命令索引（pending）。

## P3 验收与停止条件的最终审计（pending）

必须区分以下三种结案，不能把未执行当阴性：

- **P3 正向资格完成**：必要 CPU/原生协议/真实模型输出与 lifecycle gates 全通过；普通目标可重复；强简单基线完成；稀疏条件表的独立验证与支持域明确；预算/存储/原输入保全完成。完整 conditional table 仍不等于 production 状态表，P4 实际接线资格需单列，不能因表名或正确性 PASS 自动开启 dependency/interference/joint。
- **P3 阴性结案**：在已授权正常负载及冻结强简单基线中，有足够对应证据触发预冻结停止条件，例如目标不能重复、强简单基线已解决所需问题、只有人工极端限速才有收益或需要修改注意力/执行器。保留全部失败/无效/波动结果，说明哪些资格已完成及哪些后续研究停止；这不是策略成功或 P4 完成。
- **P3 证据不足/阻塞**：剩余必要运行未执行、输出/drain/权限/预算/表资格失败但停止结论尚不能确定，则明确 pending/blocked/incomplete；完成可安全执行部分后交付，不把预算用尽、GPU不可用或未完成表写为系统不可行。

当前 normal off 两次已有真实等待，暂不满足“没有可重复普通目标”的停止条件；最强简单基线是否已解决该目标、条件表是否可用仍 pending。隔离微基准和 synthetic faults 不属于收益判断证据。最终下一允许阶段必须根据这些实际结果填写；此草稿保持 P3 进行中、P4–P7关闭。
