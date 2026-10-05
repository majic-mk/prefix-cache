# Prefix Cache 增量调度系统 V1 完整实施方案

版本日期 2026-09-25

本文件合并架构、模块、控制合同、实施计划、实验验收与 Codex 指令。文中依赖兼容性、GPU 实验和性能收益均未验证。建议将整个交接包放入项目的 `docs/prefix_io_v1/`，按 `04_CODEX_EXECUTION.md` 执行。


---

<!-- BEGIN 00_START_HERE.md -->

# Prefix Cache 增量调度系统 V1

版本日期 2026-09-25

## 交付定位

这是一份交给 Codex 的研究系统实施合同，包含架构、模块职责、生命周期边界、算法第一版、实施顺序、实验和验收。它没有附带已经实现的调度器，也没有宣称在 5090 上完成验证。

固定路线如下。

> 在 py-kvcache 与作者配套 vLLM 的现成系统上，保留精确前缀缓存、成本准入、共享 staging、预加载和异步流水线。只增量增加可关闭的观测、资源释放依赖排序与解码干扰配额。

## 阅读顺序

1. `01_ARCHITECTURE.md` 定义系统边界、模块和完整请求流程。
2. `02_CONTROL_AND_CONTRACTS.md` 定义可执行的 V1 策略、完成语义、资源预算和故障处理。
3. `03_IMPLEMENTATION_AND_EXPERIMENTS.md` 定义阶段任务、测试、实验和继续／停止门槛。
4. `04_CODEX_EXECUTION.md` 是可直接给 Codex 的执行指令。
5. `05_SOURCES_AND_AUDIT.md` 区分已查到的接口事实与尚待现场验证的内容。

`FULL_PLAN.md` 合并上述文件，适合只上传一个文件时使用。规范冲突时，以分文件版本为准。模板是项目自定义配置，不能直接作为现成 vLLM 参数使用。

## 已确定与未确定

已确定的设计包括单 GPU、BF16、精确 Prefix Cache、CPU staging 中转、沿用成本规划器、不重写 attention、不引入 Source／repair，不训练 MLP／RL。

尚未确定的部署事实包括作者分支与 py-kvcache 的最终兼容提交组合、实际 Torch/CUDA/驱动版本、服务器路径、实际显存和 staging 容量、成本表、SLO 和实测收益。这些字段由 P0—P3 现场核验后冻结，不能用猜测填为通过。

## 首次执行范围

Codex 先执行 P0 和能够完成的 CPU 开发／单元测试。若已有明确授权的 GPU 环境与预算，可按阶段门禁继续。模板默认不授权租机、付款、驱动修改、远程推送、共享目录删除或完整 GPU 实验。

授权不足只阻塞相应操作，不阻塞可安全完成的代码阅读、文档、CPU 测试和兼容性检查。不得用假 GPU 结果填补缺失证据。

## 最终系统主干

```text
原生精确 Prefix Cache
         ↓
现成成本准入与等待规划
         ↓
现成恢复／写回／预加载任务
         ↓
新增可关闭策略
    先判断谁被资源卡住
    再选择能够解除阻塞的合法操作
    最后按解码干扰与实际资源额度决定批量
         ↓
现成异步流水线、完成回调与资源所有权
         ↓
请求继续执行，统计完整请求流
```

## 当前不能写进论文的结论

不能称控制器已实现、流水线已在 5090 验证、所有竞品已穷尽审计、候选机制首次提出、收益一定超过 10% 或达到某个期刊分区水平。

已存在的异步 I/O、成本准入、预加载与生命周期管理都应归功于底座。研究候选仅是额外调度机制及其新证据。

<!-- END 00_START_HERE.md -->

---

<!-- BEGIN 01_ARCHITECTURE.md -->

# 01 系统架构与模块

## 1. 目标与研究假设

系统保持 Prefix Cache 的精确复用语义。当前请求可以直接使用 GPU 命中、恢复 CPU／SSD 副本，或沿用原生 prefill 重新计算。未来缓存副本的写入由原有成本与资格规则决定。

新增策略研究两个问题。

- 某项传输完成以后，能否让当前缺少的 GPU 块或 CPU staging 重新可用，以及是否还需要其他前置操作。
- 在允许推进这些操作时，如何避免一次提交过多任务，过度影响其他活跃请求的 token 输出。

目标是改善同时满足 TTFT 与 ITL 要求的有效吞吐，兼顾完整请求流中的缓存写入、恢复、排队、后续 miss 重算和最终完成。不能承诺每个请求都更快。

本方案中的阈值、候选数量和门槛属于 V1 设计选择，未经实验验证。已有实现事实标注为 [S1] 等，来源见 `05_SOURCES_AND_AUDIT.md`。

## 2. 固定基础与不可更改边界

### 2.1 主底座

采用 `atlarge-research/py-kvcache` 与其 README 指向的 `t348575/vllm`，另复用 `t348575/kvcache-experiments` 的实验驱动。[S1][S2][S3]

py-kvcache 已核验代码提交为 `3abba7a502d553f6e7e2e58b92086487e3395d7e`。

作者 vLLM 的 `preload` 分支候选提交为 `d6eadf416bb5234047760bf55d532f2f038cf697`。本轮查到该版本具有预加载、任务 flush 和完成路径，但没有完成与 py-kvcache 的整套兼容测试。它是候选锁定点，不能当作已通过的安装组合。[S2][S4]

P0 要验证需要的真实符号、安装路径和集成测试。失败时先查作者对应分支／提交和最小兼容补丁。不能直接切到任意最新版 upstream，也不能转而从零写一个 offload 引擎。

当前项目仍以 `majic-mk/super-robot` 的本地工作区为承载入口。先读取实际工作区 AGENTS 和 Git 状态，不把历史审计 SHA 当作本地最新状态。原有文件、实验和未提交改动全部保留。

### 2.2 直接继承的能力

| 能力 | 来源与处理 |
|---|---|
| 原生模型、prefill、decode、批处理、GPU KV 分配 | 沿用选定 vLLM |
| Prefix block identity 与查找 | 沿用底座，不再建 Source 或语义匹配 |
| 外部缓存文件映射与重复副本判断 | 沿用 py-kvcache |
| 成本曲线和加载／延期／重算规划 | 沿用现有 LoadPlanner 与 break-even |
| CPU staging、共享预加载、引用和回收 | 沿用 reactor 与 staging cache |
| io_uring、CUDA 复制、复制合并与父任务完成 | 沿用现成执行路径 |
| 流水线和异步重叠 | 保留，不用串行替代 |
| 测试与实验驱动 | 复用作者工具，只补本题缺少的观测与统计 |

以上对应 [S1][S3][S5][S6]。保留能力不等于把每个开关都盲目开启。基线先在开发负载充分调优，冻结共同配置后再做策略对比。

### 2.3 第一版明确排除

不引入多 Source、d1/d2、Residual-K、repair mask、混合来源传播、Prefix shadow、KV 量化或近似复用。不要恢复旧 CacheBlend 自制逐层模型执行器。

不引入 RL／MLP、新淘汰算法、用户复用预测网络、跨机调度、多 GPU 训练或 GDS 驱动开发。

不修改 attention 数值计算、不增加热路径完整 KV hash、不为统计执行全局 GPU 同步、不把已有缓存功能包装为原创。

## 3. 三个平面与单一所有权

### 3.1 原生推理与缓存规划平面

运行原生 Prefix lookup、等待请求规划、内存分配和 prefill/decode。继续产生合法的 load/store/preload 工作。原有 load/defer/recompute 决策保持一个唯一入口。

### 3.2 新增策略平面

只接收有版本的紧凑状态快照，返回排序、批量和短期配额。策略不直接分配 GPU 块，不持有第二份资源所有权，不删除文件，不提交模型计算。

### 3.3 原生 I/O 执行平面

reactor 仍拥有 I/O ring、CUDA streams、staging slots、未完成文件和父任务 Future。执行前再次验证状态，完成后由原实现释放资源。

策略与执行器之间采用单向快照／命令，不能持有互相等待的锁。旧 epoch 的策略输出失效时，回退为合法保守调度或上游策略，不按过期资源预测强行执行。

```text
客户端与请求流驱动
    │ 真实请求、token流、到达时间
    ▼
vLLM Scheduler + 原生 Prefix Cache
    │
    ├─ GPU命中 ─────────────────→ 原生模型执行
    ├─ 无外部命中／规划DECLINE ──→ 原生prefill
    └─ CPU/SSD命中 → 原有LoadPlanner
                         │ ADMIT / DEFER / DECLINE
                         ▼
                原有外部缓存任务与预加载提示
                         │
             ┌───────────┴───────────┐
             │ 新增策略层             │
             │ 快照 → 依赖 → 候选     │
             │ 干扰估计 → 有界额度    │
             └───────────┬───────────┘
                         ▼
                原有IoReactor执行器
                 │                │
     SSD read → staging → H2D    D2H → staging → SSD write
                 │                │
                 └──完成事件与原有回收协议──┘
                         │
                  父任务完整完成通知
                         ▼
                  vLLM继续执行请求
```

图中箭头表示依赖。不同请求和不同批次之间允许重叠，同一缓冲的生产与消费严格服从事件。

## 4. 分模块合同

### M0 版本、环境与实验权限

职责是锁定兼容组合、模型 revision、目录命名空间和操作预算。

输入为实际 Git 工作区、候选依赖、服务器只读信息。输出为 `dependency-lock.json`、`environment.json`、`capability-report.json` 与 `permissions.yaml`。

必须检查安装模块来自期望路径，不能因为 import fallback 成功就认定安装正常。py-kvcache 内部存在 `VLLM_AVAILABLE`、`PLAN_API_AVAILABLE` 等能力判断，正式运行要验证真实接口与 handler，而不是 stub。[S7]

验收是实际版本可重现、CPU 测试通过、GPU 资格另有实测。禁止把 CUDA 存在等同于模型、复制 kernel 和 offload 全部可用。

### M1 精确前缀与缓存命名空间

直接沿用原身份与块映射。新增工作仅限对外部 cache root 作部署隔离，绑定模型与 tokenizer revision、缓存布局、dtype、适配器与运行配置摘要。

同一 trace 中同一配置必须使用稳定命名空间，不能每个请求生成新 namespace 而消灭复用。不同实验臂使用独立 namespace，或相同预热内容的独立副本。

第一版仅单模型、单租户、单 KV cache group。多租户隔离不能仅靠文件目录名称声称完成。KV group 限制在已查代码中有显式检查。[S7]

验收包括相同 token 前缀能命中、祖先前缀不同不能误命中、模型或布局不同不能串用、当前请求必需 KV 不受跳过写回影响。

### M2 原生成本准入

沿用 `LoadPlanner`、成本曲线、break-even 和现有的重复写入／资格判断。[S1][S6]

输出保持原有加载、延期、拒绝外部加载并原生重算三类结果。GPU 已命中部分不强行重算，拒绝外部加载只处理剩余需要恢复的部分。

第一轮主消融不更改这个规划器，也不再写第二个评分器与它竞争。计入其耗时和准入决定，后续确有证据再把准入改动作为独立实验。

若原实现可能在 handler 接收后但首次复制前进行 break-even 拒绝，必须沿用已有安全拒绝回调。新策略不得直接让已产生 H2D 的任务切换为重算。

原工具用于 break-even 的单输出 token 实验可以保留为校准，不能充当本题持续 decode 的主结果。[S3]

### M3 异步流水线执行器

完整复用 reactor、io_uring、staging、CUDA streams 和 `swap_blocks_batch` 等现有复制合并路径。[S5]

新增策略只控制尚未签发操作的选择和额度。不能拆散后端逻辑块或破坏对齐，不把一次策略批准等同于一次物理 DMA。

同请求恢复采用底座当前 ready 语义。若该组合需要整份选定 Prefix 恢复才可运行，就保持此语义。SSD 分批读取与前一批 H2D 可以重叠，其他请求照常 decode。

只有经 P0 验证存在可用的原生逐层消费机制，才保留相应实现。第一版不额外造逐层模型执行器，也不移植旧 Source／repair 代码。不能为了让新策略赢而关闭底座已有的重叠。

### M4 低开销快照与事件观察

从 scheduler、worker 和 reactor 的既有状态获取紧凑观测。新的观察字段见第二份文件。

scheduler 维护请求优先级、等待原因、活跃引用／源块保护摘要。reactor 维护 slot、I/O、copies 和父任务状态。状态拥有者在自己的线程内更新，再发布只读快照。

每个调度边界最多做一次策略重算。连续 decode 时一次迭代本身可能对应每个请求生成一个 token，不能据此声称完全没有每 token 开销。CUDA 完成轮询仍按执行器需要运行，不等待下一次复杂策略计算。

热路径只读有限任务描述与计数，避免扫描全部 GPU KV block pool。关键观测丢失或过期时标记未知，不填零。

### M5 实际释放依赖分析

新增模块，职责是建立“需要的资源 → 可解除该需求的完成条件 → 尚可执行的前置操作”的小型关系。

这里只做观测和估计，不能改变真实引用计数或 fence。物理资源的身份必须包括块所属 pool／group、allocation generation 或等价有效期，不能只用可循环复用的 block id。

区分活跃计算引用、原生可淘汰缓存引用、I/O 保护和已提交复制。对仅因写回受保护的空间，依据底座真正的释放边界计收益。若底座父任务直到所有文件写完才完成，不能把单个 D2H 的完成假设成 GPU 块已经可用。

同一资源需多个条件同时完成时按 AND 处理。多条候选路径都能提供空间时按替代集合处理，不把一种可能性伪装成唯一因果链。

当只知道 pool 缺空间，尚未知道原生 allocator 会选择哪个 victim，记录“候选释放路径”，不要称已经确认该 block 阻塞某请求。第一版只对有足够证据的路径升优先级。

### M6 干扰标定与保守额度

新增轻量策略输入，离线用匹配状态的受控测量得到实际干扰表。线上只查有限候选和更新保守误差余量。

方向、联合读写、上下文负载与批量都要区分。不能将 H2D 和 D2H 单独干扰简单相加。无法验证的组合使用测试过的保守额度。

ITL 目标与 GPU step 时间并不相同。worker step 时间可作控制信号，最终 SLO 要用规定口径的 token 输出时延验证。

### M7 增量调度策略

输入为资源快照、原有请求排序、成本表、可签发工作和必须排空任务。输出为有限有效期内的操作优先级与配额。

V1 先使用确定性规则与短路径比较。保持同一底座和准入，分别提供 fixed、pressure、interference、dependency_only、joint 等实验模式。详见控制合同。

策略开关为 off 时不创建新的引用或队列层，原有调度路径直接执行。shadow 模式只记录拟选择，不改变原动作。

### M8 正确性进展与异常处理

对接原生 `jobs_to_flush`、worker wait 和父任务 Future。[S4]

新增限流之前先接通强制排空信号，让执行器知道哪些已接受任务不能等待下一轮配额。该通道只能绕过性能额度，不能绕过内存容量、数据依赖或完成事件。

任务失败时不把未完成 KV 标为可用。已提交操作仍要按后端协议排空或在进程退出前保持其资源安全。底座目前的某些路径用 assert 处理失败，第一版可以保守地终止受影响服务／实验并报告，不能虚称已具备无损在线恢复能力。[S4]

### M9 测试、实验与证据

复用作者测试与 benchmark harness，补充持续输出、token 时刻、真实缓存层命中、源块保护等待和策略耗时。作者已有多种 replay 驱动，先检查接口再扩展，避免建立第二套服务器启动框架。[S3]

每次运行保存 config、lock、模型与 trace 摘要、原始请求记录、资源时间线、完成／失败统计、成本标定来源和统计报告。模拟器和 mock 的输出永远单独标为非 GPU 证据。

## 5. 完整运行流程

### 5.1 初始化

读取权限和锁文件 → 验证实际环境 → 冻结真实显存／staging／SSD预算 → 启动原生服务 → 加载原成本表 → 确认策略模式与共同基线配置 → 预热 → 开始计时。

策略关闭、观测打开、固定策略和完整策略都使用相同执行器、存储格式和完成语义。

### 5.2 GPU 命中

沿用原生复用，继续原生剩余 prefill／decode。新策略无需发起恢复，也不能为了测试而强制逐请求清理 GPU cache。

### 5.3 CPU／SSD 命中

原 LoadPlanner 决定 ADMIT／DEFER／DECLINE。DEFER 的已有预加载和等待截止逻辑继续有效。等待截止只是避免策略无限停车，不保证请求必定满足 SLO。

ADMIT 后，原生分配与任务保护照常生效，任务进入原 reactor。新策略每轮选择可签发子工作，数据完成后经原回调汇总。只有底座规定的恢复完成条件成立，才能恢复该请求的计算。

CPU 命中也要确认实际属于 staging cache／预加载缓存命中，不能只看一次文件曾被访问过就称 DRAM 命中。

### 5.4 直接计算与不写回

DECLINE 或真正 miss → 原生 prefill，不强制新增外部缓存 I/O。

后续写回资格仍单独判断。可以出现“本次直接计算，仍保存以后可能有价值的副本”或“本次直接计算且跳过外部副本”。第一版这些决定沿用现成规划和资格策略。

已经存在的合法 CPU／SSD 副本不能因本次选择重算而被删除。当前推理使用的 GPU KV 始终由原引擎管理。

### 5.5 新 Prefix 写回

原生资格与去重 → 接受前检查共同 admission 约束 → 沿用原有延迟到下一步的提交时机 → 原 reactor 取得安全源数据并 D2H → 原存储写入与发布 → 原完成回调。

新增策略可以推迟尚未接受的可选副本，或在已接受合法任务之间调整先后。不会自行取消进入保护协议的任务。对未被接受也未受保护的候选，源块后来被复用时必须让候选过期，不能继续使用旧地址。

### 5.6 流水线与控制的关系

控制器给出新的有限额度，reactor 消费额度。同一 epoch 的额度是累计上限，不能在每次 pump 中重新获得一份额度。

下游 completion、slot settlement、D2H 后必要写入续接和 forced drain 始终允许进展。执行器已经预留的下游额度优先保留，新策略不能通过反复停住下游制造 staging 堆积。

已经提交的 SSD 写入无法凭修改 Python 优先级而提速或撤销。能够改变的是剩余尚未签发工作和后续流量。所有“解阻收益”必须扣除该限制。

## 6. 建议文件组织

下列是 Codex 将创建的建议布局，不表示文件现在已经存在。

```text
现有项目根目录/
  docs/prefix_io_v1/                 本交接文档和现场审计报告
  src/prefix_io_control/
    config.py                       严格配置校验和默认关闭
    types.py                        小型快照、候选、配额数据类
    observation.py                  只读观测适配
    dependencies.py                 可释放条件与短路径
    cost_table.py                   标定读取、匹配与余量
    policy.py                       有限候选决策
    bridge.py                       与实际作者分支接口对接
    metrics.py                      轻量计数与事件导出
  patches/prefix_io_v1/
    common/                         两边共用的兼容／正确性修复
    observer/                       状态与进展信号
    policy/                         队列选择与额度
  tests/prefix_io_v1/                新增测试
  experiments/prefix_io_v1/
    locks/                          依赖、模型、配置和数据锁
    configs/                        权限与各实验臂
    scripts/                        复用作者harness的薄包装
    manifests/                      预先冻结的请求流
  third_party/                      依赖隔离检出，按项目现有惯例处理
  artifacts/prefix_io_v1/            不纳入普通源码提交的大型原始结果
```

若项目已有同职责目录，沿用既有组织，禁止仅为满足上述名称做大规模搬迁。外部库以固定提交加小补丁管理，保留许可证与上游归属，不建立多个互相复制的 reactor。

## 7. 第一版完成后的表述

可以说“基于现有 Prefix Cache 外部缓存执行器，实现并评估了可关闭的依赖感知调度增量”。

不能在结果出来前说“新系统彻底解决读写阻塞”。若配额、水位、缓冲隔离和延后准入已取得同样收益，应如实停止复杂机制或缩小贡献。

<!-- END 01_ARCHITECTURE.md -->

---

<!-- BEGIN 02_CONTROL_AND_CONTRACTS.md -->

# 02 控制算法与工程合同

## 1. 必须先厘清的事实

以下均为 V1 必须遵守的定义，不能让 Codex 自行更改语义。

| 概念 | 定义 |
|---|---|
| 逻辑 Prefix block | 原缓存引擎的身份与管理单位 |
| storage unit | 底座文件／存储块的最小合法工作单位，可包含多个 GPU blocks |
| I/O batch | 本次批准的若干合法 storage units 或复制映射 |
| parent job | 原 connector 对外认可的完整 load／store 任务 |
| child work | parent 内部的一项文件 I/O 或一组复制，不自行改变 parent 完成条件 |
| submitted | 已交给 CUDA／I/O 后端，包含设备排队，未必正在传输 |
| in-flight bytes | 已提交尚未完成的字节，不能等同于瞬时带宽 |
| source-safe | 源数据不再被相关异步操作读取，是否能复用仍取决于原生命周期协议 |
| buffer-reusable | 原执行器确认这批物理内存没有其他保留／在途使用者 |
| cache-visible | 底座已按既定协议发布可命中的外部缓存 |
| durable | 满足选定 fsync 等持久性语义，不等于任意 write 返回成功 |

`source-safe`、`buffer-reusable`、`cache-visible`、`durable` 可以发生在不同时间。V1 不改变它们的上游协议。

## 2. 关键实现限制

### 2.1 不能提前把 D2H 完成当成父任务完成

已核对的作者 worker 通过 completed job 更新状态，源块复用通过 `jobs_to_flush` 等协议保护。py-kvcache 的 store parent 则在文件子任务终结后汇总。[S4][S5]

因此 V1 不新增“D2H 完成就直接 complete_store”的快捷路径。

只有 P0 证明所选实际后端已有独立 source-safe 确认，才能把它用于观测和原生回收。若无此接口，GPU 释放收益必须估算到原父任务／全部相关保护完成。

将来改造两阶段完成协议，应作为单独的协议优化项目，并对基线和新策略采用相同协议。本轮不隐式加入。

### 2.2 不能把原执行器续接硬拆成新的独立队列

已查 reactor 中 D2H 完成会继续签发 SSD 写入。第一版保留这种已接受链路的续接。[S5]

只有已经在原执行器中存在、尚未签发的 ready work 才可重排。不能为了制造“先写 SSD 再 D2H”的策略空间，故意把自然续接变成额外等待。

如果一批 SSD 写入已在设备队列里，策略只能减少其他新流量、等待它完成，或者选择其他合法可推进工作。不能声称 Python 调整优先级已经重排设备队列。

### 2.3 真正的 source fence 可能有 job 粒度

若一个父 job 覆盖多个文件，只有整个 job 完成才解除一组块的 fence，则完成一个文件不会立即释放这些 GPU 块。

依赖分析必须把这组完成条件作为闭包。跨两个 store job 受保护的同一物理块，只有全部相关保护结束后才能计为可用。仍有活跃请求引用时，GPU 释放收益仍为零。

## 3. 新增模块的数据结构

这里的类名均为项目拟新增名称，不是声称 vLLM 已有同名 API。Codex 应在 P0 输出实际字段与接口映射。

### 3.1 SystemSnapshot

| 字段组 | 最少内容 |
|---|---|
| 版本 | run_id、snapshot_epoch、monotonic_ns、字段能力掩码 |
| 推理负载 | active decode 数、实际 batch、prefill token 数、上下文长度分位、近期 step 时间 |
| 请求 | 有界 waiting request 列表、原生优先顺序、到达／等待时间、已有准入决定 |
| GPU | 原 pool 总量、可立即覆盖量、带 fence 的候选量、restore 预留量 |
| staging | 实际总分配字节、空闲 slots、各阶段占用、可驱逐干净缓存量 |
| I/O | SSD read/write 未完成数与字节、H2D/D2H 未完成字节、待签发量 |
| 生命周期 | 选定 job 的块有效期、活跃引用摘要、保护者集合、完成依赖 |
| 安全进展 | 原生 jobs_to_flush、被 wait 的 parent、最后进展时间 |

不能把“free list 中存在但仍需 fence 的块”直接计为可以无等待覆盖的块。两个指标分别记录，避免总容量虚增。

调度器线程提供 GPU 所有权摘要，reactor 线程提供实际 staging/I/O。跨线程只能使用发布快照，不能在执行器锁内反向调用调度器进行复杂查询。

### 3.2 WorkDescriptor

包含 parent_id、child_id、原请求／共享消费者集合、方向与 stage、合法传输字节、原顺序、创建时间、已提交状态、所需 slots、目标块有效期、已有源事件、后端最小单位。

集合可以是有界摘要和数量，不把任意长度用户文本或完整 KV 放入策略。

### 3.3 ReleaseWitness

用于描述一次候选完成闭包能带来的新增可用资源。

- 对象包含物理资源 ID、generation／等效生命周期标识、原 ownership 来源。
- 条件包含仍需完成的所有 I/O／引用事件。
- 效果分为 GPU 可复用字节、CPU 可复用字节、直接完成某个 restore，不能无单位混加。
- `evidence_level` 区分实际已观测阻塞、原 allocator 尚未选择 victim 的候选释放路径、无法证明。
- `release_scope` 指明 per-copy、per-file、parent-job 或多保护者闭包。

未能证明释放条件的对象不参加依赖升优先级，不以“所有 pending bytes 都能释放”替代。

### 3.4 DispatchBudget

包含 epoch、到期时间、累计 SSD read/write 签发数与字节上限、累计 H2D/D2H 签发字节、各方向 in-flight 上限、staging 预留限制与共同队列上限。

“本轮还能提交多少”和“全系统现在最多在途多少”必须分别检查。一次预算发放只消费一次，不随 `_pump_once()` 重置。

控制器禁止把 MiB/s、bytes 和 operations 混为同一个数。

## 4. 资源预算的硬边界

### 4.1 CPU staging

`StagingPool.compute_slot_count()` 采用包含 `min_slots` 的下限逻辑，实际内存可能高于用户提供的 staging 配置。[S8]

P0/P1 必须记录实际 tensor storage 字节，计入共享 cache、copies 占用和真实 allocator 开销。若最低 slot 需求超出批准预算，启动失败并报告所需值，或在冻结前调低 iodepth 重新配置。不能静默超配。

容量修复是共同基线修复，不能只在新策略里执行。

### 4.2 GPU

统计模型、KV pool、workspace、CUDA graphs 和 I/O 相关临时内存。所有实验臂固定相同实际 KV pool 预算和模型设置。

同一物理块多请求共享只计一次。GPU source bytes、restore reserved bytes 与 active bytes 可能重叠，不能直接相加作为总使用量。

### 4.3 SSD

py-kvcache 当前没有完整磁盘文件淘汰与 GC。[S1]

第一版目标为“有限 GPU／CPU 缓冲、磁盘足以容纳该轮完整数据集”。每次运行预估全部唯一前缀 storage units、对齐、临时文件、元数据和日志，留明确空间余量。数据写到项目专属根目录。

达到设定 disk guard 时停止接纳新测试请求并完成合法排空，标记本轮无效或未完成。不能让满盘后静默跳过写入成为方法收益。

不运行后台手写 GC，不在正文宣称已研究 SSD 淘汰最优策略。需要 SSD 满容量场景时，以单独后续阶段评估可复用的成熟后端。

### 4.4 已接受队列

保留原队列和 parent Futures，不再默认添加一层无限等待的“第二父任务队列”。若原队列缺乏合理界限，在共同 admission 入口添加容量保护，并让所有策略共享。

队列满在接受前表达延期。不能让 `transfer_async=False` 代表“已经接受，留待以后”。所查 worker 中有 `assert success`，这会把限流当成失败。[S4]

## 5. V1 策略的确定形式

### 5.1 实验模式

| 模式 | 行为 |
|---|---|
| off | 完整保留上游动作，策略路径不产生额外资源占用 |
| shadow | 执行动作仍是上游，旁路记录候选和成本 |
| fixed | 固定批量／额度，仍保留所有正确性进展 |
| pressure | 固定额度基础上加水位、age、防饥饿与共同缓冲预留 |
| interference | 以同样基础根据 decode 干扰调整额度，不利用真实释放依赖排序 |
| dependency_only | 依赖感知排序，普通额度固定 |
| joint | 同时启用依赖排序与干扰额度 |

这些是项目自定义字段。不要把它们当作 py-kvcache 自带选项传入。

### 5.2 候选边界

建议起始实现最多看 32 个前台／已接受 parent，最多检查 64 个待签发候选，依赖追踪最多经过 3 个 I/O 阶段或生命周期事件，单次生成最多 8 个闭包。它们是控制器计算量上限的起始设计，不是性能最优值。

这些观察上限不等于全系统允许的请求并发数，也不允许丢弃窗口外的已接受任务。原队列中的其他工作按公平轮转或原顺序继续获得服务。

一个闭包可以包含同一 parent 的多个文件完成条件。不能把“深度最多3”误写成“最多处理3个文件”。超出可准确分析范围时回退到原有策略，记录 reason。

### 5.3 先选择前台目标

先处理真正的 `jobs_to_flush` 与必要 completion，无须性能优化评分。

对普通工作，使用原生优先级与已有规划器决定的等待顺序。先考虑已接近等待上限的任务，再考虑有实际资源阻塞的任务。首版不再额外设计全局请求重排算法。

等待年龄用于防止无限让路，不允许按结果随意丢弃慢请求，也不允许为了提高 goodput 提前结束长请求。

### 5.4 判断零 I/O 的解阻机会

当缺少 staging 时，先沿用底座现成的干净 cache／预加载回收机制。能够安全驱逐就不需要先额外写盘。不能把正常缓存占用伪装成必须写回的债务。

GPU 同理，存在原生可无等待复用的块就不触发所谓 emergency writeback。

只有既有办法仍不能提供资源时，才计算需要传输完成的候选闭包。

### 5.5 比较释放闭包

对选定等待目标，得到当前需求缺口 `need_gpu`、`need_slots` 或 `restore_remaining`。针对可用闭包估计：

1. 何时按原协议能获得所需资源或完成 restore。
2. 完成后的新增可用资源是否足以满足缺口。
3. 新签发动作对 decode 的附加干扰估计及其测量支持。
4. 该路径是否依赖仍然活跃的计算、尚未完成的父任务或已经不可调度的设备工作。

第一版采用可解释的字典序选择。

- 优先保持原等待目标和原生服务优先级。
- 在可满足同一目标的支持路径中，优先预测合法解阻时间较早者。
- 时间处于标定误差范围内，选择预测干扰较小者。
- 再按新增传输字节少、原队列顺序确定平局。

不对 GPU bytes、CPU bytes、ms 任意加权相加。若需要多任务才能满足缺口，只使用受限闭包估计并去重。无法估计时不承诺解阻，用回退路径推进并记录。

这里的优先级只会作用于尚未签发 work。路径中已经提交的阶段保留原执行顺序，其剩余时间计入估计。

### 5.6 选择批量与额度

对选中的方向和工作集，只比较几个按实际 storage unit 对齐的合法批量。建议初始候选为 1、2、4、8 个 storage units，实际总字节和最小可用批量在 P0/P3 冻结。它们不是固定 token block size。

令 `s_t` 为当前负载，`i_t` 为已经存在的 I/O，`a` 为本轮新增动作。

`T0_hat(s_t,i_t)` 表示不再增加此动作时，下一控制窗口的预计基线耗时。

`Delta_hat(s_t,i_t,a)` 表示相对该基线的附加成本。

候选必须同时满足硬资源限制、累计签发额度和经验时延预算，例如：

```text
T0_hat + Delta_hat + uncertainty_margin <= internal_step_budget
```

若用的是“无 I/O 基线＋联合 I/O 总成本”，就不能再额外加入同一笔已有 I/O 的成本。两个估计口径必须二选一且固定。

可行候选中优先推进足以解除所选短路径阻塞的批量。尚无解阻目标时，在当前允许范围内使用较大合法批量。不要把“最大可行批次”称为全局最优。

实际选定结果一次性消费共享读写预算。不能由 H2D、D2H 两个独立模块各自认为自己拥有整份剩余额度。

### 5.7 没有可行普通动作

暂停新的推测性预加载和未接受的可选写入。继续收取 completion、既有链路续接和合法缓冲回收。

对于等待过久或必须保持系统前进的已接受工作，允许使用事先验证过的最小进展额度，记录 `progress_override`。它可以超过经验干扰预算，但不能突破物理容量和依赖。

所有超预算造成的 token 延迟都计入统计。它只保证在可执行条件下不因本策略无限停发，不保证硬实时完成，也无法保证故障设备必定响应。

## 6. 为什么要先做 shadow

shadow 使用真实当前状态生成拟动作，却不改变底座调度。它能检查：

- 是否真的出现候选依赖。
- 预测释放条件与后续实际回收是否一致。
- 控制器是否主要在修改本来就会执行的同一操作。
- 是否把原生可直接驱逐的缓存误判成需要写回。
- 候选完成粒度是否与真正 parent completion 一致。

shadow 不能证明采用拟动作后的反事实性能收益，也不能把原来发生的时延直接算成新策略可消除的时延。

## 7. 完成状态机

### 7.1 恢复任务

```text
LOOKUP/PLAN
  ├─ DECLINE → 原生计算
  ├─ DEFER → 原有有界等待与可选预加载
  └─ ADMIT → 原任务保护/分配
                ↓
           ACCEPTED
                ↓
    READ_PENDING / READ_INFLIGHT
                ↓
          DATA_IN_STAGING
                ↓
       COPY_READY / H2D_INFLIGHT
                ↓
      原协议的全部子任务完成
                ↓
         PARENT_COMPLETED
                ↓
      请求ready，只通知一次
```

staging hit 可以跳过文件 read，不跳过 H2D 源数据保护。单个文件到达、CUDA API 返回、event 已创建均不等于完成。

### 7.2 写回任务

```text
CANDIDATE
  ├─ SKIP → 不新增外部副本
  ├─ NOT_ACCEPTED → 有效期内可继续评估
  └─ ACCEPTED → 原源块保护协议
                  ↓
          WAIT_COMPUTE_SAFE
                  ↓
        D2H_READY / D2H_INFLIGHT
                  ↓
        原有SSD写入续接路径
                  ↓
         FILE_PUBLISH_COMPLETE
                  ↓
        原父任务完成和保护解除
```

完成后 CPU 副本是否仍可缓存由原策略决定。store 完成不等于必须把 CPU 数据立即丢弃。

### 7.3 取消、失败与终止

候选取消不动当前计算 KV。已接受任务只用上游已有的取消／拒绝路径。客户端离开也不能导致 GPU 目标块在 DMA 尚未结束时被复用。

真实 I/O 错误、短读、进程退出和 parent failure 的处理，按底座实际能力 fail-closed。未提供完整在线恢复能力时，在报告中明确服务失败并保留证据，不添加看似成功的重试假象。

至少区分 `FAILED_DRAINING` 与 `FAILED_FINAL` 的观测状态。该观测状态不替代执行器内部状态机。部分子任务失败时，也必须防止剩余复制写入已经转交新请求的目标块。

## 8. 强制排空的最小对接

所查 worker 在 `handle_preemptions` 中调用 `worker.wait(jobs_to_flush)`。[S4]

加限流前先验证这条 wait 如何到达具体 handler。若 wait 只是等待 Future，新增额度可能阻止 Future 取得进展，需要最小桥接：在等待之前把受影响 parent IDs 交给 reactor 的紧急集合，reactor 独立推进其必要未提交 work。

该桥接是拟新增能力，不能假装已存在某个 `force_drain` API。Codex 需要给出真实调用链和测试，再设计窄接口。

紧急集合使用 parent ID 与 run/generation 防止串用。即使 scheduler 停在 wait，reactor 仍能收取 completion 并续接。不能持有 reactor mutex 等待 reactor 自己完成。

清理紧急集合以真实任务完成为准，不以新一轮 scheduler 到达为准。测试 shutdown、重复通知、空集合、已经完成 job 和不存在 job 的行为。

## 9. 标定与在线反馈

### 9.1 标定内容

沿用作者 break-even 数据生成脚本估算恢复与 prefill 路径。[S3]

本题额外标定持续 decode 的干扰。固定模型、上下文分布、batch、prefill 混入、kernel 模式和起始缓存状态，做成对重复测量。方向至少包含 H2D、D2H、联合读写和真实 SSD 链路。

初始状态采用稀疏设计，先识别主要变量，再增加边界。避免把所有计数器都做成全笛卡尔积。

A/B 次序随机化或 ABBA，包含热身，报告方差和置信区间。输入／输出 token 工作量固定的机制实验与自然结束的质量实验分别保存。

### 9.2 step 时间与 ITL

`decode step` 只作内部信号，客户端 ITL 是另一口径。若一次流式消息打包多个 token，记录为 chunk gap，不能给其中每个 token 伪造相同时间并宣称测到 token ITL。

主 ITL 使用可核验的原生逐 token 输出事件，客户端同时报告 TTFT 与 chunk gap。无法获得逐 token 时刻时标记 ITL 未测得，不能用 TPOT 均值替代。

内部 step budget 在开发集冻结，并预留 scheduler／sampling／输出延迟余量。禁止使用 `ITL上限 - 最近一次GPU step` 就声称存在确定的安全 slack。

### 9.3 在线误差

维护对当前总耗时预测的残差与保守分位余量。线上无对照状态时，这个残差只能说明预测失配，不能声称得到 I/O 的因果干扰。

观测不匹配、混入大 prefill、数据过期或表外状态时，不更新原因归属的干扰表，使用保守回退。线上只调整预先规定的 margin／quota，不能训练新网络，也不能修改最终 SLO 阈值。

配置变更、模型变更、传输粒度或明显硬件环境变更后，旧表失效。原始表不可原地改写，在线余量另存带 run_id 的记录。

## 10. 可执行伪代码

以下为拟实现逻辑，不是现成 API。真实函数名由 P0 的接口映射决定。

```python
# scheduler / worker boundary: publish a compact immutable snapshot.
def at_scheduler_boundary():
    publish_snapshot(collect_supported_state())
    publish_native_mandatory_jobs()  # must happen before a blocking wait

# reactor: remains the sole owner of transfer resources.
def reactor_pump():
    poll_native_io_and_cuda_completions()
    run_native_completion_and_continuation_paths()

    if has_mandatory_jobs():
        progress_mandatory_closures_with_hard_limits()

    snapshot = latest_compatible_snapshot()
    if policy_mode == "off":
        run_original_dispatch_path()
    elif not snapshot_is_fresh_and_compatible(snapshot):
        run_validated_fallback_without_blocking_progress()
    else:
        if budget_epoch_needs_refresh(snapshot):
            candidates = observe_bounded_existing_ready_work()
            witnesses = build_verified_release_closures(snapshot, candidates)
            choice = select_target_and_closure(witnesses, candidates)
            grant_epoch_budget(choose_supported_batch(choice, snapshot))
        issue_legal_work_with_remaining_credits()

    run_native_fusion_and_submit_pending()
    finish_native_parent_jobs_once()
```

不能在每次 pump 里重发整份额度。不能通过调整循环顺序破坏原 `_flush_copy_batch()` 的复制合并。融合提交本身也必须对物理总字节验额度，既不能重复扣账，也不能绕过额度。

<!-- END 02_CONTROL_AND_CONTRACTS.md -->

---

<!-- BEGIN 03_IMPLEMENTATION_AND_EXPERIMENTS.md -->

# 03 实施顺序、测试与实验合同

## 1. 操作权限与预算

本计划授权范围由使用者交给 Codex 的明确指令及 `permissions.yaml` 共同限定。文件中的建议预算不是实际租机或消耗 GPU 的授权。

默认可以做代码阅读、工作区审计、隔离依赖准备、项目内局部实现和 CPU 测试。默认不租机、不付款、不推送远端、不改驱动、不运行共享目录清理，不启动完整 GPU 实验。

若当前环境没有 GPU，完成可完成的 CPU 工作，输出 GPU 执行清单和阻塞原因。禁止把 mock、静态检查或 CUDA import 成功标成 GPU 实验通过。

建议按每阶段 GPU 小时上限管理。首轮 smoke 与稀疏标定、pilot 合并预算可先提议不超过 8 GPU 小时，由使用者在运行前确认。达到上限即停止新增实验并安全收尾。这个数值是投入上限，既非耗时预测，也不保证预算内能得到充分证据。

## 2. 分阶段任务

### P0 只读审计与实现映射

输入为实际工作区和本文候选底座。

Codex 完成以下内容。

- 读取项目规则，记录分支、HEAD、未提交与未跟踪文件，保护现有成果。
- 检查外部依赖已经是否检出，缺少时依权限放到隔离目录，不覆盖旧环境。
- 固定 py-kvcache SHA，查作者 vLLM 分支真实接口，并检查配套测试所要求的版本。
- 输出 `CAPABILITY_MATRIX.md`，逐项注明已存在、需桥接、受限、不支持和证据路径。
- 输出 `PATCH_MAP.md`，写明将改哪些实际函数，以及哪些文件保持不动。
- 输出 GPU 源块释放与 parent completion 的真实关系，区分物理 source-safe 与协议可回收。
- 记录 import fallback、单 KV group、actual staging floor、磁盘无 GC、profiler 依赖和失败 assert 的限制。

验收产物为工作区审计、候选锁文件、接口矩阵、许可清单和风险表。

禁止仅因候选分支名叫 preload 就认定兼容，也不把旧 upstream SHA 当成作者 fork 的版本。若需要突破本计划边界，先报告具体阻塞，不扩成新的引擎项目。

### P1 原版复现与共同基础修复

保留 upstream 冻结检出，同时建立可回退的实验分支／worktree。依赖和实际测试命令来自作者项目，不捏造不存在的 CLI。

先执行原测试。按 P0 报告补充最少共同修复，例如能力检查、staging 实际预算检查、实验安全目录和可选日志开关。

通过 GPU 授权后执行原生模型、GPU Prefix 命中、CPU staging 命中、真实 SSD 恢复和 store smoke。必须以实际存储 read 字节／I/O 事件证明 SSD 路径被走过。

相同前缀复用允许 naturally 冷热切换，不能靠每个请求重置缓存制造外部恢复。单次用于 path smoke 的重启或定向置冷另作诊断，不混入正式服务流。

输出 `BASELINE_REPORT.md` 与 `dependency-lock.json`。必须检查 `VLLM_AVAILABLE`、`PLAN_API_AVAILABLE` 和对应实际调用，不接受 fallback stub 通过。[S7]

### P2 只增加观测与进展保护

实现 M4 快照、资源状态摘要和 M8 mandatory bridge。新策略保持 off／shadow。

先测试 completion 与强制排空在缺额度时仍能前进，再给普通路径加额度。不要先限流再补死锁保护。

输出 `LIFECYCLE_REPORT.md`，明确哪些释放量可在线观测，哪些只能估计，哪些无法获得。无法证明 GPU release witness 时，不用假数据实现 joint。

测量 observation off／on 的 CPU 和端到端开销。若观察明显影响基线，先削减日志、采样或状态扫描。

### P3 强简单基线与稀疏标定

复用现有 LoadPlanner 和 break-even 脚本，补充持续 decode 干扰测量。调整后不修改核心模型执行或 Prefix identity。

实现 fixed、pressure 模式，充分调节 batch、iodepth、缓存／预加载容量和缓冲预留。共同 admission 与正确性修复对所有策略一致。

通过 shadow 收集真实资源等待。当绝大多数“债务”可直接驱逐或不会阻塞时，先报告该事实，不能继续人为扩大延迟来制造收益。

输出冻结的开发配置、校准表、fallback 与 `PILOT_PROBLEM_REPORT.md`。

停止条件为没有可重复的目标问题、只能极端人为限速才出现、合理简单基线已取得相同效果，或必须重写 attention 才能继续。

### P4 最小新增策略

按第二份文件实现 dependency_only、interference、joint。只加一个策略入口，保留原队列、合并复制和完整 parent 回调。

先 shadow 验证候选，再 fake-backend 事件测试，再授权的小规模真机验证。不在这个阶段加入 RL、动态 SSD 淘汰、两阶段源块释放协议或新的长期缓存价值模型。

输出补丁序列、测试结果、控制器开销、策略 reason 日志和 `IMPLEMENTATION_REPORT.md`。

### P5 单模型完整请求流与因果归因

首先在 Qwen2.5-7B-Instruct 上完成所有预注册对照、指标和负场景。策略和 SLO 在开发结束时冻结，不依据测试结果反复变更。

主报告必须展示同执行器增量收益，以及释放依赖 witness 对应的具体等待变化。shadow 的反事实猜测不能充当真实加速。

达到内部继续门槛后才扩大工作负载。失败或无收益同样输出完整报告，不自动改成新的研究题。

### P6 双模型与健壮性

第二模型建议 Mistral-7B-Instruct-v0.3。官方配置中 sliding_window 为 null、KV 头数和层数与 Qwen 不同，适合扩大本次 Full Attention 单组范围内的验证，但实际后端 group 仍须现场检查。[S11]

两个模型分别运行，均使用 BF16。权重、tokenizer、配置和结果命名空间独立。换模型重新标定成本，不假装一张表适用于所有模型。测试负载仍独立于标定数据。

对上下文、请求到达突发、复用热区切换、GPU／CPU 容量和表失配进行预先规定的敏感性测试。人工带宽节流只作机制诊断，不能成为唯一有收益的主场景。

### P7 冻结、复现与交付

生成完整版本锁、patch 清单、配置、原始数据、统计脚本、运行命令、失败清单、结果摘要和资源预算。

审核已有能力归属和论文差异表。不写未经证明的首次、无干扰、硬实时、全面优于等结论。不删除负结果，不把未跑实验写成完成。

## 3. 模型与硬件初始范围

| 项目 | V1 范围 |
|---|---|
| GPU | 单张 RTX 5090 目标环境，实际兼容性必须测试 |
| 首模型 | Qwen/Qwen2.5-7B-Instruct |
| 第二模型 | mistralai/Mistral-7B-Instruct-v0.3，P6 进入 |
| 精度 | 权重与 KV 均 BF16，不添加量化变量 |
| 软件 | Linux，独立环境，作者 fork 与依赖精确锁定 |
| 主存 | 以现有机器为准，建议 64 GiB 起、128 GiB 更宽裕，实际预算独立固定 |
| 存储 | 普通 NVMe 文件系统路径，不要求 GDS，检查 O_DIRECT 与 io_uring 实际可用性 |
| 并发与上下文 | 根据实测可用 KV pool 生成，不能在 32GB 上照搬数据中心 batch |

上述资源配置是实验建议，不是容量保证。驱动、Python、Torch、CUDA 与 attention backend 不预先瞎锁一组版本，P0/P1 按真实兼容组合冻结。[S10]

依据模型配置做 BF16 KV 本体估算：每 token 字节 = 2 × 层数 × KV头数 × 头维度 × 2字节。Qwen2.5-7B 的 8192 token 约 448 MiB，Mistral-v0.3 的 8192 token 约 1 GiB。不包含权重、workspace、分页浪费和其他缓存。[S11]

Prompt 长度加输出长度不得超过冻结的模型上下文限制。32K max context 不等于可以用 32768-token prompt 再生成额外 token。

## 4. 工作负载设计

### 4.1 三套互相隔离的数据

- calibration 使用受控 token 长度与负载组合。
- development 用于选择批量、配额、共同基线与 SLO。
- evaluation 使用没有用于调参的独立请求／会话／文档组。

同一会话或同一公共前缀家族不能跨分区导致泄漏。允许校准覆盖评估中相同的长度区间，不允许提前使用评估结果优化控制器。

### 4.2 机制实验

相同总队列量下，改变源块是否还有活跃引用。相同 staging 占用下，改变它由可驱逐缓存、H2D、D2H 或 SSD 写入占用。分别观察固定水位与依赖策略是否作出不同动作。

需要包含一个父任务必须等全部文件完成才释放保护的情况，防止把完成一个 child 就计算全部 GPU 释放收益。

故障和强制等待测试可以构造事件，但性能结论必须另有正常负载证据。

### 4.3 服务流

复用作者驱动的已实现部分，再补 token 事件采集和公平 replay。[S3]

至少覆盖共享系统／文档前缀问答、多轮会话、热门前缀切换，以及低复用、GPU命中为主、低争用场景。Prompt/上下文长度初始可选 2K、4K、8K、16K，输出长度初始可选 128、256、512。实际组合受模型总上下文与内存约束。

输出固定长度且 ignore_eos 的模式只用于明示的受控负载实验。正常问答使用统一停止规则，不能为了多次重叠让不同策略输出不同数量的 token。

请求到达与前缀内容预先冻结。首选 open-loop 到达，负载生成器与服务延迟解耦。设客户端上限时，必须记录 scheduled_arrival、actual_send 与 client_queue，避免慢系统少收到请求造成虚假优势。

后续会话依赖前一回复的闭环流另行报告，不能与固定到达流混为同一实验。重放固定对话内容时如实称 trace replay。

### 4.4 完整缓存生命周期

每个策略从相同初始缓存状态开始，在一条 trace 内连续运行，不逐请求清空缓存。冷启动与稳定热身两种起点分开。

热身不计主测量时间时，热身产生的 backlog 应先按统一规则排空。测量期间生成的缓存写入、后续命中和 miss 不能剥离到测试之外。

不使用无限增长磁盘作为“SSD 容量受限”的实验。V1 明确只测 GPU／CPU预算受限且 SSD 可容纳该批 trace 的场景。

## 5. 基线与消融矩阵

### 5.1 系统基线

| ID | 系统 |
|---|---|
| N | 相同模型与原生执行配置，只有原生GPU Prefix，外部缓存关闭 |
| U | 作者配套系统，完整原有功能、冻结调优配置、共同必要修复 |
| F | U + 固定传输额度 |
| P | U + 读优先／age／水位等强简单策略 |
| I | 共同基础 + 干扰反馈，不使用释放依赖排序 |
| D | 共同基础 + 释放依赖排序，额度固定 |
| J | 共同基础 + 两者联合 |

GPU Prefix、预加载、共享 slot、CPU cache、成本准入、复制合并、磁盘发布语义在 I/D/J 间完全一致。不能把 J 的额外缓冲空间漏算。

P 与 U 都允许在开发集充分调优。原生多级 vLLM 或其他可复现系统可作补充系统对照，但后端不同不能用于单独归因“策略增益”。

### 5.2 四臂核心消融

使用同一共同底座 C：

| 实验臂 | 依赖排序 | 干扰反馈 |
|---|---:|---:|
| C00 | 关 | 关 |
| C01 | 关 | 开 |
| C10 | 开 | 关 |
| C11 | 开 | 开 |

报告联合相对最强单机制的增益。若讨论交互项，按固定绝对指标计算 `Y11 - Y10 - Y01 + Y00`，并给置信区间，不能凭百分比相加就宣布“1+1>2”。

### 5.3 必须独立控制的工程因素

source-safe 提前释放、同请求逐层消费、改文件粒度、新 admission、新缓冲容量都可能影响结果。第一版不把它们隐式塞进 J。必要共同修复用于所有臂，有算法性质的变化另立实验。

串行版本只能作诊断性消融，不能作为主要竞争基线。

## 6. 指标的固定定义

### 6.1 首 token 与后续 token

记录每请求到达、发送、服务接收、首次实际内容 token、后续 token、结束时间。TTFT 必须扣准起点，空响应或 role-only SSE chunk 不算首内容 token。

请求 i 的 `ITL_i[k] = token_time_i[k] - token_time_i[k-1]`。主合格规则建议使用请求内 ITL 的 P95，另报告跨请求／全体 token 的 P95/P99，不能混淆样本层级。

若只拿到 chunk 时间，记录 `client_chunk_gap`，主 ITL 证据状态为缺失，不能用流文本重新 tokenize 后虚构 token 时间。

### 6.2 双 SLO goodput

正式评估前固定每类服务的 TTFT 与请求内 ITL 门槛。模板暂为 null，Codex 不得填任意数值后直接运行正式测试。

请求合格条件为正确完成、非超时／取消／失败，TTFT 达标，且输出至少两个 token 时请求内 ITL P95 达标。单 token 请求标为 ITL 不适用并单独报告，不混入主持续 decode goodput。

主实验采用固定测量请求 cohort：

```text
goodput = 合格完成请求数 / 完整测量cohort运行时长
```

运行时长从测量 cohort 第一条计划到达开始，到最后一个请求结算且该 cohort 引入的已接受 I/O 按统一规则结算为止。待签发的未承诺预加载可按共同取消规则结束，不能清除已接受 store 来缩短分母。

同时报告只截至最后响应的 makespan、末尾 drain 时间、全部请求吞吐、普通 TTFT/ITL、合格率和 offered load。固定窗口 goodput 如作补充，必须单列窗口、跨边界任务和后续缓存成本。

### 6.3 资源与机制指标

报告实际 staging 高水位、每类 slot 占用、可立即复用 GPU blocks、带 fence 的 blocks、restore 预留、逻辑与物理传输字节、SSD实读/实写、stage wait、parent wait、强制flush时间、progress_override次数、按token/字节的缓存命中、重算 token 数、控制器耗时、snapshot过期／表外回退比例。

flush 统计分清 wait(parent)、批次 flush 和存储持久性 fsync，不能全部计成同一类“清债时间”。

### 6.4 统计方法

相同冻结 trace 和 seed 作配对运行，随机化策略顺序，保持初始缓存和机器状态可比。稀疏 pilot 可先少量重复，正式实验建议至少五个独立 trace seed，次数是否足以支持尾部结论由样本量与区间宽度判断。

按 trace／会话分组 bootstrap，避免将高度相关 token 当独立样本。报告置信区间、原始数值、负收益与低争用结果，不只给最好一条配置。

## 7. 正确性测试矩阵

以下是必须实现的测试类别。测试名称是新增需求，不是现有测试结果。

| ID | 测试 | 通过标准 |
|---|---|---|
| T01 | 工作区未提交内容保护 | 审计前后用户文件不被覆盖、清理或隐式移动 |
| T02 | 依赖与能力真实可用 | 真实 import 路径、Plan API、handler 检查，不靠stub |
| T03 | 单KV组与布局检查 | 不支持的布局显式失败，不能错读张量 |
| T04 | Prefix身份 | 同前缀命中，不同祖先/模型/布局不误命中 |
| T05 | 恢复原始KV | 同一份生产KV在存储往返后的逻辑字节一致 |
| T06 | 模型结果 | 控制batch/采样下对照输出或logits差异，容差预注册 |
| T07 | 部分load完成 | 任何child未完成时不提前ready |
| T08 | CUDA事件 | event仅创建或API返回不代表DMA完成 |
| T09 | 重复completion | 父任务仅结算一次，slot仅释放一次 |
| T10 | 活跃引用 | D2H结束但仍有活跃使用时GPU释放收益为零 |
| T11 | 多保护者 | 必须等全部保护者满足，不能提前计收益 |
| T12 | parent粒度保护 | 单child完成不冒充parent资源释放 |
| T13 | block id复用 | 旧epoch/generation动作不能读写新分配块 |
| T14 | shared staging | 所有copies结束且保留条件满足后才回收 |
| T15 | clean cache | 可安全驱逐时不额外制造写回依赖 |
| T16 | 无额度 | 接受前延期或保持已接受任务，不能return False丢任务 |
| T17 | 队列上限 | 已接受/待接受边界一致，不漏账 |
| T18 | 实际staging floor | 不能静默超批准内存 |
| T19 | epoch额度 | 同一epoch多次pump不会重复发放 |
| T20 | 融合复制计费 | 实际总字节既不越限也不重复扣账 |
| T21 | 联合读写 | 使用共享额度和联合状态，不各取完整预算 |
| T22 | mandatory wait | scheduler停止迭代时reactor仍可推进必要闭包 |
| T23 | 下游续接 | D2H完成到写盘不会被新增普通额度永久阻断 |
| T24 | deadline/age | 普通低优先级任务不会只因策略无限等待 |
| T25 | snapshot过期 | 回退有进展，不执行过期资源动作 |
| T26 | 表外状态 | 成本为未知而非0，使用明确回退 |
| T27 | 客户端取消 | DMA目标不提前给下一请求 |
| T28 | 错误和短读 | 不发布不完整cache，不回报成功 |
| T29 | 异常后排空 | 已提交传输目标仍安全或服务安全终止 |
| T30 | 关闭策略 | 原成本决策、布局、功能和完成语义不变 |
| T31 | 实际SSD命中 | 真实I/O证据，不能全是GPU/CPU命中 |
| T32 | 整条trace | 缓存不逐请求reset，后续miss成本被统计 |
| T33 | token时间口径 | chunk与token不混用，TTFT起止正确 |
| T34 | GPU/noGPU证据 | mock测试不能设置gpu_verified=true |
| T35 | 磁盘隔离 | root合法、预算不足有报告、不清理用户数据 |
| T36 | 流水线重叠 | timeline证明实际重叠，不能仅看non_blocking标志 |

精确传输的验证对象是同一生产 KV 的字节，不是强行要求改变batch后的独立 BF16 重算逐bit相同。后一种数值差异需要受控对照和预注册容差，不能把所有不同答案都解释为缓存错误，也不能忽略真实错误。

## 8. 实现完成与研究继续是两套门槛

### 工程门槛

所有相关正确性测试通过，真实 backend 调用无stub，策略关闭可回退，强制排空无新增循环依赖，实际内存不超预算，性能测量包含控制开销。

观察与控制开销的初始目标可定为低争用下 goodput 退化不超过 2%，并单列CPU耗时。这个数值是内部工程目标，依据开发集测量冻结，不视为实现已达到。

### 研究门槛

至少两类正常负载对最强可比基线出现重复的正收益，配对区间支持该结论。可预注册“约 10% goodput 改善”作为投入门槛，同时要求无争用负场景开销小、没有跳过慢请求、没有透支后续命中。

这些都不是录用门槛或预期成绩。没有达到也必须交付原始结果，不能让 Codex不断调测试集直到达到数字。

## 9. 每阶段 Codex 必须输出的报告

阶段、实际完成项、修改文件列表、精确执行命令、测试通过／失败／跳过数量、GPU是否真实运行、消耗预算、对比数据、依赖版本、已知阻塞、下一允许阶段和明确的非结论。

报告必须附证据路径。不能只说“已实现完整系统”“测试通过”，省略尚未运行的集成与 GPU 测试。

## 10. 回退与止损

只要发现生命周期错误、部分KV提前使用、实际预算越界或等待死锁，立即关闭新策略，安全停止受影响实验，保留失败现场。

若瓶颈通过共同基础修复消失，应把修复贡献与算法贡献分开。若必须更换完整执行器、新写磁盘淘汰或维护逐层模型执行才能看到收益，暂停本 V1 路线，输出明确的范围变更提案，不自动扩大。

版本回退不使用 `git reset --hard` 清除工作。采用独立worktree、已记录补丁或项目允许的非破坏方式，原始结果保持只增不改。

<!-- END 03_IMPLEMENTATION_AND_EXPERIMENTS.md -->

---

<!-- BEGIN 04_CODEX_EXECUTION.md -->

# 04 交给 Codex 的执行指令

请在我当前的 Prefix Cache 项目中实施本交接包定义的 V1。先阅读 AGENTS 等项目规则，然后依次阅读 00、01、02、03、05 文档及模板。

## 固定目标

直接复用 `atlarge-research/py-kvcache` 与作者配套 `t348575/vllm`，并优先复用作者 `kvcache-experiments` 的测试／实验驱动。在现成精确 Prefix Cache、成本准入、预加载、共享 staging 和异步流水线基础上，增量实现可关闭的依赖感知操作排序与 decode 干扰额度。

不得重新设计缓存引擎、io_uring执行器、Prefix身份规则、attention、decode或模型逐层执行。不加入Source、repair、Prefix shadow、量化、MLP、RL、新淘汰算法或跨机调度。

## 先实际审计，再修改

1. 检查当前目录、Git HEAD、分支、未提交与未跟踪内容，以及已有实现和依赖。保留所有用户工作，不执行清理、强推或破坏性重置。
2. 以候选锁文件为线索，验证真实作者版本的接口和测试。候选 SHA 并不代表已经兼容或在 GPU 上通过。
3. 输出接口映射，注明原有能力、最小公共修复、新增观察桥接和研究策略。缺失接口必须写明拟新增，不能当成现成API。
4. 特别核验父任务完成与GPU源块释放关系、强制flush路径、实际staging分配量、单KV组限制，以及import fallback和profiler依赖。
5. 在项目内保留基线与补丁边界。优先小规模扩展现成函数，不另造第二套I/O队列和资源所有权系统。

## 实施要求

按 P0→P1→P2→P3→P4→P5→P6→P7 顺序推进。每阶段满足验收才能进入下一阶段，遇到环境或权限阻塞则完成其余可安全执行的工作并报告。

首先交付 P0 审计、版本锁、CAPABILITY_MATRIX、PATCH_MAP，然后执行允许范围内的CPU开发与测试。不要只返回空泛方案，实际创建对应最小文件和测试。

GPU实验只在 `permissions.yaml` 中已有明确授权、目标环境可用且预算不为空时运行。没有授权时不租服务器、不付费、不下载超预算模型、不更改驱动、不修改远端服务。准备可执行命令和GPU阶段报告，标记BLOCKED，不编造结果。

主要代码增量为紧凑观测、真实释放条件分析、有限候选策略与原reactor薄桥接。保留原LoadPlanner唯一准入入口，首轮对照不改变原准入、不切换已开始H2D的任务为重算。

在引入普通限流之前先验证mandatory排空。不能让scheduler等Future，而reactor又等下一轮scheduler额度。不能为释放GPU块而提前complete_store。下游已接受链路续接不得被普通额度永久阻断。

策略必须提供 off、shadow、fixed、pressure、interference、dependency_only、joint 模式。off回到原有执行路径，共同兼容与正确性修复对全部实验臂一致。

配置必须严格验证。`templates/*.yaml` 是我们新增的项目合同，不能直接当成现成 vLLM API。`null` 表示必须现场决定和冻结，不能当作0或默认性能事实。

## 实验要求

复用作者 benchmark 驱动，补充实际逐token时刻和持续decode测量。单token实验仅用于现有break-even标定，不作为主结果。

保证所有实验臂使用相同模型、KV与staging实际预算、初始缓存、预加载、复制合并、停止规则和生命周期协议。主比较使用相同异步执行器，不能靠关闭基线流水线制造优势。

标定、开发、评估分开，SLO预先冻结。缓存写入、恢复、排队、后续miss、重算与末尾drain全部记录。staging最低slot超配、外部SSD无淘汰等边界必须如实处理。

测试至少覆盖03文档中的正确性矩阵，结果标注CPU/mock/GPU。精确拷贝验证同一生产KV的逻辑字节，跨不同batch的独立BF16重算另外处理数值差异。

## 每次交付格式

输出本次阶段、实际修改、执行命令、测试结果、证据位置、真实GPU运行情况、预算消耗、未解决问题、下一允许动作。不得用“全部完成”代替尚缺的GPU和集成证据。

研究结果无收益也要报告。不得擅自增加新方法、扩大GPU预算、调宽测试SLO或丢弃慢请求来满足10%投入门槛。

## 最终验收

能够用锁定版本复现基线，策略可关闭，完整异步流水线保留，生命周期与资源预算正确，真实请求流证据可追溯。论文收益和新颖性根据实验与后续审计判定，不预设为真。

<!-- END 04_CODEX_EXECUTION.md -->

---

<!-- BEGIN 05_SOURCES_AND_AUDIT.md -->

# 05 来源与本轮核验边界

核验日期 2026-09-25。

## 1. 证据口径

本交接包以本次对话已确定的设计边界为需求来源，并使用官方代码／文档核对底座接口。01—04 中新增策略、候选上限、实验门槛与模块名称都是实施建议，不代表已有系统已经提供或本项目已经实现。

本轮进行了静态代码阅读，没有修改用户仓库，没有编译作者分支，没有运行GPU，没有验证两仓库完整兼容，也没有完成全部相关论文和代码的穷尽审计。

本轮不会将前文“已证明系统的优点”扩大为“当前主分支每一个新增功能都已经发表并被独立验证”。仓库当前能力、论文原版本和本地实测分别记录。

## 2. 已核验的主要来源

### [S1] py-kvcache README

- 固定提交 `3abba7a502d553f6e7e2e58b92086487e3395d7e`。
- [官方文件](https://github.com/atlarge-research/py-kvcache/blob/3abba7a502d553f6e7e2e58b92086487e3395d7e/README.md)
- 支持的事实为外部Prefix缓存、pinned staging、异步I/O、共享预加载、CPU cache、break-even和load/defer/recompute规划。
- 同时明确依赖作者vLLM分支与profiler，磁盘文件没有完整GC／淘汰。

### [S2] 作者 vLLM 分支集合

- [官方分支API](https://api.github.com/repos/t348575/vllm/branches?per_page=100)
- 本轮返回的 `preload` SHA 为 `d6eadf416bb5234047760bf55d532f2f038cf697`。
- `main`、`with_profiling`、`extended_profiling` 同时存在，不能把任意分支名当作与py-kvcache天然兼容。
- 该SHA是候选检出，不是通过测试的环境锁。

### [S3] kvcache-experiments

- [作者README](https://github.com/t348575/kvcache-experiments/blob/master/README.md)
- [仓库](https://github.com/t348575/kvcache-experiments)
- 本轮读取README，包含现成bench驱动、prefix replay、数据集replay、pareto_measure与break-even导出路径。
- README中的原始示例包含单输出token，适合相关校准，不能替代持续decode研究。
- 执行前由P0冻结整个仓库commit。README blob SHA不能代替仓库commit。
- 不直接使用其带清空目录效果的参数，除非已限制为批准的实验专属目录。

### [S4] 作者分支 OffloadingConnectorWorker

- [固定版本 worker.py](https://github.com/t348575/vllm/blob/d6eadf416bb5234047760bf55d532f2f038cf697/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py)
- 本轮读取初始化／布局注册，以及 `_submit_store_jobs`、`handle_preemptions`、`start_kv_transfers`、`prepare_store_kv`、`get_finished` 等路径。
- 关键事实为store按既有时序延期提交、提交成功断言、`jobs_to_flush`的wait、completed_jobs汇总和部分失败断言。
- 完整scheduler侧refcount/fence/allocator关系仍须P0在同一候选SHA核验，不能由worker单文件推断全部释放条件。

### [S5] py-kvcache reactor

- [固定版本 reactor.py](https://github.com/atlarge-research/py-kvcache/blob/3abba7a502d553f6e7e2e58b92086487e3395d7e/py_kvcache/reactor.py)
- 依据本对话已读取的 `_pump_once`、共享slot、completion、`_schedule_work`、`_on_store_copy_done`、父任务汇总与TransferCoordinator路径。
- 支持异步多级执行、copy headroom、父子任务汇总、共享引用和读／写／预加载排序等事实。
- 函数名用作P0定位锚点，不能以旧行号替代实际checkout核对。

### [S6] py-kvcache 成本规划器

- [固定版本 load_planner.py](https://github.com/atlarge-research/py-kvcache/blob/3abba7a502d553f6e7e2e58b92086487e3395d7e/py_kvcache/load_planner.py)
- 本对话已读取规划器输入、ADMIT／DEFER／DECLINE、待写入状态、预加载容量与等待截止逻辑。
- 本轮方案明确继承它，不声称现有规划器已经考虑本方案新增的所有资源释放依赖。

### [S7] py-kvcache vLLM 适配

- [固定版本 vllm.py](https://github.com/atlarge-research/py-kvcache/blob/3abba7a502d553f6e7e2e58b92086487e3395d7e/py_kvcache/vllm.py)
- 本轮读取import fallback、`VLLM_AVAILABLE`、`PLAN_API_AVAILABLE`及单KV group检查。
- CPU单元测试能够导入不代表实际vLLM兼容，这一差别须写入验收。

### [S8] StagingPool

- [固定版本 staging.py](https://github.com/atlarge-research/py-kvcache/blob/3abba7a502d553f6e7e2e58b92086487e3395d7e/py_kvcache/staging.py)
- 本轮读取 `compute_slot_count` 与slot申请／释放，最低slot数量可能使分配超出名义配置。
- 源代码无额外检查不自动等于运行缺陷，是否可能重复释放需要事件测试确认，不能仅凭缺少断言声称已有bug。

### [S9] profiler 包装和依赖清单

- [profiling.py](https://github.com/atlarge-research/py-kvcache/blob/3abba7a502d553f6e7e2e58b92086487e3395d7e/py_kvcache/profiling.py)
- [pyproject.toml](https://github.com/atlarge-research/py-kvcache/blob/3abba7a502d553f6e7e2e58b92086487e3395d7e/pyproject.toml)
- 本轮核对可关闭profiler包装及包定义。作者vLLM fork另有直接profiler import，不能只凭py-kvcache的noop后备认定整个组合不需要该依赖。

### [S10] 官方执行语义与安装说明

- [vLLM KV Offloading Usage Guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)
- [vLLM GPU安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- 用于核对现代offload路径、异步传输、页锁定内存及版本匹配要求。latest文档不自动适用于已锁作者fork，接口以P0实际代码为准。
- 不使用论坛AI回答作为兼容性证据。

### [S11] 模型官方配置

- [Qwen2.5-7B-Instruct config](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)
- [Mistral-7B-Instruct-v0.3 config](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/config.json)
- Qwen配置给出28层、4个KV头、hidden 3584、28个attention头，未启用sliding window。
- Mistral配置给出32层、8个KV头、hidden 4096、32个attention头，sliding_window为null。
- 这些支持理论KV字节估算，不构成GPU实际容量、模型下载许可或后端兼容证明。运行时还要冻结模型revision。

## 3. 审计到的风险与计划处理

| 风险 | V1处理 |
|---|---|
| 作者fork分支与py-kvcache是否完全匹配 | 候选版本明确标未验证，P0/P1跑真实集成 |
| import静默降级stub | 验证实际符号、能力和handler路径 |
| staging名义预算不等于实际分配 | 实际字节检查，所有实验臂一致 |
| GPU源块释放可能等parent完成 | 使用原协议闭包，不提前回报complete_store |
| 下游原本自动续接 | 保留续接，不为新策略制造额外停顿 |
| 已提交SSD操作不可任意重排 | 只调整尚未签发工作与后续压力 |
| shared cache可驱逐 | 优先原生无I/O回收，不虚构写回债务 |
| 磁盘没有完整淘汰 | 首版磁盘可容纳固定trace，容量guard，不声称SSD压力覆盖 |
| 调度器等待但策略又等新调度 | 加限流前先接通必要排空信号与测试 |
| 原失败协议较保守 | fail-closed报告，不虚称透明恢复 |
| token间隔与网络chunk混淆 | 按真实事件统计，缺失则显式标记 |
| 旧Source流水线与新目标不一致 | 只复用通用经验／测试，不导入旧执行语义 |

## 4. 尚未完成

没有下载或编译全部依赖，没有运行其GPU测试，没有确认目标服务器、GPU驱动、存储配置及现有本地修改，没有形成成本表、SLO数值或性能结果。

本轮未重新开展相关论文全量审计。实施完成后若进入论文写作，需要按实际算法与冻结代码补充新颖性比较，不能把本交接包当成“已经确认无人做过”的证明。

<!-- END 05_SOURCES_AND_AUDIT.md -->

---

# 附录 配置模板

以下是项目拟新增合同配置，不是现成推理引擎参数。

## controller_spec.yaml

```yaml
# 拟新增的控制器配置，Codex须实现解析器，不能直接传给现成vLLM。
# 数量是V1起始设计，不是已调优值，null必须现场决定，禁止当0。
schema_version: 1
mode: "off"
inherit_upstream_load_planner: true
inherit_upstream_write_admission: true
preserve_existing_pipeline: true
preserve_native_completion_semantics: true
allow_early_source_release_protocol_change: false
allow_mid_transfer_recompute_switch: false
candidate_limits:
  max_parent_jobs_observed: 32
  max_ready_work_observed: 64
  max_release_closures: 8
  max_dependency_stage_depth: 3
  storage_units_per_batch_candidates: [1, 2, 4, 8]
  # 以上是有界观察和搜索参数，不是全系统并发限制。
credits:
  grant_once_per_epoch: true
  snapshot_max_age_ms: null
  max_issued_h2d_bytes_per_epoch: null
  max_issued_d2h_bytes_per_epoch: null
  max_inflight_h2d_bytes: null
  max_inflight_d2h_bytes: null
  max_new_ssd_read_ops_per_epoch: null
  max_new_ssd_write_ops_per_epoch: null
  max_inflight_ssd_ops: null
  max_accepted_parent_jobs: null
  actual_staging_budget_bytes: null
feedback:
  calibration_manifest: null
  internal_step_budget_ms: null
  uncertainty_margin_policy: null
  table_miss_action: validated_conservative_fallback
  stale_snapshot_action: validated_progress_preserving_fallback
progress:
  mandatory_signal_verified: false
  must_preserve_native_continuations: true
  allow_performance_budget_override_for_progress: true
  may_override_memory_or_event_safety: false
  validated_min_progress_unit: null
observability:
  log_full_kv_in_hot_path: false
  scan_full_kv_pool_each_step: false
  global_gpu_sync_for_statistics: false
  record_actual_allocated_bytes: true
  record_parent_completion_granularity: true

```

## dependency_lock.candidate.json

```json
{
  "schema_version": 1,
  "audit_date": "2026-09-25",
  "status": "candidate_only_not_compatibility_certified",
  "compatibility_verified": false,
  "gpu_verified": false,
  "repositories": {
    "py_kvcache": {
      "url": "https://github.com/atlarge-research/py-kvcache",
      "candidate_commit": "3abba7a502d553f6e7e2e58b92086487e3395d7e",
      "readme_blob_sha": "309a20e60c10279b360adaa74c21158cdcdfad73",
      "source_read_verified": true,
      "installed_commit": null
    },
    "vllm_author_fork": {
      "url": "https://github.com/t348575/vllm",
      "candidate_branch": "preload",
      "candidate_commit": "d6eadf416bb5234047760bf55d532f2f038cf697",
      "worker_blob_sha": "fa79c094a2f7226e888c73e38ef054defc9f9cc8",
      "source_read_verified": true,
      "installed_commit": null,
      "must_verify_plan_api_and_full_lifecycle": true
    },
    "benchmark_harness": {
      "url": "https://github.com/t348575/kvcache-experiments",
      "candidate_commit": null,
      "readme_blob_sha": "b9b973a1f0bc543b66cbd7db7df083d87d7a51ae",
      "installed_commit": null
    },
    "simple_profiler": {
      "url": "https://github.com/t348575/simple-profiler",
      "candidate_commit": null,
      "installed_commit": null
    },
    "user_project": {
      "url": "https://github.com/majic-mk/super-robot",
      "historical_audited_commit": "7c586e1734bdef2a248adda77bb18563178a2826",
      "local_workspace": null,
      "local_head": null,
      "local_dirty_state": null,
      "do_not_reset_to_historical_commit": true
    }
  },
  "runtime": {
    "python_version": null,
    "torch_version": null,
    "cuda_runtime": null,
    "cuda_toolkit": null,
    "driver": null,
    "attention_backend": null,
    "gpu_model": null,
    "gpu_compute_capability": null,
    "vllm_import_path": null,
    "py_kvcache_import_path": null,
    "liburing_version": null,
    "filesystem": null,
    "o_direct_verified": false,
    "io_uring_verified": false
  },
  "release_capabilities": {
    "independent_source_safe_ack": null,
    "source_fence_release_scope": null,
    "mandatory_progress_bridge_verified": false,
    "native_layerwise_consume_verified": false
  },
  "common_patches": [],
  "observer_patches": [],
  "policy_patches": []
}

```

## execution_state.initial.json

```json
{
  "schema_version": 1,
  "status": "plan_not_executed",
  "phases": [
    {
      "id": "P0",
      "title": "工作区与兼容审计",
      "dependencies": [],
      "has_gpu_subtasks": false,
      "gpu_subtasks_require_explicit_permission": false,
      "status": "not_started",
      "expected_outputs": [
        "WORKSPACE_AUDIT.md",
        "CAPABILITY_MATRIX.md",
        "PATCH_MAP.md",
        "dependency-lock.json"
      ],
      "actual_outputs": [],
      "test_evidence": []
    },
    {
      "id": "P1",
      "title": "原版复现与共同基础",
      "dependencies": [
        "P0"
      ],
      "has_gpu_subtasks": true,
      "gpu_subtasks_require_explicit_permission": true,
      "status": "not_started",
      "expected_outputs": [
        "BASELINE_REPORT.md",
        "environment.json"
      ],
      "actual_outputs": [],
      "test_evidence": []
    },
    {
      "id": "P2",
      "title": "观测与正确性进展",
      "dependencies": [
        "P1"
      ],
      "has_gpu_subtasks": true,
      "gpu_subtasks_require_explicit_permission": true,
      "status": "not_started",
      "expected_outputs": [
        "LIFECYCLE_REPORT.md",
        "observation-overhead.json"
      ],
      "actual_outputs": [],
      "test_evidence": []
    },
    {
      "id": "P3",
      "title": "简单基线与稀疏标定",
      "dependencies": [
        "P2"
      ],
      "has_gpu_subtasks": true,
      "gpu_subtasks_require_explicit_permission": true,
      "status": "not_started",
      "expected_outputs": [
        "calibration-manifest.json",
        "PILOT_PROBLEM_REPORT.md"
      ],
      "actual_outputs": [],
      "test_evidence": []
    },
    {
      "id": "P4",
      "title": "最小策略增量",
      "dependencies": [
        "P3"
      ],
      "has_gpu_subtasks": true,
      "gpu_subtasks_require_explicit_permission": true,
      "status": "not_started",
      "expected_outputs": [
        "IMPLEMENTATION_REPORT.md",
        "test-results.json"
      ],
      "actual_outputs": [],
      "test_evidence": []
    },
    {
      "id": "P5",
      "title": "单模型端到端评估",
      "dependencies": [
        "P4"
      ],
      "has_gpu_subtasks": true,
      "gpu_subtasks_require_explicit_permission": true,
      "status": "not_started",
      "expected_outputs": [
        "single-model-report.md",
        "raw-results-manifest.json"
      ],
      "actual_outputs": [],
      "test_evidence": []
    },
    {
      "id": "P6",
      "title": "双模型与健壮性",
      "dependencies": [
        "P5"
      ],
      "has_gpu_subtasks": true,
      "gpu_subtasks_require_explicit_permission": true,
      "status": "not_started",
      "expected_outputs": [
        "cross-model-report.md",
        "negative-results.md"
      ],
      "actual_outputs": [],
      "test_evidence": []
    },
    {
      "id": "P7",
      "title": "冻结复现交付",
      "dependencies": [
        "P6"
      ],
      "has_gpu_subtasks": false,
      "gpu_subtasks_require_explicit_permission": false,
      "status": "not_started",
      "expected_outputs": [
        "REPRODUCE.md",
        "FINAL_REPORT.md",
        "claim-evidence-map.json"
      ],
      "actual_outputs": [],
      "test_evidence": []
    }
  ]
}

```

## experiment_spec.yaml

```yaml
# 项目自定义实验合同。正式运行前需生成单独冻结版本。
schema_version: 1
status: draft_not_authorized_for_gpu
models:
  primary:
    id: Qwen/Qwen2.5-7B-Instruct
    revision: null
    tokenizer_revision: null
  secondary:
    id: mistralai/Mistral-7B-Instruct-v0.3
    revision: null
    tokenizer_revision: null
model_dtype: bfloat16
kv_dtype: bfloat16
single_gpu: true
single_kv_cache_group_required: true
single_model_at_a_time: true
attention_backend: null
cuda_graph_mode: null
max_model_len: null
actual_gpu_kv_pool_bytes: null
actual_staging_pool_bytes: null
ssd_scope: enough_capacity_for_full_fixed_trace_no_eviction_claim
ssd_capacity_budget_bytes: null
ssd_min_free_bytes: null
cache_root: null
namespace_id: null
upstream:
  load_planner: "on"
  staging_cache: null
  enable_preload: true
  preload_share_staging: true
  preload_lookahead_requests: null
  iodepth: null
  open_lookahead: null
  sync_on_store: false
  # 具体字段仅经锁定版本解析器验证后转成真实参数。
traces:
  calibration_manifest: null
  development_manifest: null
  evaluation_manifest: null
  split_by_prefix_family: true
  no_per_request_cache_reset: true
  arrival_mode: open_loop
  initial_cache_state: null
metrics:
  ttft_limit_ms: null
  request_itl_percentile: 0.95
  request_itl_limit_ms: null
  token_timestamp_source: null
  main_cohort_requires_two_or_more_output_tokens: true
  count_failed_and_timed_out_requests: true
  include_accepted_io_drain_in_main_duration: true
  report_client_chunk_gaps_separately: true
  report_controller_overhead: true
statistics:
  pairing: same_trace_same_seed
  policy_order_randomized: true
  group_bootstrap_unit: trace_or_conversation
  formal_seed_count_target: 5
  confidence_level: 0.95
internal_gates:
  goodput_gain_target_fraction: 0.10
  low_contention_regression_target_fraction: 0.02
  targets_are_not_promised_results: true

```

## permissions.yaml

```yaml
# 项目自定义权限模板，不是 vLLM 配置。
# 修改必须来自用户明确授权，不能由 Codex 自行扩大。
schema_version: 1
allow_project_local_edits: true
allow_cpu_tests: true
allow_public_source_read: true
allow_remote_push: false
allow_new_cloud_rental: false
allow_payment: false
allow_driver_or_system_changes: false
allow_shared_data_deletion: false
allow_gpu_runs: false
allow_model_downloads: false
max_gpu_hours: null
max_model_download_gib: null
approved_gpu_ids: []
approved_experiment_root: null
approved_dependency_root: null

```
