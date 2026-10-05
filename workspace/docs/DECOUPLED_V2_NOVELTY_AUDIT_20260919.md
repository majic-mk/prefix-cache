# ProbeKV v2：正式实验前创新性审计与实施补充

检索始于2026-09-19，整理于2026-09-20。范围：现有 `majic-mk/super-robot` 的首次交接 v2 任务书、当前代码与相关论文；不是完成稿审稿，也不是新 GPU 实验报告。

## 1. 结论与权限边界

**路线有研究价值，但尚未证明足够的增量贡献。现在应增加证伪能力，而不是再叠在线机制。**

保留用户确定的主线：同一 exact Segment 的少量历史 Source；当前浅层 K-only 选择，K+V 作为消融；freeze 后 Query 关注度 × winner 偏差选择 repair；先固定15%；目标全 token/全层重算可在合法近似上游下捕获，G0/G1严格区分；仅保存目标 KV 与轻量元数据。

本补充不修改原始交接包，不改变 P0→P1-E/P1-M→P2→P3→P4→P5→P6 依赖、成功阈值、冻结分区、样本上限或预算。既有任务书优先。本轮只新增研究审计与离线分析能力，不连接服务器、不租卡、不运行 GPU、不冻结 Profile、不解锁 P1。

不能再单独作为首创主张的内容：多历史 Source、非前缀 KV 复用、Query-aware repair、偏差驱动 repair、训练免费、load/compute overlap、按 I/O 调 repair 比例、有限代来源标签。组合是否有贡献，必须由同质量、同字节预算和完整成本下的实验回答。

## 2. 检索与核验范围

用户指定的 Sider Scholar 搜索工具存在，但本轮两次调用均返回 `UNAUTHORIZED / oauth_refresh_token_missing`；没有取得可使用的 Sider 检索结果。随后改用 arXiv 原文、作者原始来源和会议官网。没有将搜索摘要、二手博客或模型生成综述作为算法证据。

检索围绕：exact chunk 的历史上下文版本、current-state KV compatibility、query-aware recomputation、mixed-source 再入库、来源污染与生命周期成本。按 arXiv ID/标题去重，固定下表版本。不是系统性穷尽查新，未检出不代表不存在；不根据预印本给出 SCI 分区或录用保证。

| 论文与固定原始来源 | 原文核验位置与直接重叠 | 对本项目的限制/区别 |
|---|---|---|
| [CacheBlend v3](https://arxiv.org/html/2405.16444v3)，[EuroSys 2025 官方目录](https://2025.eurosys.org/accepted-papers.html) | §4.1–4.3 KV偏差与稀疏重算；§5.1缓存存储、逐层I/O/计算平衡 | fixed15、偏差修复与重叠是执行基础，不是本项目首创。其 fusor 输出新 chunk 的缓存路径也使“混合执行后产生缓存”不能被直接宣称为全新思想。 |
| [Cache-Craft v1](https://arxiv.org/html/2502.15734v1) | §3.3 同 chunk 多 prefix variants，按 CFO 选择；§3.2 Eq.12，§3.2.2 question-focused chunks | **最直接的多版本先例**。必须证明浅层实测信号相对 CFO 的增量信息与净收益；同引擎 CFO 消融不等于完整 Cache-Craft 系统复现。 |
| [SparseX v2](https://arxiv.org/html/2606.01751v2) | §3.2 当前 Q 的因果 attention 重要度；§3.3 边界处理；§3.4 full→sparse | Query 引导和稀疏执行已有先例。其多 Segment 不等于一个 Segment 的多历史 Source。GPU-resident 的结果不能替我们证明 CPU/SSD 路径。 |
| [QCFuse v1](https://arxiv.org/html/2606.05875v1) | §3.3 Eq.14–17 compressed KV view；§3.4 关键层 query scoring；§3.5 fusion | anchor 压缩的是 query conditioning 的 token 视图，**不是历史 Source 的16→8裁剪**。本路线不引入 anchor；只用于相关工作及后续合法基线评估。 |
| [ProphetKV v3](https://arxiv.org/html/2602.02579v3) | §4.2 Eq.2–6 attention/value误差分解，§4.3跨层 query probe | Query×偏差有直接理论近邻，不能把乘积写成首个理论或严格质量界。需要证明 winner-specific 当前观测、有限浅层成本的价值。 |
| [CacheTune v1](https://arxiv.org/html/2605.24022v1) | §4.1离线频域token选择；§4.2稀疏传输；§4.3 Eq.9–11硬件相关repair调节 | 动态 repair 与I/O balance已有先例；当前仍fixed15，不把未来动态比例包装成本轮创新。 |
| [KVComm v2](https://arxiv.org/html/2510.12872v2) | §3.2–3.4历史anchor、token KV相似性与offset估计 | 当前状态相似性利用历史缓存并非空白；其跨agent语义/偏移近似不同于本项目 exact目标的离散完整Source选择。 |
| [SemShareKV v2](https://arxiv.org/html/2509.24832v2) | 参考prompt/token LSH匹配；附录A.4 K/V偏差 | 语义近似匹配与偏差修复已有先例。本项目坚持exact token identity，不把模糊匹配作为新增路线。 |
| [FusionRAG v1](https://arxiv.org/html/2601.12904v1) | §3.1相似片段离线融合；§4.1 alternative path/cache去冗余 | 不是在线多历史Source排名，但“更好的预构建Source + selective repair”不是空白；可作为后续强单Source近邻，不临场扩P1队列。 |
| [HijackKV v1](https://arxiv.org/html/2607.19957v1)，[USENIX Security 2026 官方页](https://www.usenix.org/conference/usenixsecurity26/presentation/zhang-yichi) | 相同可见chunk可携带出生时上游影响，跨请求缓存复用的污染风险 | exact token match、G0或G≤1都不证明无污染；权限域与可信输入边界是必要条件，不把本项目声称为抗攻击防御。 |

除明确链接的会议官方目录外，本轮不把其他条目自动标记为已核实同行评审接收。方法审计以列出的 arXiv 固定版本为准；论文作者的速度/质量数字不移植为 ProbeKV 结果。

## 3. 三份审稿视角报告

### Reviewer A：新颖性与意义

**判断：重要系统问题，创新证据不足，不宜按当前实现直接投稿。**

优势是问题可被精确定义：同样的目标 token 在不同历史上下文中产生不同 KV，未来请求可能偏好不同版本；选择成本与储存成本可被实测。最大的反对意见是 Cache-Craft 已有多版本，Query 修复也有直接近邻。如果只有模块组合、若干更快样例或“winner不同”，不足以说明方法推进。

必须回答：在固定可见候选池上，真实 QA oracle 比强单 Source 有多大空间？当前 K 信号比 CFO/K均值多提供多少有效信息？多版本是否比使用相同字节预算缓存更多独立内容划算？实验结果若不能支撑这些问题，应报告当前主假设未成立，不能用扩机制代替证据，也不擅自改成另一条研究主线。

### Reviewer B：技术可靠性与解释

**判断：来源与执行边界认真，但代理分数、任务效用和安全性必须分离。**

K_trim 是上下文兼容性启发式，不是 Query×D mask 修完后的真实残差。G≤1 是依赖传播规则，不是误差幅度上界。浅层观测相同的两个完整 Artifact 可能深层不同，任何依赖该相同观测的确定性选择器都无法凭阈值辨别。

应以匹配观测/身份的碰撞分析解释失败，并在 P2 验证真实 Q 的 RoPE/GQA/global causal-softmax、时间可用性与额外成本。不要将 oracle 计算出的未来状态、问题前置时不可见的目标attention或未计算token的零值混入在线mask。

### Reviewer C：实验与复现

**判断：CPU与协议证据不能支持实际QA、性能或多Source必要性结论。**

需要分解四个量：候选池本身的空间、selector保留的空间、repair带来的增量、整条系统支付成本后的收益。固定同engine、ratio、boundary与池才能隔离选择器；固定winner才能隔离repair。旧executor计时不能当v2在线TTFT；手选成功样本不能代替因果trace。

失败、missing、不支持和未执行必须分开；tie-aware oracle不能把任意同分winner算成选错。fit/validation按content及完整来源依赖隔离；90个开发样本不能提供1%尾部质量认证；同请求多Source、多ratio不是独立质量单位。最近邻的完整系统复现必须与同引擎组件消融分开。

### 综合判断

三份意见共同指向：**先证明可利用的互补性，再证明轻量选择保得住，最后证明修复和来源供给的净价值。** 当前没有足够证据判断“已达到一区”，也不能据文献重叠断言项目必然无价值。保留路线，缩小主张，执行已有有界实验最合适。

## 4. 重新收紧贡献表述

| 候选贡献（待验证，不是已完成成果） | 最低证据 | 不能替代它的结果 |
|---|---|---|
| 有限历史上下文版本的质量互补性，以及当前浅层K利用互补性的能力 | 同池QA矩阵、post-hoc best-fixed与部署可用强单Source、CFO/K均值/K_trim/KV消融、held-out agreement/regret与全部选择开销 | Source向量不同、d1 winner变化、deep residual更小 |
| Source freeze后的任务相关误差修复 | 同winner固定15%、同boundary下D-only/Q-only/Q×D，真实QA与完整观测成本 | 乘法公式更复杂、attention图更好看、仅支持句命中率提高 |
| 有界来源传播下的缓存供给—风险—成本闭环 | exact-only vs允许G1，同字节预算的因果trace、可用率/复用率/质量尾部/物化成本，T21/T22及传播拒绝证据 | G标签存在、目标切片CPU可读、G1被称为无损 |

值得补齐的是**“浅层状态可辨识性”实证诊断工具**，不是新selector或新研究任务：任务书P1-M/T22已经要求该诊断。本轮将其落到分析代码，量化信息什么时候足够、什么时候丢失，帮助解释选择收益及失败边界。只有真实数据支持时，才能把这项分析写成测量贡献；CPU反例不等于发现了真实workload普遍现象。

## 5. 需要修正的逻辑与落实方式

### 5.1 因果性：兼容性选择不等于问题效用选择

对于 decoder causal self-attention，若目标之前token、位置和执行历史相同，仅改变目标之后的问题，则目标浅层K不受该suffix影响。这是结构推论，不是本轮GPU观测。

因此：K-only负责历史上下文兼容性；Query×D负责已冻结winner的任务相关修复。若同浅层观测对应不同QA最优Source，记录不可辨识冲突，不调阈值“修好”。P1记录现有请求的query位置与观测签名；新增布局实验如需额外推理，另行批准，不扩大180/36上限。

### 5.2 QA oracle、固定Source和tie

对于完整固定候选集合S、请求集合Q和真实答案评分F：

\[
H_g=\frac{1}{|Q|}\sum_q\max_{s\in S}F_{q,s}
-\max_{s\in S}\frac{1}{|Q|}\sum_q F_{q,s}.
\]

它是同组事后质量空间，不是可部署策略收益。部署单Source必须用过去/fit选择，在validation冻结；post-hoc best-fixed单列为更强的描述性参照。QA并列最优都算oracle集合；计算所选Source的F1 regret，不因ID不同惩罚QA同分者。

safe coverage按原先冻结的逐请求dense F1下降标准计算，并报告池oracle、固定Source和实际选择三者。候选不完整、QA缺失时不能将缺项当零，也不能只计算已成功的行给出完整headroom。候选池变化时，不套固定池best-fixed公式。

真实最优成本必须在QA合格者中按匹配计时scope筛选；deep residual最小者只是代理参照。本轮新分析器不计算未支持的在线TTFT收益，不把可选executor duration推成系统oracle。

### 5.3 Source分数与repair目标错配

\(\rho_{trim}\) 只用于Source评分；Top-K trim positions不是运行repair mask。Q×D可能选择不同token，所以不得把K_trim命名为“实际修复后剩余误差”。P2/P3记录两种集合的交集/差异及QA影响，不据此改成联合Source–mask搜索。

### 5.4 Query×D不是误差保证

固定Q时的attention输出一阶扰动包含attention权重乘V扰动，以及K扰动引起的整行softmax权重变化；本项目的标量归一化乘积只是一种可实施代理，没有覆盖所有项。与 [ProphetKV误差分析](https://arxiv.org/html/2602.02579v3) 对照后，不主张严格bound或必然不降质。

P2按任务书实现 post-RoPE Q/K、GQA映射、全合法key的稳定softmax、目标行TopK和确定性tie。记录source/digest、query/target绝对位置、producer depth/consumer layer、causal-mask身份和额外运行时间；shared query观察只记一次。Query不可见按已登记D-only回退，不能解除causal mask。

D-only/Q-only/Q×D须固定相同producer、consumer和可选support。Legacy若采用不同的逐层刷新，只能作为整策略对照，不能把其差异全归因于打分公式。mixed-CFO必须来自真实mixed出生执行，缺失则NOT_EVALUATED，不借用exact同prompt的元数据，也不把该缺项算成CFO失败。

### 5.5 固定15%不自动带来更快模型执行

同token数、boundary、Source tier和repair量下，更好的Source首先可能改善质量；它不会自动减少相同执行kernel的工作量。当前加速证据应来自更少dense fallback、更高质量合格复用覆盖，以及支付selection/storage/materialization后的总成本。更低repair率和更浅boundary只能在后续获准阶段单独测试，不用它们解释本轮fixed15结果。

### 5.6 G0/G1、隐私与污染边界

精确地在某上游下计算，不代表那个上游可信。HijackKV表明 token相同仍可能携带出生上下文影响，见[官方论文页](https://www.usenix.org/conference/usenixsecurity26/presentation/zhang-yichi)。本项目只在任务书已批准权限域与可信数据条件下实验，保留出生上下文/执行来源身份；G0不是安全标签，G1不是风险上界。不新增攻击生成任务，不宣称已经防御缓存投毒。

## 6. 当前实现与尚未验证项

| 功能 | 现状 | 本轮可以说什么 |
|---|---|---|
| `source_comparison_v2.py` + `selection_comparison.py` | K-only normalized L2、stable trim、余项均值、候选范围收据 | P0诊断接口已有CPU实现；QA有效性未证明 |
| `source_provenance_v2.py` | 同请求传播、跨上下文generation与发布限制 | 可执行来源规则，不是数学风险界 |
| `segment_capture_v2.py` / `native_source_capture_v2.py` / `source_store_v2.py` | 独立目标切片与浅层状态；隔离pool；目标不持有前文KV | CPU接口/文件测试存在；GPU数值与性能待验证 |
| Source/Prefix shadow解耦 | v2 capture默认不创建整请求shadow | 不等于Prefix命中组合路径已通过 |
| 新Query×D repair | 尚未接入v2；P0仍是fixed15 `normalized_kv_deviation` | P2待实现，不能写成已集成 |
| 普通生产factory | 仍建立历史schema10 pool/store；v2在隔离P0桥接 | 尚非完整v2生产系统 |
| P1资格消费者 | readiness明确保留未实现阻塞 | CPU通过不能解锁P1 |
| 自然历史Source/target资格 | census不等于重新绑定的合法队列 | 不用旧诊断或受控数据充当新自然机会 |
| 本轮identifiability分析器 | 新增只读CPU证据分析 | 验证算术与拒绝边界；不产生GPU或QA结果 |

## 7. 已采纳与不采纳

已采纳、无需改变研究路线：

1. 更新最近邻矩阵与贡献措辞，保留CFO为离线同引擎强对照，不恢复为默认在线步骤。
2. 新增同池真实QA headroom、tie-aware regret和浅层碰撞只读分析。
3. 将query位置/观测可用性、score-mask错配、G代数非误差界写入实验解释合同。
4. 将Q-only/D-only/Q×D的同winner消融及全部观察成本作为已有P2任务的明确验收内容。
5. 强化论文中“已实现/CPU验证/真实模型待验/未实现”的区分。

不采纳：堆叠CFO+anchor+d1/d2裁剪；为了创新新增学习模型；以deep residual定义QA oracle；宣称G1安全；根据看过的QA调阈值；以CPU模拟解锁GPU资格。

需要确认才可执行：新增超过现有上限的布局/安全/大型baseline实验；修改路线或主评价标准；更换数据隔离；调整预算。该审计没有授予这些权限。

## 8. 更新后的执行顺序（原阶段不重排）

1. **P0受控正确性**：继续当前有界E/M动作与真实数值验证，保留失败证据；本轮分析器单测不替代它。
2. **P1-E / P1-M**：修复资格消费者并冻结合法data/lineage，按原上限采集真实QA矩阵。先审Source机会、可选择空间、完整Artifact与浅层碰撞，不盲调阈值。
3. **P2**：实现并数值核验真正Query×D；同winner下D-only/Q-only/Q×D，另保留Legacy整策略对照，共4主arms、至多120消费动作，先fixed15。query读取/softmax/TopK成本全部可追溯。
4. **P3**：相同pool/repair/engine比较CFO、K均值、K_trim、KV与真实QA oracle；d1/d2/legacy只在合法观察点使用，不以更深代理代替QA。
5. **P4**：新浅层执行边界及完整在线成本，公平Prefix baseline、fallback及setup全部计入。
6. **P5**：同总字节预算的exact-only/G1因果生命周期；报告供给收益、质量尾部与物化成本。
7. **P6**：独立双模型与完整最近邻复现；依据实际目标期刊再确定完整稿件证据要求。

任务书的P1-R一次有界救援继续保留：至多12目标、72消费动作，须先完成对应P2数值实现检查；不增加样本或尝试次数，不用救援改写首次负结果。Source构建、独立dense参照和数值检查预算始终另列。E/M诊断也不得删除S0强控制。

### 本轮代码范围

- 新增 `src/probekv/source_selection_identifiability_audit_v2.py`。
- 新增对应CPU测试。
- 新增本审计文档及分析器接口说明。
- 不改生产selector/repair、数据分区、主阈值、GPU manifest、原始交接包或旧证据。

代码使用方式和实际CPU验证结果见同目录 `DECOUPLED_V2_IDENTIFIABILITY_AUDIT.md`。源码身份仍为真实base commit加工作树摘要，不能把本次未提交文件算入旧部署包SHA；本轮未更新服务器。
