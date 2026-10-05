# P4 无卡合同审计

本审计按 00/01/02/03/04/05 完整交接文档及 5 个模板，核对当前 P3 验收、控制模块和生命周期。用户当前明确授权无卡 CPU 工作。此文档是静态验收清单，不是 P4 实现资格；本审计没有运行测试、CUDA、模型或缓存 payload。

## 结论

P4 CPU 可以完成严格数据合同、有限候选策略、依赖单独/干扰单独/联合的纯计算逻辑、fake-backend 事件测试、off/shadow 薄桥验证、CPU 开销测量、补丁和执行证据。P4 真机 shadow、真实资源释放和 production 干扰输入、策略安全 smoke 与端到端效果均保持 BLOCKED。P5-P7 不因 CPU 结果自动开启。

P3 conditional 表不是可直接上线的 production interference 表。实际 production_state_table_complete=false、p4_connected=false、七个 cells 的 production_value/delta_prediction 均为 null，基线由 min-median 有限比较得到 B=U。GPU allocator 即时可复用能力仍 UNKNOWN；策略不得将 parent完成或 D2H完成兑换为 GPU release credit。

## 可完成的 CPU 验收

- **C01 严格、非空快照合同：**run/epoch/captured_ns/expiry、owner 与 capability provenance；等待目标、原优先顺序、四 stage 计数、staging 实际容量、mandatory 集合；unknown 明确 None。空 waiting/ready 可以表示 idle，但用于 nonempty 资格的 fixture 必须有真实描述和闭包。
- **C02 严格输入拒绝：**拒绝 bool 当整数、负计数、NaN/Inf、空 owner/run、重复 ID、cross-run/epoch/generation、count-total 矛盾、缺失能力当零、未对齐 physical units、bounded 集合溢出。
- **C03 真实释放闭包语义：**活跃 refs>0 不产生可复用收益；全部保护者 AND；父粒度不能被单 copy/文件完成替代；native reusable ACK 独立于 Future；失败/unknown 保持 fail-closed。
- **C04 资源去重与零 I/O：**以 pool/group/block/generation 去重，shared 多 consumer/copies 只计一次物理资源；原生 clean evictable 路径先于 I/O；不能把候选 victim 当 observed blocker。
- **C05 有限、确定候选：**最多 32 parents、64 ready work、8 closures、3 stages/events depth；不限制原系统并发、不丢窗口外已接受工作；必要 closure 全部文件条件不受 depth=3 截断；不对已提交 work 重排。
- **C06 同目标字典序：**保留 native target/priority，比较合法解阻时间；误差范围内再看干扰，再看新 bytes、native order；GPU/CPU bytes/ms 不相加，未知估计回退。
- **C07 依赖单独消融：**dependency_only 普通额度固定，选择排序只用充分见证；活跃引用、parent 粒度、多 protector、restore 和 staging 占用相同但原因不同的 fixture 产生合理差异。
- **C08 干扰与联合 CPU 逻辑：**interference 不用依赖排序；joint 同时启用；lookup 区分方向/联合流量/context/batch/已有 I/O；误差 margin 不重复计已有 I/O；测试纯 CPU mock 表显式非 GPU/非 production。
- **C09 真实表拒绝与保守回退：**导入 P3 条件表必须拒绝 production 激活；缺 delta、生产资源 witness、冻结 step budget/margin 或 state mismatch 进入原路径/U 回退；不能产生虚构 Delta=0。
- **C10 共享预算一次性消费：**四 stage 累计 ops/physical bytes 与 inflight 各自守卫；SSD shared、copy shared 统一扣账；同 epoch 多 pump 无补发；fused 提交总 bytes 一次计费；backend 接受后失败仍计费。
- **C11 即时原生再验证：**run/generation/epoch/expiry recheck + actual issue safety；策略只返回值型计划，不拥有 Future/slot/FD/CUDA Event；旧计划不许强行签发；settlement 防重复/跨 epoch permit。
- **C12 强制进展边界：**mandatory 在 wait 前发布、reactor 独立继续；continuation/mandatory-support/age/shutdown 只绕性能额度；容量、events、dependency 均不能绕过；必要下游不因普通 quota 永久停住。
- **C13 失败、STOP 与 exactly-once：**部分提交/短读/record/query 故障、parent failed draining/final、未知 drain 保留原 owners；client cancel 不提前复用 DMA target；父结果/slot 仅结算一次；没有透明恢复假象。
- **C14 off/shadow 与现成引擎：**off 在策略 clock/snapshot/queue 创建前返回 native；shadow 拟动作不修改实际 native 顺序或签发；保留 planner唯一入口、Prefix身份、融合、共享 staging/preload；无新任务队列或源块早释放。
- **C15 CPU 开销与上限：**在服务器 CPU 对 nonempty/full-window fixtures 报 thread_time_ns、wall-time、P50/P95/P99、candidate counts、record bound；只支持 CPU 成本，不能等价端到端 ≤2%。
- **C16 证据、补丁与准备：**新 CPU patch 与 P3 frozen source 分离，已冻结 P3不改写；version/source locks、实际 guarded commands、unique tests/skips/subtests、reason logs、reproduction、禁 GPU receipt，GPU plans 仅准备不执行。

## 非空严格字段合同

### SystemSnapshot

- schema_version, run_id 非空 bounded；snapshot_epoch/captured_ns 非负 strict int，expires_ns>captured_ns；field capabilities 不得伪填
- scheduler_owner/reactor_owner provenance、各自 epoch 与发布时刻；inference state(active_decode,batch,prefill,context分位,step time) 支持或显式 unknown
- 有界 waiting target(native priority,arrival/wait,admission,need GPU/slots/restore,blocked reason)；四 stage pending/inflight counts/bytes 与 actual staging/free/clean reclaim
- resource identities/generation、active refs/protectors/completion 摘要；mandatory/awaited parent、last progress；所有缺失字段 reason 非空

### WorkDescriptor

- parent/child/run/generation/stage identity；native order/created_ns；accepted/submitted明确 bool；合法 physical bytes/min-unit
- bounded consumers 与 count；needed slots/source-event/target allocation generation；已提交只计估计不进入重排
- immutable value only，禁止原生 job/Future/FD/Event/slot/tensor ownership 引用

### ReleaseWitness

- owner identity+pool/group/block/allocation generation；GPU/CPU/restore各自单位的 effect；release_scope明确
- 所有 AND保护条件完整，protectors完整性已知；pending parent+全部 child closure，不将 unknown列表变空集合
- evidence_level=observed_blocking/candidate/unknown；native_reusable 独立ACK，候选不得实际credit；证据ref/type/scope必须非空
- closure resource dedup、remaining schedulable work、已提交不可改阶段、completion estimate+uncertainty+time origin

### CandidatePlan

- 同一 native target/priority、run/epoch/expiry；selected ready work IDs；reason与fall back scope明确
- storage-unit批次 1/2/4/8有限候选，与真实quantum绑定；closure depth/parents/work counts及truncated flags
- predicted legal unblock时间与误差、附加干扰估计支持、new physical bytes，排名过程可解释，不声明全局最优

### DispatchBudget

- run_id/epoch/issued_ns/expires_ns 非空严格；sample age；累计签发ops/bytes与simultaneous in-flight独立
- 四 stage+SSD shared ops/bytes+copy shared bytes；actual staging reservation/admission common bound/min progress unit
- epoch只发一次，accepted backend 立即settle；fused physical sum；performance override记录理由，never native hard-limit override

### InterferenceQualification

- status/scope为production必须显式资格；环境/engine/model/layout/granularity/config SHA绑定
- prediction convention delta-vs-existing 或 total联合二选一；matched state(key directions,existing traffic,batch/context/prefill)
- finite nonnegative value+uncertainty，冻结 step budget；table miss/original-native fallback；CPU-only mock永不升级 gpu_verified

## 进展边界复核

- publish mandatory before blocking native wait；不持 reactor mutex 等 self future
- poll completion/续接/slot settlement先行，普通策略只触及既有未签发work
- 同 epoch多 pump额度无补发，grant expiry不需要下一轮scheduler才可强制前进
- normal quota不足不能用 transfer_async=False 表示已接受延期；incoming/active/draining共同计数
- age/mandatory/continuation/shutdown例外只绕性能额度，资源和event安全不绕
- cross-run generation、未知/失败保护不晋升资源信用；child完成不提前 parent complete
- 原 parent exactly-once、shared/cache pin生命周期、STOP未知drain原对象保留
- 关闭研究入口直接走原阶段路径，保留共同安全修复；shadow没有签发权
- 结果分类 CPU/mock/shadow/GPU 分别保存，mock 永远不得 gpu_verified=true

## 无卡无法完成的资格

- **G01 真实 GPU/allocator owner witness：BLOCKED。**GPU generation、active ref/fence、victim selection 与 immediately reusable ACK；必须以 live owner 适配和后续实际回收核验，不由 parent/counter=0 猜测。
- **G02 真实 production 干扰表：BLOCKED。**匹配正式 native engine/model/资源状态；含方向/联合/已有 I/O、冻结 internal step budget 与 uncertainty policy、独立验证和 table miss fallback 资格；P3 conditional 表不能替代。
- **G03 P4 shadow 真机：BLOCKED。**真实当前状态非空候选与保护闭包，拟动作不改上游，预测对真实后续回收；不能当反事实速度。
- **G04 授权小规模策略真机：BLOCKED。**D/I/J 分别开关、输出一致、SSD real bytes、实际 staging/queue/inflight、原 fusion/流水线完整、mandatory 真机进展、真实 fail-closed；CPU fake events 不算 DMA 证据。
- **G05 GPU 与端到端开销/效果：BLOCKED。**真实 token 时间与全 cohort accepted-I/O drain；正确控制对照和冻结配置；P4 smoke 不是 P5 正式双 SLO goodput 或 CI。

## 当前缺口和非结论

- dependencies.py 为 CPU fixture read-only 关系，当前无 live GPU generation/reference adapter。
- ReleaseWitness 独立数据类、完整 SystemSnapshot/WorkDescriptor 及 P4 candidate/cost/policy 尚未在审计开始时存在。
- config.py 能验证所有模式名字，不等于 interference/dependency_only/joint native策略已实现；现有 native_boundary 仅 off/shadow。
- simple_stage_policy.py 只接受 P3 off/shadow/fixed/pressure，P4策略必须单独明示增量边界。
- 旧 ResourceId/Parent/Resource dataclass 未自行完整严格校验，P4入口不能把旧 fixture 承载类当合格严格 schema。
- P3 GPU owner reusable/free-capacity 保持 UNKNOWN；没有可获credit的全局 GPU owner ACK。
- P3 conditional 表 production_state_table_complete=false、p4_connected=false；七cells production_value/delta_prediction均null，scope不等生产输入。
- P3 B=U、普通 pending flush 仍有观测，但收益未证明；禁止制造人工限速或把猜测等待当可消除时延。

CPU 通过不能证明真实 DMA 完成、瞬时设备 overlap、GPU资源可回收、低争用端到端退化≤2%、SLO goodput 或论文研究优势。CPU fixture 可以显式提供有证据字段来测算法，但必须与真实现状的 UNKNOWN 分离，不能回写已有 P3 证据。

## 证据与实际操作

完整输入 path/bytes/SHA、实际表状态、16 类 CPU 验收、5 类 GPU阻塞和 guard 保存于同目录 p4-cpu-contract-audit.json。仅新建本审计目录 JSON/Markdown；显式列出的文本 SHA 被读取，运行源码未修改，测试执行 0，GPU 操作 0，GPU预算消耗 0。
