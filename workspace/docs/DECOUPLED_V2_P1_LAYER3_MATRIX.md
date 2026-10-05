# P1 第3层主矩阵实施记录（2026-09-22）

## 范围与来源

用户已批准主矩阵改为 completed_depth=2 / first selective reuse layer=3。
新入口单独支持 P1-E/P1-M 与冻结 fit/validation；原单fit诊断入口不放宽，
原layer9入口、14个QA结果和失败历史不覆盖、不混算。
Source仍来自原冻结历史，暂不按metadata/K差异重新挑选，不新调准入阈值。

实际代码基线 `cc7898b1ba59d89ce7fdbb186ded21880f1adf08` 含已记录工作区覆盖层，
不把HEAD单独声称为完整运行身份。本批新增文件由逐文件SHA及layer3_binding绑定：

- base runtime: `3428c5e95765ec4997ae0507e3cb5144d13cd5f2e68f2eb0c01b2330b98b234e`
- layer3 dispatch: `73c75f40e9741c29ae23887cd407e95fe07714dbefb5976908371fb231450bfd`
- layer3 entry: `5addf8c841755f90c9fde407eba1c1511940182e0df8d87fd9a7d50d7f3e1976`

原模型执行、Source构建及审计模块未改写；已有12份Source逐对象核验后复用。

## 实际修改

- `src/probekv/p1_layer3_matrix_v2.py`：新操作编译、层序账本、固定Source消费、
  原始事件验证、exact/mixed独立数值gate。无在线选择、经济准入或新Source发布。
- `scripts/server/run_decoupled_v2_layer3_action.py`：按旧冻结图重编译验证成员身份，
  叠加仅改变边界的新操作；校验runtime/model/GPU/资源和teacher+greedy先决证据。
- 两个新测试文件：18项新增CPU测试；分区/来源gate隔离、篡改拒绝、
  确定性动作顺序、实际目标文件复制与损坏拒绝。
- `artifacts/decoupled-v2/server31780-20260922/layer3_matrix/`：独立prepare/launch、
  原始证据聚合、有限队列watchdog、CPU终止后汇总和只读传输。

混合Source消费仅复制目标KV/SelectionState，保留共享来源manifest元数据，
不复制父KV；不删除或修改原出生池。QA每个动作独立模型进程，不能把前一arm的
模型/工作KV当作后一arm的初态。数值时的hash和进程初始化不冒充在线热路径性能。

## 冻结动作与验收

原图 `cd2a37dcd4fb95b5baf7846c55faf59f9299ff918b45400ade35ef1093b4b642`：

- P1-E：16组、29未来请求、174 QA；四历史Source＋独立文档S0＋dense。
- P1-M：3组、5未来请求、15 QA；dense/E/M1。
- 构建：89份总量；12份复用，77份待执行。
- 新数值先决：每phase首个fit请求4动作，共8；不从validation挑数值调试样本。
- 本批队列274动作；CPU preparation通过不等于模型或P1科学判定通过。

固定K/V repair=0.15，512-token目标77行；前两层full，第3层仍完成全QKV投影，
之后attention/MLP按目标support加全部非目标行执行。teacher至少32位置，
relative-L2<=1e-4、预测token相同；free greedy token IDs相同。失败停止，不放宽。
exact数值通过不得自动解锁mixed，mixed另用M1验证。没有必要数值证据不能开始对应QA。

资源维持已批准comparison2.125GiB / 总HBM6.875GiB / 安全4GiB。
每动作真实执行180s、含启动/清理进程900s watchdog；总watchdog由冻结动作数计算。
不自动租赁/续费/关机/进入P2，费用未知保持null。

## 已完成 CPU 证据

- 本地全量1790项：1789通过，1项条件跳过（本地存在GPU时跳过CPU失败封闭场景）。
- 服务器独立部署包1776项：全部通过。比本地少14项旧CPU矩阵汇总测试；
  这些旧分析辅助测试不在复用代码快照中，新18项及旧boundary测试已部署。
- compileall、合同validator、git diff --check通过。
- 服务器全部274操作编译及资源预检通过；旧Source原始GPU事件/receipt复核通过。

远端独立目录：
`/root/autodl-tmp/probekv_stage2/artifacts/p1-layer3-matrix-server31780-20260922-run1`

环境锁SHA `d2d42284e5c11a48a417184bc1a1a3ba428b927a1fcbd3b251ccc212d5d26e89`；
动作计划SHA `16b7d2432262bac85fe3642b24f7de1e69538f59255c0e94cf272e7651685928`。
GPU批次已启动；具体完成数以原始事件及新状态快照为准，不在此静态记录宣称全部通过。

首个P1-E数值前缀已完成并下载原始证据后在本地重新核验：32个teacher位置的
logits relative-L2最大值为0，预测token一致，自由greedy token IDs一致，
exact fixed15 QA允许执行。该证据不解锁P1-M；mixed独立gate仍未执行。
本地复核文件：
`artifacts/decoupled-v2/server31780-20260922/layer3_matrix/evidence-run1-1790071431/locally_reverified_exact_gate.json`。
已有固定Source QA开始执行，多Source互补性、选择准确性和净收益仍未判定。

## 解释限制与待做

当前16个E组中7组有2–4个未来请求，9组只有1个；validation只有1个多消费者组。
单消费者组不能证明跨请求互补；更不能仅凭一个曝光fit例子证明d1/d2准确。
H与coverage仍按冻结定义计算，单组、fit/validation和全F1为零的情形显式报告。
本批只回答固定候选的质量空间及mixed风险；未验证在线新增阈值、d1在线策略、
自然增长、质量尾风险或完整系统收益。metadata多样化补充候选只是建议，未启用。
完整P1-M还需实际E/M1 artifact差异、独立目标存储/消费及质量报告，不能仅靠QA均值。

保持 formal_profile_bundle_frozen / gpu_runtime_qualified / paper_evidence / locked_test_accessed=false。

## 已完成前缀的只读复核（2026-09-22，非最终判定）

不可变部分汇总 `partial-analysis-1790076046190432976/quality_matrix_report.json`
SHA256：`366e2cb7c38190ffd10a0bc3a746b571f686351da4b34fd593a0c69df27398b6`。
该快照为22/89份Source构建（含12份旧构建复核复用）、38/189个QA，失败0，
151个QA仍pending；不可把此快照当成完整矩阵的完成数。

4个已有完整对照的content组包含6个未来请求，其中一组有3个未来请求。
这6个请求各自的dense、历史Source1–4和S0输出token IDs均相同；
当前prefix的macro headroom=0、coverage space=0、complementary groups=0。
尚无完成的validation组，P1-E判定仍为INCONCLUSIVE，不能作总体No-Go结论。
F1相同甚至全为0不等于选择器准确；相对dense“无下降”也不等于答案正确。

追加只读解释审计 `inspect_arms.py`，不修改运行中的模型、队列、来源或分析阈值：

- 对汇总引用的原始GPU事件重新验签，检查Source身份、三方KV完整性、
  同请求current-K和位置、完整32层执行行及30层fixed15 support。
- 6个请求均具有4个不同的历史KV logical digest，4套不同repair mask；
  mask两两Jaccard在0.481–0.855之间。
- 每个Source arm目标512行中repair=77、reuse=435；前两层full，
  第3层QKV投影仍full，之后attention/MLP按固定support执行，非目标行不丢失。
- 因此当前的答案一致不能解释为“全部arm使用同一份KV”或“退回dense”；
  但此证据不能区分15%修复已足够、任务对来源不敏感等科学原因。
- 不增加GPU前向、不改变prompt/scorer/ratio/threshold、不按这些结果重选候选。

解释审计报告：`source-arm-audit-1790076473796246100/source_arm_report.json`，
SHA256：`4b9768f3f661feeafb89165a1cae7a6c765a5c1719dfc2a425fd4469e4390b88`。
两个报告均在远端本批根目录和本地`layer3_matrix`工件目录中保留。

审计新增6项CPU测试；连同原layer3/matrix相关测试共38项通过，compileall及
git diff --check通过。首次用包名运行旧混合工件测试时缺少tests搜索路径造成
3项import error；补齐正确PYTHONPATH后38项全部通过。此处不是新的GPU通过声明。

剩余阶段不变：完成P1-E全部fit/validation，然后独立P1-M数值gate、E/M1消费
和质量矩阵；不自动开始P1-R、P2/P3或运行时性能实验。d1/d2选择准确性仍未验证。
