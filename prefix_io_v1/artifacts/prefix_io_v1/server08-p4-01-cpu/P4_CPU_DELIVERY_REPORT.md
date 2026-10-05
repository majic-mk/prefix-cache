# SERVER08 P4 无卡阶段交付

当前结论：已完成当前冻结接口与现有测量资料下可安全执行的 CPU 合同、薄桥接和 fake 事件验证。状态为 **P4_CPU_CONTRACT_AND_MOCK_COMPLETE_PRODUCTION_GPU_BLOCKED**。完整 P4 尚未验收，方法提速尚未验证；这份交付不进入 P5–P7。

所有源码修改、测试和原始证据都在服务器 `connect.westd.seetacloud.com:20739` 的 `/root/autodl-tmp/prefix-io-v1-handoff/project` 完成。无卡来自用户当前说明，本阶段没有探测 GPU 可用性。

## 实际实现及边界

P3 的控制源码、原生工作区和结论保持原样。新增 P4 隔离工作区：

- `third_party/work/prefix-io-p4-01-cpu/src/prefix_io_control`：新增六个模块 `p4_types.py`、`p4_policy.py`、`p4_cost_table.py`、`p4_production_table_contract.py`、`p4_bridge.py`、`p4_options.py`。
- `third_party/work/py-kvcache-p4-01-cpu`：只增量修改作者 `py_kvcache/reactor.py` 和 `py_kvcache/vllm.py`。原 LoadPlanner、Prefix 身份、模型执行器、共享 staging、预加载、复制合并、原异步队列与父完成协议继续复用。
- 新测试目录 `tests/prefix_io_v1_p4_policy`、`tests/prefix_io_v1_p4_bridge`、`tests/prefix_io_v1_p4_stage`；新增四份 CPU 资格、补丁、离线复查和 GPU 阻塞准备脚本。

值快照、任务描述和释放证明严格验证 run、epoch、时间、generation、能力与来源。资源释放使用完整父任务和全部保护者 AND 条件；活跃引用、未知完成与未知 GPU 复用能力不产生优先释放额度。观察上限为 32 个 parent、64 个候选工作、3 阶段、8 个闭包。超过观察范围回原路径，不丢弃原生任务。

纯 CPU 策略已覆盖有限合法批量 1/2/4/8、原生 clean reclaim、同一目标的解阻时间及误差区间、传输字节和原序。D 不使用干扰信号。额度由既有 DispatchController 唯一消费，没有第二套队列、资源所有权或额度账本。

原生薄桥增加原 owner 的只读 inspect/publication、同 epoch 缓存、原 ready 集合内的建议投影与性能预览接点；拒绝跨线程、过期、错误代际和虚构预测。CPU/cache witness 只表示潜在可回收闭包，不等于槽立即空闲。GPU 释放量始终未知。

关闭新策略会在时钟、快照和候选扫描前短路。shadow 保持原操作与额度。现有 fixed/pressure 继续通过原 P3 配置入口使用；新增 P4 入口管理 off/shadow/D/I/J，两入口互斥。当前 I/J 无合格生产表，整体回退 U，不安装新的 fixed 控制器或性能限额。D 使用显式固定额度底座；其真实 native ETA 当前为 None，因此不会凭 CPU 模拟预测改变生产优先级。

没有新增引擎正确性修复。P3 共同基础保持，观测/薄桥接与研究策略分别保存为 `patch-roundtrip-01/0001-observer-native-glue.patch` 和 `0002-bounded-research-policy.patch`。正反应用与基线字节一致；只安装第一份补丁时 off 无研究策略或后端导入。

## 实际 CPU 结果

统一回归实际 **1,588 唯一通过、16 跳过、0 失败**，XML 总数 1,604。其中新增 P4 用例 **179 = 88 策略/生产候选合同 + 79 原生/桥接/配置 + 12 阶段门禁**。独立 88、43、79 项运行用于相互核验，不再叠加到统一数量。

16 个跳过的真实原因是：13 个只适用于未安装 vLLM 的 fallback 分支；3 个 io_uring_setup 被当前环境拒绝。获准的 Linux-AIO CPU 路径仍实际执行。作者 GPU 端到端套件没有运行；12 个冻结的 P3 旧 fixture 仅保留历史资格，未重跑、未计入当前通过。此次选取当前实现相关回归，数量不能与上一交付的历史并集 1,613 直接相减推断失败。

统一 CPU 子进程退出 0、全部清空；CUDA guard 确认未初始化、GPU 工作负载 0。70 份输入源码在测试前后 SHA 一致。两份 patch 的正向和反向应用逐文件字节一致，52 原文件恢复，58 最终文件一致。

独立审查实际发现并关闭了跨 run 身份、联合干扰错误求和、restore 误认、直接预览前置校验、预测注入、代际窗口、采样时钟与年龄等问题。旧失败或解释器、fixture、启动路径错误记录完整保留，不将其写成通过。

原 P3 的 **2,724 份锁定输入**、**21 份控制源码**及原生基线保持；注册的 **3,048 份缓存、2,796,552,192 字节**全部重算 SHA 一致，无新增 .bin 文件。没有缓存合并或删除。

## 真实剩余工作

1. D 仍缺真实、合格的父任务完成/解阻 ETA producer 和误差来源。纯 CPU 排序通过不代表真实预测已接通。
2. I/J 仍缺真实 paired GPU 测量的语义 verifier、可授予资格的 CostTable loader、生产负载状态来源、状态匹配的额度/批量激活及资格验证。当前严格 candidate parser 只绑定元数据，结果恒为 BlockedProductionCandidate；自报 PASS、CPU mock 和 P3 conditional 表不能升级资格。native safe preview 接点和人工 defer fixture也不能替代生产应用。
3. 真实 GPU generation/active refs/protectors/复用容量、新 P4 CUDA/模型完整输出、实时 shadow、干扰、开销和性能效果仍 BLOCKED。不能把剩余工作仅写成“缺 GPU 数据”。
4. 相对独立 U 的方法提升尚未成立。C00/C01/C10/C11 必须共用固定普通额度底座；C01=只开 interference，C10=只开 dependency。另与最强 P3 基线 U 比较。SLO 仍未冻结，不运行 P5 正式效果评估。

## GPU 与预算

本阶段新增真实 GPU 运行 **0**，GPU 预算预留 **0 秒**，下载 **0 字节**，驱动/系统修改 **0**。权限 YAML 与 GPU ledger 字节保持。累计 GPU 时间仍为 **16,520.562886646483 秒（4.589045 小时 / 8 小时）**，剩余约 3.410955 小时。

CPU_ONLY_AUTHORIZATION.json 是本轮明确范围记录，不改写历史 permissions.yaml。GPU 准备/启动检查实际执行，返回 BLOCKED_CPU_ONLY_NO_GPU_LAUNCH（退出码 78），在 GPU 导入与预算预留前拒绝。

下一允许动作是继续处理新出现的 CPU 具体问题；若恢复有卡阶段，先核验明确授权、获准 GPU、原存储与预算，再补齐上述 production producer/verifier/接线并完成 P4 原生/模型资格，最后才允许进入效果对照。当前不启动 GPU、P5–P7 或新范围。

## 证据与复现

- `current-integration-01/result.json`、`cpu.xml`、`cuda-guard.json`、`process.log`、`plan.json`：实际统一测试与完整 argv/env。
- `test-input-lock-v2.json`：70 份已测试输入；初版锁与启动修正保留。
- `patch-roundtrip-01/result.json` 及两份 patch：观测/策略边界和可退回证明。
- `final-input-and-P3-preservation.json`：所有 P3 字节与缓存数据保全。
- `audit-review/p4-cpu-static-review-v2.json` / `P4_CPU_STATIC_REVIEW_V2.md`：独立审查，full_P4=false。
- `P4_VERSION_LOCK.json`、`lifecycle-and-modification-map.json`、`ast-count-scope-reconciliation.json`：真实版本、修改位置、生命周期与 AST 统计范围。
- `gpu-stage-preparation-v3.json`、`gpu-launch-denied-v3.json`：GPU 阻塞与真实剩余实现。
- `offline-P3-observation-boundary.json`：旧 P3 记录的离线复查，不是 P4 live shadow，不推断新收益。
- `EXECUTED_COMMANDS.json` / `REPRODUCE_P4_CPU.md`：本轮实际命令及重跑说明。
