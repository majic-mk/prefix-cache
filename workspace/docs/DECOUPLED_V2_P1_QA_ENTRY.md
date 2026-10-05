# P1 固定 Source QA 与单动作入口

本文件仅记录执行接线，不改研究路线、数据划分或验收阈值。P0 授权不能继承给 P1；没有新的 P1 授权时只允许 CPU 准备。

## 原始清单 → 操作

`p1_qa_dispatch_v2.compile_p1_qa` 从冻结 P1-E/P1-M 图重新解析具体 arm，不接受调用方临时换题、换 Source 或自定 mask。保留完整 token 输入与 32-token 自由生成上限；独立记录 request ID namespace、content-key 适配与关闭 prefetch window。不改旧清单。

P1 的矩阵比较锁定 `legacy_normalized_kv`、15%、first reuse layer=9。dense 不绑定 Source；历史 Source、独立 S0、配对 E/M1 分别绑定实际出生对象。S0 不是第五个在线候选。

`verify_qa_source` 不只看回执：核对实际 backing、出生完整 manifest、目标 token/位置、generation 与身份。实际目标候选必须唯一。没有 Source、缺 SelectionState、边界漂移或准备失败均保留失败，不能拿 dense 答案填补该 Source arm。

## 原生执行

QA 驱动复用现有原生 context、比较、lease/HBM、loader、K/V repair 与 cleanup；不启用目标捕获或发布。指定 Source 的质量诊断不经过 residual 或经济性拒绝，但仍执行原始身份、快照、租约、空间、ready、mask 检查。普通生产 commit 不接受此诊断标记。

当前入口只记录固定 d=8 的观察，不宣称已完成 d1/d2/完整 checkpoint 选择质量比较。后者需要独立受控观察任务和 QA oracle 聚合，不能拿深层分数代替实际质量最优解。

### 实际执行身份与容量预检

冻结 operation 保留原始 token、位置和请求摘要。在打开原生 context **之前**，由实际隔离 Source store 的 `content_key()` 绑定执行副本的 Segment 内容身份；记录绑定前后的摘要。不同 authorization namespace 的 key 不能直接沿用，不修改冻结 operation，也不允许 key 与 token/位置矛盾。

worker 显式继承已审计父进程的 import 路径顺序，避免 editable `.pth` 把克隆前的实现重新放到优先位置。

比较容量依据审计过的模型 config 和完整 active request 行数，在模型加载前调用原有 `comparison_workspace_estimate`。512-token 目标不代表只有512行 current projection：2661-token 请求仍有整个活动上下文的 QKV 暂存费用。预算不足应在 CPU 预检明确失败，不自动放大预算、缩短请求或降低估算；资源变更须明确记录并重新生成操作摘要。dense 和 exact Source 构建不执行比较，不应虚构比较费用。

CPU 预检还要使用原统一 HBM manager 核算同时存活的 working KV、完整 comparison reservation、winner KV 和默认4 GiB安全余量。仅检查 comparison 自身能否装下是不够的；资源上限调整必须同时核验总池，不将“单项足够”误报为整体可执行。此预检不是实际 CUDA allocator 峰值测量，运行时原有原子预留与设备错误检查仍保留。

## 有界入口

`scripts/server/run_decoupled_v2_p1_action.py` 默认只验证；只有显式 `--execute` 才启动单动作 worker。manifest 必须绑定：

- 实际 native manifest、P0 原始证据准备合同、冻结图及具体 operation；
- base runtime、build/QA dispatcher、runner、入口、输入解析器与资格消费者的文件摘要；
- 当前实例/GPU 身份、P1-E 或 P1-M 独立授权、开始/截止、时间上限和账单模式；
- 初始化/动作/清理的保守上界，最大动作数=1；
- 静止初始 pool/catalog/registry、显式资源与 registry 预算；
- 如消费或继承 Source：构建回执及对应成功原始 result 文件摘要。

Source 构建结果必须经原始事件链、record、audit、cleanup 和 operation 摘要复核。失败批次留下的回执不算成功；CPU fixture 不能作为真实 GPU 构建。每次都从冻结图重编译 operation，单独自洽的 hash 不是任务成员资格。

外层 watchdog 覆盖 worker 预检、模型加载、执行及收尾；内层动作 deadline 最多180秒。超时停止任务、保留证据和需重新检查的池状态，不自动重试、换 Source、进入下一任务、租卡或关机。

## 计时与证据边界

QA TTFT 是请求打开至首 token（包含该动作的准备与诊断检查）。`action_total_ms` 从模型已初始化后的动作入口开始，覆盖该动作的执行、校验和收尾，**不包含此前模型初始化**；初始化及进程启动须另看 worker/session 墙钟，不能混入同一 TTFT 口径。资格诊断的 full hash 不得当作正式 online 性能。不会由固定 Source 成功推出生产 REUSE_COMMIT、系统收益或论文证据。

前置资格核验虽然不发布 Source 或刷新 LRU，但核对历史 backing 时会短暂取得 `TargetSourceStoreV2` 的独占锁。多个任务引用同一个 P0 Source witness 时，应串行执行这些前置检查；锁竞争须保留为资源阻塞，不能改标数值失败、绕过锁或重写原证据。GPU 小批仍按单活跃请求串行执行。

新入口的 CPU 回归与任务编译不继承旧 GPU 资格。真正上卡前仍须核验当前环境/导入、部署摘要、新的 P1 授权与小批执行；正式矩阵必须消费实际成功构建对象。当前没有自动批量矩阵队列，也没有输出伪造 QA/F1。
