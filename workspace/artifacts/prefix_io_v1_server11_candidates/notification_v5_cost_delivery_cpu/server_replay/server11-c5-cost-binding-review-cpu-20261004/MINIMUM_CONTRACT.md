# C5 成本绑定 CPU 准备的独立审查合同

本目录仅写 CPU 反例与审查材料。不改冻结 C5/C4、native_cost_v6、旧 runtime 或已封存结果；不导入 torch/vLLM，不连接服务器，不启动模型/GPU，不创建预算授权。新的准备对象不是 `ExactSingleFileReceipt`，不得由字典、PASS 文本、CPU 合成数据或私有 issuer 冒充可供策略消费的成本凭据。

## 保持原计量定义

固定六窗口按 AB、BA、AB 执行，前两个独立 pair 为 calibration，第三个为 validation。每窗口 129 prompt tokens、128 cached tokens、完整输出 128 tokens；原 full_decode_step CUDA Event 口径，warmup offset=[1]，measured offset=16。工作量保持 1 次 917504 bytes SSD read、原 Qwen 模型/布局、batch=1、active_decode=1、prefill=0、context=144、四种已有 I/O 均为零。原生校准 owner 保持 parent cap=8、bridge=None，不改执行器/缓存引擎。

对于两个校准 pair 的测量值 A_i、B_i，原整数公式为：

- `a = ceil(mean(A_i))`
- `d = max(0, ceil(mean(B_i - A_i)))`
- `u = max(0, max(B_i - a - d))`
- `U = a + d + u`
- 原 A-only 内部预算为 `a + max(0, max(A_i - a))`；此固定几何下等于两个校准 A 的最大值。

以上计算复用冻结 `p4_paired_measurement_verifier.py` 的原 numerical AST，不重新拟合实现。warmup 和 holdout 不进入任何上述成本或预算。原 `all_pairs_original_estimator_audit_only` 包含保留组，只能审计，不能当拟合结果。保留组 B 必须 `B_holdout <= U`，任意正 underprediction 都保持失败，不扩大阈值、不根据验证组重拟合。

## 来源及完整原始链

新 plan 必须在作业前独立冻结，实际 source-before/after、guard、six children、raw windows、原 Event 源、完整模型/布局/设备、实际 `_run.co_filename`、加载模块、同一个 collector scalar adapter 的全部 128 帧、原 I/O CQE 与排空链均应与计划对应。`analyze_paired` 单独不授予 native_execution_verified；字符串 `origin=native_gpu_recording` 不是实际执行证明。CPU 准备可校验结构及来源，但真正发行入口保持阻断。

calibration 与 validation 的 trace、prefix family、workload 哈希必须分别独立重算并不重叠。六个实际 native request 唯一，pair 内完整 output IDs 相等；请求窗口不重叠。原 serializer 要求六个 fresh original process / private storage、同一真实 guard 会话、每窗口原 shutdown 返回。不得仅拷贝旧摘要或改 `job_id` 得到新凭据。

runtime common 与 overlay refs 唯一路径、不重叠。collector 必须位于 overlay，且其 exact ref 等于新 plan.collector_source_ref；这是冻结 bridge 按真实 capture 类源定位的要求，不能把 collector 搬到 common。模型、Event、原 runner、实际 C5 reactor 要求共通锁；完整 runtime/helper/观测源也必须固定且实际调用绑定。

## on 观测开销仍有缺口

原六窗口校准的 bridge=None 不会安装 on-only Queue.get 观测和通知等待。仅把这些源码纳入 lock，不能说明其开销已测。新完整 on 入口还需要独立的实际安装/等待/wake/截止/排空证据及其成本资格，不能直接继承旧 C4/v6 成本或此前 73.6% CPU 数字。

on 延期后 I/O 的因果位置也不同于校准 B 的原 overlap-I 路径；不能将其强行套入旧 overlap 检查并放宽界限。当前准备输出必须显式保留 `on_observation_cost_measured=false`、完整入口资格 false、GPU false；不得提供可以绕过冻结 P4Policy exact-type 边界的对象。必要新源码/新原生校准/有限 GPU 授权缺一即保持阻断。

所有反例均使用显式路径参数或环境变量绑定 server project、native-v6、C5、已封存 runtime preparation；不得依赖本机隐式路径映射冒充服务器闭包。源前后 SHA、实际命令、结果与限制独立归档。
