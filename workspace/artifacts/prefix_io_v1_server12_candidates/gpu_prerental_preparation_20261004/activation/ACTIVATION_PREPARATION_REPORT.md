# 有限精确 cell 激活前置准备

本目录新增 project overlay 和只读证据发行器，不修改原作者代码、不修改上一轮冻结文件，不启动 GPU、模型、I/O 或另一个执行器。当前**真实新强域 GPU raw 数据为零、GPU-qualified cells 为零、private capabilities 发行数为零**。CPU 准备完成不表示 on、效果或正式 goodput 已验证。

## 最小 overlay

`source/prefix_io_control/` 复制前一轮冻结的 36 个原 control 模块。仅 `p4_cost_table.py` 新增私有发行入口和对已发行 capability 的查询；新 `gpu_cell_issuer.py` 完成证据重验。

原 `P4Policy` 的 `type(table) is CostTable` 检查原样保留。原 policy、bridge、startup options 和 paired verifier 文件字节保留；原 CPU prepared loader 依然不能启用 production。普通 `CostTable(...)` 构造器仍只接受 mock_only/conditional，并永久保持 production false。自行传 `production_qualified=True`、把 JSON scope 写成 GPU、CPU semantic candidate 或旧单文件 receipt 都不能打开此入口。

私有 capability 是 Python 进程内防止意外晋升的边界，不是针对任意进程代码或恶意 `object.__setattr__` / monkeypatch 的密码学安全沙箱。这与既有 private runtime binding 的边界一致。

## 唯一发行接口

```text
prefix_io_control.gpu_cell_issuer.issue_verified_gpu_table(
    project,
    plan_ref, measurements_ref, guard_ref, intent_ref,
    expected_plan_ref, expected_guard_ref, expected_intent_ref,
    expected_runtime_domain_sha256, expected_gpu_uuid,
    expected_source_lock_sha256, expected_collector_source_ref=None
) -> exact CostTable
```

expected references 来自冻结计划前及原 guard 实际收尾后的 parent workflow，不能从 raw 测量自己推导。所有引用为根目录内 relative POSIX `path/bytes/sha256`，拒绝缺文件、symlink、路径越界、重复 JSON key、非有限数值及字节漂移。

发行顺序是：

1. 新 `strong_gpu_exact_cell_prelaunch_plan_v1`，明确 native preregistration，拒绝 CPU fixture 和旧 singleton 域；核对独立 intent、实际 project/GPU/common-domain/source-lock。
2. 严格解包实际 `strong_native_u_i_cpu_configuration_v1` 的 `configurations` 字段，再调用 SHA 锁定的原 strong pair validator，复核全部 U/I 共同设置、planner on、精确 Prefix、preload、共享 staging、原融合和异步流水线。首版只接纳 enforce_eager=true。layout 引用必须是与实际模型 config 同源的 canonical compact tensor geometry JSON，无 newline，SHA 等于其实际几何摘要，不能拿 metadata 字符串代替。所有实际 issuer/table/collector/owner/CUDA Event/guard/estimator/model/layout/runtime-pair 引用必须在新源闭包中。
3. 复用原 `native_conditional_cost.validate_guard` 验证实际原 GPU guard、成功退出、自然 shutdown 和 OS session drain；要求 before/after 完整源核验 receipt。预算 guard 源 SHA 固定，发行器没有新增预算或执行权限。
4. 对每个 cell 的六个原生窗口重放原 `validate_capture`：全部 128 输出、完整 128 原始 frames 和 CUDA Event query 因果见证、prompt/cached/selected offset、GPU/source/model/layout 和源 Event 身份都一致。
5. 重放原 `original_post_shutdown_drain` / `validate_io`，核对全部真实 owner journal、CQE bytes、实际 StageAccounting、初始/末尾 inflight closure、B 的独立 SSD payload 与选中 GPU 区间因果重叠。没有 release credit。
6. 只把两个 calibration pairs 传给 SHA 固定的原数值 AST；第三 pair 为独立 public-prefix/trace holdout，绝不用于 fit。holdout 超上界直接拒绝，不能抬高阈值后重验。
7. 再读关键来源字节，才能创建进程内 capability，并返回 exact overlay CostTable。source receipts 和现场 source ancestry 仍由原 guarded parent/runtime gate 负责，不从 CPU 摘要推导 GPU 事实。

原 estimator 公式没有改变。初次数学 test 因 timing contract 的 `EvidenceRef` 对象与 dict 不一致失败 2 项；已改为原 `EvidenceRef.from_mapping` 契约，原 AST 保持不变。

## 有限范围与现场接口

最多 1–8 cells。首版每个 cell 只覆盖 eager、batch=1、active_decode=1、prefill_tokens=0、已冻结的 `context=prompt_tokens+measured_offset-1`、SSD read，以及 1/2/4/8 个完整原单位。既有 I/O 为 ZERO 必须由完整原 owner journal 因果证明，不能靠 plan 自报。任何其他 stage、bytes、batch、context、I/O vector、模型/布局/源或 kernel 状态均 UNKNOWN，保留原 U 路径。

首轮现场激活只覆盖单次普通原 preload issue/deferral。原 `choose_batch` 返回的 `production_qualified` 标签仍为 false；本目录不宣称 production batch 激活、改批量或新增融合已生效。

`qualified_identity(table)` 只对实际已发行 exact table 返回 immutable `ExactGpuIdentity`，否则返回 None。身份包含实际 UUID、common runtime domain、标定源锁、模型/布局、eager kernel、native/collector/CUDA source SHA、实际 runtime refs 和 cell signatures。

每个 signature 为 `(gpu_uuid, common_domain_sha, model_sha, layout_sha, batch, active_decode, prefill_tokens, context_length, physical_bytes)`。source/SDK/kernel 另经 immutable identity + actual runtime refs 绑定，不通过相似值扩大匹配。

完整运行 artifact 添加 table 文件后会形成新的 runtime source lock；标定源锁可作为严格字节匹配的祖先，不能要求两个全锁 hash 相同或放宽代码/模型/SDK漂移。`runtime_pair_ref` namespace/run-id 字节可不同，但双方规范化 common-domain SHA 必须一致。

runner 的现场 binding 还必须核验原 collector 的 weak runner/source 绑定、actual prepared tuple、start.query 完成但 end.record 尚未开始、时效与两次 current-step recheck、actual StageAccounting vector。只有 scheduler load 或自报 ZERO 均不足。该现场实现由 runner 目录负责，发行器不创建第二队列、owner、release protocol 或 decoder。

新 bounded collector 若由独立 source lock 预先冻结，可通过 `expected_collector_source_ref` 精确接入；不能用换名字或旧 collector SHA 掩盖新源码。其增大观测容量不能扩大已发行成本 cell 的负载范围。

## 测试与尚缺证据

本机 CPU 拒绝与原数值 AST 测试 19 项通过，0 failures/errors/skips。包括：原 mock/conditional 永久关闭、构造 flag/scope 拒绝、裸私有 token 拒绝、exact type 保留、unknown cell/I/O/stage/bytes 不估为零、fit-only 公式、holdout 超界不重拟合、128 帧要求、GPU guard 声明拒绝、旧 singleton/CPU fixture 不能发行、parent refs、实际源漂移、非法 JSON/path、真实 wrapper 解包、实际 compact geometry SHA，以及原 P4 文件不变。

实际 final/server 验收使用 `verify_activation_cpu.py --original-frozen-control OLD/frozen/control --pair-config ACTUAL_PRIVATE_U_I_CONFIG.json --output ACTUAL_RESULT.json`。该 runner 对全部新增/复制 Python 源、实际 wrapper 和 36 个原输入做 before/after SHA 检查，检查 native/GPU 客户模块导入为零及 `_ISSUED` 为空。正向 GPU 资格测试本轮**没有运行**，不以 fixture 填补。

无卡环境中的 driver library 0-byte placeholder 不作为实际驱动资产通过证据。本目录不改原 driver pin；实际加载、版本/UUID、CUDA Event 与 native wrapper 等设备事实必须到 GPU 模式重新核验。CPU 私有 SDK 链接或源文件核验不能代替该阶段。

下一允许硬件阶段仍是总交付的有界 strong U/off 资格。新强域 raw 标定、独立 deadline/reserve、实际 ordinary candidate、真实 on 输出与原 drain、自然数据和正式评估都必须各自通过才能宣称效果。本目录不承诺有收益，不扩大原 8 小时 GPU 预算。
