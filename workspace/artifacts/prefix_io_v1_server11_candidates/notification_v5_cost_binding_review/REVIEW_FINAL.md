# C5 成本绑定准备：独立 CPU 审查

结论：当前纯 CPU 准备接口和原计量边界检查通过；它仍不能签发可用于策略的原生成本凭据，不能启动 GPU。本目录没有修改任何冻结旧源，也没有执行服务器、模型、CUDA 或实际 I/O。

本地最终运行 `LOCAL_REVIEW_01/TEST_RESULT.json` 为 20/20 通过，失败/错误/跳过均为 0，耗时 2.047 秒。原公式反例 7 项，新接口攻击测试 13 项；源码锁 44 项加锁文件共 45 项前后 SHA 一致。真实命令、平台及全部引用在该结果和 `TEST_STDERR.log`。本地锁 SHA 为 `f1ec852bfb238d8a2d34f282d7c8953b7885ce24615fff5148d7efb28deee42a`；服务器必须使用它实际生成的锁、显式 roots 及独立输出目录，不能移植本地计数或标签。

审查发现并已验证修复的三处边界：

- 实际 C5 reactor 保留 common ref，collector 仅在 overlay，并等于 plan 的 collector ref。其他 C5/preparation 源与 factory 继续位于 overlay。所有 C5 源一概移到 overlay 会与原真实消费接口断路。
- owner 的 parent cap 必须 exact int 8，bridge=None 描述必须 exact bool True；warmup offset 必须 exact int 1。Python 中 `True == 1`、`8.0 == 8` 不能作为固定参数通过的依据。
- 合成公式审计保持实际单文件几何 917504 bytes；反例中的 journal、CQE、payload、drain 同步更新，不拿原泛型 8-byte fixture 当 C5 几何通过。

另外验证了：错误的外部预期 binding/plan/source-lock 引用拒绝；重复或遗漏源码拒绝；新增未冻结 Python 文件拒绝；原观测模块即使从新 common/overlay 同时删掉也不能绕过准备锁；返回 document 是副本，修改它不会改变准备对象或签发成本。真实 load_verified_single_file/launch_gpu 与 CLI 始终阻断。合成审计只调用冻结 analyze_paired，独立 spy 确认没有调用 verify_native_cell。

计量反例执行冻结原 numerical AST 和原 A-only 预算 helper。校准仍固定前两个 AB/BA pair，第三个 AB 为独立 holdout；warmup 和 holdout 极值不改变拟合成本或 A-only 预算，holdout 超过上界 1 纳秒即覆盖失败，不能调阈。输入中的 Event/native 元数据全部为明确合成值；这些通过不是实际 CUDA 测量。

完整最小契约见 `MINIMUM_CONTRACT.md`。原 bridge=None 六窗口只能为共通原路径提供新条件成本，不能证明 on-only Queue 观测/等待的完整入口成本。当前 `on_observation_cost_measured`、`full_runtime_cost_qualified`、`native_cost_qualified`、`native_execution_verified` 和 GPU 资格均为 false；有效成本上界/预算均为 null。旧 C4/v6 数值及此前 CPU 百分比不能转成新入口资格。后续真实发行仍需新的冻结完整入口、原始校准/保留组/排空/guard 证据、完整 on 观测成本资格及明确有限授权。

本报告只评审接口准备。与新的真实服务器 source metadata builder 结合，可以进一步验证实际文件闭包；该步骤也不能产生真实测量或成本资格。服务器结果由 root 另存和核验。
