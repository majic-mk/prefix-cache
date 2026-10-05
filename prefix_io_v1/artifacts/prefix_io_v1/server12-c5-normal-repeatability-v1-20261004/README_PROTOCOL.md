# 固定三次成本重复性诊断

PROTOCOL.json 在三次作业之前冻结，始终使用第四 workload（首 token 28100、seed 2829）、off、同一 917,504-byte 独立 SSD 输入、offset 16、完整 128-token 输出与原排空生命周期。每次是独立新进程，不能重用一次输出充当三次。原公开 receipt 的 upper 16,238,752 ns 与 A-only budget 13,171,328 ns 保持不变。

已有 normal off01 的实际 16,893,473 ns 超界结果、原 qualification/raw/guard 的 SHA 引用保留在协议中。启动元数据去重改变新修订的来源，所以旧结果不合并进新的 n=3，也不能把新数值变化归因为去重或方法改善。

计划每次执行 300 秒加 20 秒收尾，三次总预留 960 秒；原累计 28,800 秒上限、实际每次账本与权限检查仍必需。本协议不发行授权。单纯成本超界或被覆盖都继续完成预登记的三次；任何权限、来源、设备、预算、原始输出、完整会计、shutdown、guard 排空或超时错误立即停止。已耗尝试不能补跑。

verify_repeatability.py 只在 CPU 读取真实 config/raw/guard/before/launch/after。它哈希原 verifier，显式编译新的独立 AST 命名空间，只更换 DELIVERY、SCOPE、实际旧 canonical 路径、独立模块名称及三份 indexed source proof 路径。原 accounting、overlap、capture、deferral、migration 和数学分析函数 AST 完全不变；不改旧模块 globals 或 callables。完整 verifier 仍调用新控制器的真实权限、guard、来源和账本检查，以及旧公开 typed receipt 重放。

三个结果分类为：全部被原上界覆盖、观察到门槛两侧变化、全部超界，或无效/不完整。三次只能说明当前修订与固定条件的这三次结果，不能证明尾部界、泛化、其它模式、因果来源或收益。即使全部覆盖，normal qualification、next mode 和 P4 收益仍为 false/None；旧 normal 失败门不被替换。

纯 CPU 检查：

```text
python -B -I -S artifacts/prefix_io_v1_server12_candidates/normal_repeatability_v1/test_repeatability_protocol.py
```

真实服务器完成一份作业、after source 检查后，可执行（需要真实 PyYAML，不能使用 -S）：

```text
.venv/bin/python -B artifacts/prefix_io_v1/server12-c5-normal-repeatability-v1-20261004/verify_repeatability.py --root /root/autodl-tmp/prefix-io-v1-handoff/project --index 0 --output artifacts/prefix_io_v1/server12-c5-normal-repeatability-v1-20261004/DIAGNOSTIC_RESULT_off01.json
```

全部三次结束后，省略 --index，将 output 指定为新的 FINAL_REPEATABILITY_DIAGNOSIS.json。所有输出使用排他创建，不覆盖旧证据。Supervisor 可调用 verify_repetition(root, diagnostic_index) 与 summarize(root)：实际证据错误抛出异常，成本超界是保留的有效诊断结果，不能被当作错误自动重试。

GPU/host 绝对时钟不映射，不从 CQE 时刻推断隔离成本或 GPU overlap。只有实际记录到的温度、时钟、功率和其它负载才能进入描述；未观测因素保留未知。读进程或设备 metadata 的外部采样不得加到逐 frame 内。此目录 CPU 测试不构造阳性 native receipt、模型输出或 GPU 结果。
