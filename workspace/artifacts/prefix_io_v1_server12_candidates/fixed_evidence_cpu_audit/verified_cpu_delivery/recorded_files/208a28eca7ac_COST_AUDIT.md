这是实际已保存成本记录的独立 CPU 字段与字节审计，不发行原生资格、授权或新收据。旧冻结源码、校准上界及预算全部只读。脚本仅使用 Python 标准库；新增 GPU、模型导入、共享库显式加载与 RPC 均为 0。

本机最终执行为：

```powershell
& 'C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe' -B -I 'artifacts/prefix_io_v1_server12_candidates/fixed_evidence_cpu_audit/test_cost_evidence_cpu.py'
& 'C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe' -B -I 'artifacts/prefix_io_v1_server12_candidates/fixed_evidence_cpu_audit/analyze_cost_evidence_cpu.py' --project-root . --output 'artifacts/prefix_io_v1_server12_candidates/fixed_evidence_cpu_audit/COST_EVIDENCE_CPU_AUDIT_FINAL_V2.json'
```

最终 19/19 CPU 边界与拒绝测试通过，无跳过。真实证据审计通过，输出 `COST_EVIDENCE_CPU_AUDIT_FINAL_V2.json`，478,278 bytes，SHA-256 `41e5617f53fa6cb114cdf9c32ec86b50890e87520963acbd1520785dd4b00631`。测试中的合成元数据只验证拒绝边界，不被称为实际原生成功。开发时发现并修复的审核器问题是父校准记录显式追加的字段、汇总中有符号的超界量，以及本机归档没有打包 Event 库源码；对应旧输出原样保留，不能作为最终审计结果或 GPU 失败记录。

服务器执行使用相同源文件：

```bash
.venv/bin/python -B -I artifacts/prefix_io_v1/server12-c5-fixed-evidence-cpu-audit-20261004/test_cost_evidence_cpu.py
.venv/bin/python -B -I artifacts/prefix_io_v1/server12-c5-fixed-evidence-cpu-audit-20261004/analyze_cost_evidence_cpu.py --project-root /root/autodl-tmp/prefix-io-v1-handoff/project --output artifacts/prefix_io_v1/server12-c5-fixed-evidence-cpu-audit-20261004/COST_EVIDENCE_CPU_AUDIT_SERVER.json
```

服务器默认读取原 A 目录的 GPU、正常 off 和固定三次交付清单。本机默认读取相应已核验的平铺映射；也可显式提供 `--local-manifest`、`--calibration-manifest` 和 `--prior-manifest`。数据路径来自这些清单，内嵌 raw/config/guard 引用还须与清单逐字节引用一致。脚本拒绝重复 JSON 键、路径穿越、符号链接、错误字节数/SHA、步骤序号或 CUDA 见证漂移。输出以 `xb` 创建，已存在时拒绝覆盖。

审计核对了 6 个原校准/heldout 窗口、3 个新固定窗口和旧 off 窗口，共 **1,280 frames 与 1,280 witnesses**，以及 67 项实际读取来源的前后字节数/SHA。每窗口均保存 128 个输出。原始帧的 `gpu_elapsed_ns` 实际为空，真实 GPU 持续时间来自同一 `native_step_ordinal` 的 `torch.cuda.Event.elapsed_time` 见证；脚本逐项检查记录/查询的主机包络，未做 GPU 与主机绝对时钟映射。完整输出、prepared load、前端步骤、独立保存的 capture/frontend/window 与内嵌记录一致。offset 16 的 ordinal 为 146、context 为 144，相邻 offset 15/17 的边界一致。offset 0 为 prefill，offset 1 是原声明的排除 warmup；其他步骤的 context 不同，不能当成同一成本格的新增拟合样本。

原 actor、collector 和公开收据模块的实际源码字节已核验。本机 tar 没有包含 `.venv/.../torch/cuda/streams.py`，该项明确标记 `payload_bytes_reverified_here: false`；已保存计划引用与所有见证的 Event 来源摘要相符，不等于在本机重新核验整个服务器运行闭包。服务器若该 exact ROOT 内 `.py` 存在，脚本实际读取并严格核验 10,672 bytes、SHA `3b34ed08b67cf2ec411e1a483b31835bbff8652c915a08f2a0e4c140e342dd9c`，不会导入 Torch。保存的 guard 成功/自然排空字段亦被读取核对；本脚本不重新发行 guard/账本资格。

| 记录 | selected 完整 GPU 步（ms） | 相对冻结 U=16.238752 ms |
|---|---:|---|
| 原校准 A0 / B0 | 13.171328 / 16.000256 | 保留原角色 |
| 原校准 A1 / B1 | 13.054688 / 16.238752 | 保留原角色 |
| 原 heldout A2 / B2 | 12.964064 / 16.097504 | 不参与拟合 |
| 新 fixed off01 | 16.266945 | 超界 0.028193 |
| 新 fixed off02 | 15.458848 | 覆盖 0.779904 |
| 新 fixed off03 | 19.185921 | 超界 2.947169 |
| 旧正常 off01 | 16.893473 | 超界 0.654721，单独保留 |

预算 **13.171328 ms 是两个校准 A 的最大值**，并非独立声明的服务 SLO。baseline 13.113008 ms、增量 3.006496 ms、正残差 0.119248 ms 合成冻结 U 16.238752 ms。它与预算均按完整 GPU 步比较；冻结 U 高于预算 3.067424 ms。即使某次 selected 被 U 覆盖，U≤预算仍为假，普通准入仍应 defer。新三条 selected 也都超过该 A-only 预算。本审核不重拟合、不改变 heldout 角色、不扩大不确定性或预算。

实际条件差异已存在但原因未确定：校准首 token 为 18100/19100/20100、种子 1829/1830/1831；正常及新固定记录首 token 为 28100、种子 2829。公开成本签名绑定 model/GPU/KV layout/kernel/active decode/batch/prefill/context/physical bytes，未绑定提示词内容、种子、workload hash 或温度、GPU 时钟和 SSD/page-cache 状态。selected 的 prepared load、非存储引擎选项和已保存输出 token ID 摘要却相同；三次新记录又使用同一提示词、种子和 source revision。这些事实不支持把超界唯一归因于提示词，也不能推断硬件温度、SSD 缓存、未观测负载或仍在 JIT。

当前下一步为 CPU 证据交叉核验、原准入/推进分支说明与封存。**单纯再次运行相同配置不会改变冻结 U>预算的准入关系**。新的前瞻校准可能得到不同数值，但没有必然降低成本或通过准入的保证，不能事后用这些失败记录调整门槛；若将来确需重新标定，应另有事前固定协议和独立验证记录。保持正常 shadow/on 阶段阻塞，不将当前有限诊断或启动耗时描述当作 P4 完成、策略性能收益或论文有效提升。

稳定源码摘要：`analyze_cost_evidence_cpu.py` SHA-256 `16c2f94e087fd7122eb9bfe88d22b8c5d462150be1e64ceb832e15d297c007a0`；`test_cost_evidence_cpu.py` SHA-256 `3c937255dcc1167c7ee0ab1c06e25398d1e0d4f8594c45e81790b939cf9a3fad`。
