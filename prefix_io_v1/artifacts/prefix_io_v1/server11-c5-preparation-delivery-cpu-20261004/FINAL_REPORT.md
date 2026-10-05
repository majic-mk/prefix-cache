# C5 运行入口 CPU 准备交付

已在当前服务器完成启动接线、独立审查、历史资源结果复核和双侧备份。本轮 **CPU 接线通过，GPU 运行 0 次**；没有完成 P4 真机或性能验证，没有论文级提升结论。

## 实际改动

只在新的 `server11-c5-runtime-preparation-cpu-20261004` 目录增量复制并接入原启动器、控制器和验证器，新增 `notification_runtime_adapter.py` 与 CPU 验证工具。冻结 C5 reactor/collector、C4、原 runtime_v4/native_v6 和历史结果均未修改。

修正 capture 与 bridge 的 owner run 身份，保留 frontend/native request 身份及原 128 帧检查；在 on 安装已有 C5 通知等待和最多 16 条值观测，off/shadow 不安装新等待或 Queue 包装。Queue 实例、FIFO、get 参数、timeout、返回值和异常保持。修正通知先于 get、detach 后待原 drain 再关闭观测的时序，不写资源完成或释放状态。三个 GPU 入口均在读取配置、预算和加载模型前无条件阻断；旧 C4/v6 凭据不能用于本轮 C5。

没有重写缓存引擎或模型执行器，没有扩大候选、干扰额度或工作量，没有下载、删除已有数据、修改系统/驱动/配额、租用 GPU 或推送代码。

## 服务器实际执行

项目绝对路径：`/root/autodl-tmp/prefix-io-v1-handoff/project`。所有本轮 Python 命令使用该项目 `.venv/bin/python -B -I -S`，环境 `CUDA_VISIBLE_DEVICES=''`。

| 命令 | 真实结果 |
|---|---|
| `freeze_cpu_preparation.py --candidate-root <C5> --runtime-v4-root <runtime_v4>` | 锁定 80 个引用；SHA `8a8dbf5226cefabfa5100f24aabe9bc4bf48010996d389a96c516c6288164071` |
| `run_cpu_preparation.py --candidate-root <C5> --author-root <project> --output-dir SERVER_CPU_RESULT_01` | 22/22；失败、错误、跳过均为 0；80 个引用前后核验一致 |
| `run_review_cpu.py ... --output-dir SERVER_REVIEW_01 --location server_cpu` | 独立 22/22；130 个实际源码引用前后相同；torch/vLLM 导入为空 |
| 新 run/control/verify 三个入口分别执行 | 各退出 2，符合阻断预期；`gpu_started=false` |
| `audit_resource_results.py --benchmark-root <旧基准> --output-dir SERVER_REPLAY` | 只读核验旧 132 份结果、42 对；没有新增正式计分 |
| `audit_and_pack_preparation.py --root <project>` | 525 个旧引用保持原 SHA；旧 GPU 账本原字节相同；封存成功 |

精确路径、全部参数、环境、退出码和日志位于备份中的各组 `*_COMMAND.json`、`*_RESULT.json`、stdout/stderr。详细服务器报告为 `server_replay/server11-c5-preparation-delivery-cpu-20261004/CPU_PREPARATION_REPORT.md`；总审计为同目录 `SERVER_FINAL_AUDIT.json`。测试组分别报告，不将本地/服务器重复执行相加为额外覆盖。GPU event、native 状态和 128 帧输入是标明的 CPU 合成元数据，未执行实际 collector.install、模型或物理 I/O。

## 证据封存

`C5_PREPARATION_CPU_EVIDENCE.tar.gz` 只含四个新 CPU 目录的封存快照：67 份数据文件、801,515 字节原始内容；压缩大小 143,098 字节。

归档 SHA-256：`514dc36f9ef9104cd23f88dd306aae640792e536182f1083448def5e35fb149c`；嵌入清单 SHA-256：`df155c53d48f1e4bee35387cad20a287f026dd0b38f099c6880974c5cc1e7a22`。

本机实际执行 `verify_preparation_backup.py --archive ... --receipt ... --output-dir server_replay --result LOCAL_BACKUP_VERIFICATION.json`，先核完整 SHA、成员路径和内容，再写新目录；67 份文件全部一致。验证器自身 9/9 检查通过，覆盖恶意路径、碰撞、篡改、链接、超限和不合规 GPU/benchmark 声明。

`LOCAL_SOURCE_CLOSURE_VERIFICATION.json` 进一步核验服务器 receipt 的可信 SHA 和全部 80 个源引用：16 个准备引用、64 个前次候选引用；13 份本地准备源码与服务器完全相同。没有重新运行已通过的语义测试或原正式计分。下载后的本地核验回执和本报告属于封存后追加，未纳入原 67 文件快照，未修改归档或冻结 manifest。

## 结论与下一允许阶段

旧合成 CPU 基准的 40 ms 主场景观察到两线程 CPU 中位数下降约 73.6%。这是减少轮询的潜力证据，**不包含本轮新增 Queue 观测和完整启动器的开销**，不能作为本轮性能资格。旧 80/100 ms 对照的无限流样本不足，完整 CPU 资格依然未通过；全部限流样本保留，协议与结果不变。120 ms 总 wall 包括 producer 固定持有时间，不等于 deadline 晚 20 ms。

当前服务器仍无 NVIDIA 设备，CPU 配额为 `50000 100000`（0.5 核）、内存 2 GiB。本轮 GPU 0 次、正式基准新增 0 次；累计 GPU 用量 22,380.561257688794 秒，原 8 小时预算剩余约 6,419.44 秒（1.78 小时），无活动预留变化。数据盘剩余约 56.9 GiB，本轮归档只有约 140 KiB，不需要为本轮扩容。

下一允许 CPU 阶段是构建绑定新 C5 完整入口和观测的严格成本凭据工厂并冻结；随后须补齐足够 CPU 配额下的完整入口成本资格。真实原生校准和 off→shadow→on GPU 验证须匹配新源码、作业、明确授权及原累计预算，维持原模型、128 tokens、单文件负载与既有阈值。当前不用租 GPU，不能把 CPU 接线通过宣称为 GPU 提升或 P4 完成。
