# C5 运行入口 CPU 准备交付（2026-10-04）

本轮结论：CPU 接线与独立审查通过；未运行 GPU，未完成 P4 真机或性能验证。

实际改动限于新的准备目录：复制并接入原启动器、控制器和验证器，增加有界值观测和 CPU 反例；冻结 C5 reactor/collector、C4、原 runtime_v4/native_v6 与旧实验未修改。没有重做缓存引擎或模型执行器，没有扩展候选、干扰额度或研究任务。

## 修复内容与语义

- 将 capture 的 owner run ID 与 bridge 对齐，保持 frontend request ID 和原 native_request_id、128 帧检查。
- 只在 on 接入已有 C5 通知等待；off/shadow 不安装等待和 Queue.get 包装。Queue 实例、FIFO、原 get 参数、timeout、返回值和异常保持。记录上限为 16 条值观测、8 条错误，持有弱引用；未写资源完成或释放状态。
- 匹配 wake 可以在 get 之前到达并清 registration；该竞态只有真实匹配 token 能通过，旧 wake、一般消息或 deferral 均不足以证明通知已运行。
- detach 发布唤醒后保留原 drain 流程，待其返回才关闭和导出观测。异常路径也在原 drain/shutdown 后关闭，未伪造 arm 清理。
- 新 GPU run/control/verify 三个入口在解析配置、触碰预算或导入模型前无条件阻断。旧 C4/v6 成本凭据不能给新 C5 启动器资格。

## 真实服务器执行

全部在当前 server11 项目执行，Python 3.12.3，`CUDA_VISIBLE_DEVICES=''`，使用 `.venv/bin/python -B -I -S`。

- `freeze_cpu_preparation.py --candidate-root <C5> --runtime-v4-root <runtime_v4>`：冻结 80 个文件，锁 SHA-256 为 `8a8dbf5226cefabfa5100f24aabe9bc4bf48010996d389a96c516c6288164071`。
- `run_cpu_preparation.py --candidate-root <C5> --author-root <project> --output-dir SERVER_CPU_RESULT_01`：22/22，失败/错误/跳过均为 0，前后核验 80 个冻结文件。
- `run_review_cpu.py --preparation-root <new-runtime> --preparation-lock-sha256 <lock> --candidate-root <C5> --native-root <native-v6> --race-review-root <C5-review> --author-root <project> --output-dir SERVER_REVIEW_01 --location server_cpu`：独立 22/22；前后实际源码相同。
- 新 run、control、verify 脚本分别运行：退出码均为预期 2，返回 GPU_BLOCKED_NO_C5_RECEIPT_AND_CPU_ENVIRONMENT_QUALIFICATION，未启动 GPU。
- `audit_resource_results.py --benchmark-root <old-C5-benchmark> --output-dir SERVER_REPLAY`：只读验证旧 132 份结果、42 个计分对；没有重跑正式计分。

精确绝对路径、参数、环境、退出码、耗时、stdout/stderr 均保存于各目录 `*_COMMAND.json`、`*_RESULT.json`、`*_STDOUT.log`、`*_STDERR.log`。两组测试分别报告；本地和服务器重复执行不相加为额外覆盖。event/native/128 帧输入是明确的 CPU 合成元数据，未执行真实 collector.install、模型、CUDA 或物理 I/O。

## 证据与边界

运行结果在 runtime 的 `SERVER_CPU_RESULT_01/CPU_RESULT.json`、review 的 `SERVER_REVIEW_01/TEST_RESULT.json`、resource 的 `SERVER_REPLAY/RESOURCE_AUDIT.json`。封存审计在本目录 `SERVER_FINAL_AUDIT.json`，备份 receipt 在 `SERVER_BACKUP_RECEIPT.json`。

审计逐字节核验 519 个旧 Python 源引用及 6 个权限/协议/历史结果锚点；旧 GPU 账本和跟踪代码不变。冻结 manifest 中服务器结果为 null 的字段是冻结时状态，保留原值，实际运行记录由独立结果引用。归档只封存四个本轮目录，不复制模型、私有编译缓存或旧 GPU 数据，不删除已有数据。

## 当前结论与下一允许阶段

旧基准在 40 ms 合成场景测得 worker+producer CPU 中位数从 33.081 ms 到 8.747 ms（约 73.6%），但完整 CPU 资格因 80/100 ms 对照受半核配额限流而失败；132 个样本全部保留。120 ms 总 wall 包含 producer 固定持有时间，不等于 deadline 晚 20 ms。新 Queue 观测/完整启动器未计分，旧 73.6% 不能转作本轮入口的性能资格，更不能证明 GPU、TTFT、吞吐或论文收益。

本轮 GPU 0 次，正式计分新增 0 次；原累计 GPU 用量 22380.561257688794 秒，原 8 小时预算剩余约 6419.44 秒（1.78 小时），没有预留变化。实例仍无 NVIDIA 设备，CPU 配额 `50000 100000`（0.5 核）、内存 2 GiB；仍不具备本轮 GPU 实验条件。

下一允许 CPU 工作是构建绑定新 C5 完整入口与观测的严格成本凭据工厂并冻结；后续需要足够 CPU 配额下的完整入口成本资格。真实原生校准及 off→shadow→on GPU 验证须另有新代码、新作业和限定预算的明确授权。保持现有模型、128 tokens、单文件任务和阈值，不自动租 GPU、不扩大研究范围。现阶段不能宣称 P4 完成或已有论文级提升。
