# P3 CPU 启动额度复核

本轮开发工作树是服务器上的 `third_party/work/py-kvcache-p3-quota-cpu`，运行中的旧 P2 工作树未切换。

完整实际命令见 `artifacts/prefix_io_v1/server07-p3-08/final-cpu-command.json`；完整环境见同目录 `execution-environment.json`；阶段解释见 `docs/prefix_io_v1/SERVER07_P3_START_BUDGET_CPU_REPORT.md`。再次运行测试必须指定新的 basetemp / junitxml / guard 路径，避免 pytest 清理或覆盖已有证据。

重建：从 version-lock.json 的作者 SHA 建 worktree，应用 inherited-p2.patch，恢复交付中的 py_kvcache/linux_aio.py，再应用 start-budget-only.patch；项目 src 与 tests 随包提供。依赖运行环境仍来自原项目 .venv，本包不含模型、虚拟环境或 CUDA 工具包。

默认 start_budget=None。仅 CPU 验证的 StartBudgetConfig 不能直接塞进 vLLM 配置。不能根据本包宣称 fixed/pressure GPU 可用或有性能提升。

交付包含源码、补丁、CPU 原始证据。receipt/local-verification 记录文件数量与哈希，不把本地解包校验算作服务器 CPU 实验。


本轮 CPU scratch 曾使主盘余量低于 8 GiB，已完整校验迁移到 /root/prefix-io-v1-validation/cpu-evidence/server07-p3-08/。后续运行应使用 run_start_budget_cpu_qualification.py 的容量预检与辅助盘输出路径；本轮只验证了它的 dry-run。原始 649 项回归的实际命令仍留作历史证据，不直接复用其 basetemp。
