# Server07 P3 并发证据复核

本轮详见 SERVER07_P3_CONCURRENT_REPORT.md；源锁和命令以 artifacts/prefix_io_v1/server07-p3-06 为准。

1. 在服务器项目根使用现有 .venv 和 execution-environment.json；读取 permissions.yaml 及 GPU ledger，确认指定 GPU、累计 8 小时和辅助存储 20 GiB/空闲 8 GiB 约束。不要手动清零账本。
2. CPU 单元测试按报告命令运行；将 JUnit 写入新的审计输出，CUDA_VISIBLE_DEVICES 设为空。
3. 原生参考与成本采集使用 commands-executed.json 中的 command 数组及各 cost/reference plan。所有旧结果只读；若要再跑 GPU，先冻结新唯一标签与输出路径，不能复用旧标签。
4. verify_gate 可离线复核 c2-permit.json、d8/permit.json；d2/cost-result.json 必须继续报告 FAILED_HELDOUT_POINT_PREDICTION，不能生成许可或补填两轮重放。
5. analyze_concurrent_pilot.py 和 analyze_fixed_io_baselines.py 读取服务器原始路径；将 --out 改为新文件。离线分析已在服务器实际执行并留存命令输出。本地压缩包用于审计，不包含模型/KV 存储，不能当成独立可运行 GPU 环境。
6. 原始 GPU 结果位于 /root/prefix-io-v1-validation/runs/server07-p3-c2-*/details；process.log 与 wrapper result 在项目 experiments/prefix_io_v1/runs/ 同名目录。

全部 19 个 GPU 命令与返回值已记录；冷参考/成本在连接器关闭且模型配置相同的条件下被深度 2/8 重用，不能描述为每个配置均独立重新采集。
