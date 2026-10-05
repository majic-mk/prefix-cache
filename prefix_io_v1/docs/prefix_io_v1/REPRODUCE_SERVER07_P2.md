# Server07 P2 复现

工作目录：/root/autodl-tmp/prefix-io-v1-handoff/project。所有运行在服务器完成。

1. 核对 experiments/prefix_io_v1/configs/permissions.yaml、批准 GPU UUID 与现有累计 ledger。不要重置预算。
2. 使用 third_party/work/py-kvcache-p2-aio（版本与 source-lock.json 相符）；保留 P1 worktree 用作基线。0003 patch 追加在公共 AIO 修复及 observer/0001、0002 后。
3. CPU 环境设置 CUDA_VISIBLE_DEVICES=""、PYTHONDONTWRITEBYTECODE=1、PYTHONPATH="$PWD/third_party/work/py-kvcache-p2-aio:$PWD/src"。精确测试 argv 见 combined-cpu-tests.json、adapter-cpu-tests.json、owner-cpu-tests.json。
4. GPU 环境沿用 artifacts/prefix_io_v1/new-server-03/native-prefix-04-launch.json 中 environment_overrides，将 PYTHONPATH 设置为 "$PWD/third_party/work/py-kvcache-p2-aio:$PWD/src:$PWD/experiments/prefix_io_v1/scripts"。
5. 模型保持离线、本地模型清单和 P1 paired 成本曲线不变。GPU 入口仅使用 run_gpu_stage.py，不直接执行 GPU 子脚本。wrapper 绑定 UUID、串行预留预算、超时与收尾。
6. 完整 GPU argv 见 server07-p2-*-launch.json 的 command 字段。再次运行时必须同时把 --label 与 --output 中的运行标签改为未使用的新标签，不覆盖原始证据。最终机制脚本是 progress-05 所对应的当前脚本，失败版本保存在本次 artifacts 下。
7. qualify_p2_native_observer.py 的 --owner-probe 是独立诊断。测量观察开销时不加该参数。off/shadow 各从独立空私有 storage 开始，同一运行不逐请求清空缓存。
8. measure_observation_cpu.py 在 GPU 性能测量结束后执行。analyze_p2_qualification.py 分析此次固定命名证据，拒绝覆盖现有输出；新数据需显式更新对应输入列表，不能混入另一次测试。
9. 每次结束检查 ledger.active_reservation=null、进程退出、GPU 内存/利用率及已接受 AIO 排空。失败和超时同样计入预算并保存。

本次主报告 SERVER07_P2_REPORT.md 列明口径限制。短调用耗时不是 cohort goodput，平均 decode 间隔不是 ITL P95；不可直接用于 P4/P5 性能结论。
