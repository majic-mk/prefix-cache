# 本轮复核入口
服务器项目：/root/autodl-tmp/prefix-io-v1-handoff/project

本轮具体命令均已执行并保留在 artifacts/prefix_io_v1/server07-p3-07/*-command.json 和 *-launch.json；环境在 execution-environment.json，原始冻结参数在 run-plan.json。不要直接重放 GPU 命令：旧标签为只增不改，且新混合轮已被存储额度阻塞。

CPU 离线复核：
```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/third_party/work/py-kvcache-p2-aio:$PWD/src:$PWD/experiments/prefix_io_v1/scripts" .venv/bin/python experiments/prefix_io_v1/scripts/analyze_flush_diagnostic.py --plan artifacts/prefix_io_v1/server07-p3-07/run-plan.json --out <新的分析文件>
```

源码与证据可按 source-lock.json / delivery-manifest.json 逐项校验 SHA-256。本地包只用于读证据；GPU/CPU 开发与测试在服务器完成。完整项目还依赖之前交付中的作者工作树、构建环境、模型和共享原始 SSD，不是包含权重的可独立运行镜像。
