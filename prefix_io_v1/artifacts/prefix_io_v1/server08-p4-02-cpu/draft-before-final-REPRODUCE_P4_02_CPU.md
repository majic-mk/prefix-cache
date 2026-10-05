# P4-02 CPU 复验

服务器项目：/root/autodl-tmp/prefix-io-v1-handoff/project。无需 GPU。当前权限收据仍为 CPU_ONLY。

本轮实际统一执行命令记录在 EXECUTED_COMMANDS.json；需要复跑时换一个未用过的 --name，不覆盖旧证据：

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='third_party/work/prefix-io-p4-02-cpu/src:experiments/prefix_io_v1/scripts' .venv/bin/python experiments/prefix_io_v1/scripts/run_p4_02_cpu_qualification.py --name cpu-replay-02
```

runner 验证 CPU scope、源锁与无活动 GPU reservation，限制独立进程运行时间/新 scratch 预留，禁止 CUDA 初始化，HF/Transformers offline。统一 JUnit 去重；历史 12 个旧 P3 fixture 不计入，作者真实 GPU e2e 不执行。16 个 skip 的真实原因保留在 XML；已有单独 agent 重跑不累加到统一数量。

当前 freeze 脚本只用于最初冻结，不应覆盖现存 source lock。旧证据无需重新生成。

明天先刷新 CPU 来源/磁盘/预算及实际授权 scope 收据，再通过新的外层 --launch 门禁进入原唯一 GPU guard。现在 --launch 必须退出 78，不能把 preview/旧 P3 CLI 当成新 P4 资格。新 Python source overlay 使用锁定旧 binary fallback；没有重建镜像，ABI 仍需真实授权 GPU 验证。

先后顺序和阻塞项见 GPU_NEXT_DAY_RUNBOOK.md。原始记录离线工具仅能验证交付的 raw 数据，不能替代可信 GPU collector；CPU fixtures 不得写进真实实验结果。
