# Server07 P3 辅助存储验证：命令与复核

本次结果对应服务器 westd:38819 的项目 `/root/autodl-tmp/prefix-io-v1-handoff/project`，证据前缀 `artifacts/prefix_io_v1/server07-p3-05`。本文件描述已执行操作；不会自动启动新的 GPU 作业。

实际 23 次 GPU 封装命令和 CPU 命令逐项记录在 `commands.json` 与可阅读的 `executed-commands.txt`。后者是记录文件，不能整文件作为脚本重跑，因为旧结果、标签和失败证据必须保留。GPU 的环境覆盖在 `launch-environment.json`；CPU runner 用禁用 GPU 的守卫，测试结果为 `final-cpu.xml`（143/0/0）。

实际 GPU 命令始终通过：

```text
.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label <unique-label> --seconds <240-or-300> -- .venv/bin/python <recorded-driver> <recorded-arguments>
```

封装器使用原 permissions.yaml、同一累计账本及整进程会话清理。driver 不能绕过预算封装直接作为新 GPU 实验运行。新实验要使用新标签和输出目录；原文件不覆盖、不删除。若 GPU UUID/配置改变，须先核对现有授权与配置资格，不能把本轮候选表直接视为新环境许可。

CPU/离线复核可以在原服务器原项目读取已有证据，不需要新 GPU 运行：

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:experiments/prefix_io_v1/scripts .venv/bin/python - <<'PY'
import json
from pathlib import Path
from validate_heldout_costs import costs, reference
from analyze_cached_references import analyze as cached
from analyze_long_validation import analyze as service
root = Path.cwd()
b = Path("artifacts/prefix_io_v1/server07-p3-05")
cases = [
    (cached, "aux-reference-plan.json", "aux-reference-result.json"),
    (reference, "long-reference-plan.json", "long-reference-result.json"),
    (costs, "long-cost-plan.json", "long-cost-result.json"),
    (service, "long-replay-plan-effective.json", "long-service-result.json"),
]
for fn, plan, recorded in cases:
    actual = fn(root, b / plan)
    expected = json.loads((b / recorded).read_text())
    assert actual == expected, recorded
    print(recorded, actual["status"])
PY
```

完整 GPU 重现需要原模型和 bulk KV 数据；交付包不包含这些大文件。它包含原始运行细节、trace、冻结输入、源代码、补丁、版本锁、源哈希与逐文件清单。辅助目录数据在压缩包中映射为 `auxiliary/`；MANIFEST.json 保存每个条目的原服务器绝对路径。下载后的证据可在本地查阅及哈希校验；原绝对路径与源缓存哈希校验使分析器不能直接在 Windows 目录下原样运行。

复核顺序：包 SHA256 → MANIFEST 每项字节/SHA256 → source-lock → CPU JUnit → 30 项缓存参考 → 独立成本门槛 → 12 个 128-token 输出/1536 token 时刻 → I/O settle/来源文件/预算。失败启动 long-replay-01、旧 2K 成本失败和原候选表均保留；不纳入成功 GPU 模型运行数。

本轮曲线只取得同预算下 16256-token 独立验证许可，不能据此启动策略收益评估。下一步仍是 P3 的预冻结并发与正常混合负载、配置适用性验证及强简单基线准备。
