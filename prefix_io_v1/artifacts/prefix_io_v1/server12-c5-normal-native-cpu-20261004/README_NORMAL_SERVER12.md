# Server12 正常运行 CPU 准备

本目录只准备已有单文件 off → shadow → on 生命周期验证。成本上界为
16,238,752 ns，原 A-only 预算为 13,171,328 ns；原策略对这一条件返回
`defer`。这里没有调宽预算、重拟合成本、改变普通等待上限或添加研究策略。

当前域是 `server12-c5-native-common-cost-gpu01` 的真实完成记录及公开 typed
receipt。旧 G common_candidate、作者模型执行器、cache、preload、staging、
collector、bridge 和 reactor 保持来源及字节；新增正常运行的历史账本适配只
验证本轮完成记录与之后的合法追加，不赋予 GPU 权限。私有 SDK 使用已经通过
本服务器核验的 CUDA13 / 580.95.05 adapter，原 Ninja 与执行生命周期保留。

没有复制旧 CONFIG、binding、source lock、grant 或 scope。服务器 CPU 操作
可以重新冻结本轮真实来源和 NORMAL_RUNTIME_BINDING，准备 off 配置；这些
产物不授权启动模型。shadow/on 必须分别重验真实前驱结果，并取得各自明确的
GPU 授权、现场 context、permission 和原 budget guard 才能执行。

服务器上的 CPU 命令（`ROOT` 是实际项目根目录）：

```sh
.venv/bin/python -B artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004/audit_normal_site_ancestry.py --project-root "$ROOT" --baseline-dir artifacts/prefix_io_v1/server11-c5-normal-native-cpu-20261004 --candidate-dir artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004
.venv/bin/python -B artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004/run_cpu_normal_native.py --project-root "$ROOT" --source-lock "$ROOT/<fresh-CPU-source-lock>.json" --output-dir "$ROOT/<fresh-CPU-test-output>"
.venv/bin/python -B artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004/control_p4_single_file.py --root "$ROOT" freeze --calibration-lock-sha256 1c0e7ccaa24aef878ea513df6744ff9037c00f02329b95793801b8a05b3ac5c0
.venv/bin/python -B artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004/control_p4_single_file.py --root "$ROOT" prepare --mode off
```

冻结与配置文件只增不改；具体 CPU lock 和输出目录由父任务现场决定。
不得使用 `-S` 运行需要真实 permissions.yaml 的验证器。

来源遍历有一项必要的元数据适配：原标定锁中的 4,756 个文件仍逐个核验完整
字节和 SHA；这些已锁定资产 JSON 不再被递归解释为当前硬件，旧 595 驱动
描述只保留为历史来源。新发现的真实 parent/child/proof 仍完整遍历。ROOT
内绝对引用只在路径无跳转、无 symlink 且完整字节核验后转成等价相对引用，
原 JSON 不改。唯一允许的 ROOT 外引用为当前 580.95.05 的精确驱动 pin，
另外通过原 SDK helper 的完整资产核验，并在新锁中单列 CPU 审计来源。

原 100 项 CPU 检查及 7 项新增 host/project 引用检查只验证接口、来源、
时钟和拒绝边界。正常 GPU 运行、
实际 on 等待、完整控制成本和策略收益均未完成。后续有限 GPU 验证首先要求
off 路径仍符合冻结 upper，再在 shadow 中看到真实 defer 提案，最后在 on
中验证实际阻断与原 Queue 的 matching wake / original deadline timeout。
未观察到真实等待时保留 `NOT_EXERCISED`。全请求与 drain 计量必须保留；
gross process CPU 差值不成为孤立观察开销、goodput 改善或论文收益的证据。
