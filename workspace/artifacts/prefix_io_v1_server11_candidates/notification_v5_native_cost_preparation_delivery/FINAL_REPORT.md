# C5 原生采集代码 CPU 准备交付（2026-10-04）

本轮在指定 AutoDL 服务器 connect.westc.seetacloud.com:26909 执行，项目根目录为 `/root/autodl-tmp/prefix-io-v1-handoff/project`。当前仍为无卡实例：`/dev/nvidia*` 为空，cgroup 配额为 0.5 核 CPU、2 GiB 内存。工作遵守服务器交接包 04 文档和权限合同，没有租赁、付费、下载、系统/驱动修改、推送或旧实验数据删除。

实际完成的是 C5 原生采集与成本凭据代码的 CPU 准备和服务器源锁验证。服务器实现测试 30/30、独立审查 35/35、协议检查 17/17，共 82 项，失败、错误、跳过均为 0。没有 GPU、模型、真实 CUDA Event、原生 I/O 或真实六窗口实验运行，没有签发有效原生成本凭据。所有 native/full-runtime/P4 性能资格仍为 false。

新增四个服务器目录均位于项目 `artifacts/prefix_io_v1/`：`server11-c5-native-cost-preparation-cpu-20261004`、`server11-c5-native-cost-preparation-review-cpu-20261004`、`server11-c5-native-cost-preparation-protocol-cpu-20261004`、`server11-c5-native-cost-preparation-delivery-cpu-20261004`。本机同名功能副本位于 `artifacts/prefix_io_v1_server11_candidates/notification_v5_native_cost_preparation_*`。冻结前和封存时审计 797 个历史证据引用，旧源码与实验依据保留。

实际修改如下：

- 在新 `common_candidate` 内复制冻结 C5 的 64 个 Python 文件，只替换 canonical `prefix_io_control.p4_single_file_receipt`。C5 reactor、完整帧 collector、原策略和桥接字节保持原样；没有第二缓存引擎、队列或模型执行器。原策略仍使用同一模块中同一 `ExactSingleFileReceipt` 精确类型。
- 增量复制原生六窗口采集、重序列化和原成本验证器，将源码位置及身份绑定到 C5。五个原始数值、完整帧及释放因果函数的 AST 保持一致。固定 AB/BA/AB、两个校准对加一个独立验证对、129 prompt / 128 cached / 128 output tokens、offset 16、单次 917,504 字节 SSD read、最多八个已接受父任务，没有调宽阈值或重新选样。
- 增加明确标记为 `synthetic_cpu_contract` 的六份模拟原始文件重构及反例。CPU 重序列化必须显式传 `synthetic_cpu=True`、新 CPU 标记、C5 来源及 UUID None。内层保留原形状校验使用的历史 vocabulary，但外层输入与输出始终是模拟；通过模拟检查不能成为真实 GPU 凭据、成本或额度。
- 所有配置、计划、guard、原生执行、scope、launch 和公开成本签发入口先阻断。UUID、物理存储模板、GPU 权限文件、GPU 源锁与祖先授权均为 None。旧 GPU 授权与成本没有移植。五个实际 CLI 均输出 `GPU_BLOCKED_C5_NATIVE_COST_CODE_PREPARATION_ONLY`，退出码 2、stderr 为空，未访问 GPU 或预留预算。

公共六窗口使用 `bridge=None`，因此仅能覆盖公共路径成本；它不能证明 on-only 通知等待和 Queue 观测开销已测量。common capture.run_id 为外部 request ID，journal 为作业 LABEL；完整 on 运行的身份要求另行保留。此前完成的正常运行入口与严格来源绑定 CPU 阶段仍封存独立，不与本轮模拟记录合并成原生证据。

服务器源锁 `SOURCE_LOCK_NATIVE_PREPARATION_CPU.json` 为 159 个引用，SHA-256：`47a445ba5fc1fae5f9fba5f32871f5af771e66083753654122d7e96bc6f64e2a`。其中基线审计 JSON 也已绑定，防止从冻结到封存期间改变历史边界。独立审查另核验锁文件本身，共 160 个来源前后相同。C5 reactor SHA 为 `a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47`，collector SHA 为 `bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf`。

实际服务器执行命令以下列文件保存的完整 argv 为准：delivery 的 `SERVER_SOURCE_FREEZE_COMMAND.json`；candidate 的 `SERVER_CPU_FACTORY_COMMAND.json` 及五份 `SERVER_*_BLOCK_COMMAND.json`；review 的 `SERVER_CPU_REVIEW_COMMAND.json`；protocol 的 `SERVER_CPU_PROTOCOL_COMMAND.json`；delivery 的 `SERVER_FINAL_SEAL_COMMAND.json`。均使用项目 `.venv/bin/python -B -I -S`，并设置 `CUDA_VISIBLE_DEVICES=''`。

对应主要入口为：

```text
freeze_native_preparation.py --root <project>
run_cpu_native_preparation.py --project-root <project> --source-lock <delivery>/SOURCE_LOCK_NATIVE_PREPARATION_CPU.json --output-dir <candidate>/SERVER_CPU_01 --location server_cpu
run_review_cpu_v2.py --project-root <project> --candidate-root <frozen-C5> --native-root <native-v6> --stage-root <candidate> --original-estimator <original-file> --source-lock <delivery>/SOURCE_LOCK_NATIVE_PREPARATION_CPU.json --source-lock-sha256 47a445ba5fc1fae5f9fba5f32871f5af771e66083753654122d7e96bc6f64e2a --output-dir <review>/SERVER_REVIEW_01 --location server_cpu
run_cpu_common_protocol.py --project-root <project> --preparation-root <candidate> --candidate-root <frozen-C5> --native-root <native-v6> --source-lock <delivery>/SOURCE_LOCK_NATIVE_PREPARATION_CPU.json --output-dir <protocol>/SERVER_PROTOCOL_01
seal_native_preparation.py --root <project>
```

测试原始日志分别为 `SERVER_CPU_01/TEST_STDOUT.log`、`SERVER_REVIEW_01/TEST_STDERR.log`、`SERVER_PROTOCOL_01/CPU_PROTOCOL_TESTS.log`，对应 JSON 保存计数、来源与资格字段。没有重跑之前冻结的 132 次 CPU 对照，也没有选择、删除或改写样本。

本地开发阶段的两轮 30-case 失败及修复记录保存于 candidate 的 `LOCAL_DEVELOPMENT_RECORD.json`，不是 GPU 失败/成功。上传工具一次压缩命令过长，被远端 bash 拒绝；已限制单包 85,000 字符，重传时只核验既有相同文件并补充新文件，记录于 `UPLOAD_DEVELOPMENT_RECORD.json`。独立审查增加已解除的旧定位字段断言后采用新 v2 文件，已上传旧文件留存，实际服务器运行 v2，计数不重复。

本轮真实 GPU 运行数 0、模型进程 0、正式性能实验 0、GPU 新增消耗 0 秒。预算账本 SHA-256 保持 `bef6d78e43058eaad811ab5ecb08937356c911784f9000e76eea9de0276c0b72`，累计 `22380.561257688794` 秒，8 小时预算剩余 `6419.438742311206` 秒（约 1.78 小时），没有活动预留。

研究结论保持原证据：上一版 C4 的整体 GPU pilot 约慢 7.28%；C5 的 CPU 合成场景降低了 reactor CPU 消耗，但受 0.5 核节流影响，预设完整资格未通过。这些结果不能证明本轮新增观测的成本、C5 GPU 性能提升、论文要求或整个系统最终可行。新阶段只证明来源、接口、反例拒绝和 CPU 重构检查通过。

下一允许阶段须先具备可见且获授权的 GPU，以及不受当前节流影响的 CPU 资源。先做包含实际正常入口与观测代码的 CPU 资格检查，再以现场 GPU UUID 和新源码/作业/预算绑定准备可审查的 GPU 修订；当前 CPU 版本本身始终阻断 GPU。真实 C5 六窗口校准通过后签发真实公共成本凭据，再逐项进行 off→shadow→on 原生命周期与通知/观测成本验证，通过后才可做 P4 真机策略比较。策略没有实际延迟时须报告 NOT_EXERCISED，不能改预算或负载制造效果。

有卡和授权只能解决运行条件，不能保证结果为正。当前应保留无卡模式，避免为继续 CPU 封存支付 GPU 费用。
