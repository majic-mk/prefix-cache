# Server09 G3：CPU 就绪，三作业已获授权，等待有卡模式

当前结论：无卡阶段的准备已完成。用户在上一条三作业方案后直接回复“继续”，已按同一限定范围记录授权；无需重复授权。当前服务器仍无 GPU 设备，不能执行 GPU 验证。本轮新增 GPU 作业、GPU 操作均为 0，方法性能提升尚未验证。

## 实际改动

新增 `run_g3_calibration_pilot_v2.py` 与对应 CPU 测试。旧入口的共享缓存目录与冻结计划不同；v2 的执行、前序结果绑定、存储发布和读取核验均从已校验的 `plan.STORAGE_RELATIVE` 取得路径：
`experiments/prefix_io_v1/runs/server09-g3-calibration-01-private-storage`。

删除 v2 中 populate 的父目录创建：目标的父目录是已存在的 runs；原采集器负责创建 storage 本身。cold 不创建外部缓存；populate 原采集器创建并写入；paired 在前一作业正常 shutdown、OS 会话排空、完整源核验和存储发布全部通过后复用目录。没有新缓存引擎或模型执行器，没有修改作者采集器、runtime/result/plan 模块、安装包、系统或驱动，没有删除数据。

旧源码锁 `263b3eea9560de916860fb3340e9421ad76a49d3a15d4008857773e342fc152e` 与旧独审保留为历史。前次独审遗漏了 CLI 常量差异，本轮独审明确纠正，不再将旧入口作为可执行入口。派生执行锁仅在原 4,063 项引用后追加 v2 与测试，共 4,065 项，原引用的内容、顺序及哈希全部保持：
`7e1599a885f2bed8ad93e30057e34660e7aa0aa7d0b5888e5e53fe01feb4276d`，904,979 bytes。

## 实际命令和结果

服务器工作目录：`/root/autodl-tmp/prefix-io-v1-handoff/project`。服务器上的所有本轮 CPU 命令均设置 `CUDA_VISIBLE_DEVICES=''`。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002/test_run_g3_calibration_pilot_v2.py -v
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002/verify_g3_no_gpu_readiness.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --source-lock artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002/gpu-source-lock-storage-v2.json
```

服务器 11/11 测试通过；本地同一 11 项测试也通过。实际服务器完整读取并核验 4,065 项源码/资产引用及 1,934 份私有 SDK 工具文件（217,111,529 bytes），CPU 就绪核验退出 0、耗时约 96.67 秒。授权记录与 scope 的真实字节绑定、原 baseline 清单和预算/磁盘一致性均通过。独立有限审查通过，0 阻断项。此次修复未触及原严格 GPU 驱动门禁。

原 CPU verifier 的通用状态名包含 `REQUIRES_GPU_POWER_AND_NEW_SCOPE`；它自身不创建授权。最新包装回执另行记录实际 scope 已存在，当前状态是 `CPU_READY_GPU_AUTHORIZED_WAITING_FOR_DEVICE`，不能据旧通用状态名重复索取授权。

## 实际 GPU 与资源情况

检查时 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm` 全部不存在，`/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05` 为 0 字节占位文件。GPU UUID、空闲显存和驱动运行能力尚不能在此模式确认，GPU 阶段阻塞。没有运行 nvidia-smi、NVML、模型或 GPU 探针来替代这些事实；无卡核验不授予 GPU 运行资格。

累计 8 小时账本未变：已用 17,206.3436 秒，剩余 11,593.6564 秒（约 3.22 小时）；229 条事件，active_reservation=null，尚无 G3 作业事件。账本 SHA-256：
`7dd465ef868fb6cc4fbacddd8cc58045cc47b11aa0768765325613b454c6b744`。
磁盘可用约 11.91 GiB，满足本方案 3 GiB 计划预留及 8 GiB 保留线；每次启动前仍须重新测量。这些是预备计划，尚未实际预约或消耗 G3 GPU 时间。

## 授权与下一允许阶段

`HUMAN_AUTHORIZATION_RECORD_G3_CONTINUE_20261002.json` 保存原问题、原话“继续”、直接文本回复来源及观察时间。未伪造表单 ID，未将派生源码哈希写成用户原话；`G3_STORAGE_BINDING_REPAIR_DERIVATION.json` 明确区分原批准提案锁与修正执行接线的派生锁。固定模型、上下文、作业名称、请求数、顺序和预算不变。

用户在 AutoDL 恢复同一实例的有卡模式后回复“已开卡”。届时先按原严格门禁核验真实驱动、GPU UUID、显存、完整源码、账本和空间；全部一致后运行 cold → populate → paired。最多 3 次，每次执行 300 秒加 20 秒收尾，共计划 960 秒；18 个原请求、各输出 1 token。任一次失败即停止、不重试、不执行后续作业。原策略关闭，保留作者异步 I/O；不下载、不修改系统/驱动、不删除数据。

可审查的三条启动命令已写入 `G3_AUTHORIZED_LAUNCH_COMMANDS.json`。执行入口为 v2，显式绑定派生锁和新 scope。该轮验证原生 SSD/staging 路径与原 TTFT；不是 P4 方法效果或 P5 性能提升实验。原 strict 驱动检查在无卡模式明确不通过，不能绕过；若开卡后硬件身份改变则先报告实际差异。

## 证据位置

本目录与服务器同名交付目录镜像保存以下证据：

- `ACTUAL_G3_STORAGE_V2_SERVER_CPU_TESTS.json` / `.log`：服务器实际 11 项测试命令、输出和退出码。
- `ACTUAL_G3_STORAGE_V2_FULL_CPU_READINESS.json` / `.log`：4,065 引用与工具全量核验、真实无卡事实。
- `ACTUAL_G3_CONTINUE_SCOPE_AND_NO_GPU_CHECK.json`：小文件字节、scope/预算/目录检查。
- `G3_STORAGE_BINDING_V2.diff`：完整入口差异。
- `INDEPENDENT_G3_STORAGE_V2_SCOPE_REVIEW.json`：两位审查者实际有限审查及历史结论纠正。
- `ACTUAL_G3_CONTINUE_V2_CPU_CHECK_COMMAND.json`、`ACTUAL_G3_STORAGE_V2_FULL_CPU_READINESS_COMMAND.json`：本轮完整实际远程执行命令。
- `G3_CONTINUE_V2_DELIVERY_MANIFEST.json` 与交付校验回执：新增交付字节核验。此前 97 文件清单保持原样。
