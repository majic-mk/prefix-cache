# 新服务器迁移与有限标定入口交付

时间：2026-10-04。服务器：connect.westd.seetacloud.com:24828。
项目目录：/root/autodl-tmp/prefix-io-v1-handoff/project。
本轮阶段：C5 新物理 GPU 域的 CPU 准备和现场来源冻结。
状态：READY_FOR_EXPLICIT_FRESH_SINGLE6_GPU_GRANT；不是 P4/P5 性能通过。

## 实际现场

RTX 5090，GPU-a67d2f38-8e1b-29c3-7e5b-375e1410f8ac，
驱动 580.95.05，空闲显存 32110 MiB，没有计算进程。
容器 CPU 配额 25 核，内存上限 90 GiB，最终可用内存约 89.54 GiB。
数据盘剩余约 50.42 GiB，本轮不需要扩盘。
首次 SSH 固定主机密钥 md5：34afaa28d40dc70c5179bbe25ee2d6c5。
凭据仅存在认证 worker 内存中，没有写入本报告或工程。

新 GPU UUID 和驱动与旧卡不同。旧 GPU03 的输出、生命周期和成本证据
仍保留为旧硬件域的历史实验，不能把 UUID 改写后用于这张新卡。
旧 normal 的 100 项 CPU 测试通过结论按实际源码一致保留，未重跑全套。

## 实际修改位置

新目录：/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004。

- gpu_entry_binding.py、control_native_cost_job.py：新作业路径、现场驱动核对、
  原累计账本和原 guard、原 G 共享源码、新 SDK 元数据来源和 CPU 冻结。
- native_conditional_cost.py、p4_single_file_receipt.py：新作业与公开来源绑定；
  原数值公式、A-only 预算、留出与未知条件拒绝规则保留。
- prepare_and_verify_native_cost.py：复制到新目录，和 G 原文件逐字节相同。
- run_native_cost_experiment.py、site_sdk_binding.py：显式选择实际已有的
  580 SDK CPU 凭证，继续调用原 SDK helper；不修改原 common 模块。
- 针对性 CPU tests/runners、README、继承清单、只读审计脚本及证据。

原 G/common 的 64 个 Python 文件全部保持原字节；作者文件、缓存引擎、
模型执行、精确 Prefix、准入、staging、preload、异步拷贝与 shutdown
均未重写。共有 64 个顶层 callable AST 在当前比较中完全相同。
execute_window 仅 SDK 加载/调用发生接线变化，将这两句恢复到原调用后，
整个实际执行函数 AST 与 G 一致；没有声称整函数原始 AST 未变化。
本机 14 个原有已修改 tracked 文件的字节也保持不变。

## 实际命令与结果

均通过已认证 SSH worker 在服务器执行，CPU 命令设置
CUDA_VISIBLE_DEVICES=''、PYTHONDONTWRITEBYTECODE=1。
资源读取仅调用 nvidia-smi，不加载模型、CUDA 库或 GPU framework。
完整 argv 和原始 stdout/stderr 在下列 *_COMMAND.json/*_STDOUT.log/
*_STDERR.log/*_RESULT.json 中，不以推测补充输出。

| 实际命令记录 | 结果 |
|---|---|
| 02_CLONE_FULL_SOURCE_COMMAND.json | 原 normal 来源 4779/4779 完整字节核验通过，13.765 秒 |
| 03_SDK_REAL_ASSETS_COMMAND.json | 原 helper 完整核验真实 580 SDK 凭据和外部驱动，通过 |
| 04_NEW_SDK_ADAPTER_REAL_ASSETS_COMMAND.json | 新 adapter 对真实 8 项直接来源及完整 SDK 树复核，通过 |
| 05_SITE_SDK_TESTS_COMMAND.json | 29/29 CPU tests 通过，0 skip |
| 06_NATIVE_DOMAIN_TESTS_COMMAND.json | 12/12 CPU tests 通过，0 skip，禁止导入尝试为空 |
| 07_BINDING_CPU_TESTS_COMMAND.json | 57/57 CPU tests 通过，0 skip，前后 93 项 CPU 来源一致 |
| 08_NATIVE_ENTRY_CPU_TESTS_COMMAND.json | 23/23 CPU tests 通过，0 skip，前后 93 项 CPU 来源一致 |
| 09_UNBOUND_REVIEW_TEMPLATE_COMMAND.json | 生成明确 UNBOUND 的审查模板，没有授权/配置/作业 |
| 10_LIVE_CONTEXT_AND_SOURCE_FREEZE_COMMAND.json | 4750 项新域闭包完整核验并冻结；现场资源可用 |
| 12_SOURCE_AST_AND_COMMON_AUDIT_V2_COMMAND.json | 64 common 文件字节与 64 callable AST 审计通过 |
| 14_REAL_UNBOUND_ENTRY_REJECTION_V2_COMMAND.json | 真实 --execute CLI 在缺少新配置时返回 GPU_ENTRY_REJECTED/exit2；GPU 导入尝试为空，未创建 run |
| 15_FINAL_SMALL_SOURCE_AND_LIVE_READINESS_COMMAND.json | 2777 项冻结源码/元数据重验一致；GPU 空闲；账本不变 |

最终服务器针对性测试合计 121/121 通过。
这些是 CPU 接线、拒绝边界和来源测试，不是新卡模型或性能结果。

主要可直接执行命令（在项目根目录）：

```bash
.venv/bin/python -B -I -S artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004/test_site_sdk_binding.py -v
.venv/bin/python -B -I -S artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004/tests_native_domain.py
.venv/bin/python -B -I -S artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004/control_native_cost_job.py prepare
.venv/bin/python -B -I -S artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004/control_native_cost_job.py context
.venv/bin/python -B -I -S artifacts/prefix_io_v1/server12-c5-migration-audit-20261004/audit_site_ancestry_v2.py --root /root/autodl-tmp/prefix-io-v1-handoff/project --output <new-contained-output-path>
```

prepare/context 为追加证据操作，已有输出不会覆盖；不要把上述列表作为全套重跑脚本。
绑定和 native 的 source-lock runner 完整参数见 07/08_COMMAND.json。
原公开 SDK helper 的真实编译证据保持原日期和来源；本轮编译次数 0，
.so 加载次数 0。没有把历史编译描述成当前新编译。

保留了开发和审计失败：
1. 子代理本机 domain 测试首次错误把一个 AST 子集计为 32，实际应为 13；
   改正测试计数后通过，生产数值源码没有因此变化。
2. 本轮 11 审计脚本误用 SOURCE_REVISION.json 文件名；真正冻结文件是
   REVISION_SOURCE_LOCK.json。失败原始日志保留，追加 v2 审计后通过。
3. 本轮 13 拒绝验证 wrapper 错误预期外抛 FileNotFoundError；真实 CLI
   已正确捕获并返回 SystemExit(2)。失败 wrapper 日志保留，追加 v2
   按真实 CLI 返回校验后通过。没有改运行入口来满足测试。
冻结目录中的旧审计 helper 保留原字节供追溯，实际使用 audit 目录的 v2。
这些不是 GPU 实验失败，本轮没有 GPU 模型实验。
另一次初始只读探测发现非交互 SSH PATH 没有裸 python/python3；
之后实际命令固定使用现成 .venv/bin/python，没有安装或修改系统。

## 来源、预算和真实 GPU 状态

迁移父锁 SHA-256：
cbdaf335d5c260cfd4936ee981448fc1c1fd430121ef642c4447272221adc433。
迁移 4779 行实际摘要：
4237dceaa305b5d25eeeec5930ddd6b909e648daafa762152e80947648fe240e。

新 REVISION_SOURCE_LOCK.json：4750 项，1100072 bytes，
SHA-256：557ed20beae31b3d844cfa1db58a4538c8569d7264c9523cc350743e6e612fa5。
LIVE_BINDING_CONTEXT.json：1618 bytes，
SHA-256：c03e7368b720039560c513b3eaba2328df761b347fd0f6998672af975ba01bde。
CPU_SOURCE_LOCK.json：93 项，
SHA-256：f46da142c5ce033e43a4e6995c1890fc874361fb61a0a01374b0c8ef4d2fb55f。

本轮真实 GPU 作业 0，真实模型加载 0，GPU 实验账本新增耗时 0。
原累计 GPU 时间 23129.437302808743 秒；
8 小时预算剩余 5670.562697191257 秒（约 1 小时 34 分 31 秒），
active_reservation=null。
账本 SHA-256 始终为
4fd592aaecf9371ae050c65035e7cf037b56d876653aac2e49fa65f5fc5c3da8。
云实例 GPU 模式的租赁墙钟费用由平台计费，不等于此实验账本；本轮
CPU 期间没有消耗实验账本，不代表平台停止计费。

## 下一允许阶段与当前阻塞

CPU 准备全部完成。当前仅缺这张新卡、这份源码和该作业的一次明确 GPU 授权。
04_CODEX_EXECUTION.md 要求：
“GPU实验只在 permissions.yaml 中已有明确授权、目标环境可用且预算不为空时运行。”
旧卡的授权/成本绑定不能转移到新 UUID 和新驱动，当前目录没有
HUMAN_GPU_GRANT/EFFECTIVE_GPU_PERMISSION/GPU_ENTRY_BINDING/NATIVE_COST_CONFIG。

可审查的下一轮是一次 single6 有限原生标定：
server12-c5-native-common-cost-gpu01，最多一次 guard、六个 fresh 原模型
进程，顺序 A0/B0/B1/A1/A2/B2，每窗完整生成 128 tokens，共 768。
两组训练对照、一组独立留出，保持 BF16 eager/Triton、batch1、
active decode1、context144、SSD 单文件917504 bytes等预登记条件。
只用已有 Qwen2.5-7B 与24份真实生产 KV输入，完全离线。
执行限时1200秒、收尾20秒，总预算预留1220秒（20分20秒），
失败即停止、同一授权不重试、不用留出重拟合。
PRIMARY预留2GiB、保持8GiB可用；不下载、不删数据、
不改系统/驱动/安装包/作者文件、不扩大研究方法或GPU预算。

通过真实新标定和生命周期审核后，再根据实际成本决定 normal
off/shadow/on 资格验证的绑定与授权。尚未验证新卡收益，
尚不能承诺性能提升或论文有效性；标定上限不是取得研究结论的时间保证。

