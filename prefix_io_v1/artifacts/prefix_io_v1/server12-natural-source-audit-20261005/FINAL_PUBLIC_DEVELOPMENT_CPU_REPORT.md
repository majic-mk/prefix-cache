# 真实输入与开发入口 CPU 交付

本轮已完成真实输入准备、实际服务器 V14 冻结和 V4 生产入口 CPU 预检。下一允许阶段是开发 U/off 功能与成本采集。当前服务器无 NVIDIA 设备节点，CPU 0.5 核、内存 2 GiB，GPU 阶段如实阻塞；本轮新增 GPU 运行 0 次、模型输出 0 tokens。

## 实际改动

1. 读取原交接包及真实 permissions.yaml 后，确认公开来源读取和 CPU 开发已获授权。此前“开发前必须有人提供外部 deadline/authority”的条件来自后加的执行合同；原计划允许在开发集确定经验预算，并要求正式评估前冻结策略与服务目标。
2. 增加独立 development U/off、I/shadow 入口和 descriptor schema。该入口不生成外部 SLO、authority、deadline 或普通 I 权限。正式 effect 路径继续保留独立 SLO、真实开发 reserve、完整 native/guard/peer 校验。
3. 共同修复：从已冻结 .py 字节编译加载私有 helper，避免合法字节码缓存覆盖已核验源码。策略分派另列。原作者 drive、模型执行器、原 guard、private cost issuer、整个 finally 和 peer/capture/reserve 验证保持。
4. 对固定官方 SQuAD dev-v1.1 Git blob 获取、真实 CPU 分词和原协议 freeze_trace 重放。选择规则在读取数据及任何新 GPU 结果前固定：前三篇文章各首段，保留全部 40 条问答，按文章分 calibration/development/evaluation=30/5/5。输入保留完整原文；两处边界空白按原作者 .strip() 语义处理。没有注入人工共享前缀、删请求、按长度挑样本、截断或伪造到达轨迹。
5. 新配置只更新运行标识与各分区隔离缓存目录；共同运行域保持 f240c3738ac18ac304c3330612e03f272079fe434dc98ae61f65b94c99aa4d1c。精确 Prefix Cache、成本准入、共享 staging、preload、原复制合并及异步流水线保留。旧源与现有数据未删除。

SQuAD 是真实公开文档问答数据，不是生产请求/会话/到达轨迹。到达计划来自原作者规则，128-token 固定输出是控制负载设置，不能当作正常问答质量验证。来源：[官方 SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)，数据采用 CC BY-SA 4.0；Git commit `eee5fdbf62f8613a7812b03419e6b29617b74fd1`、blob `e9a3f913ad1468ebe105b891334ca7b0bc0e2510`，4,854,279 B，SHA-256 `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`。

## 实际测试和命令

下列均在服务器项目根 `/root/autodl-tmp/prefix-io-v1-handoff/project` 以 `CUDA_VISIBLE_DEVICES=''` 执行，未导入模型、Torch、vLLM 或初始化 CUDA。

| CPU 测试套件 | 最终实际结果 |
| --- | --- |
| 公开源适配 v2 | 20/20 |
| V13 追加冻结工具 | 7/7 |
| 开发输入适配原型 | 18/18 |
| 真实分词与家族闭合工具 | 16/16 |
| V4 生产接线/拒绝/源码缓存测试 | 26/26 |

逐套列出，不将本地与服务器重复运行相加。实际 40 条分词得到 7,079 个输入 tokens，长度 138–190；三个文档首个完整 16-token 缓存块跨分区不重复。calibration/evaluation 只准备输入，未用其结果调参或运行 GPU。开发分区请求 30–34 全部进入原入口；没有换成 qualification 标签或合成 token fixture。

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S artifacts/prefix_io_v1/server12-natural-source-audit-20261005/development_runtime_wiring/test_development_runtime_wiring_cpu.py -v

CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S artifacts/prefix_io_v1/server12-natural-source-audit-20261005/prepare_actual_development_cpu.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --action prepare

CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S artifacts/prefix_io_v1/server12-natural-source-audit-20261005/source_freeze/append_public_source_lock_v14.py --project-root /root/autodl-tmp/prefix-io-v1-handoff/project --addition-manifest artifacts/prefix_io_v1/server12-natural-source-audit-20261005/ACTUAL_V14_DEVELOPMENT_ADDITIONS_01.json --source-lock-output artifacts/prefix_io_v1/server12-natural-source-audit-20261005/PRERENT_SOURCE_LOCK_V14.json --proof-output artifacts/prefix_io_v1/server12-natural-source-audit-20261005/PRERENT_SOURCE_PROOF_V14.json

CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S artifacts/prefix_io_v1/server12-natural-source-audit-20261005/prepare_actual_development_cpu.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --action config

CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/strong_trace_runner_v4.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --config artifacts/prefix_io_v1/server12-natural-source-audit-20261005/ACTUAL_PUBLIC_DEVELOPMENT_UOFF_RUN_CONFIG_01.json --preflight
```

五条命令均 exit 0。V14 实际核验 5,014 个文件、16,093,402,743 B，完整继承 V13 的 4,990 项资产；耗时 92.715 秒。最终生产预检 exit 0，耗时 8.138 秒。快速预检的 `full_source_verified=false` 表示该命令没有重复整盘哈希；独立 V14 完整字节冻结已通过并绑定到它，不能把两者混为一项。

最终源锁 SHA-256：`427c796365d31a5ff7a725f2bfa52641a74a9c609b67fda918fa23834af343b4`。
U/off 配置 SHA-256：`923ae77ab5bd5fa47a8ead8eb2d475a99c5a6250f7e2e5fc14b4a6ea00bf7551`。
RUN_CONFIG 在其源锁外生成以避免自引用；原 guard 与入口仍逐次绑定实际字节和全部依赖。

曾失败的公开源传输、旧适配器空白检查和测试路径回放均保留。初次命令拼接在启动子进程前失败、随后一次监督器超时无法确定子进程 exit，均如实标记；另一 raw 下载明确 exit 1。最终传输校验同一冻结 Git blob 的完整 SHA-1/SHA-256 后通过。没有把这些失败计为 GPU 失败或伪造成功日志。完整命令、exit、stdout/stderr 在同目录的实际 *_COMMAND.json / *_RESULT.json / *_STDOUT.log / *_STDERR.log；详见 ACTUAL_CPU_FAILURE_RECOVERY_01.json。

## 当前能证明的内容和限制

CPU 证明完整真实输入、分词/文档家族、原解析器、模型资产、隔离命名空间、真实成本来源和原生产入口已接通；不能证明新自然请求已在 GPU 生成，不能证明策略有提升。

此前实际 GPU 校准仅覆盖 context527、batch1 等严格条件下的一项 SSD-read 成本：完整估计 16.633120 ms。本轮自然请求的 context 条件不同，禁止外推覆盖。短文本 40 条是功能先导输入，不能单凭它们形成论文性能收益结论。当前 I 预览额度仍为 null，I/shadow 如实 UNBOUND；要先取得真实 U 观测，再选择并冻结开发工程阈值。该阈值仍不构成服务 SLO 或普通 I 安全资格。

本轮 GPU 运行 0。原累计 8 小时预算不变，已用 26,222.628 秒，剩余 2,577.372 秒（约 42.96 分钟），无 active reservation。账本 SHA-256 `774847ccdb7ec4c1785c19a87e718b2a91e15fba44248eb949dab559e08fdd6b` 未变。CPU 没消耗 GPU 账本，不代表云实例租用费用为零。驱动、系统、已安装包、模型权重及已有缓存均未修改；tracked Git status 为空。

## 下一允许阶段

将当前实例切至 GPU 模式后，先确认可见 GPU UUID、实际原 guard 和私有 SDK 环境。已存在持续 GPU 授权，无需重新询问。若 UUID 变化，既有成本凭据不能冒充新设备凭据，需先按剩余预算完成对应环境重绑定/验证。

若仍为已验证的 GPU-b2de2c25-cdc7-a350-267f-56e7763a287f，可执行已生成的 U/off guard 命令（ACTUAL_PUBLIC_UOFF_CONFIG_PREPARATION_01.json）。上限 300 秒执行加 20 秒收尾，5 个开发请求各完整生成 128 tokens，共 640 tokens 计划量；PRIMARY 预留 128 MiB，磁盘至少保留 8 GiB。计划 token 数尚未发生，不能计为实测。到达时刻 78.36–87.84 秒按完整作者计划保留，启动后存在等待时间。

完整 U/off 输出、原 shutdown、native 尾部和 OS 会话排空后，闭合真实成本和观测；然后准备有真实工程预览额度的 I/shadow。正式策略 on 与效果实验仍要求对应条件成本覆盖、真实资源释放与 reserve，以及事前冻结的服务目标/评估合同。

当前数据盘实际剩余约 48.5 GiB，满足本轮预留，无需扩容。服务器证据目录为 `artifacts/prefix_io_v1/server12-natural-source-audit-20261005/`，交付包 PUBLIC_DEVELOPMENT_CPU_SOURCE_EVIDENCE.zip 含完整公开源、失败和成功证据、新代码及冻结引用；模型/SDK/私有缓存主体保留在服务器，包不是整机备份。最终压缩包验证在 PUBLIC_DEVELOPMENT_CPU_ARCHIVE_VERIFICATION.json 及本地对应验证文件。
