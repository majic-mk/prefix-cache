# 新服务器第三轮交付草稿：固定官方模型来源与有界下载，P1 继续

> DRAFT：当前证据读取时间为 2026-09-26 15:09:59 UTC。GPU 实际结果、最终下载结算与阶段状态必须由主线程读取本轮最终产物后填写下方 PENDING；本草稿不替代 canonical docs、execution_state、dependency-lock 或共享账本，也不将正在进行的作业写成成功。

执行服务器：connect.westc.seetacloud.com:24801。项目根目录：/root/autodl-tmp/prefix-io-v1-handoff/project。开发、测试、下载和证据均在该服务器执行。本轮继续固定 py-kvcache 与作者配套 vLLM 的增量路线，模型仍为 Qwen/Qwen2.5-7B-Instruct、BF16、无量化。

## 本轮实际改动

1. 新增有界模型下载入口 experiments/prefix_io_v1/scripts/download_pinned_model.py。它验证固定 revision、精确文件 URL/大小/SHA256、项目内真实路径、磁盘空间及有限权限；与 GPU runner 使用同一 flock，拒绝未结算 active_reservation，确保下载与 GPU 作业不并行。每个文件请求前将 bytes+1 的探测上限写入共享账本并 fsync；只发布完整且哈希匹配的文件。失败保留 partial，按完整预留保守收费；无隐式重试、无 Range 断点续传、无覆盖旧失败证据。
2. 修复 urllib 默认重定向处理可能先读取跳转 body 的问题。首次版本遇跳转失败关闭；实际确认官方精确 CDN 后，仅允许 ModelScope 官方主站→cdn-lfs-cn-1.modelscope.cn、该 CDN→自身，最多 3 跳。每跳不读取旧响应 body，验证 TLS、精确主机、内容地址 SHA256 和 namespace/repository/revision/filename/tag；最终仍验证下载文件大小与哈希。不接受通配符、任意 CDN、旧的临时签名或 HF→该 CDN。缺 Content-Length 时允许按 expected+1 硬上限流式读取；若提供长度则必须精确匹配。
3. 调整 native_gpu_prefix_smoke.py 的官方来源校验，使同一个作者原生 LLM 接收已验证的 ModelScope 本地 HF 格式文件。记录 provider 及 revision 命名空间，核验六类本地来源证据的大小/哈希；未引入 ModelScope SDK 或其他模型执行器。模型与配置验证、离线环境、项目内 runtime cache/temp、唯一运行输出目录均在导入 vLLM 前完成。工作目录切到本次输出目录，避免作者 profiler 相对文件覆盖其他运行。
4. 准备纯 CPU safetensors header/index/BF16 审计入口 audit_local_model_headers.py，以及获准 GPU runner 内使用的 check_selected_gpu_capacity.py。脚本存在不等于检查已执行；两项最终实际结果均留待主线程补齐。
5. 新增可交平台的独立 io_uring_setup 诊断脚本并实际运行一次，记录 errno、进程安全状态、memlock 与 ABI；补齐作者原生 staging/SSD 入口、读取字节证据及资源释放依赖的只读审查。

本轮研究策略补丁为空。原始来源 checkout 的跟踪源码干净；原有引擎、作者原生扩展、精确 Prefix 身份、成本准入、共享 staging、预加载、复制合并和异步流水线没有被重写。公共兼容修复与研究策略继续分开；新观测/策略未接入本轮 GPU smoke，关闭新策略的原有路径仍须在完整 P1/P3 中真实验收，不能由本次 CPU 检查宣称已验收。

## 版本与来源锁定

| 对象 | 固定值及证据 |
|---|---|
| py-kvcache | 3abba7a502d553f6e7e2e58b92086487e3395d7e |
| 作者 vLLM | 817a7e3124f817cd6e549581d3e5483207a753a4；继续使用第二轮 build04 的本地原生构建 |
| kvcache-experiments / simple-profiler | 0e023a84a21246b9bbc06266fa8070397eccbdc9 / ec0d563bf68856df83c5824ac579700ec076b9e2 |
| 原生 _C.abi3.so SHA256 | e30616d12f903169493f73c28e707540be17916e89794a9214b141d6e4769b94 |
| 模型 provider / source | modelscope / https://modelscope.cn |
| 模型及 revision namespace | Qwen/Qwen2.5-7B-Instruct / modelscope_git_commit |
| 固定 revision | 16c174980d8a1492910551634b4969e69cdc2444 |
| manifest | modelscope-source/modelscope-download-plan.json |
| manifest SHA256 | 9b1c79986aa1fd62eb9064ecb8b4e2a03fc235a34ac2d69c33a4fcb531a91017 |
| 清单大小 | 11 个必要文件，15,242,788,168 B；四个原始 safetensors 分片 |
| 下载脚本最终 SHA256 | c39562e4b2a6450f355b52210dc06ebd5ae499e6693349509ce6532718e08ded |
| native smoke 当前 SHA256 | 37eebb38bfa2d79af8219172c55838e2f4ad01f6d8a6cbc0e938ae1600e9d569 |

本节相对证据路径以 artifacts/prefix_io_v1/new-server-03/ 为基准。source-preservation.json 保存源码 HEAD、跟踪状态和扩展哈希；本轮未重做第二轮构建。第二轮 build01/02 失败、build03 停止、build04 成功的记录仍保留，不能改写成第三轮重新构建。

官方身份链来自 Qwen 官方发布博客的 ModelScope 组织链接及 ModelScope 返回的 Qwen/GithubAddress 信息；固定 revision REST 文件列表、config/index/tokenizer_config 均已实读核验，全零 revision 的负对照返回 404。Git smart refs 尝试返回 421，未据此声称 Git 协议可用。未证明该 ModelScope revision 与任何 Hugging Face commit 等价；不能把它标记为 HF revision。原始模型配置未经改写。详见 modelscope-source/SOURCE_REVIEW.md 与其原始 HTML/JSON/请求账单。

## 实际执行、失败与测试结果

| 操作 | 截止草稿的真实结果 | 证据 |
|---|---|---|
| 官方来源与 metadata | 11 次请求，payload 115,928 B；没有权重 body | modelscope-source/metadata-requests.jsonl、metadata-accounting.json |
| 首次下载 qwen-ms-01 | 两个小文件成功；首权重 GET 的未知重定向被拒，权重 body=0；该权重保守扣 3,945,441,441 B，整体失败记录保留 | download-launch.json、downloads/qwen-ms-01/result.json、qwen-download-01.log/.exit |
| CDN 资格审查 | 4 个官方 GET 首跳均到精确 CDN，分别 HEAD 为 200，四长度匹配；8 请求 body=0，TLS 验证开启 | redirect-review/redirect-review-public.json、allowlist-evidence.json、content-address-check.json |
| 显式重试 qwen-ms-02 | RUNNING；前两分片已完整 size/hash 校验并发布，第三分片正在下载；尚未证明 11 文件全部可用 | download-retry-launch.json、downloads/qwen-ms-02/result.json、qwen-download-02.log |
| 下载/显式续传 CPU 测试 | 最终 109 passed / 0 failed / 0 skipped；原 65 项（含更早 39 项）+ 新续传 44 项 | resume-review/first.xml |
| 模型来源/路径/KV metadata/serialization CPU 测试 | 最终 49 unique passed；包括旧 24 项，不重复计 cwd 回归 | native-prefix-kv-metadata/cpu-tests.xml |
| 交付脱敏 CPU 测试 | 15 passed；真实交付复制尚须单独执行 | delivery-redaction/cpu-tests-final.txt、verification.json |
| 本轮独立 CPU 用例总计 | 当前 173 unique，109+49+15；均为 CPU/synthetic，非真实模型或 GPU 验收 | 以上三个最终 suite；旧 89 项摘要是早期快照，封版须更新 |
| 原作者 LiburingRing(2) | 真实 EPERM，BLOCKED | io-uring-current.json |
| 独立最小 syscall 探针 | io_uring_setup(entries=2,flags=0) 返回 -1 / EPERM，exit 2；无 mmap/enter/I/O/GPU | platform-probe/setup-probe.json |
| 本地全量模型与 header 审计 | PENDING：主线程填写实际 status、文件数/字节、tensor 数、BF16/index 结果及证据 | PENDING |
| 本轮 GPU 容量/原生模型与 Prefix smoke | PENDING：主线程填写每一实际 label、退出码、时长、冷/热 cached tokens、输出 token 比较、shutdown/session drain 及证据 | PENDING |

CPU 先行失败不是最终通过结果：下载首组 39 项修复前 35 passed / 4 failed；新增跳转阶段先行 57 项中 11 failed，最终扩大为 65 项全通过。所有 before/after 记录保留。历史 388 passed / 8 skipped、第二轮 runner 21 项、UUID 16 项、旧模型计划 26 项不是本轮最终 173 个独立用例的新执行，不重复累计。

原生 GPU smoke 固定 BF16、64 MiB 配置 KV 预算、TRITON_ATTN、128-token 合成 token 输入和 16-token 输出，预期原生计数 0→112、输出 token IDs 精确相同。其实际 KV tensor 分配字段当前实现为 unknown/null，不能用配置预算充当实测分配。即使 smoke 成功，它仍只证明作者原生 GPU Prefix 路径；外部 connector/offload 被显式关闭，不证明 py-kvcache staging、SSD、生产 KV 字节一致性或完整端到端缓存。

## 授权、真实 GPU 与模型下载会计

权限仍为指定 GPU UUID GPU-8b500efe-1a50-0e8e-b21e-716807eebedf、8 GPU 小时、20 GiB 模型下载，即 21,474,836,480 B。权限文件及来源见 permissions-before.yaml 和第二轮 authorization.json。系统/驱动/网络配置修改、租赁、付费、远程发布均未由本轮新增授权。下载器与 GPU runner 共用锁，下载进行时不并行运行 GPU。

GPU 基线为第二轮累计 6 个真实作业、5 pass/1 fail、45.273048002272844 秒。第三轮最终新增作业、累计秒数及剩余小时：**PENDING**。CPU 构建、CPU 测试、模型下载墙钟不计成 GPU 时长；失败 GPU 作业仍必须计入实际累计。

模型额度始终分别报告以下三个数，另列本地完整模型逻辑大小，不能混用：

| 会计量 | 截止 2026-09-26 15:09:59 UTC 的已读快照 | 最终封版 |
|---|---:|---|
| P：已结算实际 HTTP response payload | 7,811,955,802 B | PENDING |
| C：累计预算扣费，含失败保守收费 | 11,757,397,243 B | PENDING |
| R：正在进行的文件预留 | 3,864,726,425 B，第三分片 | PENDING |
| 可新分配额度 cap−C−R | 5,852,712,812 B | PENDING |

上述 P 在文件结束后结算，第三分片正在接收的字节尚不进入该字段；不是实时累计网络流量。R 也不是全部已消耗；不可把 C+R 当作“实际下载”。完整清单逻辑大小 S=15,242,788,168 B 仅表示目标数据集，不等于本轮 P 或 C。最终仅在 result=VERIFIED_LOCAL_FILES 并完成全量验证后，才可声明该 S 全部本地可用。

首权重失败的实际 body=0，但 C 增加 3,945,441,441 B，修复/重试后也不得退还。64 MiB metadata 预留按实际 115,928 B 结算，其未用部分已释放；metadata 初始已完成请求包含在随后有界审计中，不能追溯宣称每个初始请求之前都有预留。64 KiB 重定向头审查预留实际 payload=0、charge=0，已全部释放；另记 9,680 B response headers 仅作诊断，非 payload，也不是 TLS/TCP 物理线速流量。详见 accounting-review.md、两份 budget-settlement.json 与最终共享账本。

## 实际命令和证据位置

主要入口、已执行与待执行命令分列见 REPRODUCE_DRAFT.md。命令的实际参数/时间/退出码应以 download-launch.json、download-retry-launch.json、专项 commands.jsonl、GPU 共享账本及每次 runs/<label>/ 产物为准。主线程完整远程命令快照及截止时间：**PENDING：封版时补真实路径，未保存则明确缺项**。不可仅以一条复现示例冒充已执行记录。

所有网络失败、下载失败、测试先行失败、旧构建失败与中断证据保留；精确重定向原始记录可能包含临时签名，交付阅读优先 redirect-review-public.json，不将历史签名用作下载入口。

## 限制、阶段门禁与下一允许阶段

P0 保持完成。P1 目前仍缺真实 SSD/handler/Plan、生产 KV store/restore 及生命周期的原生端到端验收；本轮 GPU 独立 smoke 最终结果即使通过，也不改变这些缺项。io_uring_setup 仍 EPERM，是真实平台资格硬门禁。Seccomp=2 只表示存在过滤模式，不能仅据它断言唯一根因；io_uring_disabled=0 和 memlock unlimited 也不等于可用。没有修改 sysctl/seccomp/权限或改用同步 I/O 绕过。

下一允许动作仍限于 P1：

1. 等单次显式下载结束，读取完整 result、逐文件 size/hash 和共享账本，确认无 active reservation；有失败或不确定预留则先停止并核对，不盲目重试。
2. 在完整本地模型和余额成立后，使用同一预算 runner 记录指定 GPU 容量，并执行同作者原生 GPU Prefix smoke。最终行为、失败原因、actual allocation unknown 等按实填入本报告，保留所有新 label 的证据。
3. 平台提供可用 io_uring 条件后，先重跑最小 setup，再原生 LiburingRing/open/完成队列与 SSD 路径；复用作者现有 handler/Plan/engine，冻结 staging/SSD/成本准入参数，完成生产 KV、精确输出、命中层次、实际读取字节和释放依赖验收。已有入口审查不等于该受预算集成配置已实现，原作者默认 E2E 命令不得直接原样运行。

真实 P1 缺项未补齐前不进入 P2–P7 真实验收，不激活有限候选调度/干扰额度研究策略，不作 TTFT/ITL、流水线重叠或论文收益结论。CPU/mock 只能验证边界逻辑，不能代替实际 SSD、CUDA event、资源复用或模型输出证据。
