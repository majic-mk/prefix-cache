# 新服务器第三轮复现草稿：已执行命令、证据与待执行入口

> DRAFT：本文件在真实下载仍运行时编写。历史命令仅用于追溯，不应直接重跑旧 label；待执行命令不代表已执行或已通过。GPU 最终运行记录、模型完整校验与最终预算由主线程填 PENDING。工作目录统一为 /root/autodl-tmp/prefix-io-v1-handoff/project，全部实际操作在服务器。

## 权限与固定输入

权限读取 experiments/prefix_io_v1/configs/permissions.yaml，预算读取 experiments/prefix_io_v1/gpu-budget-ledger.json。当前上限 8 GPU 小时、20 GiB 模型下载；唯一批准 GPU UUID 为 GPU-8b500efe-1a50-0e8e-b21e-716807eebedf。下载器和 GPU runner 共用 gpu-budget-ledger.lock；有 active_reservation 或锁被持有时拒绝启动，不手动清空以“修复”余额。

本轮模型固定输入：

- manifest：artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json；
- manifest SHA256：9b1c79986aa1fd62eb9064ecb8b4e2a03fc235a34ac2d69c33a4fcb531a91017；
- provider / revision namespace：modelscope / modelscope_git_commit；
- revision：16c174980d8a1492910551634b4969e69cdc2444；
- 目标目录：models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444；
- 11 个文件合计 15,242,788,168 B；所有大小和 SHA256 以 manifest 为准，不能用 master、模型名或缓存目录代替固定身份。

继续使用固定作者 vLLM 本地 build04。作者原生扩展、固定源码和第二轮公共 UUID 补丁的复现见 docs/prefix_io_v1/REPRODUCE_NEW_SERVER_02.md。本轮不重新构建，也不换上游 wheel。禁止将只读官方 metadata 准备脚本当成任意可重跑的未计费下载入口。

## 本轮已经执行的 CPU 命令

以下均是已保存日志中的真实入口。运行时 CUDA 隐藏；65 项下载测试使用隔离的临时项目、假 HTTP 字节流和独立账本，禁止真实 socket 连接，不改共享真实账本。

最终下载组：

~~~bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv-prefix/bin/python -m pytest -q -ra tests/prefix_io_v1_download --junitxml=artifacts/prefix_io_v1/new-server-03/download-review/redirect-after.xml
~~~

结果：65 passed / 0 failed / 0 skipped。实际命令记录 download-review/redirect-commands.jsonl，标准输出 redirect-after.txt，JUnit redirect-after.xml。原始 39 项修复后 after.xml 是这一最终 65 项的子集，不另计。先行 39 项有 4 failed、新增跳转先行 57 项有 11 failed，均保留 before 证据，不删除失败来制造全绿历史。

模型来源与路径适配：

~~~bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/prefix_io_v1_native_prefix --junitxml=artifacts/prefix_io_v1/new-server-03/native-prefix-adaptation/cpu-tests.xml
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python artifacts/prefix_io_v1/new-server-03/native-prefix-adaptation/verify_cpu.py
~~~

结果：24 passed；脚本编译、固定作者 API 字段 AST、CLI 和真实保存 manifest 的 provider/provenance 验证通过。这不是本地完整权重校验，也没有加载模型。

cwd 隔离变更后，再次执行相同 24 项及源顺序检查：

~~~bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/prefix_io_v1_native_prefix --junitxml=artifacts/prefix_io_v1/new-server-03/native-prefix-cwd/cpu-tests.xml
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python artifacts/prefix_io_v1/new-server-03/native-prefix-cwd/verify_source.py
~~~

结果：24 passed、源编译及路径验证→chdir→vLLM import 顺序通过。不能把同一组重复执行记成 48 个独立用例。上述 65+24=89 是此草稿初版的历史快照，旧 cpu-results-summary.json 不应充当最终累计。随后显式续传组扩为 109 项（含原 65），模型来源/KV metadata/serialization 组扩为 49 项（含原 24），交付脱敏 15 项；当前唯一用例总计 109+49+15=173。最终证据分别是 resume-review/first.xml、native-prefix-kv-metadata/cpu-tests.xml、delivery-redaction/cpu-tests-final.txt。各新增阶段真实命令见相应目录 README/commands.jsonl，本次计数更正没有重跑任何测试。其他历史 CPU 结果见旧报告，不称本轮重新执行。

补丁顺序：

1. 下载器 download-review/0001-bounded-response-download.patch；
2. 其上 download-review/0002-evidenced-cdn-redirect.patch；
3. native smoke 原第二轮新文件补丁；
4. 其上 native-prefix-adaptation/0001-provider-provenance.patch；
5. 再上 native-prefix-cwd/0002-native-prefix-run-directory.patch。

本轮补丁路径均从 artifacts/prefix_io_v1/new-server-03/ 起。已部署工作树已含这些修改，不重复 apply。download-review/verification.json、redirect-verification.json 和 native-prefix-cwd/verification.json 保存正反向应用/空白或顺序检查结果；它们用于恢复变化边界，不用于覆盖其他代理或用户的新变更。

## 已执行的平台诊断

作者 LiburingRing(2) 真实尝试返回 EPERM，证据 io-uring-current.json。独立标准库 syscall 探针只实际执行一次：

~~~bash
/root/miniconda3/bin/python -I -S artifacts/prefix_io_v1/new-server-03/platform-probe/io_uring_setup_probe.py
~~~

结果 exit 2、io_uring_setup(entries=2,flags=0) 返回 -1 / EPERM。没有 mmap、enter、SQE 或数据 I/O、GPU、网络、sysctl 或权限修改。命令与环境：platform-probe/commands.jsonl；结果：setup-probe-once.txt、setup-probe.json；脚本与 ABI 依据哈希：probe-manifest.json。将来平台条件改变时可复制单一脚本到获准环境运行一次，不反复轮询同一环境，也不尝试更高权限/替代 flags 绕过。setup 成功以后仍须真实原生 ring/SSD 验收。

## 实际下载命令及失败留痕

来源 metadata 的逐请求 URL、时间、状态、payload 见 modelscope-source/metadata-requests.jsonl、requests-started.jsonl、metadata-accounting.json；固定身份与六类来源证据见 SOURCE_REVIEW.md。下载第一轮原始 argv 保存在 download-launch.json，命令为：

~~~bash
timeout --signal=TERM --kill-after=15s 1800s /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python experiments/prefix_io_v1/scripts/download_pinned_model.py --manifest artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json --model-dir models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444 --label qwen-ms-01
~~~

该 label 已使用，不能直接重跑。证据 qwen-download-01.log/.exit 与 downloads/qwen-ms-01/result.json：两个小文件下载成功，随后首权重的未批准 redirect 被失败关闭；权重响应 body=0，仍保守扣该权重大小+1，即 3,945,441,441 B。旧 charge 必须保留，不能因修复后成功而返还。

随后只读审查 4 次官方固定 URL 首跳 GET（不读取 body）和 4 次对应精确 Location HEAD，全部 body=0。证据 redirect-review/requests-started.jsonl、requests-completed.jsonl、redirect-review-public.json、content-address-check.json。唯一主机 cdn-lfs-cn-1.modelscope.cn、四 HEAD 长度匹配，CDN 内容地址与 manifest SHA256 一致。没有禁用 TLS。临时签名仅用于当时的受控 HEAD，不可复制历史 auth_key 作为新下载入口。

精确 route 修复及 65 项 CPU 通过后，主线程发起一次显式重试。真实 argv 在 download-retry-launch.json：

~~~bash
timeout --signal=TERM --kill-after=15s 1800s /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python experiments/prefix_io_v1/scripts/download_pinned_model.py --manifest artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json --model-dir models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444 --label qwen-ms-02
~~~

此命令已经启动，不能在其运行时另开下载/GPU或重用同 label。截止草稿前两分片已 size/hash 通过、第三分片进行中。最终 status、exit、起止时间、全部文件数与哈希：**PENDING**。以 qwen-download-02.log、qwen-download-02.exit（作业结束后才存在）及 downloads/qwen-ms-02/result.json 为准，不能仅看日志里两个成功文件就声明全量完成。

下载器全量结束前会重验所有文件并仅在成功时写 VERIFIED_LOCAL_FILES。失败 partial 不作为可用模型，不做隐式 Range resume；显式新尝试需要新 label、有效剩余额度和无不确定预留，不能盲重试。本次成功的复用文件也会重新 size/hash 校验。

## GPU 独立 P1 smoke：待执行/待填的条件和入口

以下是未来入口，不是本草稿已执行记录。必须先完成 11 文件校验、本地模型检查，读取实际共享账本确认下载进程结束、无 active reservation、GPU 授权/剩余时间有效。任何实际 GPU 检查均必须通过 run_gpu_stage.py；容量探测本身也不绕过 runner。主线程应保存实际新 label 的完整 argv、输出与退出码。

CPU 本地 header 审计入口（需完整本地模型；本草稿未执行）：

~~~bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/prefix_io_v1/scripts/audit_local_model_headers.py --model-dir models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444 --output artifacts/prefix_io_v1/new-server-03/local-model-header-audit.json
~~~

它核对 safetensors header、索引所有权、BF16、shape/offset/文件跨度，仅读 header、不加载 tensor payload，不代替下载器的完整 SHA256。实际执行命令和结果：**PENDING**。目标输出不可覆盖既有文件。

原生 GPU smoke 模板（选择尚不存在的新 label；以下 native-prefix-01 是否已使用须主线程在执行时核对）：

~~~bash
.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label native-prefix-01 --seconds 900 -- .venv/bin/python experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py --model-dir models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444 --model-plan artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json --output-dir experiments/prefix_io_v1/runs/native-prefix-01/details
~~~

实际 GPU 容量检查 argv/label、smoke argv/label、退出码及账本事件：**PENDING**。原生模型构造由作者 vllm.LLM 执行，不使用模型 provider SDK。runner 绑定批准 UUID、设置离线环境、预留时长、清理并确认子会话。脚本冻结配置/来源/环境，使用项目内 runtime cache/temp 和本次 details 为 cwd。正常单卡 uni 路径的原生 EngineCore multiprocessing session 审查见 native-integration-review/P1_NATIVE_INTEGRATION_REVIEW.md；该源码结论不泛化为所有外部 launcher。原 kvcache-experiments server wrapper 含 start_new_session=True，不能未经适配直接塞入 runner。

必须收集而非猜测：

- runs/<实际label>/ 的 runner 实际命令、stdout/stderr、退出码、GPU 秒数和会话清理结果；
- details/frozen-config.json：作者 SHA、provider/revision、manifest SHA256、本地文件验证、有效离线/路径设置与固定 ENGINE/SAMPLING；
- details/cold.json、repeat.json：cached-token 0→112、128 输入/16 输出以及 exact token IDs；
- details/smoke-result.json：真正 PASSED_GPU_PREFIX_PATH_ONLY 或明确失败、shutdown 状态；
- 作者 profiler 输出如 results_engine_core_0.json、merge.json 及 registry，若实际产生则保存并标记来源，不能伪造不存在的文件；
- 实际 KV tensor 分配若脚本未测得，保持 null/unknown；64 MiB 只是配置值。

即使通过也只验证 native GPU Prefix；SSD/staging/生产 KV 字节一致性/end-to-end 标志保持 false，不能提升为完整 P1。若 GPU 失败按真实 error/traceback 记录并结算，不修改预期 cached-token 数或输出标准让测试通过。

## 最终结算与封版方法

读取最终 gpu-budget-ledger.json 和实际 result，保留带时间的只读快照。报告 GPU 秒数（含失败）与剩余小时，并将下载量分开：

- P=model_payload_received_bytes：已经结束并结算的实际 HTTP response payload；文件运行中该数不是实时网络读数；
- C=model_download_bytes：预算累计扣费，包括首权重 body=0 但保守扣 3,945,441,441 B；
- R=active_reservation.reserved_bytes（无预留时为 0）：已占用但未结算，不等于全部已消费；
- 可新分配额度=21,474,836,480−C−R；完整本地模型 S=15,242,788,168 B 另列，不能与 P/C 混为一谈。

最终填写位置：DELIVERY_DRAFT.md 的 PENDING 表格，以及主线程维护的正式报告/锁/状态。GPU 基线 45.273048002272844 秒；本轮最终新增和累计：**PENDING**。最终 P、C、R 和全部下载状态：**PENDING**。

64 MiB metadata 预留按 115,928 B 结算，未用部分已释放；64 KiB header 审查预留 payload/charge 均 0，已释放；9,680 B response headers 另记，不称 payload。初始 metadata 请求是在有界审计期间归集结算，不得追溯宣称每一请求前都有预留。完整说明在 accounting-review.md。

主线程远程执行命令总快照路径、截止 UTC 和 SHA256：**PENDING**。专项 commands.jsonl 已存在且应保留；若总快照未保存，最终交付明确该缺项，不虚构路径或写成“所有命令已保存”。

## 下一允许阶段

继续 P1。io_uring EPERM 未解除时，不运行依赖真实 SSD ring 的缓存集成，也不激活 P2–P7 真实验收。平台条件变化后先最小 setup，再作者原生 LiburingRing/SSD 实际完成队列；复用现有 native handler/Plan/engine 的同 session、有界、小规模编排，冻结实际 staging/SSD/成本准入参数，验证生产 KV、命中层次、实际文件读取字节和资源释放。入口和现有默认值风险已列 native-integration-review/P1_NATIVE_INTEGRATION_REVIEW.md，尚不等于完整集成驱动已就绪。

成功 foreground 原生 file_read 事件在完整 trace、冻结 io_size 和成功父任务前提下可按 count×io_size 推导该路径实际完成字节；原生 preload.file_read 直接含 raw nbytes。必须真实开启/保存作者 profiler，cache hit/共享等待者不能冒充额外 SSD 读取，失败早结算任务的 trace 不能直接当完整 I/O 总计。没有这些实际证据则保留 BLOCKED，CPU/mock 不替代验收。
