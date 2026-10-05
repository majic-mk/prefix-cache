# 新服务器第三轮交付：原生模型与 GPU Prefix 通过，完整 P1 仍受阻

服务器：connect.westc.seetacloud.com:24801。项目：/root/autodl-tmp/prefix-io-v1-handoff/project。所有开发、下载、测试在服务器完成，本地仅保存交付副本。P0 完成；本轮通过 P1 原生模型、GPU Prefix 和实际 KV tensor backing 检查。原生 io_uring 仍 EPERM，完整 P1 未验收，P2–P7 门禁关闭。

## 实际结果

native-prefix-04 退出 0。固定作者 vLLM、Qwen2.5-7B-Instruct、单 RTX 5090，权重和实际 KV 均为未量化 BF16。同一引擎顺序执行相同 128-token 输入，两次各生成 16 token：原生 cached tokens **0 → 112**，输出 token IDs 完全一致；没有逐请求 reset 或放宽容差。

通过作者现成 worker RPC，只读检查 28 层、28 个唯一 CUDA BF16 storage，实际 backing **66,977,792 B**，配置预算 **67,108,864 B**，73 blocks／1,168 tokens。该值不是 allocator reserved，也不是物理释放 witness。原生 shutdown 完成，runner 会话排空，最后指定 UUID 的 NVML 检查无计算进程。

这是 N 原生 GPU-only 短合成 token smoke，外部 KV connector 关闭。不证明真实 staging／SSD、同一生产 KV 字节往返、Plan／handler 全链调用、释放协议、mandatory 排空、服务流 ITL、吞吐或策略收益。首次形状仍出现 Triton JIT，不使用这些启动／请求耗时作性能比较。

## 实际修改

1. 新增固定官方清单下载器和显式失败分片 Range 续传入口。共享预算锁，先持久预留，校验来源、revision、大小与 SHA，完整校验后才发布；原失败费用和 partial 保留，没有盲目重下或退款。
2. 原生 smoke 增加 ModelScope 来源校验、独立工作目录和项目缓存、一次有限 KV metadata RPC。修正实验启动配置：短 IPC 路径、恢复原生 chunked prefill、auto KV dtype 继承 BF16，并严格验证实际模型及 KV tensor 精度。没有改 attention、采样器或模型执行器。
3. 修正 PATH，使启动可找到项目虚拟环境内的 Ninja；固定私有 NVCC 13.0.88、FlashInfer 项目缓存和 FLASHINFER_NO_DOWNLOAD=1。第二次失败作业曾因 FlashInfer 忽略 XDG_CACHE_HOME 在 /root/.cache/flashinfer 生成默认 JIT 文件；该偏差已记录，原文件未删，后续明确使用项目内缓存。
4. 新增权重 header 审计、单 UUID 容量检查、便携 io_uring 探针及复制时交付脱敏。更新版本锁、能力矩阵、生命周期、修改位置、阶段状态和复现命令。

主要代码位于 experiments/prefix_io_v1/scripts/：download_pinned_model.py、resume_pinned_model.py、native_gpu_prefix_smoke.py、audit_local_model_headers.py、check_selected_gpu_capacity.py。smoke 逐步 0001–0005 补丁与原件在本轮 native-prefix-* 证据目录；这些是实验工具变化，不是研究策略补丁。

作者原始仓库的跟踪源码保持干净，原生 _C 哈希未变，本轮没有新增引擎补丁。共同修复、观察桥接、研究策略继续分开；观察桥接未接入本次运行，policy_patches 为空、策略 off。原精确 Prefix、LoadPlanner、共享 staging、预加载、复制合并和异步实现保持原位；其外部 SSD 路径尚未验收。

## 版本与模型锁

| 组件 | 固定版本 |
|---|---|
| py-kvcache | 3abba7a502d553f6e7e2e58b92086487e3395d7e |
| 作者 vLLM | 817a7e3124f817cd6e549581d3e5483207a753a4 |
| 作者实验仓库 | 0e023a84a21246b9bbc06266fa8070397eccbdc9 |
| simple-profiler | ec0d563bf68856df83c5824ac579700ec076b9e2 |
| Python／Torch | 3.12.3／2.11.0+cu130 |
| 私有 NVCC／FlashInfer | 13.0.88／0.6.11.post2 |

vLLM 安装元数据为 0.1.dev1+g817a7e312，以作者源码 SHA 为准。系统 CUDA／驱动未改。

模型取自官方 ModelScope Qwen 渠道，revision **16c174980d8a1492910551634b4969e69cdc2444**，命名空间 modelscope_git_commit，不宣称与 Hugging Face commit 等价。清单 SHA256：9b1c79986aa1fd62eb9064ecb8b4e2a03fc235a34ac2d69c33a4fcb531a91017。11 文件全部验证；339 个 tensor 均 BF16，7,615,616,512 元素，header/index 一致。此前 HF 访问失败证据保留，没有改 DNS、代理或路由。

最终 smoke SHA256：7d31cce851ca2334fedbe6b66c4e2d9bfff19e8768bafaa52f68e414518d77e2。
原生 _C SHA256：e30616d12f903169493f73c28e707540be17916e89794a9214b141d6e4769b94。
完整锁：experiments/prefix_io_v1/locks/。

## 测试、失败和证据

本节 B 表示 artifacts/prefix_io_v1/new-server-03/；R 表示 experiments/prefix_io_v1/runs/。

| 检查 | 实际结果 | 证据 |
|---|---|---|
| 固定下载／Range CPU | 109 passed，包含此前 65 项 | B/resume-review/first.xml |
| 来源／原生配置／KV metadata／IPC CPU | 61 passed，包含此前 24／49／51 项 | B/native-prefix-kv-dtype/cpu-tests.xml |
| 交付脱敏 CPU | 15 passed | B/delivery-redaction/cpu-tests-final.txt |
| 本轮独立 CPU 合计 | **185 passed，0 failed，0 skipped** | B/cpu-results-summary.json |
| 模型／header | 11 文件哈希通过，339 BF16 tensor | B/model-completion-accounting.json、model-header-audit.json |
| native-prefix-01 | 失败：IPC 路径 119 B 超过 107，权重未加载 | R/native-prefix-01/ |
| native-prefix-02 | 失败：加载权重后采样 JIT 缺 Ninja PATH | R/native-prefix-02/ |
| native-prefix-03 | 失败：JIT 通过后 Triton 拒绝显式 BF16 字符串 | R/native-prefix-03/ |
| native-prefix-04 | **真实模型、0→112、输出相同、实际 BF16 KV 通过** | R/native-prefix-04/details/ |
| 原生 ring／便携探针 | **EPERM** | B/io-uring-current.json、platform-probe/ |
| 最终 preflight | exit 2，io_uring／real handler 阻塞，mode=off | B/preflight.json |

历史完整 CPU 回归 388 passed／8 skipped、第二轮针对性用例没有在本轮重跑，不重复累计。所有中间失败保留。独立交叉复核见 B/FINAL_EVIDENCE_REVIEW.md。

## GPU 与下载预算

仅使用授权 UUID GPU-8b500efe-1a50-0e8e-b21e-716807eebedf。下列包括启动、失败、JIT 和收尾；NVML 检查无 CUDA context，也按 runner 墙钟保守计入。

| 本轮作业 | 退出码 | 墙钟秒 |
|---|---:|---:|
| native-capacity-01 | 0 | 0.227859 |
| native-prefix-01 | 1 | 43.941840 |
| native-capacity-02 | 0 | 0.229365 |
| native-prefix-02 | 1 | 36.531682 |
| native-capacity-03 | 0 | 0.225150 |
| native-prefix-03 | 1 | 102.042734 |
| native-capacity-04 | 0 | 0.226993 |
| native-prefix-04 | 0 | 56.275134 |
| native-postrun-01 | 0 | 0.273678 |

本轮 239.974435192 秒；累计 **285.247483194 秒 = 0.0792354120 GPU 小时**，8 小时预算剩余 **7.9207645880 小时**。累计 15 个作业，本轮 9 个；全部本轮 session 已排空，active_reservation=null。结束后的批准设备无计算进程，空闲 33,669,644,288 B。证据：B/final-budget-summary.json、共享 ledger、每个 R/label/result.json。

| 模型账目 | 字节 |
|---|---:|
| 完整 11 模型文件 | 15,242,788,168 |
| 累计应用层返回 payload，含来源元数据 | 15,242,868,376 |
| 保守扣费，含失败预留 | 19,422,798,722 |
| 20 GiB 下载预算剩余 | 2,052,037,758 |
| 保留失败 partial，额外磁盘副本 | 3,321,888,768 |

扣费约 **18.0889／20 GiB**；应用层计数不等于 HTTP／TLS wire 流量。首次跳转白名单拒绝和 30 分钟超时未退款；经过 CPU 验证的单次 Range 仅补 234,488,904 B，完整 SHA 通过后发布。详情与精确 argv 见 B/MODEL_PREPARATION_REPORT.md。

## 下一允许阶段

仍是 **P1**。平台须先提供可用的原生 io_uring，随后验证作者 ring／mmap／完成队列，再准备本机真实成本曲线和最小原作者驱动，验证 CPU staging、实际 SSD read/write、同一生产 KV 字节往返及 parent／fence／末尾 drain。不能套用其他模型／GPU 曲线，不能以 dummy forward、固定 sleep 或文件存在替代证据，也不能绕过 LoadPlanner。

便携探针实际 flags=0、depth=2 返回 EPERM，同时记录 io_uring_disabled=0、Seccomp=2、memlock unlimited；不足以断言唯一根因。没有关闭 seccomp、改驱动或另写 I/O 引擎。平台说明见 PLATFORM_IO_URING_REQUEST.md，本任务没有向平台发送消息。

主要命令见 REPRODUCE_NEW_SERVER_03.md。交付包 server-root/commands.jsonl 是主线程命令／输出快照，专项目录另存命令。临时签名和 Cookie 仅在交付副本脱敏，服务器原始证据保留，源与副本哈希随包记录。P1 未完成前不进入 P2–P7 真实验收，不作性能或论文收益结论。
