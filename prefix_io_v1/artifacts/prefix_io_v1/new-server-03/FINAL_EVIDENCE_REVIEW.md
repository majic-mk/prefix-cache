# 最终证据独立复核：原生 GPU Prefix 短路径通过，完整 P1 仍未验收

复核方式：只读服务器已完成运行的 JSON、日志、源码/扩展哈希和 CPU 测试产物，执行 JSON/算术/哈希一致性断言；没有重新运行模型、GPU、网络或测试，没有修改正式报告、状态、权限或共享账本。本次仅新增本复核报告。

**结论：现有证据相互一致，支持 native-prefix-04 的 PASSED_GPU_PREFIX_PATH_ONLY。** 它是 N 类作者原生 GPU Prefix 的短 synthetic-token 路径，不能扩展为 SSD/staging、生产 KV 字节一致性、物理释放依赖、性能或完整缓存集成资格。三个早期真实 GPU 失败及其耗时均保留并计费。

## 模型与输出一致性

核对对象为 experiments/prefix_io_v1/runs/native-prefix-04/ 下的 result.json，以及 details/frozen-config.json、smoke-result.json、cold.json、repeat.json、kv-tensor-storage-metadata.json。

- runner 的 exit=0、child_exit=0、timed_out=false、error=null；单次计费墙钟 56.27513352409005 秒。smoke 状态 PASSED_GPU_PREFIX_PATH_ONLY，native_engine_shutdown=completed，runner session_drained=true，清理前后均未发现剩余 session member。
- 冻结作者 commit 为 817a7e3124f817cd6e549581d3e5483207a753a4，与工作树 HEAD 相同；实际 vLLM 模块路径在作者工作树。当前 smoke SHA 与 native-prefix-04-launch.json 保存值一致，原生扩展 SHA 仍与第二轮 build04 相同。
- 冻结模型是官方 ModelScope 的 Qwen/Qwen2.5-7B-Instruct，revision namespace=modelscope_git_commit，revision=16c174980d8a1492910551634b4969e69cdc2444。不是未经验证的 HF 等价 revision。冻结 manifest SHA、11 个文件的大小/哈希记录逐项与固定 manifest 一致；六类来源证据文件的当前大小/SHA 独立复核通过。未重新读取并哈希 15 GB 全量权重，而是交叉核对本次 smoke 的已验证记录与完成下载证据。
- 两个请求的 prompt_token_ids 均严格等于冻结的 1000…1127，共 128 个。cold/repeat 的结果分别为 cached-token 0、112，finish_reason 均为 length，且 smoke-result 的内嵌 results 与独立 cold/repeat 文件完全相同。
- 两次输出都是 16 个 token IDs，逐项完全相同：[11, 220, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]。没有容差替代或仅比较输出长度。

该结果证明这一固定短输入的原生缓存计数与生成结果符合冻结预期，不证明语言质量、长序列、并发、多样输入、逐 token ITL 或其他缓存层的行为。

## 实际资源证据及其边界

实际 ENGINE 为单卡 uni、模型 BF16、无量化、safetensors、TRITON_ATTN、block_size=16、max_model_len=256、max_num_seqs=1、eager、64 MiB 配置 KV 预算。外部 kv_transfer_config=null、kv_offloading_size=null、cpu_offload_gb=0。当前 cache dtype 参数是作者支持的 auto，实际模型和实际 KV Tensor dtype 都已记录为 torch.bfloat16；不能把 auto 字符串误写成实际 FP32 或未验证 dtype。

KV metadata 独立文件与 smoke-result 内嵌字段一致：

| 资源量 | 真实记录 |
|---|---:|
| CUDA BF16 KV tensor 数 | 28 |
| 唯一 backing storage 数 | 28 |
| 每个 tensor shape | [73, 2, 16, 4, 128] |
| 原生 blocks | 73 |
| 配置 KV 字节上限 | 67,108,864 |
| 原生 KV 配置声明字节 | 66,977,792 |
| 实测 unique untyped_storage().nbytes() 合计 | 66,977,792 |

28×73×2×16×4×128×2=66,977,792 B，与元数据和实际 storage 合计一致，且小于配置上限；差额 131,072 B 并不代表泄漏或额外可复用块。测量是一次原生 worker RPC 中的有限 metadata 读取，去重依据 storage backing identity，不读 KV tensor 内容、不复制 tensor、不扫描块、不挂 hook。

cuda_allocator_reserved_bytes=null、physical_release_witness=null。上述 storage bytes 不是 CUDA allocator reserved、进程总显存、模型权重/workspace大小、空闲显存或真实释放见证。native shutdown 与 session drain 是本次执行清理证据，不能替代共享 staging/CUDA event/SSD CQE/多父 store 依赖的资源释放合同验收。

主线程随后实际 native-postrun-01 只检查批准 UUID：
GPU-8b500efe-1a50-0e8e-b21e-716807eebedf。
NVML 返回 compute_process_count=0，free=33,669,644,288 B，used=521,273,344 B，总量=34,190,917,632 B，未创建 CUDA context；其 result exit=0、session_drained=true。它支持检查时该设备没有 NVML 列出的 compute process，不将设备 used_bytes=0 或长期无泄漏当作结论，也不替代对象级物理释放证据。

## 运行环境记录与历史失败

frozen-config.json 保存离线、UUID、原生缓存、序列化 opt-in、项目内 cache 与 .p1tmp/IPC 路径。真实 launch 还补充了 frozen-config 未列出的项目私有 JIT/FlashInfer 环境，复现时必须一起读取 native-prefix-04-launch.json：

- CUDA_HOME=项目 experiments/prefix_io_v1/cuda-toolkit，PATH 含项目 .venv/bin 与私有 toolkit/bin；
- FLASHINFER_WORKSPACE_BASE=项目 experiments/prefix_io_v1/runtime-cache/flashinfer；
- FLASHINFER_NVCC=项目 experiments/prefix_io_v1/cuda-toolkit/bin/nvcc，FLASHINFER_NO_DOWNLOAD=1；
- TORCH_CUDA_ARCH_LIST=12.0、NVCC_THREADS=1、MAX_JOBS=4。

这些是实际 launcher 环境记录，不能声称全都已经出现在 frozen-config 内；offline/NO_DOWNLOAD 标志也不是网络流量审计结果。可信本地 callback 的 VLLM_ALLOW_INSECURE_SERIALIZATION=1 是当前原生离线 uni 路径的已记录 opt-in，不泛化为开放服务的配置。

历史失败文件仍存在，独立 result 均 exit=1、child_exit=1、session_drained=true：

| 作业 | 计费秒数 | 已读原始证据中的失败 |
|---|---:|---|
| native-prefix-01 | 43.941840377636254 | 原 IPC 路径超过 sockaddr_un.sun_path 107 字符限制；details/smoke-result.json |
| native-prefix-02 | 36.53168174251914 | process.log 的底层 FileNotFoundError: ninja；顶层 EngineCore 初始化失败 |
| native-prefix-03 | 102.04273400548846 | process.log 的 unsupported kv_cache_dtype (str), got bfloat16；顶层 EngineCore 初始化失败 |

不将这三次写成成功，不把它们从预算中排除；native-prefix-04 使用独立新 label 的成功证据。

## CPU 与模型准备交叉核验

当前 cpu-results-summary.json 的文件 SHA 均与对应保存产物一致，XML 用例/失败/错误/跳过数重新解析一致：

| 最终 suite | 通过用例 |
|---|---:|
| resume-review/first.xml | 109 |
| native-prefix-kv-dtype/cpu-tests.xml | 61 |
| delivery-redaction/cpu-tests-final.txt | 15 |
| 当前 unique 总计 | 185 |

109 已包括早期 65，61 已包括早期 24/49/51；历史重复运行不再相加。这里没有重新跑测试，且 CPU/synthetic/local IPC 结果不冒充实际模型或 SSD 实验。

模型准备记录与本次 frozen 模型相同：完整 11 文件 15,242,788,168 B，339 个 BF16 tensor；保留第四分片失败 partial 3,321,888,768 B，目录文件逻辑字节 18,564,676,936 B。累计应用层已返回 payload=15,242,868,376 B，保守预算收费=19,422,798,722 B，模型下载剩余=2,052,037,758 B。两次下载失败完整保守收费、Range 补 234,488,904 B 和原 partial 均保留。独立模型过程说明见 MODEL_PREPARATION_REPORT.md，不在此重复下载历史。

## 最终预算与阶段门禁

已读取 final-budget-summary.json，并逐项对照共享账本全部 GPU events、各新作业 result，以及 native-postrun-01 原始 process.log/result：

- 累计 15 个受预算作业；本轮 9 个（4 次 capacity、4 次 native smoke、1 次 postrun）。
- 累计计费 285.24748319387436 秒，即 0.07923541199829844 GPU 小时；本轮 239.97443519160151 秒；8 小时额度剩余 7.920764588001702 小时。
- 各事件耗时求和与汇总一致；全部本轮 session_drained=true，批准 UUID 一致；共享账本和汇总的 active_reservation 均为 null。
- 模型 C/P/余额与完成模型会计一致。GPU 作业时长、模型下载字节和 CPU 编译/检查时间没有互相混算。

最终能支持的阶段表述为 **P1 的 N 类原生 GPU Prefix 短路径通过，完整 P1 仍受 SSD/io_uring 门禁阻塞**。smoke 中 ssd_or_staging_qualified、end_to_end_cache_qualified、production_kv_byte_identity_qualified、performance_claim 明确为 false。没有真实 SSD store/restore 完成字节、生产 KV 往返比较、缓存生命周期/物理释放或性能收益资格，也没有打开 P2–P7 的前序门禁。

## 关键证据 SHA256

| 文件 | SHA256 |
|---|---|
| native-prefix-04/details/frozen-config.json | 6c3ba3ba8cdab49e8db5a1fae3111ba6b7dd38b74d9830b73913a86db9766581 |
| native-prefix-04/details/smoke-result.json | fbb7a4a26198fab0450e9ffb2cb2e4ff87e1550454bc2bf4c6f0ff36fbc19b74 |
| native-prefix-04/details/cold.json | e16f6a6e1180703a0dcacad0f15309696cfe32371dc49a643c40cf8866abf250 |
| native-prefix-04/details/repeat.json | 939ee00201462858c7384a0ec05a1fbea9ded567d787d4183288300932785a4a |
| native-prefix-04/details/kv-tensor-storage-metadata.json | 064376188bf0eaf5e678e72fa2101d22644b7f7fc181aa5145d11f631a9a26d4 |
| native-prefix-04/result.json | 4ae0ec52d903f6784703f9076357f26fabc085d965b6006adb275439934db49e |
| new-server-03/native-prefix-04-launch.json | 8a74443c343e2277c429a736a7e90abfef8e47b5881b43617d0b0b49a55e1053 |
| new-server-03/cpu-results-summary.json | cb04597615d33cc63e1154aa871eeefdb24d9ef93008d7e749c9cb3d1dce3ad6 |
| new-server-03/final-budget-summary.json | 5bebfcaefbfe1a8a38f47d90a404679ec2a908668042a152907542e8931407fb |
| scripts/native_gpu_prefix_smoke.py | 7d31cce851ca2334fedbe6b66c4e2d9bfff19e8768bafaa52f68e414518d77e2 |
| 作者原生 vllm/_C.abi3.so | e30616d12f903169493f73c28e707540be17916e89794a9214b141d6e4769b94 |

本表 native-prefix-04 路径基于 experiments/prefix_io_v1/runs/，new-server-03 路径基于 artifacts/prefix_io_v1/，scripts 基于 experiments/prefix_io_v1/，原生扩展位于 third_party/work/vllm-author-build/。
