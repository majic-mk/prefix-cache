# 服务器09 G2 CUDA13 修复版：CPU 准备交付

截至 2026-10-01，此目录为新追加的模型资格验证候选。服务器 48 项 CPU 测试、4,035 项完整源/资产锁预检及预算/空间收口实际通过。本准备阶段新增 GPU 作业为 0，未创建新的人类授权或活动 scope。原 off01 失败证据保留，shadow01 未执行；G2 和 P4 均未因此完成，尚无性能提升结论。

## 上一轮实际 GPU 结果与原因

旧 `server09-g2-normal-off-01` 已真实加载 Qwen2.5-7B 的全部 4 个权重分片，日志模型内存约 14.29 GiB；初始化采样组件时失败，guard 退出 1，计入原预算 65.7177117979154 秒。未完成 cold/repeat 请求，未得到任何 128-token 正常输出。LLM 构造未完成，不能声称调用了原 EngineCore.shutdown；guard 确认真正会话已排空，账本无活动预留。旧 shadow01 被前置条件阻塞，没有运行。

报错表面为 FlashInfer 的 sm75 检查，前置日志明确 SM 12.x 需要 CUDA >=12.9。CPU 只读审计发现当前 torch 为 2.11.0+cu130，而 FlashInfer 原选择逻辑在未设置 CUDA_HOME/CUDA_PATH 时找到系统 nvcc12.8.93，先使用该编译器版本。因此软件选择不一致；不能据此判断 RTX5090 硬件不支持，修复后的实际模型/JIT 仍须 GPU 检验。

原始实际结果、完整日志、预算及工具链诊断见本机 `artifacts/prefix_io_v1_server09_g2_20261001/SERVER09_G2_OFF_FAILURE_AND_CPU_RECOVERY_REPORT.md`，对应服务器 `artifacts/prefix_io_v1/server09-g2-20261001/`。已核验两侧 29 文件清单，历史报告保持原字节。

## 实际改动与边界

1. 新增独立入口目录 `artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-20261001`，限定新名称 off02/shadow02；旧模型入口、源锁、旧授权及失败记录保持原字节。
2. SDK 公共修复使用服务器已经安装的 CUDA13.0.88：完整固定 1,934 个 toolkit 文件、cudart、真实 libcuda 与编译证据。未来作业仅在本次 `details/runtime-cache/cuda13-sdk` 创建六个链接，供原 FlashInfer 选择原 CUDA13 编译器和链接库使用；不替换系统 toolkit、驱动或 .venv 文件。
3. 真正 guard 会话、源/模型/权限/预算门禁通过后，先配置该作业缓存，再准备 SDK 并显式应用返回的六个进程环境键，最后才导入 framework。SDK 配置应用于 off 与 shadow，作为共同修复。证据只导出这六个普通字段和目标链接，不导出全部继承环境。
4. 原 worker `g2_worker_observation.py` 与旧版本逐字节相同；独审确认受限 SDK 接线之外 runner AST 与旧版相同。原作者 LLM、SamplingParams、add_request/step、完整 token 保存、异常和关闭路径保留。没有重写执行器、缓存引擎或策略。
5. 模型、bf16、TRITON_ATTN、精确 Prefix Cache、KV64MiB、原调度设置保留；每个作业仍 cold/repeat 各完整生成 128 tokens，预期 cached0/112。研究策略关闭；此资格作业没有 SSD I/O，native drain 为 not_applicable。事件仅诊断，无时钟映射、成本资格或效果声明。

新 runner 40,249B SHA-256 `931ec0d122ac1e688f3ec55d721b4f930cb5aad5342ae1c8821d3c73ab5a76eb`；worker 13,090B SHA `096f2fffe91de7132f45997be51fecceec57d65fdf86caab6f8bbe5c71748f7b`；计划 14,023B SHA `f027df4fc433e80c3f1d2a6c1cd35d61a084feef083b0b402dc315ed4fb398de`。

## 实际 CPU 命令和结果

所有服务器资格前准备命令均 `CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S`，无 framework 导入或 GPU 初始化。

- `artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-20261001/test_g2_worker_observation.py --source-root '/root/autodl-tmp/prefix-io-v1-handoff/project' --scalar-source artifacts/prefix_io_v1/server09-runtime-connector-20261001/p4_runtime_scalar_connector.py --frame-source artifacts/prefix_io_v1/server09-migration-20261001/collector-candidate/p4_full_step_frame_adapter.py`：22/22 PASS，0.043 秒。
- `artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-20261001/test_g2_normal_model_lifecycle.py --source-root '/root/autodl-tmp/prefix-io-v1-handoff/project'`：26/26 PASS，0.073 秒。新 48 项包含原 45 项回放和 3 项 SDK 接线检查，未作为 48 项全新检查重复累计。
- `artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-20261001/run_g2_normal_model_lifecycle.py --project '/root/autodl-tmp/prefix-io-v1-handoff/project' --preflight --source-lock artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-20261001/gpu-source-lock-candidate.json`：退出 0，SOURCE_LOCK_VERIFIED_BUT_NO_NEW_G2_HUMAN_SCOPE，source_count4035，gpu_initializedFalse，actual_gpu_runs0。
- 标准库分片组装命令实际逐项核验连续 offset、长度和 SHA，再以 xb 追加新锁及完整本地准备审计。源码预检实际读取了服务器全部锁定目标和模型文件，并由 SDK 原接口完整核验实际工具链及外部驱动。
- 标准库收口命令实际确认原始 permissions、有效权限、旧 runner/计划/G1/G2 锁字节未变，账本与 off01 会话状态保持，新的 off02/shadow02 名称及新授权文件均不存在。

此前 SDK 最终 V2 已在服务器 12/12 CPU 测试通过，并完成真实全部 1,934 文件/217,111,529B 的只读预检。真实 CUDA13 CPU 编译/主机链接四个命令均通过（含 compute_120f/sm_120f）；生成的 .so 未加载，没有执行 GPU kernel。这些证据证明 CPU 选择、布局和链接，不能代替 CUDA 模型实跑。SDK 首版真实清单排序预检失败及修复前版本均已保留，V2 独审与最终 17 文件两侧镜像已通过。

## 冻结与证据

新完整锁 895,738B / 4,035 引用，SHA-256 `f3d21df27310ce2c5199ab69f2a62f80b2111390f3ad160aba6ea843dd06f1d9`。保留旧 G1 全部 2,052 条和旧 G2 全部 2,088 条的原值、原序；新增 1,947 条。外部 libcuda 不放入普通项目路径引用，由固定原始资产清单及 helper 校验其实际字节。

- `ACTUAL_SERVER_CPU_NORMAL_MODEL_TESTS.json`：两次实际服务器 stdout/stderr、命令及退出码。
- `ACTUAL_SERVER_CPU_COMPLETE_SOURCE_PREFLIGHT.json`：真实 4,035 项服务器预检。
- `ACTUAL_CPU_SOURCE_LOCK_ASSEMBLY.json`：15 个锁分片与 28 个审计分片的真实组装。
- `ACTUAL_SERVER_CPU_BUDGET_STORAGE_CLOSURE.json`：账本、源文件、权限和空间实际收口。
- `INDEPENDENT_BOUNDARY_REVIEW.json`：独立 AST/字节审查，独立 16 项只读 CPU 回放及 SDK 接线门禁通过。
- `CPU_SOURCE_LOCK_PREPARATION_RESULT.json`：完整本地来源/构建审计，已原字节追加到服务器；其 CPU 字段不授予 GPU 权限。
- `CPU_NEXT_SCOPE_REVIEW.json`：独立新 scope 提案审查，文中服务器检查标为 pending 是其创建时状态；之后实际检查以以上 root 服务器回执为准。
- `NEXT_GPU_STAGE_PROPOSAL_NOT_AUTHORIZED.json`：可审查的新阶段命令和固定边界，尚未授权。
- `DELIVERY_MANIFEST.json` 及两侧核验回执为此次准备包交付闭环；manifest 只覆盖其冻结时的准备文件，后续人类授权及真实作业须独立证据，避免循环修改源锁。

## 真实预算与下一允许阶段

本 CPU 准备阶段新增 GPU 作业 0。原累计 GPU 用时 16604.382256507408 秒，剩余 12195.617743492592 秒（约 3.388 小时），224 个事件，active_reservation=null；账本 SHA `1e5b29fbc7fb66f84f3554149e5ac24f5a342b349f02700062d9a1bf53a0679e` 未改变。

收口时 PRIMARY 可用 12977139712B（约 12.09GiB）。拟用 128MiB 空间预留和 8GiB 可用空间下限；预留是计划数，不冒充实际编译缓存上界，真实 guard/阶段门禁在执行时再次检查。没有数据删除、下载、租赁/付款、驱动或系统修改。

下一阶段须新的直接人类授权绑定本次 f3d21d…源锁：off02 一次，只有完整输出、原关闭和 guard 排空通过才允许 shadow02 一次；每次执行 300 秒，加 20 秒收尾，两次最多预留 640 秒，继续使用原 8 小时预算。任一 off02 失败即保留日志/部分新目录，停止 shadow02 和同 scope 重试。

再次限定授权的依据是原授权记录明确绑定旧源锁、旧固定名称和资格 context；旧代码 scope 检查拒绝把已使用/失败的 off01 改名重试，未用 shadow01 也只有成功 off01 后才有效。本修复改变了工具链接线、源锁和固定名称，原回复不能伪造为新 scope 授权。

只有这些资格作业通过，才可准备后续原生 I/O/计时成本资格和严格对照的独立阶段。没有完成四组性能对照，不能声称本方法已形成提升、P4 已通过或整体系统已经可行。
