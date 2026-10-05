本次在新 AutoDL server09（SSH 端口 29815）完成 G1 后的 CPU 实施与下一阶段准备。服务器项目路径：/root/autodl-tmp/prefix-io-v1-handoff/project。没有删除模型、实验数据或历史失败记录，没有修改驱动、系统、缓存引擎或模型执行器。

实际 GPU 结果仍为已授权的 G1 off / shadow 两次，均通过原生 CUDA、Linux AIO、共享 staging、复制合并和异步流水线检查。两次 GPU 账本新增 18.101658063 秒；8 小时累计预算已用 16538.664544709492 秒，剩余 12261.335455290508 秒。G1 的输入是合成 page/tensor，没有加载模型。它证明现服务器可以运行该原生传输路径；尚不能证明方法的端到端收益，P4 尚未完成。

版本锁：项目 HEAD cc7898b1ba59d89ce7fdbb186ded21880f1adf08；原生 py-kvcache HEAD 3abba7a502d553f6e7e2e58b92086487e3395d7e；作者 vLLM HEAD 817a7e3124f817cd6e549581d3e5483207a753a4。服务端 CPython 3.12.3、torch 2.11.0、vLLM 0.1.dev1+g817a7e312、transformers 4.57.6、numpy 2.3.5、pytest 9.1.1。完整 P0 工作区、接口能力矩阵、生命周期和历史修改位置见上一份 G1 交付报告。

| 本批实际修改位置（相对服务器项目根） | 行为和边界 | 服务器实际 CPU 测试 |
|---|---|---|
| artifacts/prefix_io_v1/server09-context-v3-20261001/ | 复用原完整轨迹校验，接受真实冷 prefill prior-context=0，保留首帧与全部输出；decode selection 原规则保持。 | 27 passed |
| artifacts/prefix_io_v1/server09-runtime-connector-20261001/ | 薄封装原 execute / prepare / sample；参数、返回身份、原异常和调用次数保持。关闭观测恢复原方法。 | 38 passed |
| artifacts/prefix_io_v1/server09-paired-v3-20261001/ | baseline/action 先分别验证全部帧，再复用原 paired / IO 门禁。CPU 假时钟不发布生产成本。 | 32 passed |
| artifacts/prefix_io_v1/server09-g2-boundary-20261001/ | 复用原 NativeWindowJournal 四阶段边界，逐序号与计数守恒；缺失、未知、partial/outside 均拒绝。旧 runner 仍为 CPU stub。 | 19 passed |
| artifacts/prefix_io_v1/server09-g2-native-drain-20261001/ | 原 shutdown 一次调用且返回身份保持，随后核验实际 worker、AIO、16 个 owner 字段及四阶段守恒。实际字段缺失拒绝，禁止用 snapshot 默认零代替。 | 53 passed |
| artifacts/prefix_io_v1/server09-g2-normal-worker-20261001/ | 新的具体资格 runner 调用原 LLM、UniProc RPC、add_request / step / shutdown，紧凑观测只保留有界标量和弱引用；完整输出先写证据再导出观测。未安装 live hook。 | 22 + 23 = 45 passed |

本批服务器新增定向 CPU 测试合计 **214 passed、0 failed**。此前整体验证 1956 passed / 16 skipped、权限候选 21 passed 为各自批次，不与本批重复相加。本机和独立审查结果单独保留，未冒充服务器测试。正常 worker 的两个最终测试实际耗时分别为 0.042 和 0.061 秒；它们使用 CPU fixture，未导入 GPU 后端、未初始化模型。

生命周期约束保持：source-safe、buffer-reusable、cache-visible、durable 分别判断；Future.done、D2H 完成或空队列均不单独授予资源释放。未完成 owner generation、活动引用、真实 fence、原 allocator 确认前，不发布 release credit。关闭新观测和研究策略可回原路径；成本、ETA、真实跨时钟映射未资格时继续回退 U。

独审发现并修复了三项实质缺口：owner 字段缺失被默认零掩盖；origin/scope 任意对象可能保留资源 owner；观测导出失败会丢失已完成的完整输出。对应修复说明、前版冻结字节、失败记录保存在候选目录历史子目录。normal runner 在 frontend 收到完整 128 个 token 后立即写 exclusive cold/repeat-frontend.json，再进行 export / event query / reconciliation；观测失败不会改写模型结果或重试原执行。

模型资产只读核验实际在服务器流式读取全部 11 个文件，共 15,242,788,168 字节，339 个张量索引，与固定 ModelScope revision、已保存官方列表、下载计划和 SHA 一致。耗时 12.720283 秒，状态 PASS_OFFICIAL_ASSET_BYTES_SHA256_ONLY。首次过严要求空 LatestCommitter.Id 等于 URL revision 的失败记录保留，后续复用原验证语义；没有重新下载模型。该证据不等于运行时 ABI 或模型执行资格。

normal 候选固定使用已有 Qwen2.5-7B-Instruct、原作者 vLLM、单 GPU 单进程、原精确 Prefix Cache、温度零，每个 job 各做 cold / repeat 两个请求、每请求完整 128 输出，两个模式共 512 IDs。native IO 为 none、drain 为 not_applicable；当前 CUDA event 只 query，不 synchronize，不把未验证的时间当作成本。原 native 7 个 .so 回退与新作者 Python 源通过锁定 loader 复用，未把旧 build Python 前置替换作者源。FlashInfer / CUDA / Torch extensions 及已有缓存路径在新 run 的 details/runtime-cache 下隔离。

原 permissions.yaml SHA 795d24f9955379653e34ffbbfd7614ad9ef96ad0ac9a208c6b4baefd994fac50 不变；GPU 账本 SHA 32770dbca6f2e37d274f5007d151692e0a4c910a688d352abec79d60318ffaae、223 events、active reservation 为 null。本批 CPU 未新增 GPU 消耗。实际 tracked git diff 为空，因为实验目录未跟踪；具体源码变化用 SHA 和历史备份证明，不能说两个脚本是 Git tracked 改动。G1 中两个 canonical 脚本的授权路径参数修改维持已冻结 SHA。

存储只读检查时 PRIMARY 可用 12,986,712,064 B，满足下一轮 8 GiB floor + 128 MiB 预留。既有 FlashInfer/Triton/Inductor/vLLM 编译缓存约 32.12 MiB，是估算依据而非新启动峰值上限。原 20 GiB 字段分别对应模型下载额度与旧服务器获批 AUX 的限制，不是当前 PRIMARY 全局上限；本阶段只使用 PRIMARY，不启用 AUX，不删除任何数据。

实际测试命令（每条均在服务器根目录执行）：

```sh
cd '/root/autodl-tmp/prefix-io-v1-handoff/project' && CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S '/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-context-v3-20261001/test_p4_complete_trace_context_v3.py' --control-source third_party/work/prefix-io-p4-02-cpu/src
cd '/root/autodl-tmp/prefix-io-v1-handoff/project' && CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S '/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-runtime-connector-20261001/test_p4_runtime_scalar_connector.py' --collector-source '/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-migration-20261001/collector-candidate/p4_full_step_frame_adapter.py'
cd '/root/autodl-tmp/prefix-io-v1-handoff/project' && CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S '/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-paired-v3-20261001/test_p4_paired_context_v3.py' --control-source third_party/work/prefix-io-p4-02-cpu/src --context-source '/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-context-v3-20261001'
cd '/root/autodl-tmp/prefix-io-v1-handoff/project' && CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S '/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-g2-native-drain-20261001/test_g2_native_drain_provider.py' --source-root '/root/autodl-tmp/prefix-io-v1-handoff/project' --connector-source '/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-runtime-connector-20261001/p4_runtime_scalar_connector.py' --adapter-source '/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-migration-20261001/collector-candidate/p4_full_step_frame_adapter.py'
cd '/root/autodl-tmp/prefix-io-v1-handoff/project' && CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S '/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-g2-normal-worker-20261001/test_g2_worker_observation.py' --source-root '/root/autodl-tmp/prefix-io-v1-handoff/project' --scalar-source '/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-runtime-connector-20261001/p4_runtime_scalar_connector.py' --frame-source '/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-migration-20261001/collector-candidate/p4_full_step_frame_adapter.py'
cd '/root/autodl-tmp/prefix-io-v1-handoff/project' && CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S '/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-g2-normal-worker-20261001/test_g2_normal_model_lifecycle.py' --source-root '/root/autodl-tmp/prefix-io-v1-handoff/project'
```

boundary、模型逐字节核验、source-only preflight、SHA 与预算收口的完整原始命令和 stdout/stderr 见 ACTUAL_CPU_COMMANDS.json、ACTUAL_G2_SOURCE_LOCK_VERIFICATION.json 及各候选 SERVER_CPU_TEST_RESULT.json / SERVER09_MODEL_PAYLOAD_READONLY_RESULT_V2.json。新源锁 452304 B、SHA 5e7cc04fe759411800b895169d669f75cade6b2f0adda34d3992ecf2a7dbd760，保留原 2052 项并新增 36 项，共 2088 项。服务器已实际逐字节核验全部文件，通过 SOURCE_LOCK_VERIFIED_BUT_NO_NEW_G2_HUMAN_SCOPE，exit=0，GPU 初始化与运行均为 0。该检查只证明源和资产完整，不赋予 GPU 权限。

下一允许阶段：继续安全 CPU 分析与证据核验。下一次 GPU 为 **G2 正常模型完整输出与生命周期资格**，待单独授权后按冻结方案先 off、再 shadow，最多两次，各 300 秒 + 20 秒 guard 预留，总预留 640 秒，沿用 8 小时累计账本，不重置预算。off 失败即保留原日志、回 CPU 诊断，不自动运行 shadow。不下载模型，不修改系统/驱动，不增加租用/付款，不运行效果四臂，不宣布 P4 完成。

证据主目录已在服务器 /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-cpu-followthrough-20261001，同时镜像到本地本目录。各候选 DELIVERY_MANIFEST 与 LOCAL_AND_REMOTE_DELIVERY_VERIFICATION 记录逐文件 SHA；normal 源锁与 preflight 回执在对应候选目录，最终交付清单列出本文及原始证据。

