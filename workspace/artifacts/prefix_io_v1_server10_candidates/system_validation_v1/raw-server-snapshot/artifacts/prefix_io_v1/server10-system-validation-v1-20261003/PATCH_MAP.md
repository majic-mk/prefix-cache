# 新服务器修改位置清单

以下路径均相对服务器项目 ROOT。本轮新增的是私有迁移配置、进程内薄适配和 CPU 来源校验；既有作者源码、GPU 运行实现及数字比较器保持原字节。历史共同修复继续与研究策略分开，off 路径保留。

| 位置 | 实际改动与用途 | 分类及证据边界 |
| --- | --- | --- |
| `artifacts/prefix_io_v1/server10-sdk-rebind-v1-20261003/rebind_cuda13_cpu.py` | 将新机实际驱动绑定到克隆的 CUDA13 SDK，复用 SDK 树；追加真实 CPU 小探针编译/链接凭证。 | 环境适配。结果在 `artifacts/prefix_io_v1/server10-cuda13-cpu-20261003/SDK_REBIND_RESULT.json`；未修改驱动或系统，CPU 编译不等于 GPU 资格。 |
| `artifacts/prefix_io_v1/server10-reference-migration-v1-20261003/migration_contract.py`、`prepare_server10_reference.py` | 追加新 GPU UUID、作业名、权限和源锁；保留历史源及预算祖先。 | 迁移配置；沿用原累计预算，历史 GPU UUID 不改写。 |
| `artifacts/prefix_io_v1/server10-reference-migration-v1-20261003/run_server10_reference.py`、`server10_reference_runtime.py` | 私有 globals 绑定新机配置，执行后恢复；复用原 launcher、执行器、缓存与 guard。 | 薄适配；原父/子/运行后源校验、原 shutdown 和 OS 排空门禁保留。恢复证据位于两次 reference 作业的 `details/server10-migration-runtime-state.json`。 |
| `artifacts/prefix_io_v1/server10-reference-postprocess-v1-20261003/postprocess_reference.py` | 修复 Python 3.12 下分析器函数来源校验：从整模块编译结果取原函数代码，不再用孤立 AST 编译作比较。 | 共同 CPU 校验修复；数值比较逻辑不改，GPU 不重跑，原父进程 exit 78 保留。新结论单独写入 `REFERENCE_CPU_POSTPROCESS_RESULT.json`。 |
| `artifacts/prefix_io_v1/server10-native-qualification-v1-20261003/prepare_native_qualification.py` | 生成精确 off/shadow 两作业配置和 CPU 门禁，调用原参数化 native qualifier。 | 资格配置；原 `experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py` 与 `run_gpu_stage.py` 不改。次数、顺序及成功后继续由外层执行记录约束。 |
| `artifacts/prefix_io_v1/server10-g2-migration-v1-20261003/run_server10_g2.py`、`prepare_server10_g2.py` | 新机绑定原完整模型资格入口及 SDK 配置，保持原 off 后 shadow 顺序和完整输出/排空门禁。 | G2 迁移配置；off/shadow 真实 GPU 完整输出与关闭均 PASS，私有绑定全部恢复。 |

本轮未新增模型执行器或缓存引擎，未以新策略替换原精确 Prefix Cache、成本准入、共享 staging、预加载、复制合并和异步流水线。紧凑观测的有限运行资格已见 G1；真实资源释放、有限候选调度、干扰额度及收益仍须各自的后续证据，不能由迁移或审计结果代替。
