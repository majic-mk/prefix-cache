# 新服务器第四轮：CPU 校准候选准备，P1 仍阻塞

日期：2026-09-27（Asia/Shanghai）。操作服务器 connect.westc.seetacloud.com:24801；项目 /root/autodl-tmp/prefix-io-v1-handoff/project。开发与 CPU 检查均在该服务器执行；本地只保存交付副本并校验完整性。

## 本轮实际完成

- 重连并复查权限、工作区、固定作者仓库与阶段状态。四个冻结 upstream 的 HEAD 未变，跟踪源码无改动；P0 已完成，P1 部分通过，P2–P7 继续关闭。
- 实际重跑一次 CPU io_uring_setup(depth=2, flags=0) 探针：exit 2、EPERM。记录 io_uring_disabled=0、Seccomp=2、memlock unlimited；这些信息不足以认定唯一根因。没有更改系统权限、seccomp、驱动或服务。
- 新增 experiments/prefix_io_v1/scripts/prepare_native_calibration_plan.py 与 tests/prefix_io_v1_calibration/test_prepare_plan.py。脚本校验已通过原生 smoke 的模型/来源/版本锁、同一次预算作业成功与清理状态、GPU UUID，以及实际 BF16 KV 几何；验证原作者完整源码 SHA 后只复用 Job 和 build_job_plan。
- 用真实本地资格文件生成候选计划：名义 doc-size 64/128/192，每组3个拟测请求，共9个。原驱动按词构造 prompt，这些参数不是已验证的 token 长度。计划仍为 PLANNED_CPU_ONLY／UNEXECUTED，driver_ready=false，f/g_mem/g_ssd 全部为 null，未冻结可运行配置。
- 可选曲线检查拒绝旧 v1 标量表冒充 v2，并核对当前模型路径、dtype、几何与有限非负时间值。原生格式解析不证明真实标定；即使 v2 可解析或 golden 自洽，也始终 usable_for_planner=false。
- 补充原生曲线接口审计及独立代码复核。没有修改模型执行器、缓存引擎、准入器或研究策略；策略仍 off，原有异步流水线和共同修复保持原位。

本轮准备器 SHA256：bbea9525726a3e88a6cd2e5e7489cefe77b391606213999ad70737a36a1567f0。
候选计划 SHA256：ffb164411e2980db5f8cb393c691ed639a35ba3646e99b99645487852ea839ab。

## 测试和命令

新增 CPU suite **32 passed，0 failed，0 errors，0 skipped**。覆盖输入上限、来源/源码变动拒绝、输出身份错配、失败或未排空作业、GPU/模型锁错配、v1拒绝，以及结构检查不提升为真实资格。

历史第三轮185项没有重跑，也不在本轮重复累计。CPU preflight 本轮返回 BLOCKED／exit 2，原因是 io_uring unavailable 和真实 handler 未验证；这项阻塞不计作单元测试通过。

精确命令见 REPRODUCE_NEW_SERVER_04.md 和本轮 calibration-preparation/README.md。原始命令及输出仍保留于服务器项目上级 commands.jsonl；本次增量包包含专项运行证据。

## 核实的校准接口缺口

当前作者 emit_break_even.py 输出 v1 标量阈值；当前 py-kvcache LoadPlanner 需要 v2 三条曲线与 KV 几何。两个固定仓库中没有可直接复用的 v2 导出器。原 g_mem 还涉及从 SSD 曲线和带宽模型推导，不能描述为 RAM 实测。

f(N) 机制上可独立于 SSD，但原 launcher 的新 session、名义文本长度、默认预热与离线配置仍需执行适配；孤立 f(N) 不足以解除完整 P1。此次未为其消耗 GPU，也未把上一轮 smoke 耗时转换成曲线、套用其他设备数据或新写插值算法。详细行号和边界见 calibration-format-audit.md。

## GPU、下载与阶段

本轮真实 GPU 运行：**0**；模型下载：**0**。权限文件和共享预算账本逐字节未变。

历史累计 GPU 墙钟仍为 **285.247483194秒（0.0792354120小时）／8小时**，剩余7.9207645880小时。模型下载保守扣费仍为 **19,422,798,722 B（18.088890912 GiB）／20 GiB**，剩余2,052,037,758 B。活动预算预留为空。

上一轮真实 Qwen2.5-7B BF16 原生 Prefix 结果仍有效：0→112 cached tokens，两次16个输出 token相同；这不是本轮重跑结果，也不代表 SSD/staging 集成通过。

**下一允许阶段仍是 P1。** 需要平台先提供原生 io_uring 可用的环境，再验证作者 ring/完成队列/实际 I/O，完成同环境合法成本曲线、最小原生驱动、CPU staging、SSD store/restore、同一生产 KV 字节往返及原生末尾 drain。现 permissions.yaml 不允许驱动或系统改动，本轮未绕过该限制。P1 未验收前不进入 P2–P7。

## 证据与交付

本轮证据目录 artifacts/prefix_io_v1/new-server-04/：

- io-uring-recheck.json、preflight-command.json、preflight.json：实际平台检查和退出码。
- source-preservation.json、calibration-source-provenance.json：固定版本、干净源码与审计源文件哈希。
- calibration-format-audit.md：v1/v2、golden、数据真实性和下一入口审计。
- calibration-preparation/cpu-tests.xml、cpu-tests.log、candidate-plan.json、README.md：32项测试、实际候选计划与命令。
- cpu-results-summary.json、final-consistency.json：本轮统计、权限/账本不变及阶段一致性。
- canonical-before/、before-execution_state.json：更新前文件保留。

本轮 ZIP 是第三轮交付的增量，需与 prefix-io-v1-new-server-03-20260926.zip 配套；该基础包 SHA256 为55a7ef22048073c4ffc386eb62eb5a855b5c95b3dc8388b5096bb4c877714698。增量包仅含本轮工具、测试、报告、状态与证据，不包含模型/虚拟环境，也没有把候选计划称为真实结果。
