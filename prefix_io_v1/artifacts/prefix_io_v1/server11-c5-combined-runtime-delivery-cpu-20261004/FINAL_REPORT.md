# C5 完整入口接线与资源预检交付（2026-10-04）

已在指定 AutoDL 服务器 `connect.westc.seetacloud.com:26909`、项目 `/root/autodl-tmp/prefix-io-v1-handoff/project` 完成本轮 CPU 工作。实现测试 23/23、独立审查 23/23、资源合同 19/19，共 **65/65 通过**，失败、错误、跳过均为 0。实际资源预检结果为 **RESOURCE_LIMITED**：GPU 设备节点为空，CPU 配额 `50000 100000`（0.5 核）、内存 2 GiB。没有启动正式测量或 GPU，不能据此宣布 P4 完成或系统性能提升。

本轮修复的是实际入口的来源接线缺口。此前正常启动器仍指向旧 C5 目录内的 C4/v6 canonical receipt，而新 C5 receipt 在新 native preparation 的 `common_candidate` 中。本轮只新增 combined runtime 副本，启动器、控制器及验证器统一指向这个新源码域，包含实际 `single_file_runtime_binding.py` 的身份辅助来源。C5 reactor、collector、policy、bridge、canonical receipt 均未修改；正常通知 helper 和 Queue 观测代码与此前 Stage A 字节完全相同，SHA-256 为 `9c51a1f11fc966739ab44d9852de33ea139e28db0fa9b5f4fae6315634f594ea`。十个原始模型调用、生命周期和数值函数 AST 保留，没有重写缓存引擎或执行器。

新目录均在项目 `artifacts/prefix_io_v1/` 下：`server11-c5-combined-runtime-cpu-20261004`、`server11-c5-combined-runtime-review-cpu-20261004`、`server11-c5-combined-runtime-resource-cpu-20261004`、`server11-c5-combined-runtime-delivery-cpu-20261004`。本机交付对应 `artifacts/prefix_io_v1_server11_candidates/notification_v5_combined_runtime_*`。GPU UUID、权限文件、有效绑定、存储模板和 GPU 锁未解决，保持 None。阻断的旧 receipt/config/scope 创建体已移除；没有创建新的 GPU 配置、作业、授权、凭据或预算预留。

实际 CPU 检查执行 C5 `collector.install(origin=cpu_fixture)`，使用明确标记的模拟 worker、connector、event，保留实际四个绑定 invalidation 方法、原 prepare 调用及恢复。off/shadow 执行正常 helper 与精确类型 `queue.Queue`；原始 `_drain_incoming`、`_intake` 的源码执行覆盖旧 wake 和 STOP 的处理。CPU fixture 不是真实 vLLM/model/CUDA/native I/O。

on 的正向原生附着仍未验证。真实桥接需要有效 native typed receipt、实际 runner identity、原生 Event 来源和 capture 身份；CPU capture 正确返回未知状态，原 helper 和 reactor 正确拒绝不满足条件的 on 安装。测试验证的是这些拒绝，没有伪造 typed receipt、篡改 capture.origin 或私设桥接 weakref。`actual_on_installed`、`actual_native_bridge_attachment`、`full_entry_cpu_cost_measured`、`full_runtime_cost_qualified` 和 `on_observation_cost_measured` 均为 false，实际成本、额度及 native receipt 为 null。

本轮明确修正后续依赖顺序：**CPU 资源预检 → 获授权的真实 GPU 公共六窗口校准与严格凭据 → off→shadow→on 的真实生命周期和完整入口成本 → P4 公平策略比较**。完整 on 成本测量需要真实公共凭据，不能把它作为公共凭据生成的前置条件。之前报告中“先完整入口 CPU 资格、再真实公共校准”的表达应按本轮顺序理解；资源预检只是运行条件，不是完整成本或实验资格。没有 CPU 模拟通过可以替代这个缺失证据。

资源代码读取真实 cgroup quota/stat、affinity、GPU 节点及预算账本。有限配额至少一核或合法 unlimited、且计数器和 affinity 可读，只能满足新测量的初步资源条件，不能授予任何性能资格；未知字段保守阻断。当前半核直接报告 RESOURCE_LIMITED，测量迭代为 0。合法 `max 100000` 的解析已在冻结前修复，并有反例测试。旧 paired 无节流判定和阈值仍不变，所有限流样本须保留。

旧 132 次 CPU 对照没有重跑、改写或换阈值。该基准使用 RecordedQueue 子类和手工 observer，无法直接代表实际正常 helper 的精确 Queue 与额外观测开销；旧结果不迁移成 combined entry 成本资格。原协议 SHA-256 保持 `6c00cf687f4903988a363e18a178ccca413076ced81920271bf9c962abde1218`。nr_throttled 是服务器累计计数，不解释成本轮或某个测试窗口的节流次数。

实际服务器命令均使用项目 `.venv/bin/python -B -I -S`，并设置 `CUDA_VISIBLE_DEVICES=''`：

```text
freeze_combined_runtime.py --root <project>
run_cpu_combined_runtime.py --project-root <project> --source-lock <delivery>/SOURCE_LOCK_COMBINED_RUNTIME_CPU.json --output-dir <candidate>/SERVER_CPU_01 --location server_cpu
run_review_cpu.py --project-root <project> --source-lock <delivery>/SOURCE_LOCK_COMBINED_RUNTIME_CPU.json --stage-root <candidate> --candidate-root <StageC>/common_candidate --previous-runtime-root <StageA> --output-dir <review>/SERVER_REVIEW_01 --location server_cpu
run_cpu_resource.py --project-root <project> --source-lock <delivery>/SOURCE_LOCK_COMBINED_RUNTIME_CPU.json --output-dir <resource>/SERVER_RESOURCE_TESTS_01 --location server_cpu
resource_preflight.py --project-root <project> --source-lock <delivery>/SOURCE_LOCK_COMBINED_RUNTIME_CPU.json --output-dir <resource>/SERVER_PREFLIGHT_01
seal_combined_runtime.py --root <project>
```

全路径 argv、环境、退出码、stdout/stderr 分别保存于各目录的 `SERVER_SOURCE_FREEZE_*`、`SERVER_CPU_FACTORY_*`、`SERVER_CPU_REVIEW_*`、`SERVER_CPU_RESOURCE_TESTS_*`、`SERVER_RESOURCE_PREFLIGHT_*`、`SERVER_FINAL_SEAL_*`。候选的 run/control/verify/contract 四个 CLI 另实际执行，均退出 2、stderr 为空，输出 GPU_BLOCKED。阻断发生在配置、预算、模型与 GPU 访问之前。

结果原件为 candidate 的 `SERVER_CPU_01/CPU_RESULT.json`、review 的 `SERVER_REVIEW_01/TEST_RESULT.json`、resource 的 `SERVER_RESOURCE_TESTS_01/CPU_RESOURCE_RESULT.json` 和 `SERVER_PREFLIGHT_01/RESOURCE_PREFLIGHT.json`；旁边保留完整测试日志。source lock 为 202 个来源，SHA-256 `bb7522102bddee35e5878551e98d23ae475897e24663d4c3aa9ae235329b176d`，各运行前后核验一致；独立审查包含锁文件本身，共 203 个引用。基线 952 项历史引用在冻结前及封存时保持原 SHA，基线自身也已绑定。

本机 fixture 开发失败记录如实保留：实现测试首轮缺少正常路径的 identity helper import 搜索路径；独立审查首轮模拟 EventProxy 未先执行原 record，实际 origin/type 检查仍保留。修复只补充原路径和明确模拟 record，没有改真实验证规则。review 的 `LOCAL_DEVELOPMENT_RECORD.json` 及候选 README 保存说明；这些不是失败或成功的 GPU 实验。

真实 GPU 运行 **0 次**、模型进程 0、正式性能比较 0，本轮 GPU 新增消耗 **0 秒**。预算账本原 SHA `bef6d78e43058eaad811ab5ecb08937356c911784f9000e76eea9de0276c0b72` 未变，累计 `22380.561257688794` 秒，原 8 小时预算剩余 `6419.438742311206` 秒（约 1.78 小时），无活动预留。数据盘剩余约 56.9 GiB，本轮不需要扩容、下载模型或删除实验数据。没有系统、驱动、旧源码、旧证据、原权限或研究范围修改。

下一实际实验步骤需要恢复有卡且有足够 CPU/内存的实例，先现场核验真实资源与目标 GPU UUID。之后在原累计预算内准备并绑定新的公共六窗口 GPU 修订、作业与明确授权，使用现有 Qwen2.5-7B、固定 128 输出、相同缓存/预加载/复制合并和停止协议。当前 CPU 版本始终阻断 GPU，旧 UUID、旧授权和旧成本不自动迁移。得到严格公共凭据后再做 off/shadow/on 真实验证及 on 观测成本覆盖。策略没有实际 defer 时记录 NOT_EXERCISED，禁止通过改负载、额度、成本或选样制造收益。

本轮结论是来源接线、CPU 接口和资源阻断正确。C5 GPU 收益、完整异步系统最终可行性和论文效果仍没有验证；此前负结果和受限 CPU 结果全部保留。
