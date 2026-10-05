# C5 严格成本绑定：CPU 准备交付

本轮已在当前服务器完成新 C5 的严格源码绑定准备、原成本公式契约测试、独立审查及有限后续协议检查。**这仍是 CPU 准备，不是有效成本凭据签发、GPU 验证或 P4 完成。**本轮 GPU 与新增正式性能基准均为 0 次。

## 实际改动

新增四个 CPU 目录：`server11-c5-cost-binding-cpu-20261004`、`server11-c5-cost-binding-review-cpu-20261004`、`server11-c5-cost-protocol-cpu-20261004`、`server11-c5-cost-delivery-cpu-20261004`。准备适配器只返回私有 `C5CostBindingPreparation`；不返回 `ExactSingleFileReceipt`，有效 `cost_upper_ns`、`step_budget_ns` 和原生凭据均为空，GPU/原生/完整入口资格均为 false。

严格绑定实际 C5 reactor、collector、完整上轮启动器/观测闭包、准备适配器自身、原验证器/序列化器/数值代码，以及真实服务器的模型元数据、Event 和 runner 文件。reactor 保留 common 角色，collector 唯一保留 overlay 角色并与 plan 完全相同；重复、遗漏、漂移、错误类型或 C4 凭据均拒绝。

复用原有限六窗口、完整 128 帧和数值 AST：两校准对加一独立保留对，AB/BA/AB；原 129 prompt/128 cached/128 output、offset 16、单次 917,504 字节 SSD read、8 个 accepted parents、bridge=None。保留组不得参与拟合或抬高上界；A-only 预算不由 action/保留组数据生成。未修改 workload、公式、阈值或累计预算。

原 C5/C4、runtime_v4/native_v6、上轮准备源码和历史实验结果未改，没有重写缓存引擎、执行器或资源所有权。没有下载、删除已有数据、修改驱动/系统/配额、创建 GPU 作业、租用 GPU 或推送代码。

## 真实服务器执行与结果

项目为 `/root/autodl-tmp/prefix-io-v1-handoff/project`；命令使用该项目 `.venv/bin/python -B -I -S`，`CUDA_VISIBLE_DEVICES=''`。完整绝对路径、参数、环境、退出码及 stdout/stderr 保存在各组 `*_COMMAND.json`、`*_RESULT.json` 和日志中。

| 实际命令 | 结果 |
|---|---|
| `freeze_cost_preparation_v2.py --root <project>` | 115 个源码/元数据引用锁定；597 个旧证据引用核验相同 |
| `build_actual_source_binding.py --root <project>` | 实际服务器来源绑定通过；timing samples=0；未执行模型或物理 I/O |
| `run_cpu_cost_binding.py --candidate-root ... --preparation-root ... --native-root ... --original-estimator ... --project-root ... --source-lock ... --output-dir SERVER_CPU_01 --location server_cpu` | 39/39，无失败、错误或跳过，115 个冻结引用前后相同 |
| `run_review_cpu.py ... --source-lock-sha256 ... --output-dir SERVER_REVIEW_01 --location server_cpu` | 独立 20/20，无失败、错误或跳过；116 个实际读取文件引用前后相同（含源锁文件本身） |
| `run_cpu_protocol_v2.py ... --source-lock <protocol-v2-lock> --output-dir SERVER_PROTOCOL_V2_01` | 18/18，无失败、错误或跳过；116 个协议引用前后相同 |
| `c5_cost_binding_preparation.py` | 预期 exit 2，GPU_BLOCKED_C5_COST_BINDING_PREPARATION_ONLY，gpu_started=false |

绑定/公式输入明确为 CPU 契约示例；真实文件引用不等于执行其物理路径。未导入 torch/vLLM、未加载模型、未测新 GPU 事件。实际 source builder 读取的是服务器现有源码和元数据；从旧计划仅复用 SOURCE locators 与固定工作量常量，未复用旧成本、GPU UUID、授权或通过资格。

主绑定源锁 SHA-256：`da3c8dded24d44e34553b161021d23a5a2f7daf2257b2a06db2f5e0d33e7dd6c`（115 项）。协议修复后的独立源锁：`c268a380dd66eb6b27ceeff0147a5908d1a2b1c9bd8bff2f73602e6092ef76b8`（116 项），只增加 `run_cpu_protocol_v2.py`，原 115 项逐字节相同。39 和 20 项没有因仅协议运行器修复而重复运行。

## 保留的失败记录

首次 freezer 在检查旧历史测试的三个 64 MiB 边界 fixture 时，误套用了新源码的 10 MiB 上限，未生成锁。v2 对这些已登记旧文件按原大小分块验 SHA，没有扩大新源码或归档限制，也没有删除 fixture。

首次协议运行器把调用日志 `SERVER_CPU_COST_PROTOCOL_COMMAND.json` 当作源码，因不在源锁中于测试前退出，执行测试数为 0。v2 只选择固定六个协议文件与 v2 自身，保留所有旧源码和失败日志。两次错误均为 CPU 交付工具问题，没有运行 GPU，未伪装成成功结果。

## 本轮确定的缺口

原共同成本六窗口使用 bridge=None，所以不会安装 on-only 通知或 Queue 观测。把新文件放进源锁，不能说明它们的开销已经测量。原 migration_gate 的语义 True 也不能说明 costs 被上界覆盖。

依赖顺序不存在必须使用伪成本才能解除的环：真实 C5 共同六窗 → 通过原校准/保留组规则 → 新严格有限 common receipt → off→shadow→on 的通知实际生命周期和完整成本资格。若原公式令原 policy 直接 issue、没有真实等待，应报告 NOT_EXERCISED；不能抬高成本、调低预算、伪造 defer 或修改负载以制造通知命中。

当前冻结 C5 内的旧 receipt 仍绑定 C4/v6，真正新原生 issuance、六窗口 C5 真机 runner 和策略资格仍未实现/执行。本轮适配器有意阻断它们，不能称其已经可用于 GPU。旧合成 CPU 场景约 73.6% 降幅不包含完整新入口开销，不能折算为 GPU、TTFT、吞吐或论文收益。

## 证据、资源与下一允许阶段

核心证据在 factory 的 `SERVER_CPU_01/CPU_RESULT.json`、review 的 `SERVER_REVIEW_01/TEST_RESULT.json`、protocol 的 `SERVER_PROTOCOL_V2_01/CPU_PROTOCOL_RESULT.json`，以及 delivery 的 `ACTUAL_SOURCE_CPU_BINDING_RESULT.json`、`SERVER_FINAL_AUDIT.json`、两份源锁。封存归档仅包含本轮四目录；`SOURCE_DEPENDENCY_SNAPSHOTS` 保留 116 项源码/元数据的有界快照，不复制模型权重或私有编译缓存。

备份信息见 `SERVER_BACKUP_RECEIPT.json`，本地逐文件核验见 `LOCAL_BACKUP_VERIFICATION.json`；这些封存后回执不属于原快照，不修改原归档。本报告的本地版本和服务器版本保持同一字节。

本轮 GPU 0 次、新正式性能基准 0 次；原 GPU 账本原字节不变，累计 22,380.561257688794 秒，原 8 小时预算剩余约 6,419.44 秒（1.78 小时），没有活动预留。当前无 NVIDIA 设备，仍为半核 CPU 和 2 GiB 内存。原 132 份基准及其资源受限判定保留，未重跑、筛样或调阈。

下一允许 CPU 工作是增量接好新 C5 共同六窗的真实采集/重序列化/严格 issuance 路径，并维持 GPU 硬阻断，准备可审查的源码和有限作业参数。完整入口的 CPU 成本资格仍需足够 CPU 配额；GPU 原始校准及 off→shadow→on 须满足目标设备、明确作业授权和原预算。当前仍不用为这些 CPU 准备开卡，P4 与性能提升未证实。
