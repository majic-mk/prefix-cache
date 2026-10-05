# 固定证据 CPU 审计交付与效果阻塞判定（2026-10-04）

目前没有完成方法效果的 off/on 对照，不能判方法无效。已确认的直接阻塞是成本标定覆盖与预算构造不匹配，以及启用策略后的生命周期/释放观测验证不完整。本轮未发现新的原模型或缓存运行错误，也没有证明论文性能提升。

此前三次真实 off 均完成完整输出、真实 SSD 读取与原 shutdown/会话排空；新策略桥接为 None。原运行成功和本轮 CPU 审计通过，不等同于真实 on、依赖排序或 goodput 收益已通过。

## 三类问题分别是什么

**实验资格与设计（已确认）**：固定成本上界 16.238752 ms 在三次重复中两次被超过。预算 13.171328 ms 来自两个校准 A 步的最大值，并非开发集独立声明的服务 SLO。baseline=13.113008 ms，实际剩余额度仅 0.058320 ms；新增动作增量加不确定性为 3.125744 ms，预算缺口 3.067424 ms。这个预算可作为严格工程资格约束，但没有依据把它视为正式服务 SLO 或直接用于证明收益。冻结收据不变时，重跑同配置不会改变普通门的 defer。旧失败和全部三次结果保留；本轮没有重新拟合、放宽阈值、选通过的种子或增加候选。

**代码集成与验证（具体边界）**：单文件 interference 的原初始化明确要求 DispatchController=None，issue/fallback 回原 U；这不是必须强行接入新累计额度控制器的代码错误。源码与原函数 CPU 回放证明进度/等待期限分支仍存在，且不能绕过 native safety、容量或事件依赖；没有发现可选成本 defer 必然制造循环等待。但这些分支尚未在真实 on 中验证，当前原生发布缺少可用于收益论证的 GPU owner generation/refs/protectors 释放适配，当前请求也没有可核验的 blocked owner-release witness。完整策略尚缺真实观测与集成验收。

**方法有效性（仍未知）**：三次 off 无法证明 I/D/J 相对强基线的收益，更无法推断方法永久不可行。当前单文件候选在当前冻结条件下不满足普通干扰准入，且现有请求未展示依赖排序所需的正常资源等待问题。若完成正当开发预算与观测后仍没有可行候选/资源等待目标，应停止这个 pilot 并报告无当前机会；不能靠人为限速、改变测试门槛或关闭基线流水线制造优势。

## 本轮实际新增内容

仅在新审核目录新增独立成本证据审计、启动证据审计、原推进依赖回放、源字节保护、修正顺序/停止条件及有界归档工具；没有修改已冻结缓存引擎、模型执行、准入数学、策略源码、原事件采集器、旧数据或权限预算。

服务器目录：`/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-c5-fixed-evidence-cpu-audit-20261004`。
本地目录：`C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server12_candidates/fixed_evidence_cpu_audit`。
原工作区：`/root/autodl-tmp/prefix-io-v1-handoff/project`，HEAD cc7898b1ba59d89ce7fdbb186ded21880f1adf08。
持续 GPU 授权保留，原累计上限仍为 28,800 s，无需逐轮再请求用户授权；本次审计计划 GPU 作业数为 0。

## 实际 CPU 执行与结果

1. 成本审计实际读取并核验 68 项来源，前后 SHA 一致，核对原六条校准、三条新重复和单独旧 off 反例，共 10 个实际窗口、1,280 帧及 1,280 个 CUDA 事件见证。服务器 Event 源 streams.py 实际字节也通过核验，未导入 Torch。只有 offset16/context144 属于冻结 cost cell；其余上下文不同，不把 128 个 offset 当成额外拟合样本。raw elapsed 空占位和真实 witness 逐 ordinal 对应，无 GPU-host 绝对时钟映射或释放额度推断。
2. 新增成本证据拒绝/边界测试实际服务器 19/19 通过、0 skip。测试覆盖错误 ordinal、输出改变、上下文/偏移改变、空缺 frame、事件 enclosure、绝对时钟伪声明、重复 JSON key、SHA 漂移、路径与符号链接等。
3. 推进依赖审计实际服务器 15 份 source pins 一致，21 项静态检查及 14 项原函数 CPU 值回放通过。没有模型、真实 GPU Event、FD、staging pool 或新缓存引擎；没有伪造 typed receipt。CPU 值回放不等于 on GPU 活性已验收。
4. 启动审计实际服务器通过 28 项证据引用，核对每次 1 次公开 loader 与 7 次内部元数据重验。约 24 s 完整入口包含约 19.6 s 公开重放，约 85 s LLM 初始化为最大单独计时阶段；两者不能相加重算。每个真实 process.log 有 3 条 inference JIT warning，但日志没有 request/step/frame 对应，不能声称 selected16 的超界由 JIT 引起。温度、GPU 时钟、SSD/page cache 和未观测其他负载保持未知。18 项本地拒绝检查作为工具检查单独报告。
5. 源保护按原已审阅字节验证器执行完整 4,816 项、16,064,147,630 bytes 的实际服务器 SHA 核验，BEFORE 耗时约 92.79 s，AFTER 耗时约 93.84 s，两次实际 PASS，全部源与原 ledger 字节一致。最终归档还要求明确的 AFTER PASS、行数/phase/源锁检查。
6. 本轮归档和实际下载字节核验另有 CPU_AUDIT_DELIVERY_RESULT.json / LOCAL_CPU_AUDIT_BYTE_VERIFICATION.json 回执；本报告生成时未预先声称归档完成。归档不打包模型/SDK，不删除服务器任何旧数据。

## 命令及保存的失败

工作目录为 ROOT=`/root/autodl-tmp/prefix-io-v1-handoff/project`，P=ROOT/.venv/bin/python，D 为本审核目录。CPU 外层统一 CUDA_VISIBLE_DEVICES=''。实际 COMMAND JSON 保存完整绝对路径数组。

```text
P -B -I -S D/run_cpu_source_protection.py --project-root ROOT --phase BEFORE
P -B -I -S D/audit_progress_dependencies_cpu.py --project-root ROOT --output D/PROGRESS_DEPENDENCY_AUDIT.json
P -B -I -S D/audit_startup_evidence_cpu.py --project-root ROOT --output D/STARTUP_EVIDENCE_AUDIT.json
P -B -I D/analyze_cost_evidence_cpu.py --project-root ROOT --output D/COST_EVIDENCE_AUDIT.json
P -B -I -S -m unittest discover -s D -p test_cost_evidence_cpu.py -v
P -B -I -S D/run_cpu_source_protection.py --project-root ROOT --phase AFTER
```

主要实际日志：CPU_SOURCE_PROTECTION_BEFORE_01_*、CPU_PROGRESS_DEPENDENCIES_01_*、CPU_STARTUP_EVIDENCE_01_*、CPU_COST_EVIDENCE_02_*、CPU_COST_AUDIT_TESTS_01_*、CPU_SOURCE_PROTECTION_AFTER_01_*。本轮不是重复运行上一轮 79 个测试；是新增审计及上述实际回放/检查。

本机成本审计早期版本拒绝了签名误差字段解析、parent 校准 augmentation 与未打入本地归档的 streams.py 等条件；旧结果保留，本地最终 V2 通过。服务器第一次成本 CLI 因上传报文单文件压缩后超过 85KB 上限而未收到脚本，实际 exit=2，日志 CPU_COST_EVIDENCE_01_* 保留；改为只上传必要小源文件后，CPU_COST_EVIDENCE_02 实际 exit=0。这些均是本次审计工具/传输问题，没有产生额外 GPU 作业，没有修改原规范 SHA 来绕过漂移检查。

有界归档器独立审阅建议增加明确的 BEFORE/AFTER PASS、phase、4816 行和冻结 SHA 检查；补为 audit_delivery_bytes_v2.py，旧未执行版本保留。修复只影响本次私有证据打包门，不改生产运行。

## 因果边界及下一修正顺序

校准三族 first token 18100/19100/20100、seed1829/1830/1831；新族28100/2829。公开成本条件不包含 prompt、seed、selected ordinal 或 workload hash；这些是未绑定/变化的字段，不代表它们一定造成超界。实际 selected load、非存储引擎配置和完整输出 token ID hash 已核对一致，包装入口版本不同。没有匹配的普通负载 A/B 因果对照，不能将超界归因于 I/O、提示词、JIT、服务器或代码优化。

下一允许 CPU 工作按以下顺序推进：

- 定位真实 allocator/handler 已有 owner/refcount/generation/protector 与 parent 保护数据来源，补充有界只读释放观察；未知保持未知，不增加第二套所有权或提前 complete_store。
- 对每个已允许模式验证原 bridge/continuation 及物理限制，遵守模式原接口，不把设计要求的 controller=None 当成接线 bug 强行改掉。
- 在新数据产生前形成独立 calibration/development/evaluation 协议，冻结负载身份、真实 token 时间、热身对应、开发预算和正式服务 SLO；禁止因旧失败而事后改 U/budget 或利用 heldout 调参。
- 先确认正常负载中存在真实、可重复的资源等待与合法释放闭包，再做策略 shadow、小规模 on 活性验证及同执行器收益对照。没有观察目标就如实报告无当前依赖排序机会。

具体条件/停止规则已写 CPU_NEXT_PHASE_DECISION.json。本轮判定 NO_GO_UNCHANGED_SINGLE_FILE_EFFECT_EXPERIMENT，仅针对当前冻结单文件收益路线，不关闭全部 V1 方法可能性。没有任何新 GPU 作业，shadow/on/P4收益资格仍未打开；不能拿本次完整证据审计 PASS 替代它们。

GPU ledger 保持已用 24,678.895239 s、剩余 4,121.104761 s（约68.69分钟），没有新增 GPU 记账；AutoDL 实例开机费用另计。原代码、模型和后续实验数据保留；本阶段只需 CPU，可切无卡模式。证据补充共少量 MB，不需要因本审核扩容。

