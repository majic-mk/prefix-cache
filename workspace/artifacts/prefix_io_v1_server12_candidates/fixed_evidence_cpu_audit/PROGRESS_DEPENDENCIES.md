# 冻结源码的推进依赖 CPU 审计

当前单文件候选不满足普通成本门，并不等于原生 I/O 永远无法执行。原策略在成本判断之前保留 native progress 和原始工作年龄达到 max_wait 的推进分支；原 reactor 在 defer 后仍持有原 ready FD，释放本次临时 staging 预留，并在原 Queue 唤醒或有限期限到达后回到原调度循环。

本审计只读取实际归档映射中的冻结源码，以 SHA 验证字节；执行原有纯 CPU 值类型、P4Policy、NativeP4Bridge、DispatchController 和 dependencies.analyze。没有导入 reactor、Torch、vLLM 或 py-kvcache，没有创建/伪造 ExactSingleFileReceipt、CUDA Event、缓存引擎、原生资源、权限或新策略。CPU 值 fixture 的 acceptance 是记账回放，不是实际后端提交。该结果不能代替真实 on GPU 活性验证。

## 实际路径

源码前缀 G 为 `artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate/`。控制模块位于 G 下的 `source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/`，reactor 位于 G 下的 `source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py`。JSON 结果逐项保存完整项目相对路径、SHA、原函数行号和 AST SHA。

| 模式或接口 | 实际连接及限制 |
|---|---|
| 窄 single-file interference | reactor.py:278–282 要求 dispatch_controller 为 None；P4 只增加 defer，issue/fallback 继续原生 U 额度与安全检查。没有新 DispatchLedger 生产 permit 接线。 |
| dependency_only | 同一初始化分支要求原 DispatchController 且 mode=fixed；P4 给有限排序建议，原 controller 消费已有性能额度。不能用窄 single-file 条件推断 dependency_only 不可行。 |
| DispatchLedger | dispatch_budget.py 中提供独立 CPU 合同；不能将其类存在或 reserve 测试当作已连接原生入口。 |
| 原生资源释放 | dependencies.analyze 只有完整父依赖、引用/代际和 native_reusable 见证才能计算立即复用；父 Future 成功或 CPU completion 均不够。 |

`p4_policy.py:275` 的 issue_preview 先检查当前 run/epoch、样本新鲜度、native_ready_work、原生已接受但未提交的工作身份及代际。通过这些前提后，`:296` 对 native progress 或 `now_ns - work.created_ns >= max_wait_ns` 返回 issue/native_progress_override；`:317` 才比较完整成本上界和完整 A-only 预算。兜底不把过期或错误身份升级成合法工作。

`p4_bridge.py:391` 的 preview_issue 调用原 policy；shadow 中建议 defer 转回 native_fallback。live step 在预览期间结束也转回原路径，记录真实 deferral 前再次核对同一 live state。桥接器不持有工作队列、资源所有权或累计额度。

`reactor.py:2139` 的 _prefix_stage_decide 根据原 _stop、原 downstream continuation、原 mandatory Future、原 mandatory waiter support 设置 progress。仅在 preview.action=defer 且 progress=None 时返回 _P4Deferred；controller=None 时 issue/fallback 返回 None，让原始提交继续。普通 preload 的年龄来自原 ready.open_start_ns，重复预览不会把年龄重置。

`reactor.py:2465` 将真实 mandatory preload waiter 转为 mandatory_support。`:3129`、`:3198`、`:3235` 的原 ready 队列 defer 分支释放已预留的 slot 并返回，尚未 popleft；已接受原 FD/工作没有转交新队列。后续 CQE/DMA/收尾仍属原 reactor。

`reactor.py:2270` 的 retry metadata reuse 仅允许一个窄 preload：没有 accepted parents、其他 active/AIO/copy work、mandatory waiter、stop、drain unknown 或冲突 scheduled-load publication；原 accounting、容量、live state、epoch、时间必须一致。`:2355` 的等待期限是原 open_start+max_wait 与各个样本新鲜度期限的最小值，不能延长。`:2385` 释放 submit lock 后调用原 _incoming.get 的有限 timeout，唤醒后返回原 _intake；`:1065` 随后回到 _pump_once。`:1457` 原泵依次处理 CUDA completion、CQE、原调度、复制合并、ring submit 和 job finish。

这避免将等待问题仅交给一个可能迟到的观察通知：原 Queue 的 STOP/new job/mandatory 消息和 timeout 都是独立返回通道。该结论依赖 owner 线程继续获调度、原生容量可用和后端最终完成，不是无限外部故障下的无条件活性保证。

## 本机实际执行

```text
Python 3.12 -B -I -S audit_progress_dependencies_cpu.py --project-root <workspace> --output LOCAL_PROGRESS_DEPENDENCIES_CPU_RESULT.json
```

实际 exit=0：15 个冻结源/测试文件 before/after 字节不变，21 项静态合同检查通过，14 项原函数 CPU 值回放通过，forbidden_imports=[]。原源码在新的独立模块命名空间中完整编译执行，未改 AST、globals 或函数，也未导入/执行旧 reactor。原测试只读取其源码覆盖点，未把测试 fixture 当作 GPU 证据。

CPU 回放证据包含：原 bridge 的 mandatory、mandatory_support、continuation、shutdown 及不可重置的 age override；过期快照仍 fallback；原 controller 的零性能额度普通 defer、等待年龄后 issue、一次 acceptance 消耗一次额度；即使 mandatory，native_issue_safe=False 仍 defer；父已完整成功但没有原生释放见证时立即复用为零；缺父闭包和失败 Future 尚有 inflight 时也不计算安全释放。该零额度设置只是明确的 CPU 合同 fixture，不是新的服务器额度或候选提案。

已读原测试包含 test_single_file_policy.py、test_reactor_single_file.py、test_single_file_retry_observation.py、test_single_file_wait.py；其 progress/age、期限、提前/迟到/重复 wake、原 queue、mandatory waiter 等测试名称和行号保存在结果。已有测试中的 synthetic event 或 test-only typed receipt 没有在本审计中重新构造或运行。

服务器兼容命令（由根代理执行；本文件生成时没有连接服务器）：

```text
.venv/bin/python -B -I -S artifacts/prefix_io_v1/server12-c5-fixed-evidence-cpu-audit-20261004/audit_progress_dependencies_cpu.py --project-root /root/autodl-tmp/prefix-io-v1-handoff/project --output artifacts/prefix_io_v1/server12-c5-fixed-evidence-cpu-audit-20261004/SERVER_PROGRESS_DEPENDENCIES_CPU_RESULT.json
```

输出使用 xb，已有结果不会被覆盖。服务器实际项目文件优先，存在但 SHA 错误时硬失败；本机仅在项目路径缺失时使用 SHA 固定的真实归档映射，逐源再次核验字节。不会使用副本错误后回退至另一个“碰巧通过”的副本。

## 仍不能证明的事项与下一步

本轮三次实际 off 的 bridge=None，证明原执行、输出、记账和生命周期，不能证明 live on 的成本 defer/原 bridge/notification/最大等待期限已在 GPU 上工作。普通 cost 上界迁移失败仍保留。当前 U=16,238,752 ns 大于 A-only=13,171,328 ns，普通候选成本门不满足；本审计不更改两者，也不将推进兜底解释为性能收益。

静态和 CPU 证据未发现由可选成本 defer 形成的必然循环等待；等待只在没有其他原生待完成工作的窄状态安装，并且临时 staging 已释放。但以下仍是活性前提：原 staging/FD/设备额度可获得，原 mandatory/continuation 标记传播正确，原 CQE/CUDA fence 最终完成，owner 线程继续执行，原 shutdown/drain 安全接口返回。性能 override 不能越过这些安全依赖；未知或永久不可用的安全事实不应被强行绕过。

现有 native publication 明确没有 GPU owner generation/reference/protector release capability；dependencies.py 的 owner-published 注释也明确 CPU fixture 不构成 live GPU 释放适配。当前固定 request 没有可证明的 blocked owner-release witness，不能据此声称 dependency_only 排序有收益，也不能据此否定该策略。

下一最小 CPU 步骤是把原 owner 发布接口、mandatory/support 传播、失败/收尾与 wait wake 的已有源码/测试证据逐项对应，确认哪些事实已真实观察、哪些仍不可用；保持原成本上界、A-only 预算、候选范围及全部反例，不重新拟合、不挑选样本、不新增 GPU 运行。将来真实 on 的验证必须独立满足原正常资格/源/设备/账本门，并保留完整输出、原 shutdown 与 session drain，不能以这份 CPU 通过代替。
