# 无卡可行性审计最终裁决

本轮审计已完成。裁决是 **停止当前性能策略的付费 GPU 投入，保留可运行系统、共同修复和全部证据；暂不进入 P5**。当前 active-poll I 在已测条件没有净收益，也没有成立的新性能候选。这不声称全部 D/J 在所有负载上不可行。

## 实际工作与改动

在用户指定服务器无卡模式执行，0.5 核、2 GiB。新增只读日志分析器、CPU 通知审计、源/账本核验和本报告。冻结 C4、作者代码、策略公式、权限、旧数据和 GPU 账本均未改写。关闭新策略仍使用已有 off 路径及统一共同修复。

- 8/8 原通知/原队列/失效场景 CPU 用例通过。
- 181 次合成元数据重试保持 preview/defer/reserve/release=181、reuse=180、collect=1；未预测真实 GPU 时间。
- 75 项真实 v4 证据一致性复算通过，22 个原文件逐个匹配既有交付清单 bytes/SHA。
- 强 U 两次运行的 8 份 result/analysis/config/trace 匹配历史 SHA，直接在服务器重算。
- 前后 9 份指定源/锁/权限/证据文件 SHA 相同，GPU 账本未变化，服务器 tracked Git clean；并非重新核验全部源锁。

## 当前 I 策略

| 模式 | 原真实请求+排空 s | 选定 CUDA Event 区间 ms | 原成本覆盖 |
|---|---:|---:|---:|---|
| off | 1.678503 | 15.546 | 通过 |
| shadow | 1.677100 | 15.636 | 通过 |
| on | 1.800676 | 68.464 | 失败 |

这些是复算先前完成的 GPU 结果，本轮未重跑。on 比 off 慢 7.279%，比 shadow 慢 7.368%；请求加排空额外 122.173/123.576 ms。181 次真实延期、180 次复用，首末延期跨度 64.899 ms，受控步骤 68.464 ms。跨度、步骤与总差额范围重叠，不能相加。原生 v6 三对 B−A 为 5.839712/2.497472/3.552992 ms，不是可保证回收的收益或一般上界。

源码已定位：延期 ready 留在队头，使 _has_poll_work 为真，原 reactor 保持非阻塞 intake、原 pump 与 sleep(0)。普通入队、mandatory、STOP 可唤醒原 Queue；decode step-end/live 失效和 scheduled-load 标量更新未向该 Queue 发通知。AIO eventfd/condition 属于内部，pending CUDA 完成仍靠 query。忽略 ready 后直接使用原 0.5 秒等待有停顿风险；不存在可直接配置的安全停车方案。

既有 CPU profile 只覆盖 155 次合成 ready-drain，没有完整 pump、decode 主线程及两线程调度。GIL、主机提交间隙或 kernel 因果比例尚未测出。窄通知修复可能在原范围内，但尚未证明安全、可用或有收益，不能立即变成开卡候选。

## 强 U 真实剩余机会

| 历史 U 运行 | pending flush ms | 首目标 D2H host enqueue 之前 ms | wait/cohort |
|---|---:|---:|---:|
| server08-p3-16-mixed-01-off | 408.055 | 370.058 | 1.239% |
| server08-p3-16-mixed-06-off | 641.355 | 574.030 | 1.991% |

约九成等待在首个目标 D2H host enqueue 之前，期间后台大 parent 的 D2H 实际推进。但两次 store_readiness_probe 均关闭，缺少源 GPU 完成到 ready 连续链、瞬时 staging/copy 容量与 dispatch 拒绝原因，不能把九成判为可消除排队。时间戳是主机提交和 CUDA Event 时长，不能冒作设备绝对 start/end。

每次都是 c2-5 恢复目的地保护。59 个退休样本块仍 active_refs=1、未进 free queue；source/destination generation 已变化。解除保护允许原恢复继续，不等于新增 GPU 空闲容量。physical freed blocks 保持 unknown，release credit=false。

等待跨前端步骤且期间无 token 输出，位于请求推进路径。c2-5 比最后请求早 7.4/8.2 秒完成，不能将等待全部换算成 cohort 或吞吐收益；1.24%/1.99% 比例也不是可省时间上界。历史 U 早于 C4 共同空闲修复，不能预测当前底座等待仍有相同时长。

## 阶段、论文结论和下一允许动作

真实模型输出、原生 I/O、有限动作与排空可行性此前已有真实证据。当前研究策略提升未获验证；完整 P4、D/J 真释放闭环、正常准入下的研究效果和 P5 正式 SLO-goodput 均未通过。现在不足以支撑性能方法论文，不能承诺未来一定达到论文要求。

本轮审计已经闭合。保留 off、共同修复、代码及全部原始结果，停止当前策略的 GPU 重跑与调参。只有将来出现具体、范围内且安全的新候选和可控目标证据，才重新制定 GPU 对照。窄通知、181 次完整双线程诊断、目的地保护 ready 观测属于未来可选工作，不是本次未完成项。无需租卡等这轮审计。

## 实际命令与证据

以下命令已在服务器执行；环境、完整参数、stdout/stderr/退出码保存在同目录 *_COMMAND.json、*_RESULT.json 和日志。新运行隐藏 CUDA，未导入模型/CUDA，未安装下载或更改系统。

~~~sh
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-feasibility-audit-v1-20261003/audit_contract_v2.py --root /root/autodl-tmp/prefix-io-v1-handoff/project --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-feasibility-audit-v1-20261003/CPU_CONTRACT_BEFORE_V2.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-feasibility-audit-v1-20261003/hotpath_review/audit_notifications_cpu.py --candidate-root artifacts/prefix_io_v1/server11-p4-single-file-candidate-v4-20261003 --author-root /root/autodl-tmp/prefix-io-v1-handoff/project --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-feasibility-audit-v1-20261003/CPU_NOTIFICATIONS.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-feasibility-audit-v1-20261003/runtime_review/collect_project_evidence.py --root /root/autodl-tmp/prefix-io-v1-handoff/project --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-feasibility-audit-v1-20261003/CPU_RUNTIME_PROJECT_SELECTED_MEMBERS.json --prior-archive-verification /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-delivery-v6-20261003/LOCAL_ARCHIVE_VERIFICATION.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-feasibility-audit-v1-20261003/runtime_review/analyze_runtime_evidence.py --evidence /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-feasibility-audit-v1-20261003/CPU_RUNTIME_PROJECT_SELECTED_MEMBERS.json --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-feasibility-audit-v1-20261003/CPU_RUNTIME_REANALYSIS.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-feasibility-audit-v1-20261003/opportunity_review/audit_strong_u.py --workspace-root /root/autodl-tmp/prefix-io-v1-handoff/project --project-raw --reference-dir /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-p3-16 --checkpoint-manifest /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-p3-16/server08-p3-16-blocked-checkpoint-evidence-v3-manifest.json --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-feasibility-audit-v1-20261003/CPU_STRONG_U_REANALYSIS.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-feasibility-audit-v1-20261003/audit_contract_v2.py --root /root/autodl-tmp/prefix-io-v1-handoff/project --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-feasibility-audit-v1-20261003/CPU_CONTRACT_AFTER.json
~~~

一次初始工具核验错写 runner 文件名而失败，记录保留；新增 audit_contract_v2.py 后前后核验成功。一次半核环境的大归档流式扫描过慢且 SSH 中断，未生成完整 bundle/result，未记为通过；部分 stdout 保留。随后直接读原 22 文件并逐个匹配冻结清单完成复算，没有修改实验数据或扩大配额。

核心证据：CPU_RUNTIME_REANALYSIS.json、CPU_STRONG_U_REANALYSIS.json、CPU_NOTIFICATIONS.json、CPU_CONTRACT_BEFORE_V2.json、CPU_CONTRACT_AFTER.json、FEASIBILITY_DECISION.json。独立本地复核见 hotpath_review/HOTPATH_REVIEW.md、runtime_review/REVIEW.md、opportunity_review/OPPORTUNITY_REVIEW.md。本地复核不替代服务器执行记录。

本轮真实 GPU 操作 **0**、新预算消耗 **0 秒**。原 8 小时累计 22380.561258 秒，剩余 6419.438742 秒（约 1.78 小时），active=null。最后元数据：GPU 节点 []，数据盘可用 56.89 GiB，本轮无需扩容，未删除数据。
