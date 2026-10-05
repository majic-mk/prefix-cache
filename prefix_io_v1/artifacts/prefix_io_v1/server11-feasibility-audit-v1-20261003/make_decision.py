"""Write an evidence-bound decision after the completed server CPU audits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit-dir", required=True)
    a = p.parse_args()
    b = Path(a.audit_dir).resolve(strict=True)
    require(b.name == "server11-feasibility-audit-v1-20261003", "new audit only")
    refs = []
    def read(name):
        raw = (b/name).read_bytes()
        refs.append(dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        return json.loads(raw)
    before = read("CPU_CONTRACT_BEFORE_V2.json")
    after = read("CPU_CONTRACT_AFTER.json")
    notify = read("CPU_NOTIFICATIONS.json")
    runtime = read("CPU_RUNTIME_REANALYSIS.json")
    opportunity = read("CPU_STRONG_U_REANALYSIS.json")
    require(before["gpu_ledger"] == after["gpu_ledger"], "unchanged GPU ledger")
    require(before["reference_checks"] == after["reference_checks"], "nine references unchanged")
    require(all(x["new_gpu_seconds"] == 0 and x["gpu_ledger"]["active_reservation"] is None
                for x in (before, after)), "CPU-only and no active reservation")
    require(notify["synthetic_cpu_success"] and notify["synthetic_cpu_test_count"] == 8, "8 CPU tests")
    require(notify["synthetic_181_retry_facts"]["decisions"] == 181, "181 synthetic decisions")
    require(runtime["check_count"] == len(runtime["checks"]) == 75, "75 consistency checks")
    require(runtime["retry"]["actual_attempts"] == 181 and
            runtime["retry"]["actual_observation_reuses"] == 180, "181 real attempts, 180 reuses")
    require(not runtime["retry"]["per_attempt_actual_cpu_durations_available"], "no invented CPU costs")
    arms = runtime["arms"]
    require(all(v["limited_qualified"] for v in arms.values()) and not arms["on"]["cost_covered"],
            "finite lifecycle qualified, on cost failed")
    require(all(runtime["delta_decomposition"][k]["total_request_and_drain_ns"] > 0
                for k in ("on_minus_off", "on_minus_shadow")), "on slower than both controls")
    runs = opportunity["strong_U_runs"]
    require(len(runs) == 2 and all(not r["GPU_release_credit"] and
            r["physical_GPU_blocks_freed"] is None for r in runs), "unknown physical release")
    require(not opportunity["paid_GPU_restart_justified_by_this_audit_alone"], "no speculative restart")
    tags = ("CPU_CONTRACT_BEFORE_V2", "CPU_NOTIFICATIONS", "CPU_RUNTIME_PROJECT_COLLECT",
            "CPU_RUNTIME_REANALYZE", "CPU_STRONG_U", "CPU_CONTRACT_AFTER")
    commands = []
    for tag in tags:
        c = read(tag+"_COMMAND.json")
        result = read(tag+"_RESULT.json")
        require(result["exit"] == 0, "completed server command: " + tag)
        commands.append("CUDA_VISIBLE_DEVICES='' "+shlex.join(c["command"]))
    decision = dict(
        status="CPU_FEASIBILITY_AUDIT_COMPLETE_STOP_CURRENT_PERFORMANCE_INVESTMENT",
        created_utc=datetime.now(timezone.utc).isoformat(),
        decision=dict(system_functional_feasibility_previously_verified=True,
            current_measured_active_poll_I_net_gain=False, new_performance_candidate_established=False,
            resume_paid_GPU_now=False, complete_P4_verified=False, enter_P5_now=False,
            D_J_universally_falsified=False, paper_performance_evidence_established=False,
            required_follow_up_to_complete_this_audit=False),
        actual_changes=["Read-only source/state receipt, real-log analysis, source-extracted CPU notification tests and decision.",
                        "No frozen C4, author tree, policy, permission, GPU ledger or historical evidence changed."],
        server_CPU_results=dict(notification_cases_passed=8, synthetic_metadata_retry_attempts=181,
            real_runtime_consistency_checks_passed=75, historical_U_raw_hash_checks_passed=8,
            named_reference_checks_before_after=9, counts_not_added_as_unique_unit_tests=True),
        original_V4_reanalysis=dict(arms={k:{f:v[f] for f in (
            "whole_request_ns","total_request_and_drain_ns","selected_gpu_event_span_ns",
            "full_output_tokens","cost_covered","limited_qualified")} for k,v in arms.items()},
            retry=runtime["retry"], delta_decomposition=runtime["delta_decomposition"],
            native_v6_pairs=runtime["v6_pairs"],
            paired_B_minus_A_not_guaranteed_recoverable_gain_or_universal_upper=True),
        strong_U_remaining_opportunity=[dict(label=r["label"],
            cohort_seconds=r["cohort_seconds"], pending_flush_seconds=r["pending_flush_seconds"],
            pending_flush_over_cohort_pct=r["pending_flush_over_cohort_pct"],
            first_target_D2H_host_enqueue_after_wait_seconds=[
                f["before_first_D2H_host_enqueue_seconds"] for f in r["fences"]],
            target_finished_before_last_request_seconds=[
                f["target_finished_before_last_request_seconds"] for f in r["fences"]],
            target_eligible_earlier=[f["readiness_evidence"]["target_eligible_earlier"] for f in r["fences"]],
            store_readiness_probe_enabled=[f["readiness_evidence"]["store_readiness_probe_enabled"] for f in r["fences"]],
            sampled_retired_blocks=r["sampled_retired_blocks"],
            sampled_active_ref_histogram=r["sampled_active_ref_histogram"],
            sampled_free_queue_linked_count=r["sampled_free_queue_linked_count"],
            physical_GPU_blocks_freed=None, GPU_release_credit=False) for r in runs],
        causal_limits=[
            "Real retry window is wall time; prior 155-synthetic-retry profile omits full pump and concurrent decode.",
            "CPU tests prove notification semantics, not real GPU gains or GIL causal fractions.",
            "Before-D2H interval lacks source-completion/ready/capacity/deny-reason chain.",
            "Destination protection clearing is not new GPU free capacity while active references remain.",
            "Historical U predates C4 common fix; wait fractions are not removable-latency bounds.",
            "No normal-admission performance, formal predeclared SLO, held-out repeated gain or novelty proof."],
        next_allowed_action="Preserve off/common fixes and evidence; stop current GPU repeat/tuning. A future scope-preserving candidate needs controllable ready-work evidence and safe progress before a frozen GPU protocol. This audit has no remaining required work.",
        optional_future_candidate_gates=[
            "Bounded run/step notification with no lost wakeup, original max-wait and mandatory/fault fallback.",
            "For D/J: actual owner/generation, completion, reuse and mandatory progress; continuous ready/capacity reasons.",
            "Strong comparable baseline, normal cost admission, full request/drain accounting and predeclared balanced repeats."],
        new_GPU_operations=0, new_GPU_seconds=0, GPU_ledger=after["gpu_ledger"],
        server_final_metadata=after["server"], source_refs=refs)
    with (b/"FEASIBILITY_DECISION.json").open("x",encoding="utf-8") as f:
        json.dump(decision,f,ensure_ascii=False,indent=2);f.write("\n")
    rows = "\n".join("| %s | %.6f | %.3f | %s |" % (
        k,v["total_request_and_drain_ns"]/1e9,v["selected_gpu_event_span_ns"]/1e6,
        "通过" if v["cost_covered"] else "失败") for k,v in arms.items())
    urows = "\n".join("| %s | %.3f | %.3f | %.3f%% |" % (
        r["label"],r["pending_flush_seconds"]*1e3,
        r["fences"][0]["before_first_D2H_host_enqueue_seconds"]*1e3,
        r["pending_flush_over_cohort_pct"]) for r in runs)
    report = """# 无卡可行性审计最终裁决

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
""" + rows + """

这些是复算先前完成的 GPU 结果，本轮未重跑。on 比 off 慢 7.279%，比 shadow 慢 7.368%；请求加排空额外 122.173/123.576 ms。181 次真实延期、180 次复用，首末延期跨度 64.899 ms，受控步骤 68.464 ms。跨度、步骤与总差额范围重叠，不能相加。原生 v6 三对 B−A 为 5.839712/2.497472/3.552992 ms，不是可保证回收的收益或一般上界。

源码已定位：延期 ready 留在队头，使 _has_poll_work 为真，原 reactor 保持非阻塞 intake、原 pump 与 sleep(0)。普通入队、mandatory、STOP 可唤醒原 Queue；decode step-end/live 失效和 scheduled-load 标量更新未向该 Queue 发通知。AIO eventfd/condition 属于内部，pending CUDA 完成仍靠 query。忽略 ready 后直接使用原 0.5 秒等待有停顿风险；不存在可直接配置的安全停车方案。

既有 CPU profile 只覆盖 155 次合成 ready-drain，没有完整 pump、decode 主线程及两线程调度。GIL、主机提交间隙或 kernel 因果比例尚未测出。窄通知修复可能在原范围内，但尚未证明安全、可用或有收益，不能立即变成开卡候选。

## 强 U 真实剩余机会

| 历史 U 运行 | pending flush ms | 首目标 D2H host enqueue 之前 ms | wait/cohort |
|---|---:|---:|---:|
""" + urows + """

约九成等待在首个目标 D2H host enqueue 之前，期间后台大 parent 的 D2H 实际推进。但两次 store_readiness_probe 均关闭，缺少源 GPU 完成到 ready 连续链、瞬时 staging/copy 容量与 dispatch 拒绝原因，不能把九成判为可消除排队。时间戳是主机提交和 CUDA Event 时长，不能冒作设备绝对 start/end。

每次都是 c2-5 恢复目的地保护。59 个退休样本块仍 active_refs=1、未进 free queue；source/destination generation 已变化。解除保护允许原恢复继续，不等于新增 GPU 空闲容量。physical freed blocks 保持 unknown，release credit=false。

等待跨前端步骤且期间无 token 输出，位于请求推进路径。c2-5 比最后请求早 7.4/8.2 秒完成，不能将等待全部换算成 cohort 或吞吐收益；1.24%/1.99% 比例也不是可省时间上界。历史 U 早于 C4 共同空闲修复，不能预测当前底座等待仍有相同时长。

## 阶段、论文结论和下一允许动作

真实模型输出、原生 I/O、有限动作与排空可行性此前已有真实证据。当前研究策略提升未获验证；完整 P4、D/J 真释放闭环、正常准入下的研究效果和 P5 正式 SLO-goodput 均未通过。现在不足以支撑性能方法论文，不能承诺未来一定达到论文要求。

本轮审计已经闭合。保留 off、共同修复、代码及全部原始结果，停止当前策略的 GPU 重跑与调参。只有将来出现具体、范围内且安全的新候选和可控目标证据，才重新制定 GPU 对照。窄通知、181 次完整双线程诊断、目的地保护 ready 观测属于未来可选工作，不是本次未完成项。无需租卡等这轮审计。

## 实际命令与证据

以下命令已在服务器执行；环境、完整参数、stdout/stderr/退出码保存在同目录 *_COMMAND.json、*_RESULT.json 和日志。新运行隐藏 CUDA，未导入模型/CUDA，未安装下载或更改系统。

~~~sh
""" + "\n".join(commands) + """
~~~

一次初始工具核验错写 runner 文件名而失败，记录保留；新增 audit_contract_v2.py 后前后核验成功。一次半核环境的大归档流式扫描过慢且 SSH 中断，未生成完整 bundle/result，未记为通过；部分 stdout 保留。随后直接读原 22 文件并逐个匹配冻结清单完成复算，没有修改实验数据或扩大配额。

核心证据：CPU_RUNTIME_REANALYSIS.json、CPU_STRONG_U_REANALYSIS.json、CPU_NOTIFICATIONS.json、CPU_CONTRACT_BEFORE_V2.json、CPU_CONTRACT_AFTER.json、FEASIBILITY_DECISION.json。独立本地复核见 hotpath_review/HOTPATH_REVIEW.md、runtime_review/REVIEW.md、opportunity_review/OPPORTUNITY_REVIEW.md。本地复核不替代服务器执行记录。
"""
    ledger=after["gpu_ledger"];meta=after["server"]
    report += "\n本轮真实 GPU 操作 **0**、新预算消耗 **0 秒**。原 8 小时累计 %.6f 秒，剩余 %.6f 秒（约 %.2f 小时），active=null。最后元数据：GPU 节点 %s，数据盘可用 %.2f GiB，本轮无需扩容，未删除数据。\n" % (
        ledger["gpu_wall_seconds"],ledger["remaining_original_8h_seconds"],
        ledger["remaining_original_8h_seconds"]/3600,meta["gpu_device_nodes"],meta["disk_free_bytes"]/1024**3)
    with (b/"FEASIBILITY_FINAL_REPORT.md").open("x",encoding="utf-8") as f:
        f.write(report)
    print(json.dumps(dict(status=decision["status"],new_GPU_seconds=0)))

if __name__ == "__main__":
    main()
