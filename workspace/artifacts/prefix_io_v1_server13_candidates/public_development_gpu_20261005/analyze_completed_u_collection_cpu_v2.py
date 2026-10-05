"""Describe completed actual U raw evidence; never qualify a cost table or gain.

This local CPU consumer requires genuine completed guard and closure receipts.
No result file is created until all actual evidence inputs have been read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys


NAMES = ("actual-request-outputs.json", "actual-original-full-step-capture.json",
         "actual-native-stage-journal.json", "strong-native-workload-result.json")
STAGES = ("ssd_read", "ssd_write", "h2d", "d2h")


def require(ok, why):
    if not ok:
        raise ValueError("COMPLETED_U_ANALYSIS_REJECTED: " + why)


def load(path):
    path = Path(path).resolve()
    require(path.is_file() and path.stat().st_size <= 32 * 1024 ** 2, "bounded actual input file")
    raw = path.read_bytes()
    return json.loads(raw), dict(local_path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def describe_ns(values):
    require(values and all(type(value) is int and value >= 0 for value in values), "actual typed duration list")
    ordered = sorted(values)
    return dict(count=len(values), min_ms=min(values)/1e6,
        p50_ms=ordered[max(0, math.ceil(.50*len(ordered))-1)]/1e6,
        p95_ms=ordered[max(0, math.ceil(.95*len(ordered))-1)]/1e6,
        max_ms=max(values)/1e6, percentile_method="nearest_rank",
        description_only=True, improvement_claim=False)


def analyze(raw_directory, *, guard_path, closure_path, prior_raw_directory=None):
    require(not any(name == "torch" or name == "vllm" or name.startswith(("torch.", "vllm.")) for name in sys.modules),
            "CPU raw analysis imports no model/framework")
    directory = Path(raw_directory).resolve()
    values, sources = {}, []
    for name in NAMES:
        values[name], source = load(directory/name)
        sources.append(source)
    frontend, capture, journal, native = [values[name] for name in NAMES]
    guard, guard_ref = load(guard_path)
    closure, closure_ref = load(closure_path)
    sources.extend((guard_ref, closure_ref))
    run_id = native["run_id"]
    require(guard.get("label") == closure.get("run_id") == run_id and
            guard.get("gpu_uuid") == closure.get("gpu_uuid") == native["gpu_uuid"] and
            guard.get("gpu_job_attempted") is True and guard.get("exit") == guard.get("child_exit") == 0 and
            guard.get("timed_out") is False and guard.get("error") is None and
            guard.get("interrupted_signal") is None and guard.get("session_drained") is True and
            guard.get("session_members_before_cleanup") == guard.get("session_members_after_cleanup") == [],
            "only actual completed and naturally drained original GPU job")
    require(closure.get("schema") == "actual_uncalibrated_original_U_mixed_complete_stream_post_guard_CPU_closure_v2" and
            closure.get("status") == "PASS_ACTUAL_U_FUNCTION_AND_COMPLETE_MIXED_STREAM_OBSERVATION_ONLY" and
            closure.get("complete_stream_observation_valid") is True and closure.get("mixed_frames_preserved") is True and
            all(closure.get(name) is False for name in ("cost_qualified", "table_issued", "ordinary_I_authorized",
                "formal_goodput_allowed", "strategy_improvement_proved", "exact_finite_cost_cells_qualified")),
            "real closed U mixed stream with no calibration/table/I/effect authority")
    for expected, actual in ((closure["completed_guard_ref"], guard_ref), (closure["actual_native_result_ref"], sources[3])):
        require(expected["bytes"] == actual["bytes"] and expected["sha256"] == actual["sha256"],
                "actual guard/native bytes are the ones replayed by completed closure")
    require(native.get("guard_reservation_id") == guard.get("reservation_id") and
            native.get("frontend") == frontend and native.get("full_original_step_capture") == capture and
            native.get("actual_native_stage_journal") == journal, "actual raw documents match closed native result")
    for name, source in zip(NAMES[:2], sources[:2]):
        key = "actual_request_outputs_ref" if name == NAMES[0] else "actual_original_full_step_capture_ref"
        expected = native[key]
        require(expected["bytes"] == source["bytes"] and expected["sha256"] == source["sha256"],
                "actual raw frontend/capture digest matches native source reference")
    require(frontend.get("planned_requests") == frontend.get("successful_requests") == 5 and
            frontend.get("failure_or_unsubmitted_requests") == 0 and frontend.get("actual_request_stream_completed") is True,
            "all actual five requests complete")
    rows = frontend["rows"]
    require(len(rows) == 5 and all(row["state"] == "COMPLETED" and len(row["output_token_ids"]) ==
            len(row["token_return_ns"]) == 128 and len(row["itl_ns"]) == 127 for row in rows),
            "complete actual640 outputs/timelines")
    require(capture.get("valid") is True and capture.get("failures") == [] and
            capture.get("pending_event_pairs") == 0 and capture.get("open_event_pair") is False and
            native.get("complete_stream_observation_valid") is True and
            native.get("full_step_evidence_qualified") is False and native.get("exact_finite_cost_cells_qualified") is False,
            "actual valid mixed stream remains distinct from exact-cell costs")
    frames, witnesses = capture["frames"], capture["event_witnesses"]
    require(len(frames) == len(witnesses) == frontend["original_engine_steps"] == closure["complete_native_stream_steps"],
            "every actual original engine step has a native frame and CUDA event witness")
    native_outputs = defaultdict(list)
    for frame in frames:
        for rid, tokens in frame["outputs"]:
            native_outputs[rid].extend(tokens)
    require(dict(native_outputs) == {row["native_request_id"]: row["output_token_ids"] for row in rows},
            "all frame tokens reconstruct every actual frontend output")
    kinds = Counter(frame["prepared"]["step_kind"] for frame in frames)
    heterogeneous = sum(frame["prepared"]["exact_cell_eligible"] is False for frame in frames)
    require(capture["heterogeneous_observation"]["heterogeneous_frame_count"] == heterogeneous ==
            native["formal_complete_native_observation"]["heterogeneous_frame_count"],
            "preserved non-exact-cell frame count agrees with closed observation")
    cuda_by_kind = {}
    for kind in kinds:
        elapsed = [witness["gpu_elapsed_ns"] for frame,witness in zip(frames,witnesses) if frame["prepared"]["step_kind"] == kind]
        cuda_by_kind[kind] = describe_ns(elapsed)
    require(journal.get("valid") is True and journal.get("lost") == 0 and journal.get("run_id") == run_id,
            "complete actual native journal")
    require(all(frame.get("clock_domain") == "monotonic_ns" for frame in journal["frames"]),
            "native journal declares the original host monotonic clock for descriptive containment")
    io = {}
    for stage in STAGES:
        events = [event for event in journal["events"] if event["stage"] == stage]
        accepted = [event for event in events if event["kind"] == "accepted"]
        completed = [event for event in events if event["kind"] == "completed"]
        require(len(events) == len(accepted) + len(completed) and len(accepted) == len(completed) and
            {(event["operation_sequence"],event["physical_bytes"]) for event in accepted} ==
            {(event["operation_sequence"],event["physical_bytes"]) for event in completed} and
            all(event["result"] == event["physical_bytes"] for event in completed),
            "actual accepted/completed bytes match without failed I/O: " + stage)
        inside = sum(any(frame["start_ns"] <= event["at_ns"] <= frame["end_ns"] for frame in frames) for event in events)
        io[stage] = dict(accepted_operations=len(accepted), completed_operations=len(completed),
            completed_physical_bytes=sum(event["physical_bytes"] for event in completed),
            host_timestamp_inside_execute_sample_intervals=inside,
            host_timestamp_outside_execute_sample_intervals=len(events)-inside)
    prior = None
    if prior_raw_directory is not None:
        previous, prior_ref = load(Path(prior_raw_directory)/NAMES[0])
        sources.append(prior_ref)
        before = {row["request_id"]: row for row in previous["rows"]}
        require(set(before) == {row["request_id"] for row in rows}, "same five source request IDs as prior actual run")
        comparison = []
        for row in rows:
            earlier = before[row["request_id"]]
            require(row["prompt_sha256"] == earlier["prompt_sha256"] and
                    row["actual_prompt_token_ids"] == earlier["actual_prompt_token_ids"], "same actual source prompt and token IDs")
            comparison.append(dict(request_id=row["request_id"], prompt_token_ids_identical=True,
                output_token_ids_identical=row["output_token_ids"] == earlier["output_token_ids"],
                actual_output_tokens=128, prior_output_tokens=len(earlier["output_token_ids"])))
        prior = dict(prior_run_is_failed_capture_reference_only=True, rows=comparison,
            all_output_token_ids_identical=all(row["output_token_ids_identical"] for row in comparison),
            timing_used_as_performance_control=False)
    no_restore = io["ssd_read"]["completed_operations"] == io["h2d"]["completed_operations"] == 0
    summary = dict(schema="actual_completed_original_U_mixed_stream_descriptive_analysis_v1",
        status="ACTUAL_GUARD_CLOSED_U_FUNCTION_AND_MIXED_STREAM_DESCRIPTION_ONLY", run_id=run_id,
        gpu_uuid=native["gpu_uuid"], actual_evidence_sources=sources, GPU_operations_by_analysis=0,
        complete_requests=5, actual_output_tokens=640, original_engine_steps=frontend["original_engine_steps"],
        CUDA_event_witnesses=len(witnesses), whole_native_tokens_match_frontend=True,
        heterogeneous_non_exact_cell_frames=heterogeneous, heterogeneous_frame_fraction=heterogeneous/len(frames),
        actual_step_kind_counts=dict(kinds), original_CUDA_step_elapsed=describe_ns([row["gpu_elapsed_ns"] for row in witnesses]),
        original_CUDA_step_elapsed_by_kind=cuda_by_kind,
        actual_token_return_interval_description=describe_ns([value for row in rows for value in row["itl_ns"]]),
        cached_tokens_by_request=[dict(request_id=row["request_id"], cached_tokens=row["num_cached_tokens"]) for row in rows],
        actual_native_IO_stages=io, comparison_with_prior=prior,
        raw_capture_per_frame_IO_fields=dict(existing_IO_all_unsupplied=all(frame.get("existing_io") is None for frame in frames),
            new_IO_all_unsupplied=all(frame.get("new_io") is None for frame in frames)),
        causal_scope=dict(native_stage_completion_bytes_verified=True,
            actual_per_frame_IO_cost_association_qualified=False, capture_IO_fields_not_replaced_from_journal=True,
            host_event_interval_containment_is_descriptive_only=True,
            cross_clock_absolute_CUDA_mapping_used=False, IO_caused_CUDA_or_token_slowdown_proved=False,
            preexisting_vs_new_IO_interference_cell_qualified=False),
        workload_conditions=dict(no_SSD_read_or_H2D_observed=no_restore,
            SSD_cold_restore_exercised=not no_restore,
            no_restore_does_not_prove_method_ineffective=True, no_synthetic_IO_injected=native.get("synthetic_io_or_stall_injected") is False),
        actual_original_shutdown_returned=native["original_engine_shutdown_returned"],
        actual_native_tail_drained=native["native_tail_drained"],
        cost_qualified=False, table_issued=False, ordinary_I_authorized=False,
        formal_goodput_allowed=False, strategy_improvement_proved=False, method_ineffective_proved=False,
        no_independent_service_SLO_inferred=True)
    for source in sources:
        _, after = load(source["local_path"])
        require(after == source, "immutable evidence unchanged during analysis")
    return summary


def render_report(summary):
    io = summary["actual_native_IO_stages"]
    gpu = summary["original_CUDA_step_elapsed"]
    lines = [f"真实 U 功能与完整流观测：{summary['run_id']}", "",
        f"原 GPU guard 正常结束并排空会话；5 个请求完整生成 {summary['actual_output_tokens']} tokens，"
        f"原引擎共 {summary['original_engine_steps']} 步，{summary['CUDA_event_witnesses']} 份真实 CUDA 事件见证。"
        "逐帧输出与原前台 token 序列一致，原 shutdown 和 native tail 已闭合。", "",
        f"包含 {summary['heterogeneous_non_exact_cell_frames']} 个不适合单一精确成本单元的帧"
        f"（{summary['heterogeneous_frame_fraction']:.1%}）；所有帧均保留。"
        f"全步 CUDA elapsed 的 p50 为 {gpu['p50_ms']:.3f} ms，p95 为 {gpu['p95_ms']:.3f} ms。"
        "这是此次运行的描述性观测，未作为策略提升或成本表资格结论。", "",
        "| 原生阶段 | 完成次数 | 实际完成字节 |", "|---|---:|---:|"]
    lines += [f"| {stage} | {io[stage]['completed_operations']} | {io[stage]['completed_physical_bytes']} |" for stage in STAGES]
    lines += ["", "原生 journal 的阶段完成及字节可验证；capture 的每帧 existing_io/new_io 未通过 journal 补造。"
        "时间戳落在原 execute/sample 的 host 区间只描述时间包含关系，不能证明 I/O 导致 CUDA 或 token 延迟。", ""]
    if summary["workload_conditions"]["no_SSD_read_or_H2D_observed"]:
        lines += ["本小规模 pilot 未观察到 SSD 读/H2D，因此没有覆盖 SSD 冷前缀恢复机会。"
            "这不能证明方法无效；实际收益仍需满足成本资格、合法机会和独立对照条件后验证。", ""]
    prior = summary["comparison_with_prior"]
    if prior is not None:
        identical = prior["all_output_token_ids_identical"]
        lines += [f"与 Uoff02 的五份实际输入 token 完全一致；全部输出 token 一致性结果为 {identical}。"
            "Uoff02 的 CUDA capture 失败，未用其时延作为性能对照。", ""]
    lines += ["当前未签发成本表、普通 I 的成本与运行资格尚未成立、未验证策略性能提升，未推断服务 SLO。", "",
        "证据：", ""]
    lines += [f"- {source['local_path']}；SHA-256 {source['sha256']}" for source in summary["actual_evidence_sources"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", required=True)
    parser.add_argument("--guard-path", required=True)
    parser.add_argument("--closure-path", required=True)
    parser.add_argument("--prior-raw-directory")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args(argv)
    summary = analyze(args.raw_directory, guard_path=args.guard_path, closure_path=args.closure_path,
                      prior_raw_directory=args.prior_raw_directory)
    prefix = Path(args.output_prefix).resolve()
    json_path, md_path = Path(str(prefix)+".json"), Path(str(prefix)+".md")
    require(prefix.parent.is_dir() and not json_path.exists() and not md_path.exists(), "new append-only output files")
    with json_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    with md_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render_report(summary))
    print(json.dumps(dict(status=summary["status"], run_id=summary["run_id"], actual_output_tokens=640,
        actual_steps=summary["original_engine_steps"], heterogeneous_frames=summary["heterogeneous_non_exact_cell_frames"],
        cost_qualified=False, strategy_improvement_proved=False), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
