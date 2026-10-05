"""Append one two-job diagnostic plan/scope; no GPU operation here."""
import argparse
import copy
import datetime
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/root/autodl-tmp/prefix-io-v1-handoff/project"))
    parser.add_argument("--ancestor-lock", default="artifacts/prefix_io_v1/server10-reference-migration-v1-20261003/gpu-source-lock-server10-reference.json")
    parser.add_argument("--ancestor-sha256", required=True)
    parser.add_argument("--capture-relative", default="artifacts/prefix_io_v1/server10-kv-capture-v1-20261003/production_kv_capture.py")
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--capture-test-relative", default="artifacts/prefix_io_v1/server10-kv-capture-v1-20261003/test_production_kv_capture.py")
    args = parser.parse_args()
    root = args.project.resolve(strict=True)
    delivery = "artifacts/prefix_io_v1/server10-kv-diagnostic-v1-20261003"
    path = root / delivery / "run_server10_kv_diagnostic.py"
    spec = importlib.util.spec_from_file_location("_kv_diagnostic_cpu_prepare", path)
    entry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = entry
    spec.loader.exec_module(entry)
    entry.require(root == entry.ROOT, "fixed diagnostic server root")
    ancestor_ref = entry.ref(root, args.ancestor_lock)
    entry.require(ancestor_ref["sha256"] == args.ancestor_sha256, "explicit immutable ancestor SHA")
    capture_ref = entry.ref(root, args.capture_relative)
    entry.require(capture_ref["sha256"] == args.capture_sha256, "explicit audited capture SHA")
    inventory = dict(path="artifacts/prefix_io_v1/server10-cuda13-cpu-20261003/CUDA13_SOURCE_INVENTORY.json",
        bytes=692329, sha256="0e7cc1df1d8d7aed7271841d6e9f1e4188785f40ad4c74b4c6c78f670b7794b6")
    proof = dict(path="artifacts/prefix_io_v1/server10-cuda13-cpu-20261003/CPU_COMPILE_LINK_RESULT.json",
        bytes=5588, sha256="ef7a7482cc1cae539e2425d6aefc82ea8160e398ab7d8b14926f76537e614a25")
    for row in (inventory, proof):
        entry.checked(root, row)
    def write(name, value):
        target = entry.safe(root, delivery + "/" + name)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        return entry.ref(root, delivery + "/" + name)
    config = dict(schema_version=1, gpu_uuid=entry.GPU_UUID, job_names=entry.JOBS,
        storage=entry.STORAGE, permissions_ref=entry.ref(root, entry.PERMISSIONS), ancestor_ref=ancestor_ref,
        capture_ref=capture_ref, sdk_inventory_ref=inventory, sdk_proof_ref=proof,
        mode="diagnostic-only", load_planner="off", production_qualified=False, performance_claim=False,
        latency_fit_allowed=False, seconds_limit_per_job=300, reserved_seconds_per_job=320,
        storage_reserve_bytes_per_job=512*1024**2, total_storage_reserve_bytes=1024**3,
        storage_floor_bytes=8*1024**3, maximum_capture_binary_bytes=22020096,
        maximum_original_cache_bytes=22020096,
        expected_capture_job_counts=dict(populate=dict(store=3, load=3), paired=dict(store=0, load=6)),
        warmup_included_in_byte_capture=True, flush_dummy_excluded_from_byte_comparison_only=True,
        actual_GPU_runs=0)
    config_ref = write("KV_DIAGNOSTIC_CONFIG.json", config)
    plan = entry.configure_plan(root, entry.load(root, entry.PINNED["plan"], "prepare_plan"), config)
    new_refs = [config_ref, capture_ref, entry.ref(root, args.capture_test_relative)]
    new_refs.extend(entry.ref(root, delivery + "/" + name) for name in (
        "run_server10_kv_diagnostic.py", "prepare_server10_kv_diagnostic.py", "test_server10_kv_diagnostic.py"))
    plan_ref = write("KV_DIAGNOSTIC_CPU_PLAN.json", plan.build_calibration_plan(new_source_refs=new_refs))
    raw_ancestor = entry.checked(root, ancestor_ref).read_bytes()
    manifest = plan.assemble_source_lock(raw_ancestor, new_refs + [plan_ref])
    lock_ref = write("gpu-source-lock-kv-diagnostic.json", manifest)
    scope = plan.scope_template(lock_ref, plan_ref=plan_ref)
    human = {key: copy.deepcopy(value) for key, value in scope.items() if key not in ("status", "human_authorization_record")}
    human.update(authorization_origin="direct_human_reply", allow_gpu_initialization=True, allow_gpu_runs=True,
        question="本次新服务器GPU系统验证指令内的有界真实KV字节诊断：populate、fresh paired各一次，每次300+20秒，独立新缓存；原8小时累计预算、8GiB空间底线不变，失败停止，不重试。",
        scope_summary_is_agent_bounded_interpretation=True,
        verbatim_user_answer="旧服务器已经关机，已经把代码克隆到新的服务器，接下来验证在新服务器进行gpu实验，完成系统验证",
        reply_observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        observation_timestamp_is_not_user_sent_timestamp=True,
        question_item_id="not_applicable:direct_typed_user_message", reply_medium="direct_typed_user_message",
        credentials_omitted=True, existing_reference_or_G2_scope_reused=False)
    human_ref = write("HUMAN_AUTHORIZATION_RECORD_KV_DIAGNOSTIC.json", human)
    scope.update(status="USER_AUTHORIZED_G3_RAW_COST_PILOT", allow_gpu_initialization=True,
        allow_gpu_runs=True, human_authorization_record=human_ref)
    scope_ref = write("KV_DIAGNOSTIC_AUTHORIZED_SCOPE.json", scope)
    refs = {row["path"]: row for row in manifest["files"]}
    consistency = plan.validate_scope(root, scope, refs)
    ledger = entry.read_json(root / "experiments/prefix_io_v1/gpu-budget-ledger.json")
    import os
    stat = os.statvfs(root / "experiments/prefix_io_v1/runs")
    import time
    budget = plan.validate_budget_and_storage(ledger,
        dict(origin="actual_statvfs", captured_unix=time.time(), primary_root=str(root), free_bytes=stat.f_bavail*stat.f_frsize),
        remaining_jobs=2)
    result = dict(status="PASS_CPU_PREPARED_DISTINCT_KV_DIAGNOSTIC_SCOPE", source_lock=lock_ref,
        source_count=len(refs), scope_ref=scope_ref, human_ref=human_ref,
        scope_consistency=consistency, budget_storage=budget, GPU_runs=0,
        actual_model_source_bytes_verified=False, byte_chain_verified=False, production_qualified=False,
        performance_claim=False, latency_fit_allowed=False)
    write("KV_DIAGNOSTIC_CPU_PREPARATION_RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
