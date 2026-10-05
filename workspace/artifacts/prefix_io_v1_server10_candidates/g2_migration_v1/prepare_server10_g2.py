"""Append a distinct G2 machine scope from the user's system-verification request.

CPU preparation only. The existing GPU launcher/guard checks the whole source
closure, separate scope, actual device and remaining eight-hour budget later.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/root/autodl-tmp/prefix-io-v1-handoff/project"))
    parser.add_argument("--ancestor-lock", default="artifacts/prefix_io_v1/server10-reference-migration-v1-20261003/gpu-source-lock-server10-reference.json")
    parser.add_argument("--ancestor-sha256", required=True)
    args = parser.parse_args()
    root = args.project.resolve(strict=True)
    delivery = "artifacts/prefix_io_v1/server10-g2-migration-v1-20261003"
    path = root / delivery / "run_server10_g2.py"
    spec = importlib.util.spec_from_file_location("_server10_g2_cpu_prepare", path)
    entry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = entry
    spec.loader.exec_module(entry)
    entry.require(root == entry.ROOT, "fixed server10 project")
    ancestor_ref = entry.file_ref(root, args.ancestor_lock)
    entry.require(ancestor_ref["sha256"] == args.ancestor_sha256, "explicit frozen ancestor lock SHA")
    ancestor = entry.json_read(entry.safe(root, args.ancestor_lock))
    refs = {row["path"]: row for row in ancestor["files"]}
    entry.require(len(refs) == len(ancestor["files"]) and 2052 <= len(refs) <= 8192, "unique complete reference ancestor")
    def write(name, value):
        target = entry.safe(root, delivery + "/" + name)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        return entry.file_ref(root, delivery + "/" + name)
    permit_ref = entry.file_ref(root, entry.PERMISSIONS)
    config = dict(schema_version=1, gpu_uuid=entry.GPU_UUID, job_names=entry.JOBS,
                  permissions_ref=permit_ref, sdk_inventory_ref=entry.INVENTORY_REF,
                  sdk_proof_ref=entry.PROOF_REF, source_lock=entry.LOCK, scope_record=entry.SCOPE)
    config_ref = write("G2_MIGRATION_CONFIG.json", config)
    new_sources = [config_ref, ancestor_ref]
    for relative in (entry.SCRIPT, delivery + "/prepare_server10_g2.py", delivery + "/test_server10_g2.py"):
        new_sources.append(entry.file_ref(root, relative))
    for row in new_sources:
        entry.require(row["path"] not in refs or refs[row["path"]] == row, "preserve complete ancestor refs")
        refs[row["path"]] = row
    module, cleanup = entry.bind_original(root, refs, config)
    try:
        template = module.scope_template()
        plan_ref = write("SERVER10_G2_CPU_PLAN.json", dict(schema_version=1,
            status="CPU_ONLY_NEW_MACHINE_G2_PLAN", purpose=module.PURPOSE,
            original_runtime_ref=entry.ORIGINAL_REF, configuration=config,
            qualification_context=module.context(), maximum_jobs=2,
            seconds_limit_per_job=300, reserved_seconds_per_job=320,
            off_success_and_original_shutdown_and_session_drain_before_shadow=True,
            reuse_reference_scope=False, full_model_source_verified=False,
            performance_claim=False, GPU_runs=0))
        refs[plan_ref["path"]] = plan_ref
        lock = dict(schema_version=1, status="FROZEN_SERVER10_G2_NORMAL_MODEL_SOURCE_CLOSURE",
            ancestor_source_lock=ancestor_ref, original_G1_baseline=module.BASELINE_LOCK,
            original_G1_baseline_sha256=module.BASELINE_SHA,
            files=[refs[key] for key in sorted(refs)])
        lock_ref = write("gpu-source-lock-server10-g2.json", lock)
        ledger = entry.json_read(root / "experiments/prefix_io_v1/gpu-budget-ledger.json")
        entry.require(ledger.get("active_reservation") is None
            and type(ledger.get("gpu_wall_seconds")) in (int, float)
            and 0 <= ledger["gpu_wall_seconds"] <= 28800 - 640,
            "inactive original ledger and both G2 reservations within original eight hours")
        scope = dict(template, status="USER_AUTHORIZED_G2_NORMAL_MODEL_LIFECYCLE",
            allow_gpu_initialization=True, allow_gpu_runs=True, source_lock=entry.LOCK,
            source_lock_sha256=lock_ref["sha256"], base_permissions=permit_ref)
        human = dict(authorization_origin="direct_human_reply", purpose=module.PURPOSE,
            gpu_uuid=entry.GPU_UUID, allow_gpu_runs=True, allow_gpu_initialization=True,
            allowed_modes=["off", "shadow"], maximum_jobs=2, maximum_total_planned_reserve_seconds=640,
            permitted_run_names=entry.JOBS, source_lock=lock_ref, base_permissions=permit_ref,
            authorization_basis="NEW_SERVER_GPU_SYSTEM_VERIFICATION_DIRECT_REQUEST",
            verbatim_user_answer="旧服务器已经关机，已经把代码克隆到新的服务器，接下来验证在新服务器进行gpu实验，完成系统验证",
            scope_summary="独立G2完整模型资格验证：最多off、shadow各一次，每次300秒和20秒收尾；每次cold/repeat各完整128 tokens。off完整输出、原shutdown返回且OS会话排空后才shadow；失败停止；不重试、不下载、不改系统或驱动。沿用原8小时累计预算。",
            scope_summary_is_agent_bounded_interpretation=True,
            existing_reference_scope_reused=False, question_item_id="not_applicable:direct_typed_user_message",
            reply_medium="direct_typed_user_message", credentials_omitted=True,
            observation_timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            observation_timestamp_is_not_user_sent_timestamp=True,
            SSH_target="root@connect.westd.seetacloud.com:12351")
        human_ref = write("HUMAN_AUTHORIZATION_RECORD_SERVER10_G2.json", human)
        scope["human_authorization_record"] = human_ref
        scope_ref = write("SERVER10_G2_AUTHORIZED_SCOPE.json", scope)
    finally:
        restored = cleanup()
    entry.require(restored["status"] == "RESTORED_PRIVATE_G2_MACHINE_BINDINGS", "CPU preparation module restore")
    result = dict(status="CPU_PREPARED_SEPARATE_SERVER10_G2_SCOPE", source_lock=lock_ref,
        source_count=len(refs), scope_ref=scope_ref, human_record_ref=human_ref,
        source_lock_full_verified=False, GPU_runs=0, inherited_budget_seconds=28800,
        remaining_budget_seconds=28800-ledger["gpu_wall_seconds"], migration_cleanup=restored)
    write("G2_CPU_PREPARATION_RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
