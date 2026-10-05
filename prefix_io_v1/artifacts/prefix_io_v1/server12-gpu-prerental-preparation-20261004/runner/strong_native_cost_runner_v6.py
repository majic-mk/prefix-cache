"""Thin strong-domain parameterization of the existing six-process raw probe.

The original probe owns original LLM requests, native preload, journal, CUDA
capture and shutdown. This wrapper changes only frozen common configuration,
512-token calibration input, expected Prefix block hit, and parent metadata.
The original estimator and independent holdout verifier remain unchanged.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OLD_SUFFIX = "calibration_v2/run_native_cost_experiment.py"
OLD_SHA = "22f49f361c94406e2929d6c6472909f1b21440ed420a2bf3ac8cebe2d97d19e9"
MAX_SECONDS = 900
RESERVE = 512 * 1024**2
FLOOR = 8 * 1024**3
PROMPT_TOKENS, CACHED_TOKENS, OFFSET, OPERATIONS = 512, 512, 16, 1
INITIAL_EXECUTION_PRE_CONTEXT = PROMPT_TOKENS - 1
FIRST, SEEDS, ORDER = (40100, 41100, 42100), (4029, 4030, 4031), ("AB", "BA", "AB")
CONFIG_FIELDS = {"schema", "job_id", "gpu_uuid", "seconds_limit", "storage_reserve_bytes", "storage_floor_bytes",
    "source_lock_ref", "source_proof_ref", "plan_ref", "intent_ref", "permissions_ref", "off_qualification_ref",
    "off_guard_ref", "output_relative", "storage_template_relative", "input_manifest_ref", "runner_ref"}


def driver_module():
    name = "_strong_raw_original_guard_driver"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "strong_trace_runner.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prompt(index):
    values = list(range(1000, 1000 + PROMPT_TOKENS))
    values[0] = FIRST[index]
    return values


def cell_descriptor():
    R = driver_module()
    entries = []
    for index in range(3):
        tokens = prompt(index)
        entries.append(dict(pair_id="strong512-pair" + str(index), split="calibration" if index < 2 else "validation",
            arm_order=ORDER[index], seed=SEEDS[index], prompt_token_ids=tokens,
            prompt_sha256=R.canonical_sha(tokens), prefix_family_sha256=R.canonical_sha(tokens[:16]),
            trace_sha256=R.canonical_sha(dict(prompt_token_ids=tokens, seed=SEEDS[index], output_tokens=128))))
    return dict(id="strong512-context527-read1-existing0", prompt_tokens=PROMPT_TOKENS,
        cached_prompt_tokens=INITIAL_EXECUTION_PRE_CONTEXT,
        cached_prompt_tokens_semantics="initial_execution_pre_context",
        frontend_cached_prompt_tokens=CACHED_TOKENS, initial_execution_pre_context=INITIAL_EXECUTION_PRE_CONTEXT,
        measured_offset=OFFSET, warmup_offsets=[1], operations=OPERATIONS,
        stage="ssd_read", transfer_quantum_bytes=917504, existing_io=[dict(ops=0, bytes=0)] * 4, entries=entries)


def prepare_plan(root, *, source_lock_ref, pair_ref, input_manifest_ref, geometry_ref, output, job_id):
    """CPU plan template: null GPU and preparation flags prevent real issuance."""
    R = driver_module()
    refs = R.source_rows(root, source_lock_ref, full=True)
    for row in (pair_ref, input_manifest_ref, geometry_ref):
        R.check_ref(root, row)
    pair = R.read(R.check_ref(root, pair_ref))
    R.require(pair.get("schema") == "strong_native_u_i_cpu_configuration_v1", "strong common pair wrapper")
    validator = R.load(root, refs[R.STRONG], "_strong_cost_pair_validation_" + str(time.monotonic_ns()))
    validator.validate_runtime_pair(pair["configurations"])
    relative = Path(__file__).resolve().relative_to(root).as_posix()
    new_root = Path(relative).parent.parent
    required = dict(collector_source_ref=(Path(relative).parent / "bounded_native_full_step_collector_v2.py").as_posix(),
        original_collector_source_ref=(Path(relative).parent / "bounded_native_full_step_collector.py").as_posix(),
        cuda_event_source_ref=".venv/lib/python3.12/site-packages/torch/cuda/streams.py",
        native_source_ref=R.NATIVE + "/reactor.py", original_guard_source_ref=R.GUARD,
        validation_source_ref=(new_root / "activation/native_conditional_cost.py").as_posix(),
        strong_pair_validator_source_ref=R.STRONG, runtime_pair_ref=pair_ref["path"],
        model_manifest_ref=R.MODEL_PLAN, kv_layout_ref=geometry_ref["path"],
        original_estimator_source_ref="third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py",
        wrapper_source_ref=relative, issuer_source_ref=(new_root / "activation/source/prefix_io_control/gpu_cell_issuer.py").as_posix(),
        cost_table_source_ref=(new_root / "activation/source/prefix_io_control/p4_cost_table.py").as_posix(),
        model_config_source_ref=R.MODEL + "/config.json")
    for name, path in required.items():
        R.require(path in refs, "full cost leaf frozen before template: " + name)
        R.check_ref(root, refs[path])
    R.require(refs[required["original_collector_source_ref"]]["sha256"] ==
              "9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d",
              "immutable original full-step collector delegate")
    plan = dict(schema="strong_gpu_exact_cell_prelaunch_plan_v1", evidence_origin="native_runtime_preregistered",
        cpu_preparation_only=True, synthetic_fixture=False, gpu_uuid=None, project_root=str(root), job_id=job_id,
        journal_run_id=job_id, source_lock_ref=source_lock_ref,
        common_runtime_domain_sha256=R.common_domain_sha(pair["configurations"]), kernel_mode="eager",
        cells=[cell_descriptor()], input_manifest_ref=input_manifest_ref,
        calibration_kind="bounded_controlled_original_probe", natural_trace=False,
        actual_gpu_runs=0, formal_goodput_allowed=False, GPU_qualification_issued=False,
        required_live_binding="Parent live GPU/source/SDK gate binds UUID and derives new full lock before intent/guard")
    plan.update({key: refs[path] for key, path in required.items()})
    R.new_json(output, plan)
    return plan


def prepare_geometry(root, output):
    R = driver_module()
    row = R.ref(root, R.MODEL + "/config.json")
    model = R.read(R.check_ref(root, row))
    R.require(model.get("num_hidden_layers") == 28 and model.get("num_key_value_heads") == 4 and
              model.get("hidden_size") == 3584 and model.get("num_attention_heads") == 28, "actual fixed Qwen config geometry")
    geometry = dict(model_config_sha256=row["sha256"], num_hidden_layers=28, num_key_value_heads=4,
        head_dim=128, dtype="bfloat16", dtype_bytes=2, tokens_per_block=16, tensor_parallel_size=1,
        physical_block_bytes=28 * 2 * 4 * 128 * 2 * 16)
    with output.open("xb") as stream:
        stream.write(json.dumps(geometry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return R.ref(root, output.relative_to(root).as_posix())


def bind_live_plan(root, *, template_ref, source_lock_ref, source_proof_ref, off_qualification_ref,
                   off_guard_ref, permissions_ref, gpu_uuid, output):
    """Future live binding, requiring the real device and completed strong off."""
    R = driver_module()
    refs = R.source_rows(root, source_lock_ref, full=True)
    proof = R.read(R.check_ref(root, source_proof_ref))
    R.require(proof.get("status") == "PASS_FULL_CPU_SOURCE_BYTES" and proof.get("source_lock_ref") == source_lock_ref and
              proof.get("source_count") == len(refs) and proof.get("actual_gpu_runs") == 0, "independent full live-source proof")
    plan = deepcopy(R.read(R.check_ref(root, template_ref)))
    R.require(plan.get("cpu_preparation_only") is True and plan.get("gpu_uuid") is None and plan.get("cells") == [cell_descriptor()],
              "unchanged CPU strong cell template")
    off = R.read(R.check_ref(root, off_qualification_ref))
    R.require(off.get("status") == "PASS_STRONG_NATIVE_OFF_WORKLOAD_LIFECYCLE" and off.get("gpu_uuid") == gpu_uuid and
              off.get("common_runtime_domain_sha256") == plan["common_runtime_domain_sha256"] and
              off.get("original_engine_shutdown_returned") is True and off.get("native_tail_drained") is True, "real same-domain completed strong off")
    validation = R.load(root, plan["validation_source_ref"], "_strong_live_guard_validation_" + str(time.monotonic_ns()))
    validation.validate_guard(R.read(R.check_ref(root, off_guard_ref)), gpu_uuid=gpu_uuid, job_id=off["run_id"],
                              wrapper_path=off["runner_ref"]["path"])
    # Use the unchanged permission parser; the persistent authorization cannot
    # enlarge the original GPU budget or allow driver/system/storage changes.
    original_guard = R.load(root, refs[R.GUARD], "_strong_live_original_permission_" + str(time.monotonic_ns()))
    permission, permission_ref = original_guard.permission_source(root, permissions_ref["path"])
    R.require(permission_ref == permissions_ref and permission.get("approved_gpu_ids") == [gpu_uuid] and
              permission.get("allow_gpu_runs") is True and permission.get("max_gpu_hours") <= 8,
              "actual device-specific raw permission scope")
    ledger = R.read(R.safe(root, R.LEDGER))
    R.require(ledger.get("active_reservation") is None and ledger["gpu_wall_seconds"] + MAX_SECONDS + 20 <= R.MAX_GPU_SECONDS,
              "original idle remaining budget")
    device_helper_relative = (Path(__file__).resolve().parent.parent /
                              "raw_device_binding/uuid_minor_device.py").relative_to(root).as_posix()
    R.require(device_helper_relative in refs, "exact live UUID/minor device helper frozen")
    device_helper = R.load(root, refs[device_helper_relative], "_strong_live_uuid_minor_" + str(time.monotonic_ns()))
    device_evidence = device_helper.inspect_actual_binding(gpu_uuid)
    R.require(device_evidence["gpu_uuid"] == device_helper.uuid(gpu_uuid), "actual selected UUID/minor node binding")
    for key, row in plan.items():
        if key.endswith("_ref") and type(row) is dict and key != "source_lock_ref":
            R.require(refs.get(row["path"]) == row, "same full live calibrated source leaf: " + key)
            R.check_ref(root, row)
    R.require(shutil.disk_usage(root).free >= FLOOR + RESERVE, "raw fixed reserve and storage floor")
    plan.update(cpu_preparation_only=False, gpu_uuid=gpu_uuid, source_lock_ref=source_lock_ref,
                live_binding=dict(actual_off_ref=off_qualification_ref, actual_off_guard_ref=off_guard_ref,
                                  permissions_ref=permissions_ref, actual_uuid_checked=True,
                                  actual_device_binding=device_evidence, device_binding_source_ref=refs[device_helper_relative],
                                  source_proof_ref=source_proof_ref, current_model_execution_started=False))
    R.new_json(output, plan)
    return R.ref(root, output.relative_to(root).as_posix())


def close_actual_raw(root, *, pending_ref, plan_ref, before_ref, after_ref, guard_ref, output):
    """Future parent closure; no CPU receipt or pending child can self-qualify."""
    R = driver_module()
    plan = R.read(R.check_ref(root, plan_ref))
    raw = deepcopy(R.read(R.check_ref(root, pending_ref)))
    R.require(plan.get("cpu_preparation_only") is False and plan.get("gpu_uuid") is not None and
              raw.get("schema") == "strong_gpu_exact_cell_raw_measurements_v1" and
              raw.get("plan_ref") == plan_ref and raw.get("origin") == "native_gpu_recording" and
              raw.get("synthetic_fixture") is False and raw.get("source_verification_refs") == {}, "actual pending native raw, not CPU fixture")
    lock = R.read(R.check_ref(root, plan["source_lock_ref"]))
    for phase, row in (("before", before_ref), ("after", after_ref)):
        receipt = R.read(R.check_ref(root, row))
        R.require(receipt.get("phase") == phase and receipt.get("source_lock_ref") == plan["source_lock_ref"] and
                  receipt.get("failed") == [] and receipt.get("files_verified") == len(lock["files"]), "independent complete actual source receipt: " + phase)
    validation = R.load(root, plan["validation_source_ref"], "_strong_raw_parent_guard_" + str(time.monotonic_ns()))
    validation.validate_guard(R.read(R.check_ref(root, guard_ref)), gpu_uuid=plan["gpu_uuid"], job_id=plan["job_id"],
                              wrapper_path=plan["wrapper_source_ref"]["path"])
    for cell in raw["cells"]:
        R.require(len(cell["windows"]) == 6, "all independently guarded original windows retained")
        for window in cell["windows"]:
            report = R.read(R.check_ref(root, window["child_receipt_ref"]))
            R.require(report.get("status") == "PASS_NATIVE_FULL_STEP_RAW_CAPTURE_REQUIRES_VALIDATION" and
                      report.get("original_engine_shutdown_returned") is True and report.get("load_planner") == "on" and
                      report.get("fresh_original_process") is True and
                      report.get("subprocess_sid") == R.read(R.check_ref(root, guard_ref))["session_id"], "actual original per-window child/source/session")
    raw["source_verification_refs"] = dict(before=before_ref, after=after_ref)
    raw["completed_guard_ref"] = guard_ref
    raw["qualification_pending"] = "Real private issuer must verify exact capture, native membership, estimator and independent heldout"
    R.new_json(output, raw)
    return R.ref(root, output.relative_to(root).as_posix())


def make_actual_configuration(root, *, bound_plan_ref, source_proof_ref, intent_ref, permissions_ref,
                              off_qualification_ref, off_guard_ref, output_relative,
                              storage_template_relative, output):
    """Future append-only config; the complete CLI preflight remains mandatory."""
    R = driver_module()
    plan = R.read(R.check_ref(root, bound_plan_ref))
    R.require(plan.get("cpu_preparation_only") is False and type(plan.get("gpu_uuid")) is str,
              "actual live-bound plan before configuration")
    config = dict(schema="strong_gpu_exact_cell_acquisition_config_v1", job_id=plan["job_id"], gpu_uuid=plan["gpu_uuid"],
        seconds_limit=MAX_SECONDS, storage_reserve_bytes=RESERVE, storage_floor_bytes=FLOOR,
        source_lock_ref=plan["source_lock_ref"], source_proof_ref=source_proof_ref, plan_ref=bound_plan_ref,
        intent_ref=intent_ref, permissions_ref=permissions_ref, off_qualification_ref=off_qualification_ref,
        off_guard_ref=off_guard_ref, output_relative=output_relative, storage_template_relative=storage_template_relative,
        input_manifest_ref=plan["input_manifest_ref"], runner_ref=plan["wrapper_source_ref"])
    for key in ("source_proof_ref", "intent_ref", "permissions_ref", "off_qualification_ref", "off_guard_ref"):
        R.check_ref(root, config[key])
    R.new_json(output, config)
    verify_configuration(root, output, full=True)
    return R.ref(root, output.relative_to(root).as_posix())


def source_patch(raw):
    """Compile only three old driver functions with explicit metadata patches."""
    text = raw.decode("utf-8")
    replacements = (
        ("load_planner='off'", "load_planner='on'"),
        ("warm[0].num_cached_tokens==128", "warm[0].num_cached_tokens==CACHED_TOKENS"),
        ("cached_tokens expected128", "cached_tokens expected512"),
        ("frontend['output']['num_cached_tokens']==128", "frontend['output']['num_cached_tokens']==CACHED_TOKENS"),
        ("['block_hashes'][:8]", "['block_hashes'][:32]"),
        ("str(root/DELIVERY/'NATIVE_COST_CONFIG.json')", "CONFIG_PATH.relative_to(root).as_posix()"),
        ("str(root/SCRIPT),'--execute','--config'", "str(root/SCRIPT),'--project',str(root),'--execute','--config'"),
    )
    original_tree = ast.parse(text)
    require = driver_module().require
    require(__import__("hashlib").sha256(raw).hexdigest() == OLD_SHA, "immutable original six-process raw driver source")
    selected = ("execute_window", "execute_parent")
    methods = [node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
    require(len(methods) == 2, "original raw window and same-session parent functions")
    new_source = "\n\n".join(ast.get_source_segment(text, node) for node in methods)
    for before, after in replacements:
        require(before in new_source, "explicit strong raw parameter patch preimage: " + before)
        new_source = new_source.replace(before, after)
    before = "require(not coordinators and all(key not in kwargs for key in ('stage_accounting','observation_sink','progress_run_id','max_accepted_parents','p4_bridge')),\n                    'sole original coordinator with passive optional observation')"
    after = "require(not coordinators and kwargs.get('p4_bridge') is None and kwargs.get('max_accepted_parents') == MAX_ACCEPTED_PARENTS and kwargs.get('progress_run_id') == LABEL, 'same strong common native owner with passive journal')\n            kwargs = dict(kwargs)\n            for observed_key in ('stage_accounting', 'observation_sink', 'progress_run_id', 'max_accepted_parents', 'p4_bridge'):\n                kwargs.pop(observed_key, None)"
    require(new_source.count(before) == 1, "original common construction patch preimage")
    new_source = new_source.replace(before, after)
    # Persist actual failures before any assertion or observer detachment.
    # This records copied CPU metadata only, never qualifies an invalid stream.
    diagnostic_source = """
    rid=scalar_adapter=frontend=capture=None
    def preserve_actual_capture(phase, exported=None):
        if active_capture is None or rid is None:
            return
        row=dict(schema='strong_actual_capture_failure_diagnostic_v1', phase=phase,
            run_id=rid, origin='native_gpu_recording', production_qualified=False,
            performance_claim=False, expected_frame_count=128, diagnostic_errors=[])
        # Each read is best effort, so recording cannot prevent original cleanup.
        try:
            actual_export=exported if type(exported) is dict else active_capture.export()
            row['actual_capture_export']=actual_export
        except Exception as diagnostic_exc:
            row['diagnostic_errors'].append('export:'+type(diagnostic_exc).__name__+':'+str(diagnostic_exc)[:256])
        try:
            observer=active_capture.observer
            scalar=observer.scalar
            adapter=scalar._adapter
            actual_frames=list(adapter.frames)
            row['scalar_adapter']=dict(initial_identity=id(scalar_adapter),current_identity=id(adapter),
                same_object=adapter is scalar_adapter,actual_frame_count=len(actual_frames),
                initial_native_source_sha256=getattr(scalar_adapter,'native_source_sha256',None),
                current_native_source_sha256=adapter.native_source_sha256,
                actual_expected_native_source_ref=refs[REACTOR],
                valid=adapter.valid,enabled=adapter.enabled,last_reason=adapter.last_reason,
                frames=[dataclasses.asdict(frame) for frame in actual_frames[:128]],
                frames_truncated=len(actual_frames)>128)
            pending=adapter._pending
            if type(pending) is dict:
                row['actual_prepared_pending']=dict(ordinal=pending.get('ordinal'),phase=pending.get('phase'),
                    start_ns=pending.get('start_ns'),frame=(dataclasses.asdict(pending['frame'])
                    if dataclasses.is_dataclass(pending.get('frame')) else None))
            else:
                row['actual_prepared_pending']=None
            row['observer_stats']=dict(enabled=observer.enabled,status=observer.status,reason=observer.reason,
                scalar_enabled=scalar.enabled,scalar_status=scalar.status,scalar_last_reason=scalar.last_reason,
                scalar_adapter_last_reason=adapter.last_reason,
                event_valid=observer.events.valid,event_last_reason=observer.events.reason,
                pending_event_pairs=len(observer.events.pending),open_event_pair=observer.events.active is not None,
                actual_capture_valid=active_capture.valid,actual_capture_failures=list(active_capture.failures))
        except Exception as diagnostic_exc:
            row['diagnostic_errors'].append('scalar:'+type(diagnostic_exc).__name__+':'+str(diagnostic_exc)[:256])
        try:
            if frontend is not None:
                row['actual_frontend']=frontend
            filename=rid+'-'+phase+'-capture-diagnostic.json'
            write_json(out/filename,row)
            result.setdefault('actual_capture_diagnostics',[]).append(dict(phase=phase,path=filename))
        except Exception as diagnostic_exc:
            result.setdefault('capture_diagnostic_write_errors',[]).append(
                type(diagnostic_exc).__name__+':'+str(diagnostic_exc)[:256])
"""
    before = "    try:\n        sys.path[:0]="
    require(new_source.count(before) == 1, "original window setup diagnostics preimage")
    new_source = new_source.replace(before, diagnostic_source + before)
    before = "                capture=active_capture.export()\n                require(active_capture.observer.scalar._adapter is scalar_adapter and"
    after = "                capture=active_capture.export()\n                preserve_actual_capture('preassert',capture)\n                require(active_capture.observer.scalar._adapter is scalar_adapter and"
    require(new_source.count(before) == 1, "original frame assertion diagnostics preimage")
    new_source = new_source.replace(before, after)
    before = "        if active_capture is not None:\n            try: active_capture.detach()"
    after = "        if active_capture is not None:\n            preserve_actual_capture('finally',capture)\n            try: active_capture.detach()"
    require(new_source.count(before) == 1, "original capture cleanup diagnostics preimage")
    new_source = new_source.replace(before, after)
    new_tree = ast.parse(new_source)
    # The wrapper never alters execute/sample implementation, native ownership
    # or original estimator AST. Original source remains an immutable ref.
    return new_tree, dict(schema="strong_raw_thin_source_parameterization_v1", original_functions=list(selected),
        planner="on", prompt_tokens=512, cached_tokens=512, initial_execution_pre_context=511,
        initial_scheduled_tokens=1, cached_prompt_tokens_semantics="initial_execution_pre_context",
        original_cache_hit_rule="local_plus_external_frontend_stats_then_original_full_prompt_remote_last_token_recompute",
        original_model_and_native_engine_unchanged=True, six_fresh_processes=True, estimator_modified=False,
        fit_pairs=2, independent_validation_pairs=1, validation_used_to_fit=False, GPU_qualification_issued=False)


def verify_configuration(root, path, *, full=True):
    R = driver_module()
    value = R.read(path)
    R.require(type(value) is dict and set(value) == CONFIG_FIELDS and value["schema"] == "strong_gpu_exact_cell_acquisition_config_v1",
              "strict strong raw acquisition configuration")
    R.require(type(value["job_id"]) is str and re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value["job_id"]), "bounded raw job identity")
    R.integer(value["seconds_limit"], "raw maximum 900 seconds", 1, MAX_SECONDS)
    R.require(value["storage_reserve_bytes"] == RESERVE and value["storage_floor_bytes"] == FLOOR, "raw 512MiB reserve/8GiB floor")
    refs = R.source_rows(root, value["source_lock_ref"], full=full)
    for key in (CONFIG_FIELDS - {"schema", "job_id", "gpu_uuid", "seconds_limit", "storage_reserve_bytes", "storage_floor_bytes",
                               "output_relative", "storage_template_relative"}):
        R.check_ref(root, value[key])
    R.require(Path(__file__).resolve() == R.check_ref(root, value["runner_ref"]).resolve() and
              refs.get(value["runner_ref"]["path"]) == value["runner_ref"], "actual strong raw wrapper frozen")
    proof = R.read(R.check_ref(root, value["source_proof_ref"]))
    R.require(proof.get("source_lock_ref") == value["source_lock_ref"] and proof.get("status") == "PASS_FULL_CPU_SOURCE_BYTES" and
              proof.get("source_count") == len(refs) and proof.get("actual_gpu_runs") == 0, "complete CPU source proof")
    plan = R.read(R.check_ref(root, value["plan_ref"]))
    R.require(plan.get("schema") == "strong_gpu_exact_cell_prelaunch_plan_v1" and plan.get("cpu_preparation_only") is False and
              plan.get("synthetic_fixture") is False and plan.get("source_lock_ref") == value["source_lock_ref"] and
              plan.get("gpu_uuid") == value["gpu_uuid"] and plan.get("job_id") == value["job_id"] and
              plan.get("cells") == [cell_descriptor()] and plan.get("input_manifest_ref") == value["input_manifest_ref"],
              "actual live-bound preregistered strong cell; CPU/null-GPU plan cannot execute")
    for key, row in plan.items():
        if key.endswith("_ref") and type(row) is dict:
            R.check_ref(root, row)
            if key != "source_lock_ref":
                R.require(refs.get(row["path"]) == row, "cost plan leaf frozen: " + key)
    pair_doc = R.read(R.check_ref(root, plan["runtime_pair_ref"]))
    pair = pair_doc["configurations"]
    validator = R.load(root, refs[R.STRONG], "_strong_raw_pair_" + str(time.monotonic_ns()))
    validator.validate_runtime_pair(pair)
    R.require(R.common_domain_sha(pair) == plan["common_runtime_domain_sha256"], "same actual strong normalized U/I domain")
    intent = R.read(R.check_ref(root, value["intent_ref"]))
    R.require(intent.get("schema") == "strong_gpu_exact_cell_prelaunch_intent_v1" and intent.get("plan_ref") == value["plan_ref"] and
              intent.get("source_lock_ref") == value["source_lock_ref"] and intent.get("job_id") == value["job_id"], "parent prelaunch closed intent")
    off = R.read(R.check_ref(root, value["off_qualification_ref"]))
    R.require(off.get("status") == "PASS_STRONG_NATIVE_OFF_WORKLOAD_LIFECYCLE" and off.get("gpu_uuid") == value["gpu_uuid"] and
              off.get("common_runtime_domain_sha256") == plan["common_runtime_domain_sha256"] and
              off.get("original_engine_shutdown_returned") is True and off.get("native_tail_drained") is True, "actual prior strong off lifecycle")
    validation = R.load(root, plan["validation_source_ref"], "_strong_raw_completed_guard_" + str(time.monotonic_ns()))
    validation.validate_guard(R.read(R.check_ref(root, value["off_guard_ref"])), gpu_uuid=value["gpu_uuid"],
                              job_id=off["run_id"], wrapper_path=off["runner_ref"]["path"])
    R.require(shutil.disk_usage(root).free >= FLOOR + RESERVE, "raw disk floor and actual reserve")
    ledger = R.read(R.safe(root, R.LEDGER))
    R.require(ledger["gpu_wall_seconds"] + value["seconds_limit"] + 20 <= R.MAX_GPU_SECONDS, "original eight-hour remaining budget")
    storage = R.safe(root, value["storage_template_relative"])
    R.require(storage.is_dir(), "existing original 24-file input template")
    return dict(config=value, refs=refs, plan=plan, pair=pair, storage=storage, ledger=ledger)


def command(root, config_path, config):
    return [".venv/bin/python", "-B", config["runner_ref"]["path"], "--project", str(root), "--config", config_path, "--execute"]


def verify_guard(root, relative, gates):
    R, config = driver_module(), gates["config"]
    active = R.read(R.safe(root, R.LEDGER)).get("active_reservation")
    expected = dict(label=config["job_id"], gpu_uuid=config["gpu_uuid"], seconds_limit=config["seconds_limit"],
        reserved_seconds=config["seconds_limit"] + 20, command=command(root, relative, config), permissions=config["permissions_ref"],
        evidence=str(R.safe(root, config["output_relative"]).parent))
    R.require(os.name == "posix" and type(active) is dict and all(active.get(key) == value and type(active.get(key)) is type(value)
              for key, value in expected.items()), "actual original raw guard reservation")
    R.require(os.getsid(0) == os.getpgid(0) == active.get("session_id") == active.get("process_group") and
              active.get("state") != "cleanup_unresolved" and os.environ.get("CUDA_VISIBLE_DEVICES") == config["gpu_uuid"] and
              os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1", "same original raw Linux session/device/offline")
    tokens = (Path("/proc") / str(active["runner_pid"]) / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    R.require(R.GUARD in tokens and config["job_id"] in tokens and config["permissions_ref"]["path"] in tokens, "actual original guard process")
    return active


def configure_original(root, gates, guard, relative):
    R = driver_module()
    config, refs, plan = gates["config"], gates["refs"], gates["plan"]
    original_relative = R.PREVIOUS + "/" + OLD_SUFFIX
    R.require(refs[original_relative]["sha256"] == OLD_SHA, "actual original raw driver pin")
    old = R.load(root, refs[original_relative], "_strong_raw_old_original_" + str(time.monotonic_ns()))
    tree, patch_proof = source_patch(R.check_ref(root, refs[original_relative]).read_bytes())
    old.ROOT, old.SCRIPT, old.DELIVERY = root, config["runner_ref"]["path"], Path(config["runner_ref"]["path"]).parent.as_posix()
    old.CONFIG_PATH, old.LABEL, old.PURPOSE = R.safe(root, relative), config["job_id"], "STRONG_DOMAIN_RAW_CALIBRATION_ONLY"
    old.PROMPT_TOKENS, old.CACHED_TOKENS = PROMPT_TOKENS, CACHED_TOKENS
    old.PROMPT_FIRST, old.SEEDS, old.OPERATION_COUNT = FIRST, SEEDS, OPERATIONS
    old.SITE_SDK = R.SDK
    def strong_engine(base, storage):
        engine = deepcopy(gates["pair"]["U"]["engine"])
        extra = engine["kv_transfer_config"]["kv_connector_extra_config"]
        extra["shared_storage_path"] = str(storage)
        extra["prefix_io_parent_admission"]["run_id"] = config["job_id"]
        extra["prefix_io_observation_run_id"] = config["job_id"]
        return engine
    old.config_for_engine = strong_engine
    compat = dict(label=config["job_id"], purpose=old.PURPOSE, gpu_uuid=config["gpu_uuid"],
        out=config["output_relative"], storage=config["storage_template_relative"], input_manifest=config["input_manifest_ref"]["path"],
        collector_relative=plan["collector_source_ref"]["path"], source_lock=config["source_lock_ref"],
        gpu_entry_binding_ref=config["intent_ref"])
    def checked_execution(project, passed_config, passed_refs, passed_guard):
        R.require(project == root and passed_config == compat and passed_refs == refs, "same strict strong raw execution inputs")
        current = verify_guard(root, relative, gates)
        R.require(all(current[key] == passed_guard[key] for key in ("id", "label", "gpu_uuid", "session_id", "process_group")),
                  "same original parent reservation for all six children")
        for row in (config["plan_ref"], config["intent_ref"], config["runner_ref"], config["input_manifest_ref"]):
            R.check_ref(root, row)
        return current
    old.verify_execution_inputs = checked_execution
    exec(compile(tree, str(Path(__file__).resolve()), "exec", dont_inherit=True), vars(old))
    return old, compat, patch_proof


def normalize_actual(root, gates, report):
    R, plan = driver_module(), gates["plan"]
    R.require(report["status"] == "PASS_NATIVE_SIX_PROCESS_RAW_CAPTURE_REQUIRES_VALIDATION" and len(report["windows"]) == 6,
              "six actual complete original windows before raw packaging")
    windows = []
    for index, source in enumerate(report["windows"]):
        entry = plan["cells"][0]["entries"][index // 2]
        row = deepcopy(source)
        frontend = row["frontend"]["output"]
        R.require(row["request_id"] == frontend["request_id"] == row["capture"]["run_id"] and
                  type(frontend["native_request_id"]) is str and frontend["native_request_id"],
                  "actual original external/internal request mapping")
        row.update(pair_id=entry["pair_id"], arm="baseline" if row["condition"] == "A" else "action",
            run_id=row["request_id"], external_request_id=row["request_id"], request_id=frontend["native_request_id"],
            gpu_uuid=plan["gpu_uuid"], source_lock_sha256=plan["source_lock_ref"]["sha256"],
            model_sha256=plan["model_manifest_ref"]["sha256"], kv_layout_sha256=plan["kv_layout_ref"]["sha256"],
            output_token_ids=row["frontend"]["output"]["output_token_ids"])
        windows.append(row)
    return dict(schema="strong_gpu_exact_cell_raw_measurements_v1", origin="native_gpu_recording", synthetic_fixture=False,
        plan_ref=gates["config"]["plan_ref"], runtime_identity=dict(gpu_uuid=plan["gpu_uuid"],
        common_runtime_domain_sha256=plan["common_runtime_domain_sha256"], source_lock_sha256=plan["source_lock_ref"]["sha256"],
        model_sha256=plan["model_manifest_ref"]["sha256"], kv_layout_sha256=plan["kv_layout_ref"]["sha256"], kernel_mode="eager"),
        cells=[dict(id=plan["cells"][0]["id"], windows=windows)], source_verification_refs={},
        completed_original_guard_required=True, production_qualified=False, original_estimator_not_run_or_changed=True,
        qualification_pending="Parent closes before/after full-source receipts and actual completed original guard before issuer")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--window-index", type=int, choices=range(6))
    parser.add_argument("--prepare-plan", action="store_true", help="CPU template only; null GPU cannot be executed or issued")
    parser.add_argument("--prepare-geometry", action="store_true", help="CPU model-config geometry only, never an actual GPU layout receipt")
    parser.add_argument("--source-lock", help="relative complete source-lock JSON")
    parser.add_argument("--pair", help="relative actual PRIVATE_U_I_CONFIG.json")
    parser.add_argument("--input-manifest", help="relative original 24-file native input manifest")
    parser.add_argument("--geometry", help="relative canonical compact model-derived geometry JSON, without newline")
    parser.add_argument("--output", help="fresh relative CPU prelaunch plan template")
    parser.add_argument("--job-id", default="server12-strong-exact-cal01")
    args = parser.parse_args(argv)
    R = driver_module()
    if args.prepare_geometry:
        R.require(not args.prepare_plan and not args.execute and not args.preflight and args.config is None and args.output,
                  "explicit fresh CPU geometry metadata output")
        root = args.project.resolve(strict=True)
        geometry = prepare_geometry(root, R.safe(root, args.output))
        print(json.dumps(dict(status="CPU_MODEL_GEOMETRY_METADATA_CREATED", actual_gpu_runs=0, actual_GPU_layout_verified=False,
                              source_ref=geometry)))
        return 0
    if args.prepare_plan:
        R.require(not args.execute and not args.preflight and args.config is None and args.window_index is None and
                  all((args.source_lock, args.pair, args.input_manifest, args.geometry, args.output)), "explicit complete CPU template inputs")
        root = args.project.resolve(strict=True)
        plan = prepare_plan(root, source_lock_ref=R.ref(root, args.source_lock), pair_ref=R.ref(root, args.pair),
                            input_manifest_ref=R.ref(root, args.input_manifest), geometry_ref=R.ref(root, args.geometry),
                            output=R.safe(root, args.output), job_id=args.job_id)
        print(json.dumps(dict(status="CPU_STRONG_RAW_PLAN_TEMPLATE_CREATED", actual_gpu_runs=0,
                              gpu_uuid=plan["gpu_uuid"], executable=False, fit_pairs=2, heldout_pairs=1)))
        return 0
    R.require(args.config and (args.execute != args.preflight), "one explicit CPU preflight or original guarded execution")
    root = args.project.resolve(strict=True)
    gates = verify_configuration(root, R.safe(root, args.config), full=True)
    if args.preflight:
        print(json.dumps(dict(status="PASS_CPU_STRONG_RAW_ENTRY_INPUTS", actual_gpu_runs=0, runtime_qualified=False,
            original_guard_command=[".venv/bin/python", "-B", R.GUARD, "--permissions-path", gates["config"]["permissions_ref"]["path"],
                "--label", gates["config"]["job_id"], "--seconds", str(gates["config"]["seconds_limit"]), "--", *command(root, args.config, gates["config"])])))
        return 0
    guard = verify_guard(root, args.config, gates)
    old, compat, proof = configure_original(root, gates, guard, args.config)
    report = (old.execute_parent(root, compat, gates["refs"], guard) if args.window_index is None
              else old.execute_window(root, compat, gates["refs"], guard, args.window_index))
    if args.window_index is None and report["status"] == "PASS_NATIVE_SIX_PROCESS_RAW_CAPTURE_REQUIRES_VALIDATION":
        R.new_json(R.safe(root, gates["config"]["output_relative"]) / "strong-exact-cell-raw-pending-parent-closure.json", normalize_actual(root, gates, report))
        R.new_json(R.safe(root, gates["config"]["output_relative"]) / "THIN_STRONG_SOURCE_PROOF.json", proof)
    print(json.dumps(dict(status=report["status"], actual_gpu_jobs=1, completed_windows=len(report["windows"]), production_qualified=False)))
    return 0 if report["status"].startswith("PASS_NATIVE_") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(json.dumps(dict(status="STRONG_RAW_ENTRY_REJECTED", reason=str(exc), GPU_job_started=False)))
        raise SystemExit(2)
