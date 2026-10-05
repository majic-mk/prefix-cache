"""Site-authorized preregistration and raw reserialization for real native cost data."""
import argparse
import ast
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys


_spec = importlib.util.spec_from_file_location("_server11_c5_gpu_entry_native_contract",
    Path(__file__).with_name("native_conditional_cost.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = C
_spec.loader.exec_module(C)
require = C.require
ORIGINAL = "third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py"


def relative(root, value):
    require(type(value) is str and value and "\\" not in value and ":" not in value and
            all(p not in ("", ".", "..") for p in value.split("/")) and not value.startswith("/"),
            "bounded relative path")
    path = root
    for part in value.split("/"):
        path /= part
        require(not path.is_symlink(), "symlink metadata refused")
    require(root in path.resolve().parents, "project path confinement")
    return path


def ref(root, value):
    path = relative(root, value)
    require(path.is_file() and 0 < path.stat().st_size <= 32 * 1024**2, "bounded metadata/source file")
    raw = path.read_bytes()
    return dict(path=value, bytes=len(raw), sha256=sha256(raw).hexdigest())


def read(root, value):
    return C.EvidenceRef.from_mapping(ref(root, value)).json(root)


def write(root, value, obj):
    path = relative(root, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return ref(root, value)


def constants(path):
    """Read literal configuration only; never execute/import a GPU launcher."""
    values = {}
    def evaluate(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.Name) and node.id in values: return values[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            result = [evaluate(x) for x in node.elts]
            return tuple(result) if isinstance(node, ast.Tuple) else result
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = evaluate(node.left), evaluate(node.right)
            require(type(left) is type(right) is str, "literal text concatenation only")
            return left + right
        raise ValueError("nonliteral configuration expression")
    for node in ast.parse(path.read_bytes()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try: values[node.targets[0].id] = evaluate(node.value)
            except ValueError: pass
    return values


def create_plan(root, *, source_lock_relative, config_relative, output_relative):
    root = Path(root).resolve(strict=True)
    config,pinned,binding=C.binding_api().verify_configuration(root,relative(root,config_relative))
    require(config['source_lock']==source_lock_relative, 'actual final site source closure required')
    require(config_relative==C.ENTRY+'/NATIVE_COST_CONFIG.json', 'new fixed native config')
    lock_ref = ref(root, source_lock_relative)
    lock = C.EvidenceRef.from_mapping(lock_ref).json(root)
    rows = lock["files"]
    pinned = {r["path"]: r for r in rows}
    require(len(pinned) == len(rows), "unique source closure")
    def frozen(value):
        actual = ref(root, value)
        require(pinned.get(value) == actual, "unfrozen config/source: " + value)
        return actual
    frozen(config_relative)
    config = read(root, config_relative)
    require(config["source_lock"] == source_lock_relative, "config/source lock differs")
    wrapper_relative = str(Path(config_relative).parent).replace("\\", "/") + "/run_native_cost_experiment.py"
    wrapper_ref = frozen(wrapper_relative)
    h = constants(relative(root, wrapper_relative))
    require(h["SCRIPT"] == wrapper_relative and config["label"] == h["LABEL"] and
            h["ORDER"] == (("A", "B"), ("B", "A"), ("A", "B")) and
            h["PROMPT_FIRST"] == (18100, 19100, 20100) and h["SEEDS"] == (1829, 1830, 1831) and
            type(h.get("PROMPT_TOKENS")) is int and h["PROMPT_TOKENS"] == 129 and
            type(h.get("OPERATION_COUNT")) is int and h["OPERATION_COUNT"] == 1,
            "harness fixed actual experimental constants")
    require(h['OVERLAY'] == h['CANDIDATE']+'/source' and
            h['REACTOR'] == h['OVERLAY']+'/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py' and
            h['MAX_ACCEPTED_PARENTS'] == 8 and
            config['collector_relative'] == h['CANDIDATE']+'/native_full_step_collector.py',
            'actual common candidate source and collector configuration')
    frozen(h["CANDIDATE"]+"/single_file_runtime_binding.py")
    frozen(h["COMMON"])
    common = constants(relative(root, h["COMMON"]))
    model_ref = frozen(common["MODEL_PLAN"])
    config_ref = frozen(common["MODEL"] + "/config.json")
    model_config = C.EvidenceRef.from_mapping(config_ref).json(root)
    head_dim = model_config["hidden_size"] // model_config["num_attention_heads"]
    require(model_config["hidden_size"] % model_config["num_attention_heads"] == 0, "integer model head dimension")
    geometry = dict(model_config_sha256=config_ref["sha256"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_key_value_heads=model_config["num_key_value_heads"], head_dim=head_dim,
        dtype="bfloat16", dtype_bytes=2, tokens_per_block=16, tensor_parallel_size=1,
        physical_block_bytes=h["FILE_BYTES"])
    require(h["FILE_BYTES"] == geometry["num_hidden_layers"] * 2 * geometry["num_key_value_heads"] *
            head_dim * geometry["dtype_bytes"] * geometry["tokens_per_block"], "model-derived exact KV storage quantum")
    entries = []
    for index, (first, seed, order) in enumerate(zip(h["PROMPT_FIRST"], h["SEEDS"], h["ORDER"])):
        ids = list(range(1000, 1000 + h["PROMPT_TOKENS"])); ids[0] = first
        entries.append(dict(pair_id="pair-" + str(index), split="calibration" if index < 2 else "validation",
            arm_order="".join(order), seed=seed, prompt_token_ids=ids,
            prompt_sha256=C.canonical_hash(ids), prefix_family_sha256=C.canonical_hash(ids[:16]),
            trace_sha256=C.canonical_hash(dict(prompt_token_ids=ids, seed=seed, output_tokens=128)),
            workload_sha256=C.canonical_hash(dict(prompt_token_ids=ids, output_tokens=128,
                                                 temperature=0, ignore_eos=True))))
    event_ref = frozen(".venv/lib/python3.12/site-packages/torch/cuda/streams.py")
    require(event_ref["bytes"] == 10672 and event_ref["sha256"] ==
            "3b34ed08b67cf2ec411e1a483b31835bbff8652c915a08f2a0e4c140e342dd9c", "current actual installed event source")
    original_ref = frozen(ORIGINAL)
    C.original_estimator(relative(root, ORIGINAL))
    plan = dict(scope="server11_preregistered_native_conditional_cell_v1", schema_version=1,
        qualification_rule="zero_observed_holdout_underprediction_no_refit_v1",
        qualification_origin="new_finite_engineering_gate_not_original_SLO", internal_step_budget_ns=None,
        evidence_origin="native_runtime_preregistered",cpu_preparation_only=False,
        site_binding_ref=config["gpu_entry_binding_ref"],config_ref=frozen(config_relative),
        project_root=str(root).replace("\\", "/"), gpu_uuid=config["gpu_uuid"], job_id=config["label"],
        journal_run_id=config["label"], source_lock_ref=lock_ref, source_lock_files=len(rows),
        wrapper_source_ref=wrapper_ref, collector_source_ref=frozen(config["collector_relative"]),
        cuda_event_source_ref=event_ref, model_plan_ref=model_ref, model_sha256=model_ref["sha256"],
        model_config_ref=config_ref, kv_layout=geometry, kv_layout_sha256=C.canonical_hash(geometry),
        native_source_sha256=frozen(h["REACTOR"])["sha256"], original_estimator_ref=original_ref,
        native_source_ref=frozen(h['REACTOR']), common_overlay_relative=h['OVERLAY'],
        common_owner_parameters=dict(max_accepted_parents=8, bridge_is_none=True),
        input_manifest_ref=frozen(config["input_manifest"]), stage="ssd_read", transfer_quantum_bytes=h["FILE_BYTES"],
        units=h["OPERATION_COUNT"], operations=h["OPERATION_COUNT"],
        cached_prompt_tokens=128, prompt_tokens=129, output_tokens=128,
        measured_offset=16, warmup_offsets=[1], external_warmup_output_tokens=128,
        external_flush_output_tokens=1, entries=entries,
        process_design="six_fresh_original_processes_and_handlers_one_guard", production_qualified=False)
    C.validate_native_plan(root,plan)
    plan_ref = write(root, output_relative, plan)
    return plan_ref


def validate_common_source_binding(root, plan, child):
    """Require actual new owner and unchanged full-frame adapter identity."""
    C.validate_native_plan(root,plan)
    require(child.get('origin')=='native_gpu_recording' and
            child.get('site_binding_ref')==plan['site_binding_ref'] and
            child.get('synthetic_cpu_contract') is not True and
            child.get('cpu_preparation_only') is not True,
            'real child native origin/site binding; synthetic data refused')
    native = plan['native_source_ref']
    require(native.get('sha256') == 'a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47',
            'actual unchanged C5 native source required')
    collector = plan['collector_source_ref']
    require(collector.get('path') == 'artifacts/prefix_io_v1/server11-c5-gpu-entry-device-revision-20261004/common_candidate/native_full_step_collector.py' and
            collector.get('sha256') == 'bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf' and ref(root, collector['path']) == collector,
            'actual unchanged C5 plan collector required')
    require(native['sha256'] == plan['native_source_sha256'] and
            native['path'] == plan['common_overlay_relative']+
                '/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py',
            'plan actual common native source')
    require(ref(root, native['path']) == native, 'actual common native source bytes')
    owner = plan['common_owner_parameters']
    require(type(owner) is dict and set(owner) == {'max_accepted_parents', 'bridge_is_none'} and
            type(owner.get('max_accepted_parents')) is int and owner['max_accepted_parents'] == 8 and
            owner.get('bridge_is_none') is True,
            'fixed common owner parameters')
    binding = child['native_source_binding']
    require(binding.get('actual_source_ref') == native and binding.get('original_run_code_verified') is True and
            type(binding.get('max_accepted_parents')) is int and binding.get('max_accepted_parents') == 8 and
            binding.get('bridge_is_none') is True,
            'actual common owner binding')
    loaded = binding.get('loaded_native_modules')
    require(type(loaded) is list and 1 <= len(loaded) <= 128 and
            len({row['module'] for row in loaded}) == len(loaded), 'bounded unique actual loaded native modules')
    for row in loaded:
        name, source = row['module'], row['source_ref']
        require((name in ('py_kvcache','prefix_io_control') or
                 name.startswith(('py_kvcache.','prefix_io_control.'))) and
                source['path'].startswith(plan['common_overlay_relative']+'/') and
                ref(root,source['path']) == source, 'actual native module source escaped or changed')
    require(any(row['module']=='py_kvcache.reactor' and row['source_ref']==native for row in loaded),
            'actual reactor module absent')
    adapter_binding = child['collector_native_source_binding']
    require(type(adapter_binding) is dict and adapter_binding.get('same_adapter_all_frames') is True and
            type(adapter_binding.get('full_frame_count')) is int and adapter_binding['full_frame_count'] == 128 and
            adapter_binding == dict(
        legacy_lookup_key='third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py',
        actual_source_ref=native, adapter_native_source_sha256_before=native['sha256'],
        adapter_native_source_sha256_after=native['sha256'], same_adapter_all_frames=True,
        full_frame_count=128), 'actual common adapter source and all 128 frames')


def serialize_runtime_record(root, *, runtime_relative, plan_ref, source_verification_refs):
    root = Path(root).resolve(strict=True)
    plan = C.EvidenceRef.from_mapping(plan_ref).json(root)
    C.validate_native_plan(root,plan)
    C.validate_prelaunch_plan(root,plan_ref)
    runtime_ref = ref(root, runtime_relative)
    report = C.EvidenceRef.from_mapping(runtime_ref).json(root)
    require(report.get("site_binding_ref")==plan["site_binding_ref"] and
            report.get("cpu_preparation_only") is not True and
            report.get("synthetic_cpu_contract") is not True and
            report.get("status") == "PASS_NATIVE_SIX_PROCESS_RAW_CAPTURE_REQUIRES_VALIDATION" and
            report.get("origin") == "native_gpu_recording" and report.get("label") == plan["job_id"] and
            report.get("gpu_uuid") == plan["gpu_uuid"] and report.get("original_model_subprocesses_started") == 6 and
            report.get("actual_guarded_gpu_job_count") == 1 and report.get("input_template_unchanged") is True and
            report.get("original_engine_shutdown_returned") is True and report.get("load_planner") == "off",
            "actual six-process parent runtime receipt failed")
    rows, children = report["windows"], report["children"]
    require(len(rows) == len(children) == 6, "six actual original child receipts")
    windows, pids, storages = [], set(), set()
    for index, (row, child) in enumerate(zip(rows, children)):
        require(child["window_index"] == index and type(child["exit"]) is int and child["exit"] == 0,
                "sequential actual child exit")
        receipt = C.EvidenceRef.from_mapping(row["child_receipt_ref"]).json(root)
        validate_common_source_binding(root, plan, receipt)
        require(receipt.get("status") == "PASS_NATIVE_FULL_STEP_RAW_CAPTURE_REQUIRES_VALIDATION" and
                receipt.get("window_index") == index and receipt.get("fresh_original_process") is True and
                receipt.get("original_engine_shutdown_returned") is True and
                receipt.get("gpu_uuid") == plan["gpu_uuid"] and receipt.get("source_lock") == plan["source_lock_ref"]["path"] and
                receipt["model"]["manifest_sha256"] == plan["model_sha256"] and
                receipt["model"]["plan_sha256"] == plan["model_plan_ref"]["sha256"], "actual child source/model/native lifecycle")
        pid = C.integer(receipt["subprocess_pid"], "actual child PID", 1)
        require(pid not in pids and receipt["private_storage"] not in storages and
                receipt["subprocess_sid"] == report["guard"]["session_id"], "fresh original process/storage same original guard session")
        pids.add(pid); storages.add(receipt["private_storage"])
        require(len(receipt["windows"]) == 1, "one measured window per fresh original process")
        actual = receipt["windows"][0]
        warmups = receipt["warmups"]
        require(type(warmups) is list and len(warmups) == 1 and
                len(warmups[0]["output_token_ids"]) == plan["external_warmup_output_tokens"] and
                len(warmups[0]["flush"]["output_token_ids"]) == plan["external_flush_output_tokens"] and
                warmups[0]["prompt_token_ids"] == row["prompt_token_ids"] and warmups[0]["seed"] == row["seed"],
                "actual separate full-output warmup and original one-token flush")
        for key in actual:
            require(row.get(key) == actual[key], "parent/child raw window disagreement: " + key)
        require(row["native_journal"] == receipt["native_journal"] and
                row["native_post_shutdown"] == receipt["native_post_shutdown"], "parent/child actual owner evidence disagreement")
        entry = plan["entries"][index // 2]
        arm = "baseline" if row["condition"] == "A" else "action"
        require(row["condition"] in ("A", "B") and row["pair_index"] == index // 2 and
                row["prompt_token_ids"] == entry["prompt_token_ids"] and row["seed"] == entry["seed"] and
                row["frontend"]["output"]["request_id"] == row["request_id"] == row["capture"]["run_id"] and
                row["frontend"]["output"]["num_cached_tokens"] == 128 and
                row["frontend"]["output"]["output_token_ids"] == [token for frame in row["capture"]["frames"]
                    for _, tokens in frame["outputs"] for token in tokens], "actual frontend/scalar/request correspondence")
        windows.append(dict(pair_id=entry["pair_id"], arm=arm, seed=row["seed"],
            prompt_token_ids=row["prompt_token_ids"], output_token_ids=row["frontend"]["output"]["output_token_ids"],
            request_id=row["frontend"]["output"]["native_request_id"], external_request_id=row["request_id"],
            run_id=row["capture"]["run_id"], capture=row["capture"],
            gpu_uuid=receipt["gpu_uuid"], source_lock_sha256=plan["source_lock_ref"]["sha256"],
            model_sha256=receipt["model"]["manifest_sha256"], kv_layout_sha256=plan["kv_layout_sha256"],
            independent_payload=row["independent_payload"], native_journal=row["native_journal"],
            native_post_shutdown=row["native_post_shutdown"], native_tail_assertions=receipt["native_tail_assertions"],
            raw_child_ref=row["child_receipt_ref"], subprocess_pid=pid, private_storage=receipt["private_storage"]))
    return dict(scope="server11_native_paired_measurements_v1", origin="native_gpu_recording", plan_ref=plan_ref,
        site_binding_ref=plan['site_binding_ref'],windows=windows,
        source_verification_refs=source_verification_refs,raw_runtime_ref=runtime_ref,
        actual_subprocess_count=len(pids),production_qualified=False)



def verify_raw_window(root, *, plan_ref, child_relative):
    """Cheap early semantic check after a child naturally exits; no final grant."""
    root = Path(root).resolve(strict=True)
    plan = C.EvidenceRef.from_mapping(plan_ref).json(root)
    C.validate_native_plan(root,plan)
    C.validate_prelaunch_plan(root,plan_ref)
    child_ref = ref(root, child_relative)
    child = C.EvidenceRef.from_mapping(child_ref).json(root)
    validate_common_source_binding(root, plan, child)
    require(child.get("status") == "PASS_NATIVE_FULL_STEP_RAW_CAPTURE_REQUIRES_VALIDATION" and
            child.get("fresh_original_process") is True and child.get("original_engine_shutdown_returned") is True and
            child.get("gpu_uuid") == plan["gpu_uuid"] and child.get("source_lock") == plan["source_lock_ref"]["path"] and
            child["model"]["manifest_sha256"] == plan["model_sha256"], "original closed child failed")
    require(len(child["windows"]) == 1, "one measured window")
    row = child["windows"][0]
    index = C.integer(child["window_index"], "actual window index")
    require(index < 6, "bounded actual window index")
    entry = plan["entries"][index // 2]
    arm = entry["arm_order"][index % 2]
    require(row["condition"] == arm and row["pair_index"] == index // 2 and row["seed"] == entry["seed"] and
            row["prompt_token_ids"] == entry["prompt_token_ids"] and row["frontend"]["output"]["num_cached_tokens"] == 128,
            "actual selected child frozen workload")
    capture = row["capture"]
    event = capture["event_source"]
    expected_event = plan["cuda_event_source_ref"]
    require(event["sha256"] == expected_event["sha256"] and event["bytes"] == expected_event["bytes"] and
            event["path"] == plan["project_root"].rstrip("/") + "/" + expected_event["path"],
            "real event source mismatch")
    require(row["frontend"]["output"]["request_id"] == row["request_id"] == capture["run_id"],
            "original external request/capture run mapping")
    frames = C.validate_capture(capture, run_id=row["request_id"],
        request_id=row["frontend"]["output"]["native_request_id"],
        output_ids=row["frontend"]["output"]["output_token_ids"], prompt_tokens=129,
        measured_offset=16, warmup_offsets=[1], cached_tokens=128)
    drain = C.original_post_shutdown_drain(child["native_post_shutdown"], child["native_tail_assertions"],
        run_id=plan["journal_run_id"], native_source_sha256=plan["native_source_sha256"])
    io = C.validate_io(child["native_journal"], drain, capture=capture, frames=frames,
        run_id=plan["journal_run_id"], native_source_sha256=plan["native_source_sha256"],
        arm="baseline" if arm == "A" else "action", measured_offset=16,
        physical_bytes=plan["transfer_quantum_bytes"] * plan["units"], operations=plan["operations"],
        independent_payload=row["independent_payload"])
    return dict(status="PASS_NATIVE_RAW_WINDOW_REQUIRES_FINAL_GUARD_VALIDATION", origin="native_gpu_recording",
        native_execution_verified=False, native_cost_qualified=False, valid_native_receipt=None,
        full_runtime_cost_qualified=False, on_observation_cost_measured=False, gpu_launch_allowed=False,
        effective_cost_upper_ns=None,effective_step_budget_ns=None,window_index=index,
        child_ref=child_ref,plan_ref=plan_ref,full_output_tokens=128,full_frames=128,
        selected_gpu_elapsed_ns=frames[16]['gpu_elapsed_ns'],native_io=io,
        conditional_cost_cell_qualified=False,production_qualified=False)



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--window-check", action="store_true")
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--source-lock")
    parser.add_argument("--config")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-ref", required=True)
    parser.add_argument("--runtime")
    parser.add_argument("--guard")
    parser.add_argument("--before-source")
    parser.add_argument("--after-source")
    parser.add_argument("--measurements")
    parser.add_argument("--result")
    args = parser.parse_args(argv)
    root = args.project.resolve(strict=True)
    if args.prepare:
        plan_ref = create_plan(root, source_lock_relative=args.source_lock,
                              config_relative=args.config, output_relative=args.plan)
        receipt_ref = write(root, args.plan_ref, plan_ref)
        print(json.dumps(dict(status="FROZEN_SITE_AUTHORIZED_NATIVE_COST_PLAN", plan_ref=plan_ref,
                              receipt_ref=receipt_ref, gpu_operations=0)))
    elif args.window_check:
        plan_ref = read(root, args.plan_ref)
        require(plan_ref["path"] == args.plan, "expected prelaunch plan reference")
        result = verify_raw_window(root, plan_ref=plan_ref, child_relative=args.runtime)
        result_ref = write(root, args.result, result)
        print(json.dumps(dict(status=result["status"], result_ref=result_ref,
                              window_index=result["window_index"], production_qualified=False)))
    else:
        plan_ref = read(root, args.plan_ref)
        require(plan_ref["path"] == args.plan, "expected prelaunch plan reference")
        source_refs = {phase: ref(root, value) for phase, value in
                       (("before", args.before_source), ("after", args.after_source))}
        record = serialize_runtime_record(root, runtime_relative=args.runtime, plan_ref=plan_ref,
                                          source_verification_refs=source_refs)
        measurements_ref = write(root, args.measurements, record)
        guard_ref = ref(root, args.guard)
        result = C.verify_native_cell(root, plan_ref=plan_ref, measurements_ref=measurements_ref,
            guard_ref=guard_ref, expected_plan_ref=plan_ref, expected_guard_ref=guard_ref,
            original_source_path=relative(root, ORIGINAL))
        result_ref = write(root, args.result, result)
        print(json.dumps(dict(status="NATIVE_COST_EVIDENCE_VERIFIED", result_ref=result_ref,
            conditional_cost_cell_qualified=result["conditional_cost_cell_qualified"],
            calibration_predicted_upper_ns=result["calibration_predicted_upper_ns"],
            heldout_errors=result["heldout_errors"], production_qualified=False)))
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError,KeyError) as exc:
        print(json.dumps(dict(status="GPU_ENTRY_REJECTED",reason=str(exc),gpu_started=False)))
        raise SystemExit(2)
