"""One guarded original natural-prompt A or B raw cost window.

This thin consumer uses the unchanged original execution, single-cell CUDA/I/O
validator and cleanup. It does not run the six-window estimator, issue a table,
reuse an old GPU qualification, assert a SLO or make a strategy-benefit claim.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

P = "artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004"
DRIVER = P + "/runner/strong_trace_runner_v7.py"
DRIVER_SHA = "0d916dedd52c2aaec3822adff6560987d443425619f96d57a027a60be10fe9dd"
STRONG_RAW = P + "/runner/strong_native_cost_runner_v7.py"
STRONG_SHA = "d07c54e330b0004a20fc380ab47a7e742f05736ad61180f0a42b38b9c635de55"
NATURAL = dict(path="artifacts/prefix_io_v1/server12-natural-source-audit-20261005/ACTUAL_PUBLIC_ORIGINAL_MANIFEST_01.json",
    bytes=113401, sha256="01bb329d696f61bc9363aaaff553dd380ce457a9f01713f0991562aa92c8d340")
SDK_PATH = P + "/raw_unit_cost/raw_sdk_binding.py"
CONFIG_FIELDS = {"schema", "job_id", "gpu_uuid", "seconds_limit", "window_index", "source_lock_ref", "source_proof_ref",
    "plan_ref", "permissions_ref", "actual_u_collection_parent_ref", "actual_u_collection_guard_ref",
    "output_relative", "storage_template_relative", "input_manifest_ref", "runner_ref"}
FALSE_FIELDS = ("table_issued", "cost_qualified", "ordinary_I_authorized", "formal_goodput_allowed", "strategy_improvement_proved")
MAX_SECONDS = 300
RESERVE = 512 * 1024**2
FLOOR = 8 * 1024**3


def require(value, reason):
    if not value:
        raise ValueError("NATURAL_RAW_UNIT_REJECTED: " + reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _source_module(path, sha, name):
    path = Path(path).resolve()
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == sha, "actual pinned pure source bytes")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def driver(root):
    return _source_module(Path(root) / DRIVER, DRIVER_SHA, "_natural_raw_driver_" + str(time.monotonic_ns()))


def selected_natural_record(document):
    require(type(document) is dict and document.get("schema") == "natural_trace_workload_v1" and
            document.get("partition_counts") == {"calibration": 30, "development": 5, "evaluation": 5},
            "same complete frozen natural workload")
    records = document.get("records")
    require(type(records) is list and len(records) == 40 and all(type(r) is dict for r in records), "complete original40 records")
    row = records[0]
    require(row.get("request_id") == 0 and type(row.get("request_id")) is int and row.get("split") == "calibration" and
            row.get("seed") == 0 and type(row.get("seed")) is int and
            row.get("min_tokens") == row.get("max_tokens") == 128 and
            type(row.get("prompt_token_ids")) is list and len(row["prompt_token_ids"]) == 185 and
            all(type(token) is int and token >= 0 for token in row["prompt_token_ids"]) and
            hashlib.sha256(canonical(row["prompt_token_ids"])).hexdigest() ==
            "0f5bdcf71eb1e2e3cfdf76c1a157f88b0e91def2526018bed387378cedb9536c" and row.get("prompt_sha256") ==
            "a61dd783a3740974b3678b695d6b080686680237b4af2096cb7a37e455f580d3", "exact actual calibration first request, no padding/injection")
    return deepcopy(row)


def cell_descriptor(record):
    tokens = record["prompt_token_ids"]
    require(len(tokens) == 185 and record["request_id"] == 0 and record["split"] == "calibration", "original calibrated source record")
    entry = dict(pair_id="natural-cal0-unit185", split="raw_only_calibration", arm_order="AB", seed=record["seed"],
        prompt_token_ids=deepcopy(tokens), prompt_sha256=hashlib.sha256(canonical(tokens)).hexdigest(),
        source_prompt_text_sha256=record["prompt_sha256"],
        prefix_family_sha256=hashlib.sha256(canonical(tokens[:16])).hexdigest(),
        trace_sha256=hashlib.sha256(canonical(dict(prompt_token_ids=tokens, seed=record["seed"], output_tokens=128))).hexdigest(),
        original_natural_request_id=0, original_natural_split="calibration")
    return dict(id="natural185-context200-read1-existing0-rawonly", prompt_tokens=185,
        cached_prompt_tokens=176, cached_prompt_tokens_semantics="initial_execution_pre_context",
        frontend_cached_prompt_tokens=176, initial_execution_pre_context=176, measured_offset=16, warmup_offsets=[1],
        operations=1, stage="ssd_read", transfer_quantum_bytes=917504,
        existing_io=[dict(ops=0, bytes=0)] * 4, entries=[entry])


def make_cpu_plan(root, refs, *, source_lock_ref, pair_ref, input_manifest_ref, geometry_ref,
                  runner_ref, parent_ref, guard_ref, gpu_uuid, job_id):
    """Pure metadata factory; bound refs never grant table or GPU execution."""
    R = driver(root)
    pair = R.read(R.check_ref(root, pair_ref))["configurations"]
    record = selected_natural_record(R.read(R.check_ref(root, NATURAL)))
    required = dict(collector_source_ref=P+"/runner/bounded_native_full_step_collector_v2.py",
        original_collector_source_ref=P+"/runner/bounded_native_full_step_collector.py",
        cuda_event_source_ref=".venv/lib/python3.12/site-packages/torch/cuda/streams.py",
        native_source_ref=R.NATIVE+"/reactor.py", original_guard_source_ref=R.GUARD,
        validation_source_ref=P+"/activation/native_conditional_cost.py", strong_pair_validator_source_ref=R.STRONG,
        model_manifest_ref=R.MODEL_PLAN, original_estimator_source_ref=
        "third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py",
        model_config_source_ref=R.MODEL+"/config.json", raw_sdk_binding_ref=SDK_PATH,
        strong_raw_wrapper_source_ref=STRONG_RAW, original_raw_driver_source_ref=R.PREVIOUS+"/calibration_v2/run_native_cost_experiment.py")
    for name, path in required.items():
        require(path in refs, "actual full inherited/current leaf: " + name)
        R.check_ref(root, refs[path])
    for row in (pair_ref, input_manifest_ref, geometry_ref, runner_ref, parent_ref, guard_ref):
        R.check_ref(root, row)
    value = dict(schema="current_gpu_natural_calibration_raw_unit_plan_v1", gpu_uuid=gpu_uuid, job_id=job_id,
        journal_run_id=job_id, source_lock_ref=source_lock_ref, natural_manifest_ref=NATURAL,
        natural_source_record=record, input_manifest_ref=input_manifest_ref, cells=[cell_descriptor(record)],
        runtime_pair_ref=pair_ref, kv_layout_ref=geometry_ref, wrapper_source_ref=runner_ref,
        actual_u_collection_parent_ref=parent_ref, actual_u_collection_guard_ref=guard_ref,
        common_runtime_domain_sha256=R.common_domain_sha(pair), natural_trace=False,
        independent_family_holdout_qualified=False, table_issued=False, cost_qualified=False,
        ordinary_I_authorized=False, formal_goodput_allowed=False, strategy_improvement_proved=False)
    value.update({key: refs[path] for key, path in required.items()})
    return value


def verify_sources(root, config, refs, proof, *, R):
    # Same actual V14 denominator plus ALL new/current required byte leaves.
    # This is explicitly a raw-only source inheritance proof, not PASS_FULL.
    B = _source_module(Path(root) / (P + "/u_collection_bridge/u_collection_bridge_v3.py"),
        "0ca9de84a285180248368245f7a6e2a9b3bf9342e7aa1d1ddaca9e1eb3968877", "_raw_source_ancestry_" + str(time.monotonic_ns()))
    require(type(proof) is dict and set(proof) == B.PROOF_FIELDS and
            proof.get("schema") == "current_gpu_raw_unit_inherited_source_proof_v1" and
            proof.get("status") == "PASS_INHERITED_V14_PLUS_TARGETED_CPU_BYTES" and
            proof.get("source_lock_ref") == config["source_lock_ref"] and
            proof.get("ancestor_source_lock_ref") == B.ANCESTOR_LOCK and proof.get("ancestor_source_proof_ref") == B.ANCESTOR_PROOF,
            "distinct honest raw-only current source ancestry")
    ancestor = R.read(R.check_ref(root, B.ANCESTOR_LOCK))
    prior = R.read(R.check_ref(root, B.ANCESTOR_PROOF))
    require(prior.get("status") == "PASS_FULL_CPU_SOURCE_BYTES" and prior.get("source_lock_ref") == B.ANCESTOR_LOCK and
            prior.get("source_count") == 5014 and prior.get("actual_gpu_runs") == 0,
            "actual historical full proof, never current-host whole hashing")
    rows = ancestor.get("files")
    require(type(rows) is list and len(rows) == 5014 and len({r["path"] for r in rows}) == len(rows) and
            all(refs.get(row["path"]) == row for row in rows), "all historical source/model rows precisely inherited")
    inherited = {row["path"] for row in rows}
    added = [refs[path] for path in sorted(set(refs) - inherited)]
    required = [refs[path] for path in R.REQUIRED]
    require(proof.get("source_count") == len(refs) and type(proof.get("source_count")) is int and
            proof.get("ancestor_source_count") == 5014 and proof.get("targeted_source_refs") == added and
            proof.get("required_source_refs") == required and type(proof.get("actual_gpu_runs")) is int and
            proof["actual_gpu_runs"] == 0 and proof.get("full_source_verified") is False and
            proof.get("current_host_whole_source_hash_performed") is False, "complete actual targeted denominator, no fake whole-host proof")
    for row in added + required:
        R.check_ref(root, row)


def verify_prior_u(root, config, plan, *, R):
    parent = R.read(R.check_ref(root, config["actual_u_collection_parent_ref"]))
    require(parent.get("schema") == "actual_uncalibrated_original_U_mixed_complete_stream_post_guard_CPU_closure_v2" and
            parent.get("status") == "PASS_ACTUAL_U_FUNCTION_AND_COMPLETE_MIXED_STREAM_OBSERVATION_ONLY" and
            parent.get("gpu_uuid") == config["gpu_uuid"] and parent.get("completed_guard_ref") == config["actual_u_collection_guard_ref"] and
            parent.get("function_verified") is True and parent.get("complete_stream_observation_valid") is True and
            parent.get("original_guard_natural_session_drained") is True and parent.get("original_engine_shutdown_returned") is True and
            parent.get("actual_native_tail_drained") is True and parent.get("cost_qualified") is False and
            parent.get("ordinary_I_authorized") is False and parent.get("table_issued") is False,
            "actual current-device U function/mixed observation closure, no old off/table alias")
    actual_config = R.read(R.check_ref(root, parent["actual_run_config_ref"]))
    actual_refs = R.source_rows(root, actual_config["source_lock_ref"], full=False)
    pair = R.read(R.check_ref(root, actual_config["pair_config_ref"]))["configurations"]
    require(R.common_domain_sha(pair) == plan["common_runtime_domain_sha256"], "same actual U common model/native domain")
    gates = dict(config=actual_config, refs=actual_refs, pair=pair)
    gates["uncalibrated_u_collection"] = R.u_collection_gate(root, actual_config, actual_refs, pair)
    native = R.read(R.check_ref(root, parent["actual_native_result_ref"]))
    for row in parent["actual_raw_source_refs"]:
        R.check_ref(root, row)
    frontend = R.read(R.check_ref(root, native["actual_request_outputs_ref"]))
    capture = R.read(R.check_ref(root, native["actual_original_full_step_capture_ref"]))
    source_ref = gates["uncalibrated_u_collection"]["collector_ref"]
    ordinals = [frame["native_step_ordinal"] for frame in capture["frames"]]
    require(ordinals == list(range(ordinals[0], ordinals[0]+len(ordinals))), "whole current U native ordinals")
    replay = R.validate_u_collection_capture(root, gates, capture, frontend, source_ref=source_ref, expected_ordinals=ordinals)
    require(replay == parent["whole_native_capture_replay"], "real current U immutable full CUDA/output replay")
    runtime = R.load(root, actual_config["runtime_ref"], "_raw_prior_U_tail_" + str(time.monotonic_ns()))
    require(native.get("original_engine_shutdown_returned") is True and native.get("native_tail_drained") is True and
            runtime.verify_original_tail(native.get("post_original_shutdown")), "actual original current U native tail")
    validation = R.load(root, plan["validation_source_ref"], "_raw_prior_U_guard_" + str(time.monotonic_ns()))
    validation.validate_guard(R.read(R.check_ref(root, config["actual_u_collection_guard_ref"])),
        gpu_uuid=config["gpu_uuid"], job_id=parent["run_id"], wrapper_path=actual_config["runner_ref"]["path"])
    return parent


def inherited_model_binding(root, gates, model_dir, plan_path, *, R, previous_stats=None):
    """Prior actual Uoff03 hashes plus current exact leaf geometry; no weight read."""
    plan = gates["plan"]
    parent = gates["prior_U"]
    native = R.read(R.check_ref(root, parent["actual_native_result_ref"]))
    identity = native["model_identity"]
    require(Path(model_dir).resolve() == R.safe(root, R.MODEL).resolve() == Path(identity["model_directory"]).resolve() and
            Path(plan_path).resolve() == R.check_ref(root, plan["model_manifest_ref"]).resolve() and
            identity.get("manifest_sha256") == identity.get("plan_sha256") == plan["model_manifest_ref"]["sha256"] ==
            "9b1c79986aa1fd62eb9064ecb8b4e2a03fc235a34ac2d69c33a4fcb531a91017",
            "same actual original model directory, official plan and current U identity")
    official = R.read(plan_path)
    require(identity.get("model_id") == official["model_id"] and identity.get("revision") == official["revision"] and
            identity.get("provenance") == official["provenance"], "same original official immutable model source")
    expected = {record["path"]: {key: record[key] for key in ("bytes", "hash_algorithm", "hash")} for record in official["files"]}
    require(identity.get("verified_local_files") == expected, "actual preceding Uoff03 full model hashes; never synthetic CPU receipt")
    stats = {}
    for name, record in expected.items():
        path = Path(model_dir) / name
        relative = (Path(R.MODEL) / name).as_posix()
        require(type(name) is str and Path(name).name == name and not path.is_symlink() and path.is_file() and
                gates["refs"].get(relative) == dict(path=relative, bytes=record["bytes"], sha256=record["hash"]) and
                record["hash_algorithm"] == "sha256", "complete historical and current model leaf identity")
        stat = path.stat()
        require(stat.st_size == record["bytes"], "current same model leaf size")
        stats[name] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
        if not name.endswith(".safetensors"):
            R.check_ref(root, gates["refs"][relative])
    require(previous_stats is None or stats == previous_stats, "current model file geometry unchanged since actual CPU preflight")
    inherited = deepcopy(identity)
    inherited["historically_verified_local_files"] = inherited.pop("verified_local_files")
    inherited.update(validation_basis="actual_Uoff03_verified_model_hashes_plus_current_source_plan_and_leaf_stat",
        actual_preceding_U_result_ref=parent["actual_native_result_ref"], current_model_leaf_stats=stats,
        current_whole_model_rehash_performed=False, table_issued=False, ordinary_I_authorized=False)
    return Path(model_dir).resolve(), inherited, stats


def verify_configuration(root, path):
    root = Path(root).resolve()
    R = driver(root)
    config = R.read(path)
    require(type(config) is dict and set(config) == CONFIG_FIELDS and
            config.get("schema") == "current_gpu_natural_calibration_raw_unit_config_v1", "exact raw-only configuration")
    R.integer(config["seconds_limit"], "single raw original-guard limit", 1, MAX_SECONDS)
    require(type(config["window_index"]) is int and config["window_index"] in (0, 1), "only one A or B window; no estimator/holdout")
    refs = R.source_rows(root, config["source_lock_ref"], full=False)
    for key in CONFIG_FIELDS - {"schema", "job_id", "gpu_uuid", "seconds_limit", "window_index", "output_relative", "storage_template_relative"}:
        R.check_ref(root, config[key])
    require(R.check_ref(root, config["runner_ref"]).resolve() == Path(__file__).resolve() and
            refs.get(config["runner_ref"]["path"]) == config["runner_ref"], "actual frozen raw-only entry")
    verify_sources(root, config, refs, R.read(R.check_ref(root, config["source_proof_ref"])), R=R)
    plan = R.read(R.check_ref(root, config["plan_ref"]))
    record = selected_natural_record(R.read(R.check_ref(root, NATURAL)))
    require(plan.get("schema") == "current_gpu_natural_calibration_raw_unit_plan_v1" and
            plan.get("gpu_uuid") == config["gpu_uuid"] and plan.get("job_id") == config["job_id"] and
            plan.get("journal_run_id") == config["job_id"] and plan.get("source_lock_ref") == config["source_lock_ref"] and
            plan.get("natural_manifest_ref") == NATURAL and plan.get("natural_source_record") == record and
            plan.get("input_manifest_ref") == config["input_manifest_ref"] and plan.get("cells") == [cell_descriptor(record)] and
            all(plan.get(key) is False for key in FALSE_FIELDS) and plan.get("natural_trace") is False and
            plan.get("independent_family_holdout_qualified") is False, "original natural input controlled raw single cell, no table or service trace")
    for key, row in plan.items():
        if key.endswith("_ref") and type(row) is dict and key != "source_lock_ref":
            R.check_ref(root, row)
    expected = {
        "collector_source_ref": dict(path=P+"/runner/bounded_native_full_step_collector_v2.py", bytes=13016,
            sha256="a710f3f55ecbc6d8fbce8f3cf85ce4587daf0e1a863783df25a4ba8bc37e2aa0"),
        "original_collector_source_ref": dict(path=P+"/runner/bounded_native_full_step_collector.py", bytes=25070,
            sha256="9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d"),
        "validation_source_ref": dict(path=P+"/activation/native_conditional_cost.py", bytes=41753,
            sha256="675be4f91821863e23cc82128aadc040d8a9513c3aaf9b407d6c3fccd978ccb8"),
        "raw_sdk_binding_ref": dict(path=SDK_PATH, bytes=5515,
            sha256="5ad867f5fd2517ecc792a89e6225338a6e2ac6336e81b086af22bc27ed105dda"),
    }
    require(all(plan.get(key) == row and refs.get(row["path"]) == row for key, row in expected.items()) and
            plan.get("actual_u_collection_parent_ref") == config["actual_u_collection_parent_ref"] and
            plan.get("actual_u_collection_guard_ref") == config["actual_u_collection_guard_ref"] and
            plan.get("wrapper_source_ref") == config["runner_ref"], "exact unmodified observer/validator and bounded current source dispatch")
    require(refs.get(STRONG_RAW, {}).get("sha256") == STRONG_SHA and refs.get(DRIVER, {}).get("sha256") == DRIVER_SHA,
            "unchanged thin wrapper/common driver source pins")
    pair = R.read(R.check_ref(root, plan["runtime_pair_ref"]))["configurations"]
    validator = R.load(root, refs[R.STRONG], "_raw_original_pair_" + str(time.monotonic_ns()))
    validator.validate_runtime_pair(pair)
    require(R.common_domain_sha(pair) == plan["common_runtime_domain_sha256"], "same original common runtime")
    parent = verify_prior_u(root, config, plan, R=R)
    sdk = R.load(root, plan["raw_sdk_binding_ref"], "_raw_pure_sdk_binding_" + str(time.monotonic_ns()))
    sdk.preflight_binding(root, refs)
    ledger = R.read(R.safe(root, R.LEDGER))
    require(ledger["gpu_wall_seconds"] + config["seconds_limit"] + 20 <= R.MAX_GPU_SECONDS, "original cumulative GPU budget")
    require(__import__("shutil").disk_usage(root).free >= FLOOR + RESERVE, "original fixed reserve and floor")
    storage = R.safe(root, config["storage_template_relative"])
    require(storage.is_dir(), "existing unchanged original SSD input template")
    gates = dict(config=config, refs=refs, plan=plan, pair=pair, storage=storage, ledger=ledger, prior_U=parent)
    _, _, gates["current_model_leaf_stats"] = inherited_model_binding(root, gates, R.safe(root, R.MODEL),
        R.check_ref(root, plan["model_manifest_ref"]), R=R)
    # Exercise actual source-derived metadata wiring before a paid GPU job.
    # No execution/guard/function body is called here: only the private source
    # compilation, exact prompt and existing 24-file input identity checks.
    relative = Path(path).resolve().relative_to(root).as_posix()
    _, old, compat, _ = configure_window(root, gates, {}, relative, R=R)
    require(old.prompt(0) == record["prompt_token_ids"] and old.CACHED_TOKENS == 176,
            "actual original raw consumer metadata dispatch completed on CPU")
    old.input_groups(root, compat)
    return R, gates


def configure_window(root, gates, guard, relative, *, R):
    config, refs, plan = gates["config"], gates["refs"], gates["plan"]
    S = R.load(root, refs[STRONG_RAW], "_raw_original_strong_wrapper_" + str(time.monotonic_ns()))
    # Private source-bound metadata dispatch. No original engine function or
    # cleanup statement changes. The original six-window parent is not called.
    S.driver_module = lambda: R
    S.PROMPT_TOKENS, S.CACHED_TOKENS, S.INITIAL_EXECUTION_PRE_CONTEXT = 185, 176, 176
    # The old private acquisition consumer calls this generic binding slot.
    # Its actual byte reference is our distinct raw plan, never an old intent
    # schema or old off/table qualification receipt.
    acquisition_gates = dict(gates, config=dict(config, intent_ref=config["plan_ref"]))
    original_patch = S.source_patch
    def source_patch(raw):
        tree, proof = original_patch(raw)
        class Slice(ast.NodeTransformer):
            def __init__(self): self.count = 0
            def visit_Subscript(self, node):
                if isinstance(node.slice, ast.Slice) and isinstance(node.slice.upper, ast.Constant) and node.slice.upper.value == 32:
                    self.count += 1
                    node.slice.upper.value = 11
                return self.generic_visit(node)
        patch = Slice(); tree = patch.visit(tree)
        require(patch.count == 1, "single original copied foreground block-hash geometry slice")
        expected = ast.dump(ast.parse("base.validate_local_model(safe(root,common.MODEL),safe(root,common.MODEL_PLAN))", mode="eval").body,
                            include_attributes=False)
        class Model(ast.NodeTransformer):
            def __init__(self): self.count = 0
            def visit_Call(self, node):
                if ast.dump(node, include_attributes=False) == expected:
                    self.count += 1
                    node.func = ast.Name(id="_inherited_actual_model", ctx=ast.Load())
                return self.generic_visit(node)
        model_patch = Model(); tree = model_patch.visit(tree)
        require(model_patch.count == 1, "sole source-bound raw model metadata dispatch; original model loader untouched")
        return ast.fix_missing_locations(tree), dict(proof, prompt_tokens=185, cached_tokens=176,
            initial_execution_pre_context=176, initial_scheduled_tokens=9, qualified_family_holdout=False,
            original_cache_hit_rule="prospective_actual_foreground176_then_execution176_prefill9",
            fit_pairs=0, independent_validation_pairs=0, six_fresh_processes=False,
            raw_only=True, estimator_run=False, GPU_qualification_issued=False)
    S.source_patch = source_patch
    old, compat, proof = S.configure_original(root, acquisition_gates, guard, relative)
    record = plan["natural_source_record"]
    def prompt(index):
        require(type(index) is int and index == 0, "only preregistered calibration request0")
        require(selected_natural_record(R.read(R.check_ref(root, NATURAL))) == record, "actual exact source row recheck")
        return deepcopy(record["prompt_token_ids"])
    old.prompt = prompt
    old.PROMPT_TOKENS, old.CACHED_TOKENS = 185, 176
    old.SEEDS, old.PROMPT_FIRST = (record["seed"],)*3, (record["prompt_token_ids"][0],)*3
    old.SITE_SDK = plan["raw_sdk_binding_ref"]["path"]
    require(old.SITE_SDK == SDK_PATH, "new current-driver raw SDK consumer; no old580 site adapter")
    def inherited_actual_model(model_dir, plan_path):
        model, identity, _ = inherited_model_binding(root, gates, model_dir, plan_path, R=R,
            previous_stats=gates["current_model_leaf_stats"])
        return model, identity
    old._inherited_actual_model = inherited_actual_model
    return S, old, compat, proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root = args.project.resolve()
    R, gates = verify_configuration(root, root / args.config)
    if not args.execute:
        print(json.dumps(dict(status="PASS_CPU_NATURAL_UNIT_RAW_INPUT_SOURCE_AND_PRIOR_U_ONLY", GPU_job_started=False,
            prompt_tokens=185, expected_cached_tokens=176, measured_context200=200,
            table_issued=False, cost_qualified=False, ordinary_I_authorized=False)))
        return 0
    config = gates["config"]
    S = R.load(root, gates["refs"][STRONG_RAW], "_raw_guard_only_" + str(time.monotonic_ns()))
    S.driver_module = lambda: R
    guard = S.verify_guard(root, args.config, gates)
    S, old, compat, proof = configure_window(root, gates, guard, args.config, R=R)
    out = R.safe(root, config["output_relative"])
    require(not out.exists(), "fresh append-only raw window output")
    details = out / "windows" / ("%02d" % config["window_index"]) / "details"
    details.mkdir(parents=True, exist_ok=False)
    result = old.execute_window(root, compat, gates["refs"], guard, config["window_index"])
    index = config["window_index"]
    passed = result.get("status") == "PASS_NATIVE_FULL_STEP_RAW_CAPTURE_REQUIRES_VALIDATION"
    if passed:
        require(len(result.get("windows", [])) == 1, "sole complete original raw request")
        memory_window = result["windows"][0]
        disk_window = R.read(details / (memory_window["request_id"]+"-window.json"))
        require(disk_window == json.loads(json.dumps(memory_window, allow_nan=False)),
                "actual immutable saved window agrees with all in-memory primitive values")
        old._strict_completed_window(disk_window, index)
    summary = dict(schema="actual_current_gpu_natural_unit_raw_window_v1", status=
        "PASS_ACTUAL_NATURAL_UNIT_RAW_WINDOW_REQUIRES_PARENT_AB_COMPARISON" if passed else "FAILED_ACTUAL_NATURAL_UNIT_RAW_WINDOW",
        job_id=config["job_id"], gpu_uuid=config["gpu_uuid"], condition="A" if index == 0 else "B",
        plan_ref=config["plan_ref"], runner_ref=config["runner_ref"], original_raw_result=result,
        natural_manifest_ref=NATURAL, original_natural_request_id=0, original_natural_split="calibration",
        source_parameterization=proof, original_engine_shutdown_returned=result.get("original_engine_shutdown_returned"),
        source_record_unchanged=True, natural_trace_execution=False, independent_family_holdout_qualified=False,
        estimator_run=False, source_whole_model_rehash_performed=False, h2d_claim=False,
        table_issued=False, cost_qualified=False, ordinary_I_authorized=False, formal_goodput_allowed=False, strategy_improvement_proved=False)
    R.new_json(out / "natural-unit-raw-window-result.json", summary)
    print(json.dumps(dict(status=summary["status"], condition=summary["condition"], table_issued=False, ordinary_I_authorized=False)))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
