"""Thin, guarded driver of the unchanged author LLMEngine and native connector.

CPU imports/preflight never import torch, vLLM, a native I/O backend or a model.
Qualification measures natural request outputs; it is not a strategy experiment.
The generalized interference activation gap is rejected, not silently called I.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

SCHEMA = "strong_native_trace_run_v1"
GUARD = "experiments/prefix_io_v1/scripts/run_gpu_stage.py"
GUARD_SHA = "3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a"
LEDGER = "experiments/prefix_io_v1/gpu-budget-ledger.json"
MAX_GPU_SECONDS = 28800
MAX_SECONDS = 300
FLOOR = 8 * 1024**3
RESERVE = 128 * 1024**2
PREVIOUS = "artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004"
CANDIDATE = "artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate"
OVERLAY = CANDIDATE + "/source"
NATIVE = OVERLAY + "/third_party/work/py-kvcache-p4-02-cpu/py_kvcache"
CONTROL = OVERLAY + "/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control"
AUTHOR = "third_party/work/vllm-author-p4-02-cpu"
COMMON = "artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/run_g2_normal_model_lifecycle.py"
SMOKE = "experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py"
QUALIFIER = "experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py"
SDK = PREVIOUS + "/calibration_v2/site_sdk_binding.py"
DRAIN = "experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py"
TAIL = "artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002/g3_calibration_runtime_metrics_v2.py"
STRONG = PREVIOUS + "/strong_baseline/strong_baseline_config.py"
AUTHORIZATION = PREVIOUS + "/calibration_v2/standing_gpu_authorization.py"
MODEL = "models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444"
MODEL_PLAN = "artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json"
REQUIRED = (GUARD, COMMON, SMOKE, QUALIFIER, SDK, DRAIN, TAIL, STRONG, AUTHORIZATION,
            MODEL_PLAN, NATIVE + "/reactor.py", NATIVE + "/vllm.py",
            NATIVE + "/load_planner.py", CONTROL + "/p4_options.py",
            CONTROL + "/p4_production_table_contract.py", CONTROL + "/p4_startup_evidence.py", CONTROL + "/p4_bridge.py",
            CONTROL + "/p4_cost_table.py", CONTROL + "/p4_verified_cost_loader.py",
            AUTHOR + "/vllm/entrypoints/llm.py", AUTHOR + "/vllm/v1/engine/llm_engine.py",
            AUTHOR + "/vllm/v1/engine/core_client.py", AUTHOR + "/vllm/v1/core/sched/scheduler.py",
            AUTHOR + "/vllm/v1/worker/gpu_model_runner.py",
            AUTHOR + "/vllm/v1/executor/uniproc_executor.py",
            AUTHOR + "/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py")


def require(value, message):
    if not value:
        raise ValueError("STRONG_TRACE_REJECTED: " + message)


def integer(value, message, lower=0, upper=None):
    require(type(value) is int and value >= lower and (upper is None or value <= upper), message)
    return value


def numeric(value, message, lower=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= lower, message)
    return value


def safe(root, relative):
    root = Path(root).resolve(strict=True)
    require(type(relative) is str and relative and not relative.startswith("/") and
            ":" not in relative and "\\" not in relative and "\0" not in relative and
            all(x not in ("", ".", "..") for x in relative.split("/")), "project relative path")
    path = root
    for part in relative.split("/"):
        path /= part
        require(not path.is_symlink(), "symlink source/evidence")
    require(path.resolve().is_relative_to(root), "project containment")
    return path


def read(path, limit=32 * 1024**2):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, "bounded JSON")
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, "nonfinite JSON"))


def ref(root, relative):
    path = safe(root, relative)
    require(path.is_file(), "regular file reference")
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    require(path.stat().st_size == size, "file changed while hashing")
    return dict(path=relative, bytes=size, sha256=digest.hexdigest())


def check_ref(root, row):
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "exact source reference")
    integer(row["bytes"], "reference size")
    require(type(row["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", row["sha256"]), "reference SHA")
    require(ref(root, row["path"]) == row, "source/evidence byte drift: " + row["path"])
    return safe(root, row["path"])


def load(root, row, name):
    path = check_ref(root, row)
    require(path.suffix == ".py" and row["bytes"] <= 4 * 1024**2, "source-only bounded module")
    require(name not in sys.modules, "fresh private module")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        raw = path.read_bytes()
        require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
                "actual frozen source-only module bytes")
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
        check_ref(root, row)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def source_rows(root, lock_ref, *, full=False):
    lock = read(check_ref(root, lock_ref))
    require(type(lock) is dict and type(lock.get("files")) is list and 2000 <= len(lock["files"]) <= 8192,
            "inherited complete source/model/SDK lock")
    refs = {}
    for row in lock["files"]:
        require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "exact source row")
        require(row["path"] not in refs, "duplicate source path")
        safe(root, row["path"])
        integer(row["bytes"], "source row size")
        require(type(row["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", row["sha256"]), "source row SHA")
        refs[row["path"]] = row
    require(set(REQUIRED) <= refs.keys(), "original execution/lifecycle/guard/table closure")
    require(refs[GUARD]["sha256"] == GUARD_SHA, "original guard immutable")
    for path in refs if full else REQUIRED:
        check_ref(root, refs[path])
    return refs


def canonical_sha(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def project_ref(root, row):
    """Normalize historical absolute references without changing their bytes."""
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "original source ref")
    normalized = dict(row)
    if type(row["path"]) is str and Path(row["path"]).is_absolute():
        normalized["path"] = Path(row["path"]).relative_to(Path(root).resolve()).as_posix()
    check_ref(root, normalized)
    return normalized


def validate_workload(document):
    require(type(document) is dict and document.get("schema") in
            ("natural_off_qualification_workload_v1", "controlled_original_p3_qualification_workload_v1"),
            "explicit natural or inherited P3 controlled qualification workload")
    controlled = document["schema"] == "controlled_original_p3_qualification_workload_v1"
    if controlled:
        require(document.get("workload_kind") == "controlled_original_P3_mechanism" and
                document.get("natural_trace_bound") is False and type(document.get("source_manifest_ref")) is dict,
                "controlled original P3 is not a natural/evaluation stream")
    require(document.get("fit_or_evaluation_input_allowed") is False,
            "qualification data cannot become fitting or evaluation data")
    require(document.get("initial_cache_state") == "fresh_equal_namespace_preserved_through_whole_partition",
            "equal fresh namespaces; no midstream reset")
    records = document.get("records")
    require(type(records) is list and 1 <= len(records) <= 32, "bounded real request stream")
    require(type(document.get("selected_prompt_count")) is int and document["selected_prompt_count"] == len(records),
            "request denominator")
    integer(document.get("unselected_tail_count"), "explicit omitted natural tail")
    concurrency = integer(document.get("max_concurrency"), "original concurrency", 1, 8)
    for key in (("workload_sha256", "model_manifest_sha256") if controlled else
                ("workload_sha256", "dataset_sha256", "model_manifest_sha256")):
        require(type(document.get(key)) is str and re.fullmatch("[0-9a-f]{64}", document[key]), "workload provenance " + key)
    without_digest = {key: value for key, value in document.items() if key != "workload_sha256"}
    require(canonical_sha(without_digest) == document["workload_sha256"], "actual complete workload digest")
    seen = set()
    previous = -1
    for record in records:
        required_fields = {"request_id", "prompt", "prompt_sha256", "scheduled_time_s", "scheduled_ns", "max_tokens",
                           "min_tokens", "seed", "split", "prefix_family", "prompt_token_ids"}
        if controlled:
            required_fields.add("original_request_spec")
        require(type(record) is dict and set(record) == required_fields,
            "exact natural request row")
        identifier = record["request_id"]
        require(type(identifier) in (str, int) and type(identifier) is not bool,
                "original request identity")
        require(str(identifier) and len(str(identifier)) <= 128 and str(identifier) not in seen, "unique request identity")
        seen.add(str(identifier))
        prompt = record["prompt"]
        if controlled:
            tokens = record["prompt_token_ids"]
            require(prompt is None and type(tokens) is list and 1 <= len(tokens) <= 4096 and
                    all(type(token) is int and 0 <= token < 152064 for token in tokens), "original controlled P3 IDs")
            require(canonical_sha(tokens) == record["prompt_sha256"], "unchanged original P3 prompt IDs")
        else:
            require(type(prompt) is str and prompt and len(prompt.encode()) <= 256 * 1024, "original prompt text")
            require(hashlib.sha256(prompt.encode()).hexdigest() == record["prompt_sha256"], "unchanged author prompt")
        numeric(record["scheduled_time_s"], "arrival clock")
        integer(record["scheduled_ns"], "arrival ns")
        require(abs(record["scheduled_time_s"] * 1e9 - record["scheduled_ns"]) <= 1.0001,
                "same author arrival clock")
        require(record["scheduled_ns"] >= previous, "author order unchanged")
        previous = record["scheduled_ns"]
        integer(record["min_tokens"], "full min output", 128, 128)
        require(type(record["max_tokens"]) is int and record["max_tokens"] == record["min_tokens"], "full fixed output")
        integer(record["seed"], "frozen request seed")
        require(record["split"] == "qualification" and record["prefix_family"] is None and
                (controlled or record["prompt_token_ids"] is None), "qualification IDs never become fit/evaluation families")
    return concurrency


def activation_gap(root, refs):
    """Report the existing production qualification barrier with actual source lines."""
    found = []
    for relative, needle in ((CONTROL + "/p4_options.py", "GPU-qualified activation is not implemented"),
                             (CONTROL + "/p4_production_table_contract.py", "NOT a qualified CostTable"),
                             (CONTROL + "/p4_bridge.py", "production interference table is not GPU qualified")):
        if relative not in refs:
            continue
        path = check_ref(root, refs[relative])
        rows = path.read_text(encoding="utf-8").splitlines()
        matches = [dict(path=relative, line=index + 1, source=text.strip(), sha256=refs[relative]["sha256"])
                   for index, text in enumerate(rows) if needle in text]
        found.extend(matches)
    return dict(status="BLOCKED_GENERALIZED_I_ACTIVATION", qualified_general_table_loader_available=False,
                current_condition_binding_verified=False, old_single_file_receipt_covers_natural_trace=False,
                actual_source_rejections=found)



FORMAL_SCHEMA = "natural_trace_workload_v1"
FORMAL_NAMESPACE_BRIDGE_SHA = "30d6aa7aaae0808fd021c2d6c629066854a7b651a8de9b683eeb025788013ab3"
FORMAL_BINDING_SHA = "bab370bf83cf4736b03dc3fdc80f07b6efc40b15425ed43e3581fbcfe14f2d9b"
FORMAL_ROLES = {("development", "off", "U"): "development",
                ("development", "shadow", "I"): "development",
                ("effect", "off", "U"): "evaluation", ("effect", "on", "I"): "evaluation"}


DEVELOPMENT_BRIDGE_SHA = "e7a10755f42f9edf24c04cbc1461bed6f5e15c7ec3cac3c53cdb43431bfec8f7"
_DEVELOPMENT_MODULE_SEQUENCE = 0


def development_module_name(prefix):
    # A clock reading is not a unique identifier (notably on Windows CPU tests).
    global _DEVELOPMENT_MODULE_SEQUENCE
    _DEVELOPMENT_MODULE_SEQUENCE += 1
    return prefix + str(time.monotonic_ns()) + "_" + str(_DEVELOPMENT_MODULE_SEQUENCE)


def development_bridge_module(root, config, refs):
    relative = (Path(config["runtime_ref"]["path"]).parent.parent /
                "development_runtime_bridge/development_diagnostic_bridge.py").as_posix()
    require(relative in refs and refs[relative]["sha256"] == DEVELOPMENT_BRIDGE_SHA,
            "exact new scoped development-only CPU adapter source")
    return load(root, refs[relative], development_module_name("_strong_development_bridge_"))


def development_diagnostic(root, config, refs):
    if "formal_trace_binding_ref" not in config:
        return False
    module = development_bridge_module(root, config, refs)
    return module.is_diagnostic(root, config, refs, driver=sys.modules[__name__])


def activation_gate_module(root, config, refs):
    relative = Path(config["runtime_ref"]["path"]).with_name("activation_request.py").as_posix()
    require(relative in refs, "original finite activation implementation frozen")
    original = load(root, refs[relative], development_module_name("_strong_original_finite_gate_"))
    if not development_diagnostic(root, config, refs):
        return original
    bridge = development_bridge_module(root, config, refs)
    return bridge.bind_activation(original, root=root, source_ref=refs[relative], config=config,
                                  refs=refs, driver=sys.modules[__name__])


def formal_binding_module(root, config, refs):
    relative = (Path(config["runtime_ref"]["path"]).parent.parent /
                "formal_trace_binding/formal_trace_binding.py").as_posix()
    require(relative in refs and refs[relative]["sha256"] == FORMAL_BINDING_SHA,
            "exact unchanged formal CPU input contract source")
    module = load(root, refs[relative], development_module_name("_strong_formal_binding_"))
    # Common source provenance fix only: every formal private F instance uses
    # its exact frozen helper bytes. Effect validate_* functions stay original.
    development = development_bridge_module(root, config, refs)
    module.load_pure = development.source_only_pure_loader(module)
    bridge = formal_namespace_module(root, config, refs)
    module = bridge.bind(module, config=config, refs=refs, pair=None, driver=sys.modules[__name__])
    if development_diagnostic(root, config, refs):
        module = development.bind_input(module, root=root, source_ref=refs[relative],
                                        config=config, refs=refs, driver=sys.modules[__name__])
    return module


def formal_namespace_module(root, config, refs):
    relative = (Path(config["runtime_ref"]["path"]).parent.parent /
                "formal_runtime_bridge/namespace_bridge.py").as_posix()
    require(relative in refs and refs[relative]["sha256"] == FORMAL_NAMESPACE_BRIDGE_SHA,
            "exact new current-arm namespace CPU adapter source")
    return load(root, refs[relative], development_module_name("_strong_formal_namespace_"))


def formal_role(config):
    role = config["phase"], config["mode"], config["arm"]
    require(role in FORMAL_ROLES, "formal development Uoff/Ishadow or evaluation Uoff/Ion only")
    require(type(config.get("formal_trace_binding_ref")) is dict and
            type(config.get("activation_ref")) is dict, "closed real formal input and finite activation references")
    peer = config.get("formal_peer_closed_ref")
    require(peer is None or type(peer) is dict, "actual closed peer ref or explicitly first-arm fresh pair")
    if config["phase"] == "development":
        require((config["arm"] == "U" and peer is None) or (config["arm"] == "I" and type(peer) is dict),
                "development U first, then I shadow with same-partition guard-closed U peer")
    return FORMAL_ROLES[role]


def select_formal_workload(root, config, refs, pair, document):
    """Replay actual formal input bytes and copy unchanged selected token IDs.

    The complete manifest stays under its original workload_ref. This in-memory
    drive view selects only the prospective partition; it never freezes a new
    manifest, rewrites qualification rows, or invents text/IDs/arrival clocks.
    """
    partition = formal_role(config)
    module = formal_binding_module(root, config, refs)
    binding = module.validate_formal_workload(document, root=root, workload_ref=config["workload_ref"],
        binding_ref=config["formal_trace_binding_ref"], refs=refs, pair=pair, partition=partition)
    require(binding["common_runtime_domain_sha256"] == common_domain_sha(pair) and
            binding["skip_tokenizer_init"] is True and binding["gpu_eligible"] is False,
            "same original real token-ID common domain; metadata is no GPU authority")
    require(all(type(row["min_tokens"]) is int and row["min_tokens"] == 128 and
                type(row["max_tokens"]) is int and row["max_tokens"] == 128 for row in binding["records"]),
            "existing finite raw validator requires complete128 outputs before GPU launch")
    selected = deepcopy(document)
    selected["records"] = [dict(deepcopy(row), prompt=None) for row in binding["records"]]
    require(len(selected["records"]) == binding["complete_selected_record_count"] and
            [row["request_id"] for row in selected["records"]] == binding["source_request_ids"],
            "complete formal partition denominator/order")
    return selected, binding


def preflight_formal_runtime(root, gates):
    """CPU byte replay only; guarded execution repeats it before model imports.

    The isolated CPU reserve helper reissues a private table only for the
    existing read-only native reserve verifier. No capability crosses that
    boundary; the guarded runtime independently reissues before use.
    """
    config, refs, pair = gates["config"], gates["refs"], gates["pair"]
    if "formal_trace_binding_ref" not in config:
        return None
    original = gates.get("formal_original_workload")
    require(type(original) is dict and original.get("schema") == FORMAL_SCHEMA,
            "original full formal manifest retained in memory")
    selected, binding = select_formal_workload(root, config, refs, pair, original)
    require(selected == gates["workload"] and binding == gates.get("formal_workload_binding"),
            "no forged selected formal token IDs/family/partition view")
    finite = gates.get("finite_activation")
    require(type(finite) is dict and finite.get("phase") == config["phase"] and
            finite.get("actual_table_issued") is False and
            finite.get("formal_effect_qualified") is False and finite.get("runtime_refs") == refs,
            "original closed finite CPU gate required for every formal arm")
    descriptor, plan = finite.get("descriptor"), finite.get("calibration_plan")
    diagnostic = development_diagnostic(root, config, refs)
    require(type(descriptor) is dict and type(plan) is dict and
            plan.get("common_runtime_domain_sha256") == binding["common_runtime_domain_sha256"] and
            plan.get("gpu_uuid") == config["gpu_uuid"], "same real calibration domain/device")
    if diagnostic:
        require(descriptor.get("schema") == "development_diagnostic_gpu_activation_request_v1" and
                "independent_deadline_ref" not in descriptor and binding["independent_deadline_ref"] is None and
                binding["service_SLO"] is None and finite.get("development_diagnostic_only") is True,
                "separate no-SLO development diagnostic; no external deadline or effect promotion")
        development_bridge_module(root, config, refs).verify_diagnostic_budget(config, finite["independent_budget"])
    else:
        require(descriptor.get("independent_deadline_ref") == binding["independent_deadline_ref"],
                "same independent formal SLO/deadline")
    if config["phase"] == "development":
        budget = finite["independent_budget"]
        require(budget.get("observation_only") is True and budget.get("ordinary_I_authorized") is False and
                budget.get("reserve_not_measured_yet") is True,
                "development observes only; no fabricated measured controller reserve")
        replay = None
    else:
        require(Path(config["runtime_ref"]["path"]).name in ("native_runtime_v3.py", "native_runtime_v4.py"),
                "formal both-arm native-reserve replay implementation frozen")
        runtime = load(root, config["runtime_ref"], "_strong_formal_reserve_cpu_" + str(time.monotonic_ns()))
        replay = runtime.preflight_formal_reserve_replay(root, gates, driver=sys.modules[__name__])
    module = formal_binding_module(root, config, refs)
    result = module.validate_formal_phase(config, root=root, refs=refs, pair=pair,
        formal_workload=binding, finite_activation=finite, development_replay=replay)
    require(result.get("input_bindings_validated") is True and result.get("gpu_eligible") is False and
            result.get("formal_goodput_allowed") is False, "CPU result cannot grant GPU/effect authority")
    gates["formal_phase_preflight"] = result
    gates["development_reserve_replay"] = replay
    return result


def validate_formal_frontend_records(gates, frontend):
    """Read-only identity closure after the unchanged original drive returns."""
    if "formal_trace_binding_ref" not in gates["config"]:
        return None
    records = gates["formal_workload_binding"]["records"]
    require(type(frontend) is dict and frontend.get("status") == "PASS_COMPLETE_ORIGINAL_REQUEST_OUTPUTS" and
            frontend.get("actual_request_stream_completed") is True and
            type(frontend.get("planned_requests")) is int and type(frontend.get("successful_requests")) is int and
            frontend["planned_requests"] == frontend["successful_requests"] == len(records) and
            type(frontend.get("failure_or_unsubmitted_requests")) is int and
            frontend["failure_or_unsubmitted_requests"] == 0, "every actual formal request completed")
    rows = frontend.get("rows")
    require(type(rows) is list and len(rows) == len(records), "full formal output denominator")
    for record, row in zip(records, rows):
        require(row.get("request_id") == str(record["request_id"]) and row.get("state") == "COMPLETED" and
                row.get("prompt_sha256") == record["prompt_sha256"] and
                type(row.get("actual_prompt_token_ids")) is list and
                row["actual_prompt_token_ids"] == record["prompt_token_ids"],
                "actual original frontend IDs match exact submitted formal IDs/order")
        ids, times, itl = row.get("output_token_ids"), row.get("token_return_ns"), row.get("itl_ns")
        require(type(ids) is list and len(ids) == record["max_tokens"] and
                all(type(x) is int and x >= 0 for x in ids) and
                type(times) is list and len(times) == len(ids) and
                all(type(x) is int and x > 0 for x in times) and times == sorted(times) and
                type(itl) is list and itl == [b-a for a,b in zip(times,times[1:])],
                "complete actual outputs and token timeline without selection")
    return dict(schema="actual_formal_original_frontend_identity_closure_v1",
                successful_requests=len(rows), partition=gates["formal_workload_binding"]["partition"],
                original_manifest_workload_sha256=gates["formal_workload_binding"]["original_manifest_workload_sha256"],
                actual_original_prompt_ids_verified=True, formal_goodput_allowed=False)


def verify_formal_off_prerequisite(root, config, refs, pair, binding):
    """Require a real same-formal-input U development run closed by old guard."""
    prerequisite = read(check_ref(root, config["off_qualification_ref"]))
    require(prerequisite.get("status") == "PASS_FORMAL_DEVELOPMENT_U_WORKLOAD_ONLY_REQUIRES_GUARD_CLOSURE" and
            prerequisite.get("phase") == "development" and prerequisite.get("mode") == "off" and
            prerequisite.get("arm") == "U" and prerequisite.get("formal_effect_qualified") is False and
            prerequisite.get("workload_ref") == config["workload_ref"] and
            prerequisite.get("formal_trace_binding_ref") == config["formal_trace_binding_ref"] and
            prerequisite.get("formal_partition") == "development" and
            prerequisite.get("gpu_uuid") == config["gpu_uuid"] and
            prerequisite.get("common_runtime_domain_sha256") == common_domain_sha(pair) and
            prerequisite.get("original_engine_shutdown_returned") is True and
            prerequisite.get("native_tail_drained") is True and
            prerequisite.get("os_session_drained") is True, "actual same-formal-input U development prerequisite")
    completed = prerequisite.get("completed_guard_ref")
    require(type(completed) is dict, "formal U child alone cannot prove original outer OS drain")
    check_ref(root, completed)
    validation_path = PREVIOUS + "/calibration_v2/native_conditional_cost.py"
    require(validation_path in refs, "unchanged actual completed-guard validator frozen")
    validator = load(root, refs[validation_path], "_strong_prior_formal_U_guard_" + str(time.monotonic_ns()))
    validator.validate_guard(read(check_ref(root, completed)), gpu_uuid=config["gpu_uuid"],
        job_id=prerequisite["run_id"], wrapper_path=prerequisite["runner_ref"]["path"])
    formal_input = prerequisite.get("formal_phase_preflight")
    require(type(formal_input) is dict and formal_input.get("role") == ["development", "off", "U"] and
            formal_input.get("input_bindings_validated") is True and formal_input.get("gpu_eligible") is False,
            "real U development consumed formal contract; CPU tags are insufficient")
    require(type(prerequisite.get("formal_frontend_identity_closure")) is dict and
            prerequisite["formal_frontend_identity_closure"].get("actual_original_prompt_ids_verified") is True,
            "all actual formal U token-ID outputs verified")
    input_descriptor = read(check_ref(root, config["formal_trace_binding_ref"]))
    namespace_ref = input_descriptor.get("namespace_contract_ref")
    require(type(namespace_ref) is dict and refs.get(namespace_ref.get("path")) == namespace_ref,
            "actual full partition namespace contract frozen")
    namespaces = read(check_ref(root, namespace_ref))["partition_namespaces"]
    peer_config = dict(config, formal_peer_closed_ref=config["off_qualification_ref"])
    bridge = formal_namespace_module(root, config, refs)
    full_manifest = read(check_ref(root, config["workload_ref"]))
    bridge.verify_closed_peer(root, peer_config, refs, pair, partition="development", peer_arm="U",
        peer_storage=namespaces["development"]["U"], manifest=full_manifest,
        driver=sys.modules[__name__])
    return prerequisite


def check_phase(config, *, gap, formal=False):
    phase, mode, arm = config["phase"], config["mode"], config["arm"]
    if formal:
        formal_role(config)
        return gap
    require((phase, mode, arm) in (("qualification", "off", "U"), ("shadow", "shadow", "I"),
                                  ("development", "shadow", "I"),
                                  ("effect", "on", "I")), "finite qualified arm/phase")
    if phase in ("development", "effect"):
        require(type(config.get("activation_ref")) is dict,
                "effect interface blocked: closed real finite activation inputs required; "
                "table=None and context144/offset16 diagnostic receipts cannot activate this path")
    if phase == "shadow":
        require(type(config.get("off_qualification_ref")) is dict, "shadow requires completed same-domain original off")
    return gap


def verify_configuration(root, config_path, *, full=False):
    config = read(config_path, 1024**2)
    fields = {"schema", "phase", "arm", "mode", "run_id", "source_lock_ref", "source_proof_ref", "pair_config_ref",
              "workload_ref", "permissions_ref", "gpu_uuid", "seconds_limit", "storage_reserve_bytes",
              "storage_floor_bytes", "output_relative", "runner_ref", "runtime_ref", "off_qualification_ref"}
    if type(config) is dict and config.get("phase") in ("development", "effect"):
        fields.add("activation_ref")
    if type(config) is dict and "formal_trace_binding_ref" in config:
        fields.update(("formal_trace_binding_ref", "formal_peer_closed_ref"))
    require(type(config) is dict and set(config) == fields and config["schema"] == SCHEMA, "strict bounded run config")
    require(type(config["run_id"]) is str and re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,79}", config["run_id"]), "run label")
    integer(config["seconds_limit"], "bounded original guard seconds", 1, MAX_SECONDS)
    require(config["storage_reserve_bytes"] == RESERVE and type(config["storage_reserve_bytes"]) is int and
            config["storage_floor_bytes"] == FLOOR and type(config["storage_floor_bytes"]) is int,
            "original bounded storage reservation/floor")
    require(type(config["gpu_uuid"]) is str and re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                                                         config["gpu_uuid"]), "live bound one GPU UUID")
    refs = source_rows(root, config["source_lock_ref"], full=full)
    for key in ("runner_ref", "runtime_ref", "pair_config_ref", "workload_ref", "permissions_ref"):
        check_ref(root, config[key])
        require(refs.get(config[key]["path"]) == config[key], "config asset inside frozen closure: " + key)
    require(Path(__file__).resolve() == check_ref(root, config["runner_ref"]).resolve(), "actual thin runner source")
    proof = read(check_ref(root, config["source_proof_ref"]))
    require(proof.get("schema") == "strong_trace_source_proof_v1" and proof.get("status") == "PASS_FULL_CPU_SOURCE_BYTES" and
            proof.get("source_lock_ref") == config["source_lock_ref"] and proof.get("source_count") == len(refs) and
            proof.get("actual_gpu_runs") == 0 and type(proof.get("actual_gpu_runs")) is int,
            "actual prelaunch full CPU source proof")
    pair_doc = read(safe(root, config["pair_config_ref"]["path"]))
    require(pair_doc.get("schema") == "strong_native_u_i_cpu_configuration_v1", "original strong bridge configurations")
    validator = load(root, refs[STRONG], "_strong_trace_config_" + str(time.monotonic_ns()))
    pair = pair_doc["configurations"]
    validator.validate_runtime_pair(pair)
    config_engine = pair[config["arm"]]["engine"]
    require(config_engine.get("distributed_executor_backend") == "uni" and config_engine.get("async_scheduling") is False,
            "existing original Uni executor/scheduler route")
    require(config_engine.get("disable_log_stats") is True, "common supported metrics switch")
    require(config_engine.get("speculative_config") is None, "one actual token per step for exact ITL")
    require(config_engine["kv_transfer_config"]["kv_connector_extra_config"]["prefix_io_parent_admission"]["run_id"] ==
            config["run_id"], "same runtime parent/journal identity")
    storage = Path(config_engine["kv_transfer_config"]["kv_connector_extra_config"]["shared_storage_path"])
    require(storage.is_absolute(), "arm private native namespace")
    relative_storage = storage.relative_to(Path(root).resolve()).as_posix()
    require(relative_storage.startswith("experiments/prefix_io_v1/runs/"), "storage inside bounded native runs")
    storage = safe(root, relative_storage)
    require(not storage.exists(), "fresh append-only arm namespace; no cache reset/purge")
    out = safe(root, config["output_relative"])
    require(config["output_relative"].startswith("experiments/prefix_io_v1/runs/") and out.name == "details" and
            not out.exists(), "fresh original guarded run details")
    workload = read(safe(root, config["workload_ref"]["path"]))
    formal = type(workload) is dict and workload.get("schema") == FORMAL_SCHEMA
    formal_binding = None
    original_workload = deepcopy(workload) if formal else None
    if formal:
        require(Path(config["runtime_ref"]["path"]).name in ("native_runtime_v3.py", "native_runtime_v4.py"),
                "formal source-bound runtime before engine/model import")
        check_ref(root, config["formal_trace_binding_ref"])
        require(refs.get(config["formal_trace_binding_ref"]["path"]) == config["formal_trace_binding_ref"],
                "complete formal input descriptor inside frozen closure")
        workload, formal_binding = select_formal_workload(root, config, refs, pair, workload)
        concurrency = formal_binding["max_concurrency"]
    else:
        require("formal_trace_binding_ref" not in config, "legacy qualification never consumes formal binding fields")
        concurrency = validate_workload(workload)
    controlled = workload["schema"] == "controlled_original_p3_qualification_workload_v1"
    require(config_engine.get("skip_tokenizer_init") is (controlled or formal),
            "unchanged original text versus actual token-ID runtime domain")
    if controlled:
        manifest_ref = project_ref(root, workload["source_manifest_ref"])
        index_ref = project_ref(root, workload["source_index_ref"])
        source_manifest = read(check_ref(root, manifest_ref))
        require(refs.get(manifest_ref["path"]) == manifest_ref and refs.get(index_ref["path"]) == index_ref,
                "actual P3 manifest/index source frozen")
        require(source_manifest.get("profile") == "low_contention" and type(source_manifest.get("requests")) is list and
                len(source_manifest["requests"]) == len(workload["records"]), "complete unchanged original P3 partition")
        for original, actual in zip(source_manifest["requests"], workload["records"]):
            require(original["request_id"] == actual["request_id"] and
                    original["prompt_token_ids"] == actual["prompt_token_ids"] and
                    original["scheduled_time"] == actual["scheduled_time_s"] and
                    source_manifest["output_tokens"] == actual["max_tokens"] and
                    {key: value for key, value in original.items() if key != "prompt_token_ids"} == actual["original_request_spec"],
                    "original P3 full spec/IDs/arrival/outputs unchanged")
    require(config_engine["max_num_seqs"] >= concurrency, "common engine supports frozen concurrency")
    require(workload["model_manifest_sha256"] == refs[MODEL_PLAN]["sha256"], "actual model tokenizer manifest")
    curve = config_engine["kv_transfer_config"]["kv_connector_extra_config"]["prefix_cache_break_even_path"]
    curve_relative = Path(curve).relative_to(Path(root).resolve()).as_posix()
    require(curve_relative in refs, "original LoadPlanner break-even source frozen")
    check_ref(root, refs[curve_relative])
    gap = activation_gap(root, refs)
    check_phase(config, gap=gap, formal=formal)
    finite_activation = None
    if config["phase"] in ("development", "effect"):
        activation = activation_gate_module(root, config, refs)
        finite_activation = activation.verify(root, config["activation_ref"], refs=refs, pair=pair,
                                              gpu_uuid=config["gpu_uuid"], driver=sys.modules[__name__], phase=config["phase"])
    if formal and (config["phase"], config["mode"], config["arm"]) != ("development", "off", "U"):
        verify_formal_off_prerequisite(root, config, refs, pair, formal_binding)
    if not formal and config["phase"] in ("shadow", "development", "effect"):
        prerequisite = read(check_ref(root, config["off_qualification_ref"]))
        require(prerequisite.get("status") == "PASS_STRONG_NATIVE_OFF_WORKLOAD_LIFECYCLE" and
                prerequisite.get("formal_effect_qualified") is False and
                prerequisite.get("workload_ref") == config["workload_ref"] and
                prerequisite.get("gpu_uuid") == config["gpu_uuid"] and
                prerequisite.get("common_runtime_domain_sha256") == common_domain_sha(pair) and
                prerequisite.get("original_engine_shutdown_returned") is True and
                prerequisite.get("native_tail_drained") is True, "same-domain actual off prerequisite")
        completed = prerequisite.get("completed_guard_ref")
        require(type(completed) is dict, "off child alone cannot prove original outer OS session drain")
        check_ref(root, completed)
        validation_path = PREVIOUS + "/calibration_v2/native_conditional_cost.py"
        require(validation_path in refs, "original completed-guard validator frozen")
        validator = load(root, refs[validation_path], "_strong_prior_actual_guard_" + str(time.monotonic_ns()))
        validator.validate_guard(read(check_ref(root, completed)), gpu_uuid=config["gpu_uuid"],
                                 job_id=prerequisite["run_id"], wrapper_path=prerequisite["runner_ref"]["path"])
    grant = load(root, refs[AUTHORIZATION], "_strong_trace_standing_" + str(time.monotonic_ns()))
    _, grant_ref = grant.standing_grant(root)
    ledger = read(safe(root, LEDGER))
    used = numeric(ledger.get("gpu_wall_seconds"), "original cumulative usage")
    require(used + config["seconds_limit"] + 20 <= MAX_GPU_SECONDS, "remaining original eight-hour budget")
    require(shutil.disk_usage(root).free >= FLOOR + RESERVE, "space floor and fixed resource reservation")
    actual_run_config_ref = ref(root, config_path.resolve().relative_to(Path(root).resolve()).as_posix())
    gates = dict(config=config, refs=refs, pair=pair, workload=workload, storage=storage, out=out, ledger=ledger,
                actual_run_config_ref=actual_run_config_ref,
                standing_authorization_ref=grant_ref, activation_gap=gap, full_source_verified=full,
                source_count=len(refs), common_runtime_domain_sha256=common_domain_sha(pair), finite_activation=finite_activation)
    if formal:
        gates.update(formal_original_workload=original_workload, formal_workload_binding=formal_binding,
                     formal_phase_preflight=None, development_reserve_replay=None)
    if config["phase"] in ("development", "effect"):
        require(Path(config["runtime_ref"]["path"]).name in
                (("native_runtime_v3.py", "native_runtime_v4.py") if formal else ("native_runtime_v2.py",)),
                "finite startup requires exact new runtime collector binding before GPU")
        selected_runtime = load(root, config["runtime_ref"], "_strong_finite_cpu_collector_gate_" + str(time.monotonic_ns()))
        collector_ref = selected_runtime.preflight_collector_binding(root,gates,driver=sys.modules[__name__])
        gates["collector_binding_preflight"] = dict(source_ref=collector_ref, source_only=True,
                                                   actual_table_issued=False, GPU_qualification_issued=False)
    if formal:
        preflight_formal_runtime(root, gates)
    return gates


def common_domain_sha(pair):
    normalized = deepcopy(pair["U"])
    extra = normalized["engine"]["kv_transfer_config"]["kv_connector_extra_config"]
    extra.pop("shared_storage_path")
    extra.pop("prefix_io_p4_policy")
    extra["prefix_io_parent_admission"].pop("run_id")
    extra.pop("prefix_io_observation_run_id", None)
    return canonical_sha(normalized)


def child_command(root, config_relative, runner_relative):
    return [".venv/bin/python", "-B", runner_relative, "--project", str(root), "--config", config_relative, "--execute"]


def guard_command(root, config_relative, config):
    return [".venv/bin/python", "-B", GUARD, "--permissions-path", config["permissions_ref"]["path"],
            "--label", config["run_id"], "--seconds", str(config["seconds_limit"]), "--",
            *child_command(root, config_relative, config["runner_ref"]["path"])]


def verify_guard(root, config_relative, gates):
    config = gates["config"]
    require(os.name == "posix", "actual server Linux guard session")
    ledger = read(safe(root, LEDGER))
    reservation = ledger.get("active_reservation")
    expected = dict(label=config["run_id"], gpu_uuid=config["gpu_uuid"], seconds_limit=config["seconds_limit"],
                    reserved_seconds=config["seconds_limit"] + 20,
                    command=child_command(root, config_relative, config["runner_ref"]["path"]),
                    permissions=config["permissions_ref"], evidence=str(gates["out"].parent))
    require(type(reservation) is dict and all(type(reservation.get(k)) is type(v) and reservation[k] == v
            for k, v in expected.items()), "actual original guard active reservation")
    require(type(reservation.get("id")) is str and reservation["id"], "actual reservation identity")
    deadline = time.monotonic() + 1
    while reservation.get("session_id") is None and time.monotonic() < deadline:
        time.sleep(.01)
        newer = read(safe(root, LEDGER)).get("active_reservation")
        require(type(newer) is dict and newer.get("id") == reservation["id"] and
                all(type(newer.get(k)) is type(v) and newer[k] == v for k, v in expected.items()), "reservation drift")
        reservation = newer
    session = integer(reservation.get("session_id"), "actual guard session", 1)
    require(os.getsid(0) == os.getpgid(0) == session == reservation.get("process_group") and
            reservation.get("state") != "cleanup_unresolved", "same resolved original guard session")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == config["gpu_uuid"] and
            os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1", "offline UUID guard environment")
    pid = integer(reservation.get("runner_pid"), "actual guard PID", 1)
    tokens = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    require(GUARD in tokens and config["permissions_ref"]["path"] in tokens and config["run_id"] in tokens,
            "original bound guard process")
    return reservation


def new_json(path, document):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def drive_original_engine(engine, sampling_factory, records, *, max_concurrency, deadline_seconds,
                          clock=time.monotonic_ns, sleeper=time.sleep, on_progress=None):
    """Only original add_request/has_unfinished_requests/step implement execution.

    Token clocks are exact original step-return times, not CUDA kernel timers or
    HTTP streaming chunks. This measurement never changes cache/owner state.
    """
    integer(max_concurrency, "bounded original concurrency", 1, 8)
    numeric(deadline_seconds, "request-stream lifecycle deadline", 0.001)
    require(not engine.has_unfinished_requests(), "fresh original request engine")
    start = clock()
    deadline = start + int(deadline_seconds * 1e9)
    rows = [dict(request_id=str(r["request_id"]), native_request_id=None, state="NOT_SUBMITTED",
                 scheduled_ns=start + r["scheduled_ns"], submission_ns=None, finished_ns=None,
                 prompt_sha256=r["prompt_sha256"], actual_prompt_token_ids=None,
                 output_token_ids=[], token_return_ns=[], itl_ns=[], ttft_ns=None,
                 latency_ns=None, num_cached_tokens=None, finish_reason=None, error=None) for r in records]
    active = {}
    next_index = steps = 0
    try:
        while next_index < len(records) or active or engine.has_unfinished_requests():
            now = clock()
            require(now <= deadline and steps < 131072, "whole original request stream deadline/step bound")
            while (next_index < len(records) and len(active) < max_concurrency and
                   start + records[next_index]["scheduled_ns"] <= now):
                record, row = records[next_index], rows[next_index]
                sampling = sampling_factory(record)
                row["submission_ns"] = clock()
                row["state"] = "SUBMIT_ATTEMPTED"
                prompt = ({"prompt_token_ids": record["prompt_token_ids"].copy()} if record["prompt_token_ids"] is not None
                          else record["prompt"])
                native = engine.add_request(row["request_id"], prompt, sampling)
                require(type(native) is str and 0 < len(native) <= 256, "actual original native request ID")
                row["native_request_id"], row["state"] = native, "ACCEPTED"
                active[row["request_id"]] = row
                next_index += 1
                now = clock()
            if not active:
                require(not engine.has_unfinished_requests(), "original engine lost frontend request identities")
                if next_index < len(records):
                    sleeper(min(.01, max(0, (start + records[next_index]["scheduled_ns"] - now) / 1e9)))
                continue
            outputs = engine.step()
            returned = clock()
            steps += 1
            require(type(outputs) is list and len(outputs) <= max_concurrency, "actual bounded original output cohort")
            observed = set()
            for output in outputs:
                rid = output.request_id
                require(type(rid) is str and rid in active and rid not in observed, "original output identity/cohort")
                observed.add(rid)
                row = active[rid]
                require(len(output.outputs) == 1, "one deterministic original completion")
                completion = output.outputs[0]
                tokens = list(completion.token_ids)
                require(all(type(t) is int and t >= 0 for t in tokens), "actual integer output tokens")
                prior = row["output_token_ids"]
                require(tokens[:len(prior)] == prior, "original cumulative tokens cannot retract/change")
                fresh = tokens[len(prior):]
                require(len(fresh) <= 1, "exact ITL requires one actual new token per original step; chunked samples rejected")
                prompt_ids = list(output.prompt_token_ids)
                require(prompt_ids and all(type(t) is int and t >= 0 for t in prompt_ids), "actual original tokenizer IDs")
                require(row["actual_prompt_token_ids"] is None or row["actual_prompt_token_ids"] == prompt_ids,
                        "same original prompt IDs in all outputs")
                row["actual_prompt_token_ids"] = prompt_ids
                if fresh:
                    previous = row["token_return_ns"]
                    require(not previous or returned >= previous[-1], "monotonic token clock")
                    if previous:
                        row["itl_ns"].append(returned - previous[-1])
                    else:
                        row["ttft_ns"] = returned - row["scheduled_ns"]
                    previous.append(returned)
                    row["output_token_ids"] = tokens
                if output.finished:
                    expected = records[rows.index(row)]["max_tokens"]
                    require(len(tokens) == expected and len(row["token_return_ns"]) == expected and
                            len(row["itl_ns"]) == expected - 1, "complete output and every actual token timestamp")
                    integer(output.num_cached_tokens, "actual original cached-token count")
                    row.update(state="COMPLETED", finished_ns=returned, latency_ns=returned - row["scheduled_ns"],
                               num_cached_tokens=output.num_cached_tokens, finish_reason=completion.finish_reason)
                    active.pop(rid)
            if on_progress is not None:
                on_progress(rows, steps)
        require(all(r["state"] == "COMPLETED" for r in rows), "all selected natural requests complete")
        return dict(status="PASS_COMPLETE_ORIGINAL_REQUEST_OUTPUTS", planned_requests=len(records),
                    successful_requests=len(rows), failure_or_unsubmitted_requests=0, rows=rows,
                    original_engine_steps=steps, measurement_clock="monotonic_ns immediately after original engine.step",
                    kernel_cost_measurement=False, formal_goodput_allowed=False,
                    actual_request_stream_completed=True)
    except Exception as exc:
        for row in rows:
            if row["state"] != "COMPLETED":
                row["state"] = "FAILED_OR_UNSUBMITTED"
                row["error"] = dict(type=type(exc).__name__, message=str(exc)[:2000])
        return dict(status="FAILED_ORIGINAL_REQUEST_OUTPUTS", planned_requests=len(records),
                    successful_requests=sum(r["state"] == "COMPLETED" for r in rows),
                    failure_or_unsubmitted_requests=sum(r["state"] != "COMPLETED" for r in rows), rows=rows,
                    original_engine_steps=steps, measurement_clock="monotonic_ns immediately after original engine.step",
                    kernel_cost_measurement=False, formal_goodput_allowed=False,
                    actual_request_stream_completed=False, error=dict(type=type(exc).__name__, message=str(exc)[:2000]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--config", required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--full-source", action="store_true")
    actions.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root = args.project.resolve(strict=True)
    try:
        gates = verify_configuration(root, safe(root, args.config), full=args.full_source)
        if not args.execute:
            print(json.dumps(dict(status="CPU_CONFIGURATION_VERIFIED_NO_GPU_LAUNCH", source_count=gates["source_count"],
                full_source_verified=gates["full_source_verified"], phase=gates["config"]["phase"],
                actual_gpu_runs=0, runtime_qualified=False, formal_effect_qualified=False,
                guarded_future_command=guard_command(root, args.config, gates["config"]), activation_gap=gates["activation_gap"])))
            return 0
        reservation = verify_guard(root, args.config, gates)
        runtime = load(root, gates["config"]["runtime_ref"], "_strong_trace_native_runtime_" + str(time.monotonic_ns()))
        result = runtime.execute(root, gates, reservation, driver=sys.modules[__name__])
        print(json.dumps(dict(status=result["status"], actual_gpu_runs=1,
                             original_engine_shutdown_returned=result["original_engine_shutdown_returned"],
                             formal_effect_qualified=False)))
        return 0 if result["status"] in ("PASS_STRONG_NATIVE_OFF_WORKLOAD_LIFECYCLE", "PASS_STRONG_NATIVE_SHADOW_WORKLOAD_ONLY",
            "PASS_FINITE_DEVELOPMENT_SHADOW_REQUIRES_RESERVE_GUARD_JOIN",
            "PASS_FINITE_QUALIFIED_I_WORKLOAD_LIFECYCLE_REQUIRES_EFFECT_ANALYSIS",
            "PASS_FORMAL_DEVELOPMENT_U_WORKLOAD_ONLY_REQUIRES_GUARD_CLOSURE",
            "PASS_FORMAL_EFFECT_U_WORKLOAD_ONLY_REQUIRES_GUARD_CLOSURE") else 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps(dict(status="BLOCKED_CPU_GATE_NO_FRAMEWORK_IMPORT", reason=str(exc), actual_gpu_runs=0,
                              runtime_qualified=False, formal_effect_qualified=False)))
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
