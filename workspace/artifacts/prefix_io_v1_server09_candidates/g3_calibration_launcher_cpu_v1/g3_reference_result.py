"""Finite reference-only CPU gates around the original cached numerical analyzer.

No model, backend, authorization, curve export or numerical comparator is added.
The parent owns actual raw-byte/source/guard provenance and loads the original
analyzer with its original storage authorization. Caller dictionaries and CPU
fixtures never acquire GPU, production, physical release or cost qualification.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

PURPOSE = "CURRENT_CONTEXT_CACHED_NUMERICAL_REFERENCE_ONLY"
LABELS = {"cold": "server09-g3-reference-native-01", "paired": "server09-g3-reference-paired-01"}
RESULT_SHA = "83ae8bb14181d51e9d4b0e40bca4123d30578cce43f581d851f97033f155eb9e"
ORIGINAL_PINS = {
    "cached": ("analyze_cached_references.py", "2aaed52742c23750d068b16f4ad51e1caaa9427dc1ff39fdd09e3440460bbc1c"),
    "repeated": ("analyze_repeated_native_costs.py", "21bf7fc7562dcf1711a0e3dbad9b740e122a7126304d7f048258f79b32dc773a"),
    "storage": ("experiment_storage.py", "ec5fdb9d9f9001de1080aadb740a87320ccc783f95032c5fdff310dc869a127f"),
}
UNQUALIFIED = {"GPU_qualified": False, "production_qualified": False,
    "effect_verified": False, "physical_release_verified": False,
    "KV_byte_consistency_verified": False, "GPU_clock_mapping_verified": False,
    "runtime_curve_exported": False, "P4_performance_verified": False,
    "cost_qualified": False, "full_domain_validated": False}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _object_sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _legacy():
    path = Path(__file__).resolve().with_name("g3_calibration_result.py")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == RESULT_SHA, "original G3 closure helper source drift")
    name = "_g3_reference_original_result_" + RESULT_SHA
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    return sys.modules[name]


def _numeric_shape(row):
    """Validate complete returned metadata; all numerical comparison stays original."""
    values = row.get("output_logprobs")
    require(type(values) is list and len(values) == 1 and type(values[0]) is dict,
            "one returned logprob step required")
    values = values[0]
    require(5 <= len(values) <= 6, "complete logprobs5 values required")
    ranks = []
    for key, item in values.items():
        require(type(key) is str and key.isdecimal() and str(int(key)) == key,
                "canonical returned token ID required")
        require(type(item) is dict and set(item) == {"logprob", "rank"}, "returned logprob metadata")
        require(type(item["logprob"]) is float and math.isfinite(item["logprob"]), "finite original float logprob")
        require(type(item["rank"]) is int and item["rank"] > 0, "positive original rank")
        ranks.append(item["rank"])
    require(len(set(ranks)) == len(ranks) and set(range(1, 6)) <= set(ranks), "complete original top5 ranks")
    require(str(row["output_token_ids"][0]) in values, "selected token absent from returned logprobs")


def _configuration_view(mode, config):
    require(type(config) is dict and type(config.get("sampling")) is dict
            and type(config["sampling"].get("logprobs")) is int and config["sampling"]["logprobs"] == 5,
            "reference sampling requires exact logprobs5")
    require(config.get("native_hot_diagnostic") is (mode == "cold")
            and config.get("cached_reference_logprobs") is (mode == "paired"), "reference diagnostic flags")
    view = copy.deepcopy(config)
    view["native_hot_diagnostic"] = view["cached_reference_logprobs"] = False
    view["sampling"].pop("logprobs")
    _legacy().validate_frozen_config(mode, view)
    return view


def validate_reference_job(mode, report, runtime, guard, binding, config):
    """Validate six real reference requests, using check-only legacy closure views.

    Views normalize only diagnostic flags and the known native-hot cache shape
    for reuse of existing non-numerical structural checks. They are never saved
    or returned as a raw acquisition verdict, and never replace original rows.
    """
    require(type(mode) is str and mode in LABELS, "fixed cold/paired reference mode")
    R = _legacy()
    raw_config = _configuration_view(mode, config)
    require(type(binding) is dict and binding.get("expected_label") == LABELS[mode], "fixed reference label")
    require(type(binding.get("expected_model")) is dict and config["model"] == binding["expected_model"],
            "independently frozen current model identity")
    require(type(binding.get("expected_alias_target")) is str
            and config["alias_target"] == binding["expected_alias_target"], "frozen current model target")
    require(type(report) is dict and report.get("mode") == mode
            and report.get("status") == "PASSED_NATIVE_" + mode.upper() + "_ACQUISITION"
            and report.get("native_hot_diagnostic") is (mode == "cold")
            and report.get("cached_reference_logprobs") is (mode == "paired"), "actual original reference report")
    require(type(runtime) is dict and runtime.get("purpose") == PURPOSE, "reference-only runtime purpose")
    require(runtime.get("frozen_config") == config, "runtime/report frozen config identity")
    require(type(runtime.get("requests")) is list and len(runtime["requests"]) == 6,
            "six actual original reference requests required")
    require(type(runtime.get("sentinel_observations")) is list and not runtime["sentinel_observations"],
            "reference has no populate sentinel")
    rows = report.get("rows")
    kinds = ("f", "gpu_hot") if mode == "cold" else ("g_ssd", "g_mem")
    require(type(rows) is list and len(rows) == 6, "six original reference rows required")
    require(type(report.get("storage_preflight")) is dict
            and report["storage_preflight"].get("path") == binding.get("expected_storage"),
            "reference fixed original storage target")
    for ordinal, row in enumerate(rows):
        rep, position = divmod(ordinal, 2)
        require(type(row) is dict and row.get("kind") == kinds[position]
                and type(row.get("rep")) is int and row["rep"] == rep
                and type(row.get("prefix_tokens")) is int and row["prefix_tokens"] == 128
                and row.get("warmup") is (rep == 0), "ordered original reference row identity")
        require(row.get("prompt_token_ids") == R.expected_prompt(rep), "fixed original reference prompt")
        R._tokens(row.get("output_token_ids"), 1, "reference output")
        require(type(row.get("num_cached_tokens")) is int
                and row["num_cached_tokens"] == (0 if row["kind"] == "f" else 128), "actual reference cache shape")
        _numeric_shape(row)
        request = runtime["requests"][ordinal]
        require(type(request) is dict and set(request) == {"ordinal", "prompt_token_ids", "output_token_ids",
                "num_cached_tokens", "sentinel", "request_id"} and type(request["ordinal"]) is int
                and request["ordinal"] == ordinal and request["sentinel"] is False,
                "actual original reference generate schema/order")
        require(all(request[key] == row[key] for key in ("prompt_token_ids", "output_token_ids",
                    "num_cached_tokens", "request_id")), "runtime reference request differs from original row")
    # Paired uses one original six-row view; native checks both three-row views.
    for position in ([0, 1] if mode == "cold" else [None]):
        view = copy.deepcopy(report)
        view["native_hot_diagnostic"] = view["cached_reference_logprobs"] = False
        projected_runtime = copy.deepcopy(runtime)
        projected_runtime["frozen_config"] = raw_config
        if position is not None:
            view["rows"] = copy.deepcopy(rows[position::2])
            projected_runtime["requests"] = copy.deepcopy(runtime["requests"][position::2])
            for ordinal, (row, request) in enumerate(zip(view["rows"], projected_runtime["requests"])):
                row["kind"] = "f"
                row["num_cached_tokens"] = request["num_cached_tokens"] = 0
                request["ordinal"] = ordinal
        R.validate_job_result(mode, view, projected_runtime, guard, binding, config=raw_config)
    for key in ("production_qualified", "performance_claim", "cost_qualified"):
        require(runtime.get(key) is False, "reference runtime must not claim " + key)
    return {"status": "PASS_REFERENCE_SIX_REQUEST_PATH_CLOSURE_ONLY", "mode": mode, "purpose": PURPOSE,
            "label": LABELS[mode], "requests": 6, "rows": 6, "all_warmups_in_numeric_gate": True,
            "process_identity": copy.deepcopy(runtime["process_identity"]),
            "canonical_report_sha256": _object_sha(report), "canonical_config_sha256": _object_sha(config),
            "source_lock_sha256": binding["expected_source_lock_sha256"],
            "scope_sha256": binding["expected_scope_sha256"], "storage": binding["expected_storage"],
            "model": copy.deepcopy(config["model"]), "alias_target": config["alias_target"],
            "gpu_uuid": config["gpu_uuid"], "latency_fit_allowed": False,
            "native_cold_hot_bit_equality_required": False,
            "raw_byte_provenance": "PARENT_MUST_BIND_REAL_REPORT_RUNTIME_GUARD_BYTES",
            **UNQUALIFIED}


def validate_reference_plan(plan, *, expected_gpu_uuid, expected_native_storage, expected_paired_storage):
    require(type(plan) is dict and set(plan) == {"native_label", "sizes", "reps", "domain", "gpu_uuid",
            "kv_budget_bytes", "staging_budget_bytes", "external_groups", "run_details", "native_storage_path"},
            "finite original reference plan fields")
    require(plan["native_label"] == LABELS["cold"] and plan["sizes"] == [128]
            and type(plan["sizes"]) is list and type(plan["sizes"][0]) is int
            and type(plan["reps"]) is int and plan["reps"] == 3
            and type(plan["domain"]) is int and plan["domain"] == 1024, "fixed reference plan point")
    require(type(expected_gpu_uuid) is str and expected_gpu_uuid.startswith("GPU-")
            and plan["gpu_uuid"] == expected_gpu_uuid, "reference plan GPU identity")
    for key, value in (("kv_budget_bytes", 268435456), ("staging_budget_bytes", 134217728)):
        require(type(plan[key]) is int and plan[key] == value, "reference plan " + key)
    require(type(expected_native_storage) is str and type(expected_paired_storage) is str
            and expected_native_storage != expected_paired_storage
            and plan["native_storage_path"] == expected_native_storage, "separate native unused storage target")
    # Original analyzer eagerly evaluates group['storage'] in the get default,
    # even when storage_path is present. Keep both original fields consistent.
    storage_name = expected_paired_storage.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    require(plan["external_groups"] == [{"label": LABELS["paired"], "sizes": [128],
            "storage": storage_name, "storage_path": expected_paired_storage}],
            "one exact original paired storage group")
    details = plan["run_details"]
    require(type(details) is dict and set(details) == set(LABELS.values()), "two exact reference run_details")
    for mode, label in LABELS.items():
        path = details[label]
        require(type(path) is str and path.replace("\\", "/").endswith("/" + label + "/details/acquisition"),
                "reference details must name actual original acquisition")
    return {"status": "PASS_FINITE_REFERENCE_PLAN_CPU_ONLY", "purpose": PURPOSE, **UNQUALIFIED}


def _pinned(root, reference, basename, expected_sha=None):
    require(type(reference) is dict and set(reference) == {"path", "bytes", "sha256"}, "exact source/evidence ref")
    path = Path(reference["path"])
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in reference["path"], "relative source ref")
    path = root / path
    require(path.name == basename and path.is_file() and not path.is_symlink(), "bounded pinned source/evidence")
    data = path.read_bytes()
    require(type(reference["bytes"]) is int and len(data) == reference["bytes"] <= 4 * 1024**2
            and hashlib.sha256(data).hexdigest() == reference["sha256"], "pinned source/evidence drift")
    if expected_sha:
        require(reference["sha256"] == expected_sha, "original analyzer source identity")
    return path.resolve(), data


def _original_function(function, path, raw, name):
    require(type(function) is types.FunctionType and Path(function.__code__.co_filename).resolve() == path,
            "original analyzer callable filename: " + name)
    tree = ast.parse(raw, filename=str(path))
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    require(node is not None, "original analyzer function absent: " + name)
    # Original storage signatures bind ROOT as a default at definition time.
    # Reuse globals only while defining the pinned function, never invoke them.
    namespace = dict(function.__globals__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec", dont_inherit=True), namespace)
    # Code-object equality compares executable code/constants recursively.
    # marshal byte output also reflects incidental string-intern/reference state.
    require(function.__code__ == namespace[name].__code__,
            "original analyzer callable code drift: " + name)


def verify_reference_pair(root, plan_path, *, expected_plan_ref, native_check, paired_check,
                          analyzer_refs, original_analyzer):
    """Run the original analyzer after finite reference checks; no new comparison."""
    root = Path(root).resolve()
    path, raw_plan = _pinned(root, expected_plan_ref, Path(plan_path).name)
    require(Path(plan_path).resolve() == path, "original reference plan path binding")
    plan = json.loads(raw_plan.decode("utf-8"))
    for mode, check in (("cold", native_check), ("paired", paired_check)):
        require(type(check) is dict and check.get("status") == "PASS_REFERENCE_SIX_REQUEST_PATH_CLOSURE_ONLY"
                and check.get("mode") == mode and check.get("label") == LABELS[mode]
                and check.get("purpose") == PURPOSE and check.get("requests") == check.get("rows") == 6,
                "both six-request reference path gates required")
        require(all(check.get(key) is False for key in UNQUALIFIED), "CPU path gates cannot promote qualification")
        folder = Path(plan["run_details"][LABELS[mode]])
        report = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "frozen-config.json").read_text(encoding="utf-8"))
        require(_object_sha(report) == check["canonical_report_sha256"]
                and _object_sha(config) == check["canonical_config_sha256"], "reference original report/config changed")
    validate_reference_plan(plan, expected_gpu_uuid=native_check["gpu_uuid"],
            expected_native_storage=native_check["storage"], expected_paired_storage=paired_check["storage"])
    require(all(native_check[key] == paired_check[key] for key in ("source_lock_sha256", "scope_sha256",
            "model", "alias_target", "gpu_uuid")), "reference sessions current source/model identity")
    require(all(native_check["process_identity"][key] != paired_check["process_identity"][key]
            for key in ("pid", "sid")), "two fresh reference processes required")
    require(type(analyzer_refs) is dict and set(analyzer_refs) == set(ORIGINAL_PINS), "three original analyzer refs")
    pins = {key: _pinned(root, analyzer_refs[key], *pin) for key, pin in ORIGINAL_PINS.items()}
    require(type(original_analyzer) is types.ModuleType
            and Path(original_analyzer.__file__).resolve() == pins["cached"][0], "original analyzer module path")
    for name in ("analyze", "canonical_top5", "compare_rows", "validate_external_row"):
        function = getattr(original_analyzer, name, None)
        _original_function(function, *pins["cached"], name)
        require(function.__globals__ is original_analyzer.__dict__,
                "original analyzer function has a substituted global namespace: " + name)
    for name in ("check_drains", "require"):
        _original_function(getattr(original_analyzer, name, None), *pins["repeated"], name)
    for name in ("details_path", "authorized_path"):
        _original_function(getattr(original_analyzer, name, None), *pins["storage"], name)
    require(original_analyzer.check_drains.__globals__.get("require") is original_analyzer.require,
            "original drain validator require binding")
    require(original_analyzer.details_path.__globals__.get("authorized_path") is original_analyzer.authorized_path,
            "original storage path authorization binding")
    result = original_analyzer.analyze(root, path)
    require(type(result) is dict and result.get("status") in ("PASSED_EXACT_CACHED_REFERENCE", "FAILED_EXACT_CACHED_REFERENCE")
            and result.get("latency_fit_allowed") is False and result.get("runtime_curve_exported") is False,
            "original analyzer reference verdict required")
    # Recheck exact source/plan bytes after invoking the original method.
    _pinned(root, expected_plan_ref, path.name)
    for key, pin in ORIGINAL_PINS.items():
        _pinned(root, analyzer_refs[key], *pin)
    return {"status": result["status"], "purpose": PURPOSE, "original_analysis": result,
            "original_analyzer_refs": copy.deepcopy(analyzer_refs), "reference_plan_ref": copy.deepcopy(expected_plan_ref),
            "reference_requests": 12, "all_warmups_in_numeric_gate": True, "latency_fit_allowed": False,
            "independent_content_validation": False, "original_storage_authorization_owner": "PARENT_ORIGINAL_HELPER_CONTEXT",
            "no_new_numerical_comparator": True, **UNQUALIFIED}
