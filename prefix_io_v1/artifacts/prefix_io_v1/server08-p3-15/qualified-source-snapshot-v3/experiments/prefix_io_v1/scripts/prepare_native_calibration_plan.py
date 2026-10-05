#!/usr/bin/env python3
"""Prepare candidate native cold-prefill jobs on CPU; never run calibration."""
from __future__ import annotations
import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
AUTHOR_COMMIT = "817a7e3124f817cd6e549581d3e5483207a753a4"
PARETO = "third_party/upstream/kvcache-experiments/scripts/pareto_measure.py"
PARETO_COMMIT = "0e023a84a21246b9bbc06266fa8070397eccbdc9"
PARETO_SHA = "47f13dc13abbf042954f5a30f50aa7129b932f3f88fb02fd2b6871277dd4476c"
BREAK_EVEN = "third_party/upstream/py-kvcache/py_kvcache/break_even.py"
BREAK_EVEN_COMMIT = "3abba7a502d553f6e7e2e58b92086487e3395d7e"
BREAK_EVEN_SHA = "2419c9470df0e43558ccc729aa3927ebd4d1381e00276d4caaa57b3cf19c42a6"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local(path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(ROOT), "input must be inside the project")
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    path = local(path)
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def author_module(relative, expected_hash, names=None):
    """Execute only verified original source, never an arbitrary supplied file."""
    path = local(ROOT / relative)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_hash,
            "pinned author source SHA mismatch; no source executed")
    tree = ast.parse(raw, filename=str(path))
    if names is not None:
        selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                    and n.name in names]
        require({n.name for n in selected} == set(names), "original API nodes missing")
        tree = ast.Module(body=selected, type_ignores=[])
        ast.fix_missing_locations(tree)
    name = "_prefix_io_verified_cpu_" + expected_hash[:12]
    module = types.ModuleType(name)
    module.__dict__["dataclass"] = dataclass
    sys.modules[name] = module
    try:
        exec(compile(tree, str(path), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


def prior_qualification(smoke_dir):
    directory = local(smoke_dir)
    frozen, frozen_ref = read_json(directory / "frozen-config.json")
    result, result_ref = read_json(directory / "smoke-result.json")
    require(result.get("status") == "PASSED_GPU_PREFIX_PATH_ONLY"
            and result.get("native_engine_shutdown") == "completed",
            "completed native GPU Prefix qualification required")
    require(frozen.get("author_commit") == AUTHOR_COMMIT, "wrong author commit")
    runner, runner_ref = read_json(directory.parent / "result.json")
    environment, environment_ref = read_json(ROOT / "experiments/prefix_io_v1/locks/environment.json")
    locked, lock_ref = read_json(ROOT / "experiments/prefix_io_v1/locks/dependency-lock.json")
    require(all(type(runner.get(key)) is int and runner[key] == 0 for key in ("exit", "child_exit"))
            and runner.get("timed_out") is False and runner.get("session_drained") is True
            and runner.get("interrupted_signal") is None and runner.get("error") is None
            and runner.get("gpu_job_attempted") is True
            and runner.get("session_members_after_cleanup") == [],
            "successful drained budget-runner evidence required")
    gpu_uuid = runner.get("gpu_uuid")
    require(isinstance(gpu_uuid, str)
            and re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu_uuid)
            and gpu_uuid == frozen["environment"]["CUDA_VISIBLE_DEVICES"] == environment["gpu_uuid"],
            "historical runner/frozen/current lock GPU UUID mismatch")
    require(local(runner["evidence"]) == directory.parent
            and local(ROOT / environment["native_GPU_Prefix_evidence"]) == directory / "smoke-result.json",
            "runner or current lock points to a different qualification")
    commits = {"vllm-author": AUTHOR_COMMIT, "py-kvcache": BREAK_EVEN_COMMIT,
               "kvcache-experiments": PARETO_COMMIT}
    require(all(locked["repositories"][name]["commit"] == commit for name, commit in commits.items()),
            "dependency version lock differs from pinned author source")
    model = frozen["model"]
    require(model.get("model_id") == MODEL_ID
            and re.fullmatch(r"[0-9a-f]{40}", str(model.get("revision", ""))),
            "wrong fixed model or revision")
    manifest, manifest_ref = read_json(model["plan_path"])
    require(manifest_ref["sha256"] == model["manifest_sha256"] == model["plan_sha256"],
            "official manifest hash differs from successful smoke")
    for key in ("model_id", "revision", "source", "provider"):
        require(manifest.get(key) == model.get(key), "manifest identity mismatch: " + key)
    require(model.get("source") in ("https://modelscope.cn", "https://huggingface.co"),
            "unsupported model provenance")
    model_dir = local(frozen["model_dir"])
    model_lock = locked["model_lock"]
    require(model_lock.get("id") == MODEL_ID
            and all(model_lock.get(key) == model.get(key) for key in
                    ("provider", "source", "revision", "revision_namespace", "manifest_sha256"))
            and model_lock.get("verified_local_weights") is True
            and local(model_lock["local_path"]) == model_dir
            and local(ROOT / model_lock["manifest"]) == local(model["plan_path"]),
            "current model lock differs from historical model qualification")
    command = runner["command"]
    for flag, expected_path in (("--model-dir", model_dir), ("--model-plan", local(model["plan_path"])),
                                ("--output-dir", directory)):
        require(command.count(flag) == 1 and command.index(flag) + 1 < len(command),
                "historical runner command missing unique " + flag)
        require(local(ROOT / command[command.index(flag) + 1]) == expected_path,
                "historical runner command path mismatch")
    cfg, cfg_ref = read_json(model_dir / "config.json")
    record = next(x for x in manifest["files"] if x["path"] == "config.json")
    require(record["hash_algorithm"] == "sha256" and record["hash"] == cfg_ref["sha256"],
            "local config differs from official manifest")
    require(cfg.get("num_hidden_layers") == 28 and cfg.get("num_key_value_heads") == 4
            and cfg.get("hidden_size") == 3584 and cfg.get("num_attention_heads") == 28,
            "unexpected Qwen geometry")
    require(cfg.get("dtype", cfg.get("torch_dtype")) == "bfloat16"
            and not cfg.get("quantization_config"), "unquantized BF16 config required")
    engine = frozen["engine"]
    require(engine["dtype"] == "bfloat16" and engine["kv_cache_dtype"] == "auto"
            and engine["quantization"] is None and engine["kv_cache_memory_bytes"] == 67108864,
            "successful smoke is not the fixed BF16/64 MiB configuration")
    require(result["effective_model_dtype"] == result["actual_kv_tensor_dtype"] == "torch.bfloat16"
            and result["effective_model_quantization"] is None
            and result["effective_kv_cache_dtype"] == "auto"
            and result["effective_kv_cache_memory_bytes"] == engine["kv_cache_memory_bytes"],
            "effective BF16 or KV configuration evidence missing")
    cold, repeat = result["results"]
    expected = frozen["expected_cached_tokens"]
    require(expected == [0, 112] and [cold["num_cached_tokens"], repeat["num_cached_tokens"]] == expected,
            "native cached-token evidence mismatch")
    require(cold["prompt_token_ids"] == repeat["prompt_token_ids"] == frozen["prompt_token_ids"]
            and cold["output_token_ids"] == repeat["output_token_ids"]
            and len(cold["output_token_ids"]) == frozen["sampling"]["max_tokens"],
            "native prompt/output equality evidence mismatch")
    kv_bpt = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * (
        cfg["hidden_size"] // cfg["num_attention_heads"]) * 2
    storage = result["kv_tensor_storage_metadata"]
    require(storage["tensor_count"] == 28 and storage["dtype"] == "torch.bfloat16"
            and storage["device_type"] == "cuda" and storage["num_blocks"] > 0,
            "native storage metadata missing")
    require(storage["actual_kv_tensor_allocation_bytes"] == storage["declared_kv_tensor_bytes"]
            == result["actual_kv_tensor_allocation_bytes"]
            == storage["num_blocks"] * engine["block_size"] * kv_bpt
            <= engine["kv_cache_memory_bytes"], "KV geometry/backing evidence mismatch")
    return frozen, {"model_id": MODEL_ID, "model_name_for_runtime": str(model_dir),
        "provider": model["provider"], "source": model["source"], "revision": model["revision"],
        "revision_namespace": model["revision_namespace"], "manifest": manifest_ref,
        "model_config": cfg_ref, "prior_frozen_config": frozen_ref, "prior_result": result_ref,
        "prior_budget_runner": runner_ref, "dependency_lock": lock_ref, "environment_lock": environment_ref,
        "gpu_uuid": gpu_uuid, "gpu_identity_scope": "historical evidence matched to current lock; no fresh GPU probe",
        "kv_dtype_string": "auto", "actual_kv_dtype": "torch.bfloat16",
        "kv_bytes_per_token": kv_bpt, "weight_files_rehashed_now": False,
        "qualification_scope": "previous native GPU Prefix only; not calibration or SSD"}


def check_v2_structure(path, identity):
    data, reference = read_json(path)
    require(isinstance(data.get("curves"), dict)
            and set(data["curves"]) == {"f", "g_ssd", "g_mem"},
            "v1/scalar-only file is not v2 curves; version labels are insufficient")
    require(data.get("model_name") == identity["model_name_for_runtime"],
            "v2 model_name must equal the exact local runtime model path")
    require(data.get("kv_dtype") == identity["kv_dtype_string"]
            and data.get("kv_bytes_per_token") == identity["kv_bytes_per_token"],
            "v2 dtype or KV geometry mismatch")
    module = author_module(BREAK_EVEN, BREAK_EVEN_SHA)
    curves = module.load_curves(str(local(path)), model_name=identity["model_name_for_runtime"],
                                kv_dtype=identity["kv_dtype_string"])
    require(curves is not None, "native v2 parser returned no curves")
    for curve in (curves.f, curves.g_ssd, curves.g_mem):
        require(all(math.isfinite(x) and x >= 0 for x in
                    [curve.floor, *[value for _, value in curve.knots]]),
                "curve times must be finite and nonnegative")
    return {"file": reference, "native_parser": "load_curves", "status": "STRUCTURE_ONLY",
            "golden_points": len(curves.golden), "measurement_provenance_verified": False,
            "usable_for_planner": False}


def prepare(smoke_dir, doc_sizes, repeats, curve_file=None):
    frozen, identity = prior_qualification(smoke_dir)
    require(type(repeats) is int and 1 <= repeats <= 10, "repeats must be 1..10")
    require(1 <= len(doc_sizes) <= 8 and len(set(doc_sizes)) == len(doc_sizes),
            "require 1..8 distinct candidate doc sizes")
    # This is only a conservative nominal cap. Original driver builds words,
    # so even candidates below the cap do NOT prove actual token fit.
    require(all(type(n) is int and 0 < n < frozen["engine"]["max_model_len"] for n in doc_sizes),
            "nominal candidate must be positive and below the prior max-model-len")
    original = author_module(PARETO, PARETO_SHA, {"Job", "build_job_plan"})
    original.SERVER_CONFIGS = {"baseline": {"--no-enable-prefix-caching": None}}
    original.DOC_SIZES = sorted(doc_sizes)
    original.N_REPEATS_SERIAL = repeats
    original.COLD_ONLY = True
    jobs = [asdict(job) for job in original.build_job_plan(["baseline"])]
    require(all(j["curve"] == "cold_prefill" and j["server_config"] == "baseline"
                and j["concurrency"] == 1 for j in jobs), "unexpected native job kind")
    checked_curve = check_v2_structure(curve_file, identity) if curve_file else None
    return {"schema_version": 1, "artifact_kind": "candidate_calibration_plan_not_cost_curves",
        "status": "PLANNED_CPU_ONLY", "execution": "UNEXECUTED", "gpu_executed": False,
        "network_used": False, "budget_reservation": "NOT_RESERVED", "driver_ready": False,
        "configuration_frozen_for_execution": False, "p1_complete": False, "p3_activated": False,
        "identity": identity,
        "author_reuse": {"path": PARETO, "commit": PARETO_COMMIT, "sha256": PARETO_SHA,
            "nodes_executed": ["Job", "build_job_plan"], "entire_source_hash_verified_first": True,
            "calls_excluded": ["main", "select_model_preset", "start_server", "run_benchmark"]},
        "candidate_jobs": jobs, "planned_measured_requests": sum(j["n_requests"] for j in jobs),
        "doc_size_semantics": "original driver nominal word-derived argument; actual token counts unverified",
        "actual_token_fit_verified": False, "untimed_pre_warmup_requests": None,
        "prior_engine_reference": frozen["engine"], "cold_control_required": {"enable_prefix_caching": False},
        "calibration_measurements": {"f": None, "g_mem": None, "g_ssd": None},
        "optional_curve_review": checked_curve,
        "non_ssd_feasibility": {"f": "does not require io_uring; requires a reviewed real GPU driver and authorized budget",
            "g_mem": "not measured; original emit estimator depends on g_ssd and actual disk bandwidth",
            "g_ssd": "blocked by io_uring EPERM and unqualified original storage path"},
        "blocking_adaptations": [
            "Exact-token adapter/actual token evidence; original prompt uses words and defaults chat templates",
            "Explicit bounded pre-warmup (original benchmark defaults five 256-word prompts)",
            "Same-session reviewed launcher; original vllm_server.start_server calls setsid via start_new_session",
            "Local noninteractive model/config selection and full prior offline/private-cache/JIT launch environment",
            "Real repetitions and request/TTFT evidence with failures retained; no smoke wall time converted to f",
            "Native staging/SSD lifecycle and real CQE evidence before g_mem/g_ssd or complete v2 export",
            "Original emit_break_even emits scalar v1 only; do not relabel it as v2"],
        "future_v2_contract": {"native_reader": BREAK_EVEN, "reader_commit": BREAK_EVEN_COMMIT,
            "reader_sha256": BREAK_EVEN_SHA, "required_curves": ["f", "g_ssd", "g_mem"],
            "scalar_gate_fields_must_be_explicit": ["break_even_ssd_tokens", "break_even_mem_tokens", "safety_margin_tokens"],
            "range_policy": "do not silently extrapolate outside observed calibration domain",
            "identity_required": ["exact runtime model_name", "kv_dtype=auto", "actual BF16 provenance", "kv_bytes_per_token"],
            "algorithm": "reuse original interpolation; no curve algorithm implemented here"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--doc-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--curve-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = prepare(args.smoke_dir, args.doc_sizes, args.repeats, args.curve_file)
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / "artifacts/prefix_io_v1"), "output must be project artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(plan, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "status": plan["status"],
                      "candidate_jobs": len(plan["candidate_jobs"]), "gpu_executed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
