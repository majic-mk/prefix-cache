"""Freeze shared strong U/I values from original P3 inputs without GPU/device binding."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = next((p for p in (HERE.parents[1] / "server12-i-pilot-cpu-preparation-20261004",
                           HERE.parents[1] / "i_pilot_cpu_preparation_20261004") if p.is_dir()),
                HERE.parents[1] / "server12-i-pilot-cpu-preparation-20261004")
SOURCE_INPUTS = HERE.parent / "source_inputs"
ROOT = PurePosixPath("/root/autodl-tmp/prefix-io-v1-handoff/project")
NATIVE_RUNS = "experiments/prefix_io_v1/runs"


def require(value, message):
    if not value:
        raise ValueError("PRIVATE_PAIR_CPU_REJECTED: " + message)


def checksum(path):
    raw = path.read_bytes()
    return dict(path=path.as_posix(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def pinned(path, digest):
    require(path.is_file() and not path.is_symlink() and checksum(path)["sha256"] == digest, "original input drift")
    return path


def constants(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in names):
            result[node.targets[0].id] = ast.literal_eval(node.value)
    require(set(result) == set(names), "original source constants")
    return result


def original_extra(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == "extra" for t in n.targets))
    extra = {}
    for keyword in node.value.keywords:
        value = keyword.value
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div):
            extra[keyword.arg] = ast.literal_eval(value.left) / ast.literal_eval(value.right)
        else:
            try:
                extra[keyword.arg] = ast.literal_eval(value)
            except ValueError:
                pass  # Actual per-run path/identity are bound below, not guessed.
    extra.pop("shared_storage_path", None)
    return extra


def build(*, server_root=ROOT, run_u="server12-prerent-strong-off01", run_i="server12-prerent-strong-on01",
          source_inputs=SOURCE_INPUTS, previous=PREVIOUS):
    server_root = PurePosixPath(str(server_root).replace("\\", "/"))
    require(server_root.is_absolute() and str(server_root).startswith("/root/"), "explicit current server workspace")
    inputs = previous / "source_inputs"
    smoke = pinned(inputs / "native_gpu_prefix_smoke.py", "7d31cce851ca2334fedbe6b66c4e2d9bfff19e8768bafaa52f68e414518d77e2")
    p3_driver = pinned(inputs / "run_p3_native_pilot.py", "ff0905821eddc368dbfe168fdb735603dcfedbddcec96d86dfd01614dfbdf855")
    manifest = pinned(source_inputs / "low_contention.json", "3ef3b7a65d87cb8f4d1cb88c885137ec29e1eebdfebe0aedf420e803d9d0a5d9")
    curve = pinned(source_inputs / "ORIGINAL_PLANNER_CURVES_V2.json", "f7cd071cde00aa18edcc463be0b83886bb4240291acabb4c9f4adb30b8c6311e")
    curve_ref = json.loads((source_inputs / "ORIGINAL_PLANNER_CURVES_INPUT.json").read_bytes())
    require(curve_ref["sha256"] == checksum(curve)["sha256"] and PurePosixPath(curve_ref["path"]).is_relative_to(server_root),
            "same actual original planner curve reference")
    manifest_data = json.loads(manifest.read_bytes())
    base = constants(smoke, ("ENGINE", "SAMPLING"))
    engine = dict(base["ENGINE"], **manifest_data["engine"])
    # Supported common switches already proven by original normal-model runs.
    engine.update(async_scheduling=False, disable_log_stats=True, speculative_config=None,
                  prefix_caching_hash_algo="sha256", skip_tokenizer_init=True)
    sampling = dict(base["SAMPLING"], min_tokens=128, max_tokens=128)
    extra = original_extra(p3_driver)
    extra["prefix_cache_break_even_path"] = curve_ref["path"]
    if extra.get("prefix_io_observation_mode") == "shadow":
        extra["prefix_io_observation_run_id"] = "builder_binds_common_identity"
    helper_path = previous / "strong_baseline/strong_baseline_config.py"
    pinned(helper_path, "d2d1d8e35c582df3a6f4857f91b21c224d524ed2cb0ca9a35948f19195396478")
    name = "_prerent_original_strong_configuration"
    spec = importlib.util.spec_from_file_location(name, helper_path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[name] = helper
    spec.loader.exec_module(helper)
    policy = dict(schema_version=1, run_id=run_i, mode="interference", sample_max_age_ns=100_000_000,
                  max_wait_ns=100_000_000, internal_step_budget_ns=None, candidate_batches=[1],
                  fixed_stage_policy=None, cost_table=None)
    pair = helper.build_runtime_pair(engine, sampling, extra,
        storage_paths={"U": str(server_root / NATIVE_RUNS / run_u / "storage"),
                       "I": str(server_root / NATIVE_RUNS / run_i / "storage")},
        run_ids={"U": run_u, "I": run_i}, i_policy=policy, max_accepted_parents=8)
    return dict(schema="strong_native_u_i_cpu_configuration_v1", configurations=pair,
                contract=helper.validate_runtime_pair(pair), gpu_uuid=None, GPU_operations=0,
                native_execution_verified=False, gpu_effect_qualified=False, formal_goodput_allowed=False,
                independent_internal_step_budget_ns=None, development_deadline_ns=None, SLO=None, cost_receipt=None,
                source_inputs=[checksum(p) for p in (smoke, p3_driver, manifest, curve, helper_path)],
                original_planner_curve_ref=curve_ref,
                original_planner_curve_status="STRUCTURAL_ORIGINAL_LOADPLANNER_INPUT_ONLY_NOT_CURRENT_GPU_COST_COVERAGE",
                original_recompute_overpriced_warning_preserved=True,
                qualification_workload_kind="controlled_original_P3_mechanism", natural_trace_bound=False,
                future_effect_activation="requires independently frozen budget and newly qualified finite strong-domain cells")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-root", type=PurePosixPath, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-u", default="server12-prerent-strong-off01")
    parser.add_argument("--run-i", default="server12-prerent-strong-on01")
    args = parser.parse_args(argv)
    document = build(server_root=args.server_root, run_u=args.run_u, run_i=args.run_i)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status="PASS_CPU_SHARED_STRONG_CONFIG_UNBOUND_DEVICE", actual_gpu_runs=0,
                         output=str(args.output), gpu_uuid=None, internal_step_budget_ns=None)))


if __name__ == "__main__":
    main()
