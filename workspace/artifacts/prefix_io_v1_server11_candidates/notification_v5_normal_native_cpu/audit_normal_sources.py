"""Readonly ancestry audit for the finite normal compatibility copy."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

D = "artifacts/prefix_io_v1/server11-c5-normal-native-cpu-20261004"
G = "artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004"
OLD = "artifacts/prefix_io_v1/server11-c5-combined-runtime-cpu-20261004"
PURE_RUNNER = ("config_for_engine", "frontend_capture", "original_drain",
    "install_owner_snapshot_observation", "native_handler", "zero_snapshot",
    "input_groups", "prompt", "prepare_window_storage", "require_cold_inputs")

def require(value, reason):
    if not value:
        raise ValueError(reason)

def definitions(path):
    tree = ast.parse(path.read_bytes())
    return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    root = a.root.resolve(strict=True)
    output = a.output.absolute()
    require(output.resolve().is_relative_to(root) and not output.exists(), "new project evidence")
    proof = []
    old, new = definitions(root / OLD / "run_p4_single_file_experiment.py"), definitions(root / D / "run_p4_single_file_experiment.py")
    for name in PURE_RUNNER:
        require(old[name] == new[name], "original native/model helper changed: " + name)
        proof.append("run_p4_single_file_experiment.py:" + name)
    for file, excluded in (
        ("native_conditional_cost.py", {"binding_api"}),
        ("prepare_and_verify_native_cost.py", set()),
        ("p4_single_file_receipt.py", {"_runtime_refs", "_load_frozen_serializer", "load_verified_single_file"}),
    ):
        ancestor = root / G / file if file != "p4_single_file_receipt.py" else root / G / "common_candidate/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py"
        old, new = definitions(ancestor), definitions(root / D / file)
        require(set(old) == set(new), "compatibility copy changed callable set: " + file)
        for name in sorted(set(old) - excluded):
            require(old[name] == new[name], "original math/serializer/receipt AST changed: " + file + ":" + name)
            proof.append(file + ":" + name)
    old = root / OLD / "notification_runtime_adapter.py"
    new = root / D / "notification_runtime_adapter.py"
    require(old.read_bytes() == new.read_bytes(), "notification adapter changed")
    serial = root / D / "prepare_and_verify_native_cost.py"
    require(serial.read_bytes() == (root / G / serial.name).read_bytes(), "serializer bytes changed")
    common = json.loads((root / D / "COMMON_SOURCE_LOCK.json").read_bytes())
    locked = {r["path"]: r for r in common["files"]}
    site = json.loads((root / G / "SITE_SOURCE_LOCK.json").read_bytes())
    preserved = []
    for row in site["files"]:
        if row["path"].startswith(G + "/common_candidate/") and row["path"].endswith(".py"):
            path = root / row["path"]
            raw = path.read_bytes()
            require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "G core drift")
            require(locked.get(row["path"]) == row, "G core outside shared normal closure")
            preserved.append(row["path"])
    document = {
        "schema": "normal_native_frozen_original_source_ancestry_v1",
        "status": "PASS_ORIGINAL_ENGINE_AND_NUMERICAL_ANCESTRY",
        "unchanged_callable_AST": proof, "unchanged_callable_count": len(proof),
        "frozen_G_common_Python_files_verified": preserved,
        "frozen_G_common_Python_file_count": len(preserved),
        "serializer_byte_identical": True, "notification_adapter_byte_identical": True,
        "only_historical_binding_api_changes_in_native_copy": True,
        "canonical_metadata_history_source_functions_changed": ["_runtime_refs", "_load_frozen_serializer", "load_verified_single_file"],
        "canonical_numeric_budget_kernel_and_type_AST_unchanged": True,
        "normal_execute_window_has_explicit_authority_canonical_registration_and_gross_meter_changes": True,
        "no_claim_of_whole_execute_window_AST_identity": True,
        "GPU_jobs": 0, "model_loads": 0,
        "normal_runtime_qualified": False, "strategy_effect_verified": False}
    with output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(document, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({k: v for k, v in document.items() if k not in ("unchanged_callable_AST", "frozen_G_common_Python_files_verified")}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

