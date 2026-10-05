"""CPU-only next-day command preparation over real project CLI source.

This module does not import a GPU backend, start any command, reserve a budget,
change permissions or probe a GPU. --check-launch always denies in this scope.
The listed legacy P3 commands are compatibility references, never P4 authority.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil

ART = "artifacts/prefix_io_v1/server08-p4-02-cpu"
OUT = ART + "/gpu-next-day"
PERMISSION = "experiments/prefix_io_v1/configs/permissions.yaml"
LEDGER = "experiments/prefix_io_v1/gpu-budget-ledger.json"
AUTH_CHOICES = (ART + "/CPU_ONLY_AUTHORIZATION.json",
                "artifacts/prefix_io_v1/server08-p4-01-cpu/CPU_ONLY_AUTHORIZATION.json")
NATIVE = "third_party/work/py-kvcache-p4-02-cpu"
CONTROL = "third_party/work/prefix-io-p4-02-cpu/src"
AUTHOR = "third_party/work/vllm-author-p4-02-cpu"
BINARY = "third_party/work/vllm-author-build"
MODEL = "models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444"
MODEL_PLAN = "artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json"
FLOOR = 8 * 1024**3
ROUND_RESERVE = 3 * 1024**3
RUNNER_RESERVE_SECONDS = 20
SCRIPTS = "experiments/prefix_io_v1/scripts"
LEGACY = {
 "qualify_simple_stage_gpu_p316.py": [
   "native.__file__ must be under py-kvcache-p3-16-cpu; no P4 bridge/config"],
 "qualify_native_fault_drain_gpu.py": [
   "source list and native.__file__ pin py-kvcache-p3-15-cpu"],
 "run_decode_interference_calibration_p316.py": [
   "NATIVE_ROOTS/source_names/main only accept P3-16; control source locks target src/prefix_io_control",
   "worker probe/control producer lacks P4 load and paired table qualification"],
 "run_concurrent_capacity_p316.py": [
   "only --simple-stage-config; no P4 option or qualified cost-table input",
   "capacity permit and worker probe must be rebound to new P4 source/geometry"],
}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def safe_path(root, relative):
    require(type(relative) is str and relative and "\\" not in relative,
            "project-relative POSIX path required")
    p = Path(relative)
    require(not p.is_absolute() and all(x not in ("", ".", "..") for x in relative.split("/")),
            "path traversal or absolute path")
    dest = root / p
    cursor = dest
    while cursor != root:
        require(not cursor.is_symlink(), "symlink evidence path")
        cursor = cursor.parent
    require(dest.resolve().is_relative_to(root.resolve()), "outside project")
    return dest

def ref(root, relative):
    p = safe_path(root, relative)
    length=p.stat().st_size
    binary=(relative.startswith(BINARY+"/vllm/") and Path(relative).suffix==".so")
    require(length <= (512 if binary else 64) * 1024**2,
            "source metadata/binary size bound")
    h=hashlib.sha256()
    with p.open("rb") as handle:
        for raw in iter(lambda:handle.read(1024**2),b""):
            h.update(raw)
    return dict(path=relative, bytes=length, sha256=h.hexdigest())

def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            name + " must be nonnegative finite number")
    return value

def permission_fields(raw):
    """Read only required existing YAML scalar/list fields, without backend imports."""
    lines = raw.splitlines()
    values = {}
    for key in ("allow_gpu_runs", "max_gpu_hours", "approved_experiment_root"):
        found = [l.split(":", 1)[1].strip() for l in lines
                 if l.startswith(key + ":")]
        require(len(found) == 1, "missing/duplicate permission field: " + key)
        values[key] = found[0]
    idx = [i for i, line in enumerate(lines) if line == "approved_gpu_ids:"]
    require(len(idx) == 1, "one approved UUID list required")
    entries = []
    for line in lines[idx[0] + 1:]:
        if line.startswith("- "):
            entries.append(line[2:].strip())
        elif line and not line.startswith("#"):
            break
    require(len(entries) == 1 and re.fullmatch(
        r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", entries[0]),
        "exact one approved GPU UUID required")
    values["approved_gpu_ids"] = entries
    values["max_gpu_hours"] = number(float(values["max_gpu_hours"]), "GPU hours")
    require(values["max_gpu_hours"] > 0, "GPU budget must be positive")
    require(values["allow_gpu_runs"] in ("true", "false"), "explicit permission boolean")
    values["allow_gpu_runs"] = values["allow_gpu_runs"] == "true"
    return values

def inspect_authorization(root):
    candidates = [s for s in AUTH_CHOICES if (root / s).is_file()]
    require(candidates, "recorded CPU-only authorization required")
    selected = candidates[0]
    data = json.loads(safe_path(root, selected).read_text())
    require(type(data.get("schema_version")) is int and data["schema_version"] == 1,
            "CPU authorization schema")
    for key, expected in {"allow_project_local_edits": True, "allow_cpu_tests": True,
                         "allow_gpu_initialization": False, "allow_gpu_runs": False,
                         "allow_model_downloads": False,
                         "allow_driver_or_system_changes": False}.items():
        require(type(data.get(key)) is bool and data[key] is expected,
                "explicit CPU-only scope: " + key)
    require(data["base_permissions"] == ref(root, PERMISSION),
            "base permissions changed after CPU authorization")
    return ref(root, selected)

def cli_inventory(root, script):
    raw = safe_path(root, SCRIPTS + "/" + script).read_text()
    tree = ast.parse(raw)
    declarations = [ast.unparse(n) for n in ast.walk(tree)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "add_argument"]
    relevant = [{"line": i + 1, "source": l.strip()}
                for i, l in enumerate(raw.splitlines())
                if any(k in l for k in ("NATIVE_ROOTS", "p3-16-cpu", "p3-15-cpu",
                                        "source_sha256", "source_names", "simple-stage-config"))]
    return dict(source=ref(root, SCRIPTS + "/" + script),
                argparse_declarations=declarations, source_selection_checks=relevant)

def command(root, label, seconds, child, env, storage, role):
    guard = [".venv/bin/python", SCRIPTS + "/run_gpu_stage.py",
             "--label", label, "--seconds", str(seconds), "--", *child]
    return dict(label=label, seconds_limit=seconds,
                termination_reserve_seconds=RUNNER_RESERVE_SECONDS,
                planned_reserved_seconds=seconds + RUNNER_RESERVE_SECONDS,
                child_argv=child, guard_argv=guard, environment_preview=env,
                shell_preview=None,
                internal_guard_reference_only=True,
                new_storage_reserve_bytes=storage, role=role,
                launch_now_authorized=False, executed=False)

def required_source_paths(root):
    required={NATIVE+"/py_kvcache/reactor.py",NATIVE+"/py_kvcache/vllm.py"}
    required.update(AUTHOR+"/vllm/distributed/kv_transfer/kv_connector/v1/offloading/"+n+".py"
                    for n in ("common","scheduler","worker"))
    for directory in (CONTROL+"/prefix_io_control",NATIVE+"/py_kvcache",AUTHOR+"/vllm"):
        for path in (root/directory).rglob("*.py"):
            required.add(path.relative_to(root).as_posix())
    for name in ("prepare_p4_gpu_next_day.py","qualify_p4_native_gpu.py",
                 "prepare_p4_calibration_startup.py","prepare_p4_raw_pair.py"):
        if (root/SCRIPTS/name).is_file():required.add(SCRIPTS+"/"+name)
    required.update(BINARY+"/vllm/"+name for name in ("_C.abi3.so","_C_stable_libtorch.abi3.so"))
    return required

def validate_source_lock(root, relative):
    if relative is None:
        return None
    raw = json.loads(safe_path(root, relative).read_text())
    require(type(raw) is dict and type(raw.get("files")) is list and raw["files"],
            "final lock requires nonempty files refs")
    names = set()
    for entry in raw["files"]:
        require(type(entry) is dict and set(entry) == {"path", "bytes", "sha256"},
                "strict final source ref")
        require(entry["path"] not in names and entry == ref(root, entry["path"]),
                "stale/duplicate final source lock")
        names.add(entry["path"])
    required=required_source_paths(root)
    require(required<=names and any(p.startswith(CONTROL+"/prefix_io_control/p4_") for p in names),
            "source lock must cover all actual new Python sources, adapters and locked binary fallback")
    return ref(root, relative)

def plan(root, source_lock=None, disk_usage=shutil.disk_usage):
    root = root.resolve()
    authorization = inspect_authorization(root)
    permission = permission_fields(safe_path(root, PERMISSION).read_text())
    expected_runs = root / "experiments/prefix_io_v1/runs"
    require(Path(permission["approved_experiment_root"]).resolve() == expected_runs,
            "approved PRIMARY root mismatch")
    ledger_raw = safe_path(root, LEDGER).read_bytes()
    ledger = json.loads(ledger_raw)
    used = number(ledger["gpu_wall_seconds"], "GPU wall seconds")
    remaining = permission["max_gpu_hours"] * 3600 - used
    require(remaining >= 0, "GPU budget already exceeded")
    active = ledger.get("active_reservation")
    free = disk_usage(expected_runs).free
    require(type(free) is int and free >= 0, "actual PRIMARY free byte observation")
    environment = {
        "PYTHONPATH": ":".join(str(root/p) for p in (CONTROL, NATIVE, AUTHOR, SCRIPTS)),
        "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"
    }
    output = lambda name: str(expected_runs / name / "details")
    library = BINARY + "/vllm/_C.abi3.so"
    previews = [
        command(root, "server08-p4-02-author-copy-01", 60,
            [".venv/bin/python", SCRIPTS+"/qualify_author_copy.py", "--library", library,
             "--gpu-uuid", permission["approved_gpu_ids"][0],
             "--output", output("server08-p4-02-author-copy-01")+".json"],
            environment, 16*1024**2, "reusable common CUDA primitive; does not qualify P4"),
        command(root, "server08-p4-02-native-prefix-01", 600,
            [".venv/bin/python", SCRIPTS+"/native_gpu_prefix_smoke.py",
             "--model-dir", MODEL, "--model-plan", MODEL_PLAN,
             "--output-dir", output("server08-p4-02-native-prefix-01")],
            {**environment,"PYTHONPATH":str(root/"third_party/work/vllm-author-build")+":"+str(root/SCRIPTS)},
            512*1024**2,
            "common model sanity using existing compiled author source; no new P4 source qualification"),
    ]
    for mode in ("off","shadow"):
        name="server08-p4-02-native-"+mode+"-01"
        entry=command(root,name,180,
            [".venv/bin/python",SCRIPTS+"/qualify_p4_native_gpu.py","--mode",mode,
             "--name",name,"--execute","--scope-record",ART+"/GPU_STAGE_AUTHORIZATION.json",
             "--source-lock",source_lock or OUT+"/final-source-lock.json"],
            environment,128*1024**2,"new P4 native source-bound G1; GPU-unqualified, explicit scope restore required")
        launch=[".venv/bin/python",SCRIPTS+"/qualify_p4_native_gpu.py","--mode",mode,
                "--name",name,"--launch","--scope-record",ART+"/GPU_STAGE_AUTHORIZATION.json",
                "--source-lock",source_lock or OUT+"/final-source-lock.json"]
        entry["launch_argv"]=launch
        entry["shell_preview"]=" ".join(shlex.quote(k+"="+v) for k,v in environment.items())+" "+shlex.join(launch)
        previews.append(entry)
    inventory = {name: cli_inventory(root,name) for name in
        ["run_gpu_stage.py","qualify_author_copy.py","native_gpu_prefix_smoke.py",
         "qualify_p4_native_gpu.py",*LEGACY]}
    final_lock = validate_source_lock(root, source_lock)
    rows = []
    for c in previews:
        rows.append(dict(label=c["label"],
            budget_fit=active is None and c["planned_reserved_seconds"] <= remaining,
            PRIMARY_storage_fit=free-c["new_storage_reserve_bytes"] >= FLOOR,
            reusable_common_command=c["role"].startswith("reusable common") or c["role"].startswith("common model"),
            planned_P4_native_G1=c["role"].startswith("new P4 native"), GPU_qualified=False, qualifies_P4=False))
    return dict(schema_version=1, status="P4_NEXT_DAY_DRY_RUN_GPU_BLOCKED_CPU_SCOPE",
        user_facing_planned_local_date="2026-10-02",
        current_scope="CPU_ONLY; tomorrow availability is not current GPU authorization",
        authorization=authorization, permissions=ref(root,PERMISSION), ledger=ref(root,LEDGER),
        new_gpu_runs=0, gpu_initialized=False, gpu_availability_probed=False,
        budget_reserved_seconds=0, commands_executed=0, prepared_command_is_authorization=False,
        approved_gpu_uuid_from_file=permission["approved_gpu_ids"][0],
        gpu_identity_revalidated=False, gpu_uuid_change_requires_explicit_scope_update=True,
        budget=dict(max_seconds=permission["max_gpu_hours"]*3600, used_seconds=used,
                    remaining_seconds=remaining, active_reservation=active,
                    per_command_termination_reserve_seconds=RUNNER_RESERVE_SECONDS,
                    preview_total_reserved_seconds=sum(c["planned_reserved_seconds"] for c in previews)),
        storage=dict(root=str(expected_runs), free_bytes=free, minimum_free_bytes=FLOOR,
            bytes_available_above_floor=max(0,free-FLOOR),
            standard_model_round_reserve_bytes=ROUND_RESERVE,
            standard_model_round_fit=free-ROUND_RESERVE >= FLOOR,
            no_cache_merge_or_deletion_authorized=True, AUX_writes_not_prepared=True),
        workspaces=dict(native=NATIVE,control=CONTROL,author=AUTHOR),
        final_source_lock=final_lock, source_freeze_pending=final_lock is None,
        cli_inventory=inventory, command_previews=previews, command_gates=rows,
        legacy_cli_compatibility={k:dict(qualifies_P4=False,direct_P4_launch_allowed=False,
            required_new_adapter_changes=v) for k,v in LEGACY.items()},
        pending_freeze=[
            "completed P4-02 native/control/runner source refs",
            "actual device UUID and CUDA/native ABI under authorized GPU",
            "new P4 native qualification runner + exact byte/lifecycle receipt",
            "source/layout/model/kernel/quantum-bound load and ETA qualification",
            "paired production cost table + independent semantic verifier receipt + heldout uncertainty",
            "same fixed ordinary caps and parent/KV/staging/preload/fusion/stop-drain configuration",
            "actual per-stage storage reservation from new runner; 3GiB default may be blocked",
            "P4-only small pilot source-bound candidate and all mode activation reasons"],
        factorial=dict(C00=dict(dependency=False,interference=False),
                       C01=dict(dependency=False,interference=True),
                       C10=dict(dependency=True,interference=False),
                       C11=dict(dependency=True,interference=True),
                       ordinary_caps="identical frozen fixed caps",
                       independent_reference="U", SLO=None, P5_allowed=False),
        next_gates=[
            "authorized GPU scope + approved actual device + storage/budget preflight",
            "source-locked P4 off/shadow real CUDA/AIO, exact KV/full parent/zero ordinary quota drain",
            "production load/ETA/table qualification tied to actual source and frozen geometry",
            "I/D/J real lifecycle/model/live shadow qualification",
            "bounded P4 pilot including independent U and same-base four-arm evidence"],
        effect_verified=False, full_P4_complete=False)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--source-lock", help="final append-only JSON with exact P4-02 files refs")
    ap.add_argument("--check-launch", action="store_true")
    args = ap.parse_args(argv)
    root = args.project.resolve()
    target = args.output
    if target.is_absolute():
        relative = target.relative_to(root).as_posix()
    else:
        relative = target.as_posix()
    require(relative.startswith(OUT+"/"), "receipt must be in bounded next-day artifact directory")
    target = safe_path(root, relative)
    require(not target.exists(), "append-new receipt required")
    result = plan(root,args.source_lock)
    if args.check_launch:
        result["status"] = "BLOCKED_CPU_ONLY_NO_GPU_LAUNCH"
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False);handle.write("\n")
    print(json.dumps({k:result[k] for k in ("status","new_gpu_runs","gpu_initialized",
                                          "budget_reserved_seconds","source_freeze_pending")}))
    return 78 if args.check_launch else 0

if __name__ == "__main__":
    raise SystemExit(main())
