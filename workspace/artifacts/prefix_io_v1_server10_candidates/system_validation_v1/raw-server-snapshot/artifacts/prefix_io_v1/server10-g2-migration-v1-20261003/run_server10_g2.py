"""Machine binding of the unchanged original normal-model G2 launcher.

Original scope, source, budget, output, shutdown and off-before-shadow gates
remain authoritative. No model executor, collector or cache is implemented.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
DELIVERY = "artifacts/prefix_io_v1/server10-g2-migration-v1-20261003"
SCRIPT = DELIVERY + "/run_server10_g2.py"
CONFIG = DELIVERY + "/G2_MIGRATION_CONFIG.json"
LOCK = DELIVERY + "/gpu-source-lock-server10-g2.json"
SCOPE = DELIVERY + "/SERVER10_G2_AUTHORIZED_SCOPE.json"
PERMISSIONS = "experiments/prefix_io_v1/configs/permissions.server10.reference.yaml"
GPU_UUID = "GPU-4b4d17ec-95a3-4efd-2bc4-1613333e949f"
JOBS = dict(off="server10-g2-normal-off-01", shadow="server10-g2-normal-shadow-01")
ORIGINAL_REF = dict(path="artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/run_g2_normal_model_lifecycle.py",
    bytes=46746, sha256="82cea015522596a89a89eb1ea0245f9ea111c942b74456cd87288507aec54422")
INVENTORY_REF = dict(path="artifacts/prefix_io_v1/server10-cuda13-cpu-20261003/CUDA13_SOURCE_INVENTORY.json",
    bytes=692329, sha256="0e7cc1df1d8d7aed7271841d6e9f1e4188785f40ad4c74b4c6c78f670b7794b6")
PROOF_REF = dict(path="artifacts/prefix_io_v1/server10-cuda13-cpu-20261003/CPU_COMPILE_LINK_RESULT.json",
    bytes=5588, sha256="ef7a7482cc1cae539e2425d6aefc82ea8160e398ab7d8b14926f76537e614a25")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def safe(root, relative):
    require(type(relative) is str and relative and "\\" not in relative and not Path(relative).is_absolute()
            and all(part not in ("", ".", "..") for part in relative.split("/")), "safe G2 migration relative path")
    path = root / relative
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), "G2 migration symlink rejected")
        cursor = cursor.parent
    require(path.resolve().is_relative_to(root), "G2 migration path outside project")
    return path


def json_read(path):
    require(path.is_file() and 0 < path.stat().st_size <= 2 * 1024**2, "bounded G2 migration JSON")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate G2 migration JSON key")
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("finite G2 migration JSON")))


def file_ref(root, relative):
    path = safe(root, relative)
    require(path.is_file() and 0 < path.stat().st_size <= 2 * 1024**2, "bounded G2 bootstrap source")
    raw = path.read_bytes()
    return dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def checked(root, row):
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}
            and type(row["bytes"]) is int and type(row["sha256"]) is str, "exact G2 migration source ref")
    require(file_ref(root, row["path"]) == row, "G2 migration source drift: " + str(row["path"]))
    return safe(root, row["path"])


def bind_original(root, refs, config):
    """CPU factory used by preparation/tests; caller verifies frozen config."""
    root = Path(root).resolve(strict=True)
    require(config == dict(schema_version=1, gpu_uuid=GPU_UUID, job_names=JOBS,
        permissions_ref=config.get("permissions_ref"), sdk_inventory_ref=INVENTORY_REF,
        sdk_proof_ref=PROOF_REF, source_lock=LOCK, scope_record=SCOPE), "fixed G2 migration configuration")
    require(config["permissions_ref"]["path"] == PERMISSIONS, "fixed G2 machine permissions")
    for row in (ORIGINAL_REF, INVENTORY_REF, PROOF_REF, config["permissions_ref"]):
        require(refs.get(row["path"]) == row, "frozen G2 migration source mapping")
        checked(root, row)
    before_modules = dict(sys.modules)
    old_environment, old_path, old_argv, old_cwd = dict(os.environ), list(sys.path), sys.argv, Path.cwd()
    path = checked(root, ORIGINAL_REF)
    name = "_server10_g2_original_" + ORIGINAL_REF["sha256"][:20]
    require(name not in sys.modules, "fresh G2 original private module")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    saved = {}
    def patch(key, value):
        saved[key] = getattr(module, key)
        setattr(module, key, value)
    replacement = {module.SDK_INVENTORY: INVENTORY_REF, module.SDK_PROOF: PROOF_REF}
    new_sdk = tuple((replacement[p]["path"], replacement[p]["bytes"], replacement[p]["sha256"])
                   if p in replacement else (p, n, sha) for p, n, sha in module.SDK_SOURCE_REFS)
    for key, value in (("GPU_UUID", GPU_UUID), ("SCRIPT", SCRIPT), ("PERMISSIONS", PERMISSIONS),
                       ("SDK_INVENTORY", INVENTORY_REF["path"]), ("SDK_PROOF", PROOF_REF["path"]),
                       ("SDK_SOURCE_REFS", new_sdk)):
        patch(key, value)
    original_scope_template = module.scope_template
    original_verify = module.verify_source_lock
    def scope_template():
        value = original_scope_template()
        value["permitted_run_names"] = dict(JOBS)
        return value
    def verify_source_lock(project, relative, baseline_relative=module.BASELINE_LOCK):
        require(Path(project).resolve(strict=True) == root and relative == LOCK, "fixed new G2 migration source lock")
        result = original_verify(project, relative, baseline_relative)
        require(result == refs, "actual complete source refs equal bootstrap mapping")
        for relative in (SCRIPT, CONFIG, ORIGINAL_REF["path"], INVENTORY_REF["path"], PROOF_REF["path"], PERMISSIONS):
            checked(project, result[relative])
        require(json_read(checked(project, result[CONFIG])) == config, "unchanged G2 machine configuration")
        return result
    patch("scope_template", scope_template)
    patch("verify_source_lock", verify_source_lock)
    require(all(getattr(module, key).__globals__ is module.__dict__ for key in (
        "context", "execute_guarded", "scope_source_gates", "execution_gates", "shadow_prerequisite", "child_command")),
        "original G2 actual function globals")
    evidence = dict(status="PRIVATE_G2_MACHINE_BINDINGS_ACTIVE", gpu_uuid=GPU_UUID,
        original_ref=ORIGINAL_REF, changed_globals=list(saved), original_source_modified=False,
        executor_changed=False, guard_changed=False, collector_changed=False,
        all_bindings_restored=False, all_new_private_modules_unloaded=False)
    finished = False
    def cleanup():
        nonlocal finished
        require(not finished, "G2 migration cleanup exactly once")
        finished = True
        for key, value in saved.items():
            setattr(module, key, value)
        evidence["all_bindings_restored"] = all(getattr(module, key) is value for key, value in saved.items())
        removed = []
        for key in tuple(sys.modules):
            if key not in before_modules and key.startswith(("_server10_g2_original_", "_g2_lifecycle_")):
                sys.modules.pop(key)
                removed.append(key)
        sys.path[:] = old_path
        sys.argv = old_argv
        os.chdir(old_cwd)
        os.environ.clear()
        os.environ.update(old_environment)
        evidence.update(private_modules_unloaded=sorted(removed),
            all_new_private_modules_unloaded=all(key not in sys.modules for key in removed),
            preexisting_modules_preserved=all(sys.modules.get(key) is value for key, value in before_modules.items()),
            environment_restored=dict(os.environ) == old_environment, path_restored=sys.path == old_path)
        evidence["status"] = "RESTORED_PRIVATE_G2_MACHINE_BINDINGS" if all(evidence[key] for key in (
            "all_bindings_restored", "all_new_private_modules_unloaded", "preexisting_modules_preserved",
            "environment_restored", "path_restored")) else "FAILED_G2_MIGRATION_RESTORE"
        return evidence
    return module, cleanup


def build_facade(root, source_lock=LOCK):
    root = Path(root).resolve(strict=True)
    require(root == ROOT and source_lock == LOCK, "fixed server10 G2 project/lock")
    lock = json_read(safe(root, source_lock))
    require(type(lock.get("files")) is list and 2052 <= len(lock["files"]) <= 8192, "complete G2 source list")
    refs = {row["path"]: row for row in lock["files"]}
    require(len(refs) == len(lock["files"]) and SCRIPT in refs and CONFIG in refs, "unique G2 bootstrap refs")
    checked(root, refs[SCRIPT])
    require(Path(__file__).resolve() == safe(root, SCRIPT).resolve(), "G2 bootstrap actual source path")
    config = json_read(checked(root, refs[CONFIG]))
    return bind_original(root, refs, config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=tuple(JOBS), default="off")
    parser.add_argument("--name")
    parser.add_argument("--source-lock", default=LOCK)
    parser.add_argument("--scope-record", default=SCOPE)
    actions = parser.add_mutually_exclusive_group()
    for action in ("preflight", "scope-template", "launch", "execute"):
        actions.add_argument("--" + action, action="store_true")
    args = parser.parse_args(argv)
    require(args.name in (None, JOBS[args.mode]), "fixed new G2 job name")
    require(args.scope_record == SCOPE, "independent server10 G2 scope only")
    module, cleanup = build_facade(args.project, args.source_lock)
    evidence = None
    try:
        command = ["--project", str(args.project), "--mode", args.mode, "--name", JOBS[args.mode],
                   "--source-lock", args.source_lock, "--scope-record", args.scope_record]
        for action in ("preflight", "scope_template", "launch", "execute"):
            if getattr(args, action):
                command.append("--" + action.replace("_", "-"))
        return module.main(command)
    finally:
        evidence = cleanup()
        # The original child alone creates its approved details directory.
        if args.execute:
            details = args.project / "experiments/prefix_io_v1/runs" / JOBS[args.mode] / "details"
            if details.is_dir():
                with (details / "server10-g2-migration-state.json").open("x", encoding="utf-8") as stream:
                    json.dump(evidence, stream, indent=2, allow_nan=False)
                    stream.write("\n")
        require(evidence["status"] == "RESTORED_PRIVATE_G2_MACHINE_BINDINGS", "G2 machine binding cleanup")


if __name__ == "__main__":
    raise SystemExit(main())
