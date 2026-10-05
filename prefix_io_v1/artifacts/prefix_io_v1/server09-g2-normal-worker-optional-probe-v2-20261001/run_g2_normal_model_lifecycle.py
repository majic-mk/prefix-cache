"""Bounded original-LLM normal-output qualification candidate, not an experiment.

Import, --preflight and --scope-template use stdlib only. GPU/backend imports
occur only after a distinct G2 human scope, complete source lock, existing
budget-guard reservation, offline model bytes and storage checks all pass.
"""
from __future__ import annotations
import argparse
import dataclasses
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

DEST = "artifacts/prefix_io_v1/server09-g2-normal-worker-optional-probe-v2-20261001"
SCRIPT = DEST + "/run_g2_normal_model_lifecycle.py"
WORKER = DEST + "/g2_worker_observation.py"
SCALAR = "artifacts/prefix_io_v1/server09-runtime-connector-20261001/p4_runtime_scalar_connector.py"
FRAME = "artifacts/prefix_io_v1/server09-migration-20261001/collector-candidate/p4_full_step_frame_adapter.py"
AUTHOR = "third_party/work/vllm-author-p4-02-cpu"
RUNNER = AUTHOR + "/vllm/v1/worker/gpu_model_runner.py"
CONTEXT = ".venv/lib/python3.12/site-packages/torch/utils/_contextlib.py"
BASELINE_LOCK = "artifacts/prefix_io_v1/server09-g1-20261001/gpu-source-lock.json"
BASELINE_SHA = "5b438ebc1ec014c22d1d501915600b4cd9bc777ad9f05da1f7ac9b1a18cc1181"
GUARD = "experiments/prefix_io_v1/scripts/run_gpu_stage.py"
PREPARE = "experiments/prefix_io_v1/scripts/prepare_p4_gpu_next_day.py"
QUALIFY = "experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py"
SMOKE = "experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py"
PERMISSIONS = "experiments/prefix_io_v1/configs/permissions.server09.g1.yaml"
MODEL_PLAN = "artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json"
MODEL = "models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444"
GPU_UUID = "GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2"
PURPOSE = "NORMAL_MODEL_FULL_OUTPUT_LIFECYCLE_QUALIFICATION_ONLY"
SECONDS = 300
RESERVE = 128 * 1024**2
FLOOR = 8 * 1024**3
PROMPT = list(range(1000, 1128))
SAMPLING = dict(temperature=0.0, seed=0, max_tokens=128, min_tokens=128,
                ignore_eos=True, detokenize=False)
RUNTIME_CACHE_SOURCE_REFS = (
    (".venv/lib/python3.12/site-packages/flashinfer/jit/env.py", 6142,
     "09d2e6d6d03770d765acb8d89db2b58cd9600babacf6fbaad9dde0002a7d2cb9"),
    (".venv/lib/python3.12/site-packages/flashinfer/jit/core.py", 17095,
     "9879a6f02b6be03654a8f2107ae28d64418658619a31f10cad6960596a1b9596"),
    (".venv/lib/python3.12/site-packages/torch/utils/cpp_extension.py", 141604,
     "1517eb2ac276065210d7c861becc5bf3a5404796da16d35af9b9466667fef904"),
)
EXTRA_CACHE_DIRS = {
    "FLASHINFER_WORKSPACE_BASE": "flashinfer-workspace",
    "CUDA_CACHE_PATH": "cuda-driver",
    "TORCH_EXTENSIONS_DIR": "torch-extensions",
}
SDK_DEST = "artifacts/prefix_io_v1/server09-cuda13-toolchain-v2-20261001"
SDK_MODULE = SDK_DEST + "/cuda13_sdk_overlay.py"
SDK_PLAN = SDK_DEST + "/CUDA13_SDK_CPU_PLAN.json"
SDK_INVENTORY = "artifacts/prefix_io_v1/server09-cuda13-cpu-20261001/CUDA13_SOURCE_INVENTORY.json"
SDK_PROOF = "artifacts/prefix_io_v1/server09-cuda13-cpu-20261001/CPU_COMPILE_LINK_RESULT.json"
SDK_SOURCE_REFS = (
    (SDK_MODULE, 16929, "ed846361788e9bdde853e6e0aa16e361329e3f2dffbfbe2c4f396acf9e9d1ecf"),
    (SDK_PLAN, 10025, "8044db267791d5f217e3dd9c478375787aa73a74d53d19117f6bcebbc1f0a659"),
    (SDK_INVENTORY, 692329, "2ed079701c53d3e37005cf33432937e6c442f06cad16c5aded06efffcc7c36e0"),
    (SDK_PROOF, 5589, "3c3f9b092aea3ca89e07f31530a2ae3b38ab65cf0a8868821031f1d7f8d14e3f"),
    (".venv/lib/python3.12/site-packages/flashinfer/compilation_context.py", 3909,
     "e55c7f58a83810590e18686954527cc62735a6e91c3fa67286a5e7f00afc9be5"),
    (".venv/lib/python3.12/site-packages/flashinfer/jit/cpp_ext.py", 12445,
     "9d20f28baed969411220456b140a2786a081a2b7195d2433a70daec3c8ae7403"),
    (".venv/lib/python3.12/site-packages/flashinfer/jit/utils.py", 2644,
     "a32b738a91bafbe392c69e10fd11533ff98adbecbba9d769a19f2c5e920f5f31"),
    (".venv/lib/python3.12/site-packages/torch/version.py", 317,
     "323d35171ef1184f1d7db3bbd1f3d3e227e0e826be8fd52200778346e17c873f"),
)
SDK_PROCESS_KEYS = ("CUDA_HOME", "CUDA_PATH", "FLASHINFER_NVCC", "CUDACXX", "PATH", "LD_LIBRARY_PATH")
NINJA = ".venv/bin/ninja"
NINJA_SOURCE_REF = (NINJA, 370448, "08639e194fffa7f08b259fc4abfa4803aff66b64de52549cee42ec527d55cea6")
OPTIONAL_MODULE = DEST + "/optional_capability_probe.py"
OPTIONAL_SOURCE_REFS = (('artifacts/prefix_io_v1/server09-g2-normal-worker-optional-probe-v2-20261001/optional_capability_probe.py', 13976, '3cae91ffc978698fa10d002a5a3355b027e4ad0e953d7d8b323799e1136968fc'), ('experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py', 29692, '5d9604fc07d80495a49febfcd488cb2601f74a2bef107261c8e623a4d0a1c77d'), ('third_party/work/vllm-author-p4-02-cpu/vllm/third_party/__init__.py', 0, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'), ('third_party/work/vllm-author-p4-02-cpu/vllm/utils/import_utils.py', 14589, 'f14b07131a4d49955150016dbfa34f7d8d5f586b1545a1173e0d4f904578b18b'), ('third_party/work/vllm-author-p4-02-cpu/vllm/utils/deep_gemm.py', 19901, '99e316620e3cdda2db7d9fd7a4ae5968e9c9ef98d7399d62a2c4a5c506627854'), ('third_party/work/vllm-author-p4-02-cpu/vllm/model_executor/warmup/deep_gemm_warmup.py', 13044, 'b05db0127b19fc38da78e96c74df279e2cd53e0c262e9a7cad53486cf21f3414'))


def require(ok, reason):
    if not ok: raise ValueError(reason)


def same(actual, expected, reason):
    require(type(actual) is type(expected), reason)
    if type(expected) is dict:
        require(set(actual) == set(expected), reason)
        for key in expected: same(actual[key], expected[key], reason)
    elif type(expected) is list:
        require(len(actual) == len(expected), reason)
        for left, right in zip(actual, expected): same(left, right, reason)
    else: require(actual == expected, reason)


def safe(root, relative):
    require(type(relative) is str and relative and "\\" not in relative and
            not Path(relative).is_absolute() and
            all(p not in ("", ".", "..") for p in relative.split("/")),
            "bounded project-relative POSIX path")
    path = root / relative
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), "symlink source/evidence rejected")
        cursor = cursor.parent
    require(path.resolve().is_relative_to(root.resolve()), "path outside project")
    return path


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""): digest.update(block)
    return digest.hexdigest()


def read_json(path):
    require(path.stat().st_size <= 4 * 1024**2, "bounded JSON evidence")
    def unique(rows):
        out = {}
        for key, value in rows:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def checked_ref(root, row):
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "exact source ref")
    payload = type(row["path"]) is str and row["path"].startswith(MODEL + "/")
    limit = 4 * 1024**3 if payload else 512 * 1024**2
    require(type(row["bytes"]) is int and 0 <= row["bytes"] <= limit and
            type(row["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
            "bounded exact source bytes/hash")
    path = safe(root, row["path"])
    require(path.is_file() and path.stat().st_size == row["bytes"] and
            digest_file(path) == row["sha256"], "source bytes/hash drift")
    return path


def load_ref(root, row):
    path = checked_ref(root, row)
    require(path.suffix == ".py" and row["bytes"] <= 4 * 1024**2, "small source-only module")
    name = "_g2_lifecycle_" + row["sha256"][:20]
    if name in sys.modules:
        module = sys.modules[name]
        require(Path(module.__file__).resolve() == path.resolve(), "private module identity")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    try: spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None); raise
    return module


def context():
    return dict(purpose=PURPOSE, model_relative=MODEL, model_plan_relative=MODEL_PLAN,
        normal_outputs_per_request=128, request_phases_per_job=["cold", "repeat"],
        total_requests_across_two_jobs=4, total_outputs_across_two_jobs=512,
        expected_cached_tokens=[0, 112], prompt_token_ids=PROMPT, sampling=SAMPLING,
        engine_source=SMOKE, engine_overrides=dict(async_scheduling=False),
        scalar_source=SCALAR, frame_source=FRAME,
        author=AUTHOR, new_executor=False, native_io="none", strategies="off",
        current_stream_event_diagnostics_only=True, cross_clock_mapping_verified=False,
        production_qualified=False, performance_claim=False, seconds_limit=SECONDS,
        storage_reserve_bytes=RESERVE, storage_floor_bytes=FLOOR,
        multiprocessing=False, maximum_steps_per_request=4096,
        additional_process_cache_keys=EXTRA_CACHE_DIRS,
        sdk_toolchain=dict(purpose="EXISTING_INSTALLED_CUDA13_LAYOUT_ONLY", enabled_in_both_modes=True,
            source_refs=[dict(path=p, bytes=n, sha256=s) for p, n, s in SDK_SOURCE_REFS],
            CPU_only_inventory_preflight_required=True, GPU_model_or_JIT_runtime_qualified=False,
            original_arch_checks_preserved=True, no_JIT_bypass=True,
            environment_keys=list(SDK_PROCESS_KEYS), stubs_on_runtime_library_path=False),
        build_tool=dict(purpose="EXISTING_PINNED_NINJA_PRIVATE_PATH_ONLY", enabled_in_both_modes=True,
            source_ref=dict(path=NINJA_SOURCE_REF[0], bytes=NINJA_SOURCE_REF[1], sha256=NINJA_SOURCE_REF[2]),
            exact_regular_ELF_executable_required=True, no_whole_virtualenv_bin_PATH=True,
            CUDA13_bin_remains_first=True, GPU_model_or_JIT_runtime_qualified=False),
        optional_capability_probe=dict(purpose="ONE_PHYSICALLY_ABSENT_OPTIONAL_QUERY_ONLY",
            enabled_in_both_modes=True, optional_fullname="vllm.third_party.deep_gemm",
            source_refs=[dict(path=p, bytes=n, sha256=s) for p, n, s in OPTIONAL_SOURCE_REFS],
            original_finder_unchanged=True, author_files_unchanged=True,
            original_cached_query_first=True, matched_original_exception_origin_required=True,
            absent_optional_is_unavailable=True, required_imports_unchanged=True,
            reversible_process_only=True, GPU_model_or_full_output_qualified=False))


def scope_template():
    return dict(schema_version=1, status="NOT_AUTHORIZED_CPU_TEMPLATE",
        allow_gpu_initialization=False, allow_gpu_runs=False,
        purpose=PURPOSE, gpu_uuid=GPU_UUID, allowed_modes=["off", "shadow"],
        maximum_jobs=2, maximum_total_planned_reserve_seconds=640,
        permitted_run_names=dict(off="server09-g2-normal-off-04", shadow="server09-g2-normal-shadow-04"),
        qualification_context=context(), source_lock=None, source_lock_sha256=None,
        baseline_source_lock=BASELINE_LOCK, baseline_source_lock_sha256=BASELINE_SHA,
        base_permissions=None, human_authorization_record=None,
        new_executor=False, allow_model_downloads=False)


def verify_source_lock(root, relative, baseline_relative=BASELINE_LOCK):
    require(type(relative) is str and relative != baseline_relative, "new G2 source lock required")
    baseline_path = safe(root, baseline_relative)
    require(digest_file(baseline_path) == BASELINE_SHA, "actual old G1 source-lock bytes required")
    baseline = read_json(baseline_path)
    raw = read_json(safe(root, relative))
    require(type(raw) is dict and type(raw.get("files")) is list and
            2052 <= len(raw["files"]) <= 8192, "complete bounded G2 source file list")
    refs = {}
    for row in raw["files"]:
        checked_ref(root, row)
        require(row["path"] not in refs, "duplicate source-lock path")
        refs[row["path"]] = row
    old = {row["path"]: row for row in baseline["files"]}
    require(len(old) == 2052 and all(refs.get(path) == row for path, row in old.items()),
            "G2 must preserve every old frozen G1 source ref")
    required = {SCRIPT, WORKER, DEST + "/G2_NORMAL_MODEL_CPU_PLAN.json",
                GUARD, PREPARE, QUALIFY, SMOKE, RUNNER, CONTEXT, MODEL_PLAN, baseline_relative, SCALAR, FRAME,
                PERMISSIONS, AUTHOR + "/vllm/v1/executor/uniproc_executor.py",
                AUTHOR + "/vllm/v1/worker/worker_base.py",
                AUTHOR + "/vllm/v1/worker/gpu_worker.py",
                AUTHOR + "/vllm/entrypoints/llm.py",
                AUTHOR + "/vllm/v1/engine/llm_engine.py",
                AUTHOR + "/vllm/v1/engine/core_client.py", AUTHOR + "/vllm/v1/engine/core.py"}
    required.update(path for path, _, _ in SDK_SOURCE_REFS)
    required.add(NINJA)
    required.update(path for path, _, _ in OPTIONAL_SOURCE_REFS)
    require(required <= set(refs), "new runner/worker/plan and actual normal runtime source pins missing")
    for relative, size, sha in RUNTIME_CACHE_SOURCE_REFS:
        require(refs.get(relative) == dict(path=relative, bytes=size, sha256=sha),
                "actual supported process-cache environment source pins required")
    require(refs[GUARD]["bytes"] == 13013 and refs[GUARD]["sha256"] ==
            "3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a",
            "original current budget guard pin")
    require(refs[PREPARE]["bytes"] == 17280 and refs[PREPARE]["sha256"] ==
            "3059e689f1dcbedc450c867d84e6ba1d6393c0400d710bbbd5dde5495c871fe3",
            "original stdlib permission parser pin")
    require(refs[MODEL_PLAN]["bytes"] == 7310 and refs[MODEL_PLAN]["sha256"] ==
            "9b1c79986aa1fd62eb9064ecb8b4e2a03fc235a34ac2d69c33a4fcb531a91017",
            "original fixed official model-plan bytes")
    plan = read_json(safe(root, MODEL_PLAN))
    require(type(plan.get("files")) is list and len(plan["files"]) == 11,
            "complete fixed offline model file list")
    for file in plan["files"]:
        name = file["path"]
        require(type(name) is str and re.fullmatch(r"[A-Za-z0-9_.-]+", name) and
                name not in (".", "..") and file["hash_algorithm"] == "sha256",
                "fixed model filename/content SHA required")
        row = dict(path=MODEL + "/" + name, bytes=file["bytes"], sha256=file["hash"])
        require(refs.get(row["path"]) == row, "all real offline model bytes must be in G2 lock")
    # Pure original provider evidence verification, no framework/model load.
    base = load_ref(root, refs[SMOKE]); base.ROOT = root
    base.validate_provider(plan)
    # Existing compiler/layout assets are byte-verified on CPU even when no
    # new human GPU scope exists. This creates no overlay or runtime import.
    load_sdk_assets(root, refs)
    verify_ninja(root, refs)
    load_optional_probe(root, refs)
    return refs


def load_sdk_assets(root, refs):
    for path, nbytes, sha in SDK_SOURCE_REFS:
        require(refs.get(path) == dict(path=path, bytes=nbytes, sha256=sha),
                "exact locked CUDA13 SDK/selection source refs required")
        checked_ref(root, refs[path])
    module = load_ref(root, refs[SDK_MODULE])
    inventory = dict(refs[SDK_INVENTORY], path=str(safe(root, SDK_INVENTORY)))
    proof = dict(refs[SDK_PROOF], path=str(safe(root, SDK_PROOF)))
    return module, module.load_audited_assets(inventory, proof)


def prepare_process_sdk(root, refs, out):
    module, pin = load_sdk_assets(root, refs)
    evidence = module.prepare_overlay(approved_run_root=out.parent,
        runtime_cache=out / "runtime-cache", pin=pin, inherited_environment=dict(os.environ))
    selected = dict(evidence.environment)
    # Deliberate change to this already-guarded child process only. The helper
    # itself returns a copy; no global/system configuration is modified.
    os.environ.update(selected)
    return dict(overlay=evidence.overlay, links=evidence.links,
        selected_environment={key: selected[key] for key in SDK_PROCESS_KEYS},
        inventory_ref=refs[SDK_INVENTORY], compiler_proof_ref=refs[SDK_PROOF],
        helper_ref=refs[SDK_MODULE], CPU_assets_verified=True,
        GPU_model_or_JIT_runtime_qualified=False, production_qualified=False,
        stubs_on_runtime_library_path=False)


def verify_ninja(root, refs):
    path, nbytes, sha = NINJA_SOURCE_REF
    require(refs.get(path) == dict(path=path, bytes=nbytes, sha256=sha), "exact pinned existing Ninja ref")
    target = checked_ref(root, refs[path])
    require(target.is_file() and not target.is_symlink() and target.resolve(strict=True) == target and
            os.access(target, os.X_OK), "existing real executable Ninja only")
    with target.open("rb") as stream:
        require(stream.read(4) == b"\x7fELF", "existing pinned Ninja must be ELF")
    return target


def prepare_process_ninja(root, refs, out, sdk_evidence):
    """Link only the existing binary after actual guard gates and SDK setup."""
    require(os.name == "posix", "actual server Linux private tool layout")
    target = verify_ninja(root, refs)
    cache = out / "runtime-cache"
    require(out.is_absolute() and out.is_dir() and out.resolve(strict=True) == out and
            cache.is_dir() and cache.resolve(strict=True) == cache, "actual approved new details/cache")
    sdk_bin = Path(sdk_evidence["overlay"]) / "bin"
    require(Path(sdk_evidence["overlay"]) == cache / "cuda13-sdk", "exact current private SDK overlay")
    selected = sdk_evidence["selected_environment"]["PATH"]
    require(type(selected) is str and os.environ.get("PATH") == selected and
            selected.split(os.pathsep)[0] == str(sdk_bin), "SDK selected PATH drift")
    private_root = cache / "build-tools"
    private_bin = private_root / "bin"
    require(str(root / ".venv/bin") not in selected.split(os.pathsep), "no full virtualenv bin in inherited PATH")
    private_root.mkdir(mode=0o755, exist_ok=False)
    private_bin.mkdir(mode=0o755, exist_ok=False)
    link = private_bin / "ninja"
    os.symlink(target, link)
    verify_ninja(root, refs)
    require(link.is_symlink() and Path(os.readlink(link)) == target and link.resolve(strict=True) == target and
            [p.name for p in private_bin.iterdir()] == ["ninja"], "single exact existing Ninja link")
    tail = selected.split(os.pathsep)[1:]
    os.environ["PATH"] = os.pathsep.join([str(sdk_bin), str(private_bin), *tail])
    return dict(private_bin=str(private_bin), link=str(link), target=str(target), source_ref=refs[NINJA],
        PATH=os.environ["PATH"], CPU_tool_bytes_verified=True, GPU_model_or_JIT_runtime_qualified=False,
        production_qualified=False, whole_virtualenv_bin_PATH=False)



def load_optional_probe(root, refs):
    """Read exact original source and physical absence; no framework or hook."""
    for path, nbytes, sha in OPTIONAL_SOURCE_REFS:
        require(refs.get(path) == dict(path=path, bytes=nbytes, sha256=sha),
                "exact locked optional query helper/original source refs required")
        checked_ref(root, refs[path])
    module = load_ref(root, refs[OPTIONAL_MODULE])
    return module, module.verify_optional_absence(root, refs)


def child_command(args):
    return [".venv/bin/python", "-B", SCRIPT, "--project", str(args.project.resolve()),
            "--mode", args.mode, "--name", args.name, "--scope-record", args.scope_record,
            "--source-lock", args.source_lock, "--execute"]


def scope_source_gates(root, args):
    # Scope precedes every runtime/module/permission/model/source probe.
    require(args.scope_record is not None and safe(root, args.scope_record).is_file(),
            "NO_NEW_G2_HUMAN_SCOPE")
    require(args.scalar_relative == SCALAR and args.frame_relative == FRAME,
            "frozen scalar/frame source paths cannot be overridden")
    scope = read_json(safe(root, args.scope_record))
    require(set(scope) == set(scope_template()), "exact independent G2 scope fields")
    require(scope["status"] == "USER_AUTHORIZED_G2_NORMAL_MODEL_LIFECYCLE" and
            scope["allow_gpu_initialization"] is True and scope["allow_gpu_runs"] is True,
            "new G2 human scope required; CPU template/G1 cannot authorize")
    same(scope["qualification_context"], context(), "exact typed fixed qualification context")
    same(scope["allowed_modes"], ["off", "shadow"], "exact G2 off/shadow modes")
    require(scope["schema_version"] == 1 and type(scope["schema_version"]) is int and
            type(scope["purpose"]) is str and scope["purpose"] == PURPOSE and
            type(scope["gpu_uuid"]) is str and scope["gpu_uuid"] == GPU_UUID and
            scope["new_executor"] is False and scope["allow_model_downloads"] is False,
            "fixed normal-model-only scope/context")
    same(scope["permitted_run_names"], scope_template()["permitted_run_names"], "fixed exact run names")
    require(type(scope["maximum_jobs"]) is int and scope["maximum_jobs"] == 2 and
            type(scope["maximum_total_planned_reserve_seconds"]) is int and
            scope["maximum_total_planned_reserve_seconds"] == 640 and
            scope["permitted_run_names"][args.mode] == args.name, "two fixed bounded G2 jobs")
    human_path = checked_ref(root, scope["human_authorization_record"])
    human = read_json(human_path)
    require(human.get("authorization_origin") == "direct_human_reply" and
            human.get("purpose") == PURPOSE and human.get("gpu_uuid") == GPU_UUID and
            human.get("allow_gpu_runs") is True and human.get("allowed_modes") == ["off", "shadow"] and
            type(human.get("maximum_jobs")) is int and human.get("maximum_jobs") == 2 and
            type(human.get("maximum_total_planned_reserve_seconds")) is int and
            human.get("maximum_total_planned_reserve_seconds") == 640,
            "separate frozen direct-human record; a scope template is not approval")
    require(scope["source_lock"] == args.source_lock and
            scope["baseline_source_lock"] == BASELINE_LOCK and
            scope["baseline_source_lock_sha256"] == BASELINE_SHA and
            scope["source_lock_sha256"] == digest_file(safe(root, args.source_lock)),
            "new G2 full source-lock binding")
    refs = verify_source_lock(root, args.source_lock)
    permit_path = checked_ref(root, scope["base_permissions"])
    require(scope["base_permissions"] == refs[PERMISSIONS], "effective existing permission ref")
    prepare = load_ref(root, refs[PREPARE])
    permit = prepare.permission_fields(permit_path.read_text())
    require(permit["allow_gpu_runs"] is True and permit["approved_gpu_ids"] == [GPU_UUID] and
            permit["max_gpu_hours"] == 8 and
            Path(permit["approved_experiment_root"]).resolve() == root / "experiments/prefix_io_v1/runs",
            "existing single UUID/PRIMARY/cumulative eight-hour permit")
    ledger = read_json(safe(root, "experiments/prefix_io_v1/gpu-budget-ledger.json"))
    used = ledger.get("gpu_wall_seconds")
    require(type(used) in (int, float) and math.isfinite(used) and used >= 0 and
            used + SECONDS + 20 <= permit["max_gpu_hours"] * 3600, "cumulative budget exhausted")
    # Prior G1 events do not consume this scope, and cannot authorize this scope.
    events = ledger.get("events")
    require(type(events) is list and len(events) <= 10000, "actual bounded existing guard ledger")
    names = set(scope["permitted_run_names"].values())
    prior = [event for event in events if type(event) is dict and event.get("label") in names]
    require(len(prior) < 2 and not any(event.get("label") == args.name for event in prior),
            "G2 scope job used; no overwrite/retry without new scope")
    primary = root / "experiments/prefix_io_v1/runs"
    require(primary.is_dir() and not primary.is_symlink() and
            shutil.disk_usage(primary).free - RESERVE >= FLOOR, "PRIMARY storage floor/reserve")
    out = safe(root, "experiments/prefix_io_v1/runs/" + args.name + "/details")
    require(not out.exists(), "append-new result path only")
    return dict(scope=scope, refs=refs, permit=permit, ledger=ledger, out=out)


def execution_gates(root, args):
    gates = scope_source_gates(root, args)
    active = gates["ledger"].get("active_reservation")
    require(type(active) is dict and active.get("label") == args.name and
            active.get("gpu_uuid") == GPU_UUID and active.get("seconds_limit") == SECONDS and
            active.get("reserved_seconds") == SECONDS + 20 and
            active.get("command") == child_command(args) and
            active.get("session_id") == os.getsid(0) and
            active.get("permissions") == gates["scope"]["base_permissions"],
            "exact active existing-guard reservation/session/command required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID and
            os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "actual guarded UUID/offline environment required")
    return gates


def shadow_prerequisite(root, args, gates):
    if args.mode != "shadow": return
    off = gates["scope"]["permitted_run_names"]["off"]
    events = [event for event in gates["ledger"]["events"]
              if type(event) is dict and event.get("label") == off]
    require(len(events) == 1 and type(events[0].get("child_exit")) is int and
            events[0].get("child_exit") == 0 and events[0].get("exit") == 0 and
            events[0].get("session_drained") is True, "off must pass and drain before shadow")
    result = read_json(safe(root, "experiments/prefix_io_v1/runs/" + off + "/details/normal-model-lifecycle-result.json"))
    require(result.get("status") == "PASSED_NORMAL_MODEL_FULL_OUTPUT_ONLY" and
            result.get("purpose") == PURPOSE and result.get("original_engine_shutdown_returned") is True and
            result.get("source_lock_sha256") == gates["scope"]["source_lock_sha256"],
            "actual off result/source/shutdown prerequisite")


def new_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")


def configure_process_caches(out, base):
    """Supported process-local keys, applied before any framework imports."""
    require(out.is_dir() and not out.is_symlink(), "actual new run directory")
    os.environ.update(VLLM_USE_MODELSCOPE="0", VLLM_NO_USAGE_STATS="1", VLLM_DO_NOT_TRACK="1",
                      VLLM_ALLOW_INSECURE_SERIALIZATION="1")
    cache_root = out / "runtime-cache"
    require(not cache_root.is_symlink(), "runtime cache directory cannot be a symlink")
    directories = dict(base.RUNTIME_CACHE_SUBDIRS, **EXTRA_CACHE_DIRS)
    paths = {}
    for key, name in directories.items():
        path = cache_root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(out.resolve()),
                "process cache must remain within this run")
        path.mkdir(parents=True, exist_ok=True)
        paths[key] = str(path)
    temporary = cache_root / "tmp"
    require(not temporary.is_symlink(), "runtime temporary symlink rejected")
    temporary.mkdir(parents=True, exist_ok=True)
    paths["TMPDIR"] = paths["VLLM_RPC_BASE_PATH"] = str(temporary)
    os.environ.update(paths)
    return paths


def install_worker_observation(worker, payload):
    # Off does not read worker, runner, dependencies or factory at all.
    if payload["mode"] == "off":
        return dict(status="OFF_ORIGINAL_PATH", runtime_hook_status="not_installed")
    root = Path(payload["root"])
    module = load_ref(root, payload["refs"][WORKER])
    scalar = module.load_pinned(safe(root, payload["scalar"]), *module.FROZEN["scalar"])
    frame_path = safe(root, payload["frame"])
    binding = scalar.MethodBinding(tuple(scalar.SourceRef(str(safe(root, name)),
                payload["refs"][name]["bytes"], payload["refs"][name]["sha256"])
                for name in (CONTEXT, RUNNER)),
                tuple("GPUModelRunner." + name for name in scalar.METHODS), "native_candidate")
    # Backend import is guarded by the driver, never reached by pure preflight.
    import torch
    observer = module.connect_worker_observation(worker, scalar_source=safe(root, payload["scalar"]),
        frame_source=frame_path, binding=binding, run_id=payload["run_id"],
        native_source_sha256=payload["refs"]["third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"]["sha256"],
        event_factory=torch.cuda.Event, enabled=True)
    require(observer.enabled, "source/domain preflight refused worker observer: " + observer.reason)
    require(not hasattr(worker, "_prefix_g2_observer"), "fresh observer slot required")
    worker._prefix_g2_observer = observer
    return dict(status="INSTALLED_UNQUALIFIED_NATIVE_OBSERVER", runtime_hook_status="installed_unqualified",
                GPU_collector_verified=False, production_qualified=False)


def worker_runtime_identity(worker, payload):
    """Explicit source/ABI observation RPC, separate from the optional hook.

    Loaded file identity is a necessary source witness, never a claim of model
    numerics, kernel compatibility or production KV/physical release success.
    """
    root = Path(payload["root"]); refs = payload["refs"]
    module = load_ref(root, refs[WORKER])
    scalar = module.load_pinned(safe(root, payload["scalar"]), *module.FROZEN["scalar"])
    binding = scalar.MethodBinding(tuple(scalar.SourceRef(str(safe(root, name)),
        refs[name]["bytes"], refs[name]["sha256"]) for name in (CONTEXT, RUNNER)),
        tuple("GPUModelRunner." + name for name in scalar.METHODS), "native_candidate")
    runner = worker.model_runner
    scalar._check_domain(runner)
    originals = scalar._check_original_methods(runner, binding)
    method_files = []
    for original_ref in originals:
        function = original_ref().__func__
        for _ in range(8):
            path = Path(function.__code__.co_filename).resolve()
            relative = path.relative_to(root).as_posix()
            require(relative in refs, "actual runner decorator/raw file outside source lock")
            checked_ref(root, refs[relative])
            method_files.append(dict(qualname=function.__qualname__, source=relative,
                                     sha256=refs[relative]["sha256"]))
            if not hasattr(function, "__wrapped__"): break
            function = function.__wrapped__
        else: raise ValueError("bounded original decorator chain")
    loaded = []
    for name, imported in tuple(sys.modules.items()):
        if name != "vllm" and not name.startswith("vllm."): continue
        origin = getattr(imported, "__file__", None)
        if origin is None: continue
        path = Path(origin).resolve(); relative = path.relative_to(root).as_posix()
        require(relative in refs and len(loaded) < 2052, "actual loaded author/binary source not locked")
        checked_ref(root, refs[relative])
        loaded.append(dict(module=name, source=relative, sha256=refs[relative]["sha256"]))
    require(loaded, "actual author runtime module witness")
    return dict(status="ACTUAL_LOADED_SOURCE_IDENTITIES_ONLY", runner_methods=method_files,
                loaded_author_modules=loaded, binary_compatibility_qualified=False,
                production_qualified=False, performance_claim=False)


def export_worker_observation(worker, mode):
    if mode == "off": return dict(status="OFF_ORIGINAL_PATH", frames=[], diagnostics=[], io="not_applicable")
    observer = worker._prefix_g2_observer
    observer.resolve_ready()  # query only; no wait/synchronize or copied tensor
    result = dict(status=observer.status, valid=observer.enabled,
        reason=observer.reason, frames=[dataclasses.asdict(frame) for frame in observer.frames],
        diagnostics=[dict(**dataclasses.asdict(row), mapped_start_ns=None, mapped_end_ns=None,
                          gpu_elapsed_ns=None, cross_clock_mapping_verified=False,
                          GPU_collector_verified=False, production_qualified=False,
                          performance_claim=False) for row in observer.events.diagnostics],
        pending_event_pairs=len(observer.events.pending), open_event_pair=observer.events.active is not None,
        io="not_applicable", native_drain="not_applicable", GPU_collector_verified=False,
        production_qualified=False, performance_claim=False)
    observer.detach(); del worker._prefix_g2_observer
    return result


def detach_worker_observation(worker):
    observer = getattr(worker, "_prefix_g2_observer", None)
    if observer is not None:
        observer.detach(); del worker._prefix_g2_observer
    return dict(status="OBSERVER_DETACHED_NO_NATIVE_RELEASE")


def run_original_request(engine, sampling, frontend_id, *, deadline_seconds=60):
    """Call only original add_request/step. No model executor is copied here."""
    require(not engine.has_unfinished_requests(), "previous original request still unfinished")
    native_id = engine.add_request(frontend_id, {"prompt_token_ids": PROMPT.copy()}, sampling)
    require(type(native_id) is str and 0 < len(native_id) <= 128, "actual add_request returned native identity")
    started = time.monotonic()
    final = None
    steps = 0
    while engine.has_unfinished_requests():
        require(time.monotonic() - started <= deadline_seconds and steps < 4096,
                "bounded original request lifecycle deadline/steps")
        outputs = engine.step()
        steps += 1
        require(type(outputs) is list and len(outputs) <= 1, "single original request cohort")
        for output in outputs:
            require(output.request_id == frontend_id, "original frontend request identity")
            if output.finished:
                require(final is None, "duplicate original completed output")
                require(list(output.prompt_token_ids) == PROMPT and len(output.outputs) == 1,
                        "actual prompt/completion cohort")
                completion = output.outputs[0]
                tokens = list(completion.token_ids)
                require(len(tokens) == 128 and all(type(token) is int and token >= 0 for token in tokens),
                        "complete real 128-token output, no truncation")
                require(type(output.num_cached_tokens) is int, "actual prefix cached-token count")
                final = dict(request_id=frontend_id, native_request_id=native_id, prompt_token_ids=PROMPT.copy(),
                    output_token_ids=tokens, num_cached_tokens=output.num_cached_tokens,
                    finish_reason=completion.finish_reason)
    require(final is not None, "original engine completed without full output")
    return final


def verify_frame_outputs(observation, frontend):
    require(observation["valid"] is True and observation["open_event_pair"] is False,
            "closed valid original execute/sample observations")
    frames = observation["frames"]
    require(type(frames) is list and 1 <= len(frames) <= 4096, "complete original frame trace")
    ids = set(); actual = []
    for row in frames:
        require(row["gpu_elapsed_ns"] is None, "no qualified GPU cost field")
        for rid, tokens in row["outputs"]:
            ids.add(rid); actual.extend(tokens)
    require(ids == {frontend["native_request_id"]} and actual == frontend["output_token_ids"],
            "every original sampled ID reconstructs complete independent frontend outputs")
    return dict(status="FULL_FRONTEND_OUTPUT_RECONCILIATION_ONLY", frame_count=len(frames),
                native_request_id=next(iter(ids)), complete_output_count=len(actual),
                cost_qualified=False, GPU_collector_verified=False)


def execute_guarded(root, args, gates):
    """Concrete future GPU entry, reached only after execution_gates succeeds."""
    refs = gates["refs"]
    base = load_ref(root, refs[SMOKE]); base.ROOT = root
    # Existing pinned validator checks real current model bytes before import.
    model_dir, model = base.validate_local_model(safe(root, MODEL), safe(root, MODEL_PLAN))
    out = gates["out"]; out.mkdir(parents=True, exist_ok=False)
    old_cwd = Path.cwd()
    llm = None; finder = None; capability_probe = None
    result = dict(status="FAILED_NORMAL_MODEL_LIFECYCLE", purpose=PURPOSE, native_io="none",
        native_drain="not_applicable", SSD_qualified=False, effect_verified=False,
        production_qualified=False, GPU_collector_verified=False, new_executor=False,
        performance_claim=False, phases=[], gpu_uuid=GPU_UUID, sampling=dict(SAMPLING),
        source_lock=args.source_lock, source_lock_sha256=gates["scope"]["source_lock_sha256"],
        scope_record=args.scope_record, scope_sha256=digest_file(safe(root, args.scope_record)),
        framework_import_attempted=False, gpu_initialization_attempted=False, actual_guarded_job_count=1)
    try:
        result["runtime_cache_environment"] = configure_process_caches(out, base)
        result["sdk_toolchain_environment"] = prepare_process_sdk(root, refs, out)
        result["build_tool_environment"] = prepare_process_ninja(root, refs, out, result["sdk_toolchain_environment"])
        optional_module, absence = load_optional_probe(root, refs)
        result["optional_capability_probe"] = dict(source_ref=refs[OPTIONAL_MODULE],
            CPU_absence_gate=absence, installed=False, original_finder_modified=False)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # The original source finder and individually locked binary fallback are
        # reused in this original in-process Uni execution configuration.
        source_loader = load_ref(root, refs[QUALIFY])
        finder = source_loader.BoundAuthorFinder(root, refs); sys.meta_path.insert(0, finder)
        require(not any(name == "vllm" or name.startswith("vllm.") for name in sys.modules),
                "source finder must precede actual author runtime import")
        os.chdir(out)
        # Import itself may execute CUDA capability probes and then fail.
        # Attempt flags are separate from actual model/output qualification.
        result["framework_import_attempted"] = True
        result["gpu_initialization_attempted"] = True
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        import vllm.utils.import_utils as author_import_utils
        capability_probe = optional_module.install_capability_probe(author_import_utils, finder, root, refs)
        result["optional_capability_probe"]["installed"] = True
        result["optional_capability_probe"]["installation"] = capability_probe.evidence()
        require(Path(vllm.__file__).resolve().is_relative_to(root / AUTHOR), "actual author Python import")
        source_loader.require(source_loader.normalize_uuid(torch.cuda.get_device_properties(0).uuid) ==
                              source_loader.normalize_uuid(GPU_UUID), "actual GPU UUID differs")
        engine_config = dict(base.ENGINE, async_scheduling=False)
        result["engine_config"] = engine_config
        require(len(PROMPT) + SAMPLING["max_tokens"] <= engine_config["max_model_len"],
                "fixed full output fits original model context")
        result["model_initialization_attempted"] = True
        llm = LLM(model=str(model_dir), **engine_config)
        base.assert_no_external_cache()
        require(llm.llm_engine.vllm_config.scheduler_config.async_scheduling is False,
                "resolved original scheduler must remain synchronous")
        result["model"] = model
        result["effective_config"] = base.validate_effective_config(llm.llm_engine.vllm_config, torch.bfloat16)
        payload = dict(root=str(root), mode=args.mode, refs=refs,
                       scalar=args.scalar_relative, frame=args.frame_relative)
        identities = llm.collective_rpc(worker_runtime_identity, args=(payload,), timeout=30)
        require(type(identities) is list and len(identities) == 1, "one actual worker source witness")
        result["worker_runtime_source_identity"] = identities[0]
        for phase in ("cold", "repeat"):
            rid = args.name + "-" + phase
            payload["run_id"] = rid
            install = llm.collective_rpc(install_worker_observation, args=(payload,), timeout=10)
            require(type(install) is list and len(install) == 1, "single actual Uni worker")
            frontend = run_original_request(llm.llm_engine, SamplingParams(**SAMPLING), rid)
            item = dict(phase=phase, frontend=frontend, installation=install[0])
            result["phases"].append(item)
            # Persist completed original IDs before observation/export can fail
            # or the original budget guard can time out a later phase.
            new_json(out / (phase + "-frontend.json"), frontend)
            exported = llm.collective_rpc(export_worker_observation, args=(args.mode,), timeout=10)
            require(type(exported) is list and len(exported) == 1, "single actual worker observation")
            item["observation"] = exported[0]
            if args.mode == "shadow": item["reconciliation"] = verify_frame_outputs(exported[0], frontend)
        cold, repeat = (item["frontend"] for item in result["phases"])
        require(cold["num_cached_tokens"] == 0 and repeat["num_cached_tokens"] == 112 and
                cold["output_token_ids"] == repeat["output_token_ids"], "original exact Prefix/output equality")
        base.assert_no_external_cache()
        result["status"] = "PASSED_NORMAL_MODEL_FULL_OUTPUT_ONLY"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:1000]
        import traceback
        result["error_traceback"] = traceback.format_exc(limit=24)[-12000:]
    finally:
        if llm is not None:
            try:
                if args.mode == "shadow":
                    try:
                        llm.collective_rpc(detach_worker_observation, timeout=10)
                    except Exception as exc:
                        result["observation_cleanup_error_type"] = type(exc).__name__
            finally:
                # Cleanup observation failure/cancellation cannot skip original
                # model shutdown; this never closes a native I/O owner itself.
                try:
                    llm.llm_engine.engine_core.shutdown(timeout=15.0)
                    result["original_engine_shutdown_returned"] = True
                except Exception as exc:
                    result["status"] = "FAILED_ORIGINAL_ENGINE_SHUTDOWN"
                    result["shutdown_error_type"] = type(exc).__name__
        if capability_probe is not None:
            try:
                restored = capability_probe.detach()
                result["optional_capability_probe"]["restored"] = restored
                if restored is not True:
                    result["status"] = "FAILED_OPTIONAL_PROBE_RESTORE_OR_OVERRIDE"
            except Exception as exc:
                result["status"] = "FAILED_OPTIONAL_PROBE_CLEANUP"
                result["optional_probe_cleanup_error_type"] = type(exc).__name__
        if finder is not None and finder in sys.meta_path: sys.meta_path.remove(finder)
        os.chdir(old_cwd)
        try:
            verify_source_lock(root, args.source_lock)
            result["source_lock_unchanged_after_original_shutdown"] = True
        except Exception as exc:
            result["status"] = "FAILED_POSTRUN_SOURCE_LOCK"
            result["postrun_source_error_type"] = type(exc).__name__
        new_json(out / "normal-model-lifecycle-result.json", result)
    return 0 if result["status"] == "PASSED_NORMAL_MODEL_FULL_OUTPUT_ONLY" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("off", "shadow"), default="off")
    parser.add_argument("--name", default="server09-g2-normal-off-04")
    parser.add_argument("--source-lock")
    parser.add_argument("--scope-record")
    parser.add_argument("--scalar-relative", default=SCALAR)
    parser.add_argument("--frame-relative", default=FRAME)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--scope-template", action="store_true")
    actions.add_argument("--launch", action="store_true")
    actions.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root = args.project.resolve()
    if args.scope_template:
        print(json.dumps(scope_template(), indent=2)); return 0
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.name), "safe append-new run name")
    try:
        if args.preflight and args.scope_record is None:
            refs = verify_source_lock(root, args.source_lock)
            print(json.dumps(dict(status="SOURCE_LOCK_VERIFIED_BUT_NO_NEW_G2_HUMAN_SCOPE",
                source_count=len(refs), gpu_initialized=False, actual_gpu_runs=0,
                runtime_hook_status="not_installed", production_qualified=False)))
            return 0
        gates = execution_gates(root, args) if args.execute else scope_source_gates(root, args)
        shadow_prerequisite(root, args, gates)
        if args.preflight or not (args.launch or args.execute):
            print(json.dumps(dict(status="PURE_CPU_NEW_SCOPE_SOURCE_VALIDATED_NO_LAUNCH",
                                  actual_gpu_runs=0, gpu_initialized=False))); return 0
        if args.launch:
            require(gates["ledger"].get("active_reservation") is None, "unresolved original guard reservation")
            import subprocess
            command = [".venv/bin/python", "-B", GUARD, "--permissions-path", PERMISSIONS,
                       "--label", args.name, "--seconds", str(SECONDS), "--", *child_command(args)]
            return subprocess.run(command, cwd=root, env=dict(os.environ,
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")).returncode
        return execute_guarded(root, args, gates)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps(dict(status="BLOCKED_PURE_CPU_G2_GATE", reason=str(exc),
            actual_gpu_runs=0, gpu_initialized=False, runtime_hook_status="not_installed",
            production_qualified=False, performance_claim=False)))
        return 78


if __name__ == "__main__": raise SystemExit(main())
