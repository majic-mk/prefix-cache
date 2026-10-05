"""CPU-only P316 patch composition qualification in a new isolated scratch Git repo.

No real GPU/IO qualification is performed. This runner must be scheduled by root
after the live GPU work is idle; it never edits the current project source.
"""
from __future__ import annotations
import argparse
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
OUT_REL = "artifacts/prefix_io_v1/server08-p3-16"
P313_NATIVE = "third_party/work/py-kvcache-p3-shadow-cpu"
NATIVE = "third_party/work/py-kvcache-p3-15-cpu"
P316_NATIVE = "third_party/work/py-kvcache-p3-16-cpu"
BOUNDARY = "artifacts/prefix_io_v1/server08-p3-15/patch-boundary-plan.json"
BOUNDARY_SHA = "edd031173dc6ea9e1736d23ed2a46d38a4dbb9096515c8656142d277755db799"
SEED = "artifacts/prefix_io_v1/server08-p3-15/native-worktree-seed.json"
DRIVER_BEFORE = "artifacts/prefix_io_v1/server08-p3-15/driver-before.py"
DRIVER = "experiments/prefix_io_v1/scripts/run_concurrent_pilot.py"
PATCHES = (
    ("common315", "artifacts/prefix_io_v1/server08-p3-15/p315-common-native.patch",
     "68046b5c12af4ea8f09d44928d0d22470ec5b515720ed2cd53b4620a1170569e", None),
    ("increment316", OUT_REL + "/p316-common-native-increment.patch",
     "46a50e0ceecd93a50ad825e1213142a852b43e7f4c0e50dd382be6f071ea236e", NATIVE),
    ("research315", "artifacts/prefix_io_v1/server08-p3-15/p315-research-policy.patch",
     "d0d2e8aabcdb4a67995f55ec4f5a705edab56a664399dab8f5a6fc791d492fcb", None),
)
NEW_POLICY = (
    "src/prefix_io_control/simple_stage_policy.py",
    "src/prefix_io_control/simple_stage_options.py",
    "experiments/prefix_io_v1/scripts/simple_stage_worker_probe.py",
)
CONTROL_FILES = ("off-control.json", "fixed-control.json", "pressure-control.json")
MAX_FILE = 2 * 1024**2
MAX_INPUT_BYTES = 16 * 1024**2
RESERVE_BYTES = 64 * 1024**2
MIN_FREE_BYTES = 8 * 1024**3
SHA = re.compile(r"[0-9a-f]{64}\Z")

def require(ok, message):
    if not ok:
        raise ValueError(message)

def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()

def relative_name(value):
    require(type(value) is str and value != "" and "\\" not in value and "\x00" not in value and "\n" not in value, "relative source name required")
    p = PurePosixPath(value)
    require(not p.is_absolute() and str(p) == value and
            not any(x in ("", ".", "..", ".git", "__pycache__") for x in p.parts),
            "unsafe source path")
    return value

def source_path(root, relative):
    relative_name(relative)
    raw = Path(root) / relative
    resolved = raw.resolve()
    require(resolved.is_relative_to(Path(root).resolve()) and resolved == raw.absolute(),
            "source symlink/path outside exact project")
    require(raw.is_file() and not raw.is_symlink(), "regular source file required: " + relative)
    require(raw.stat().st_size <= MAX_FILE, "source file exceeds bounded text limit")
    require(raw.suffix in (".py", ".json", ".patch"), "text source type required")
    return raw

def bounded_bytes(root, relative):
    data = source_path(root, relative).read_bytes()
    data.decode("utf-8")
    return data

def read_json(root, relative, expected=None):
    data = bounded_bytes(root, relative)
    if expected is not None:
        require(hash_bytes(data) == expected, "frozen JSON changed: " + relative)
    def unique(items):
        answer = {}
        for k, v in items:
            require(k not in answer, "duplicate JSON key")
            answer[k] = v
        return answer
    return json.loads(data, object_pairs_hook=unique)

def expected_native(lock, prefix):
    prefix = relative_name(prefix) + "/"
    selected = {k[len(prefix):]:v for k,v in lock.items() if k.startswith(prefix)}
    require(len(selected) == 31 and all(k.endswith(".py") for k in selected),
            "exact 31 native Python sources required")
    require(all(type(v) is str and SHA.fullmatch(v) for v in selected.values()),
            "invalid native source lock")
    for k in selected:
        relative_name(k)
    return selected

def seed_manifest(seed, locked):
    require(type(seed.get("files")) is dict and len(seed["files"]) == 31,
            "P315 seed receipt must contain 31 sources")
    manifest = {}
    for name, row in seed["files"].items():
        relative_name(name)
        require(type(row) is dict and type(row.get("sha256")) is str and
                SHA.fullmatch(row["sha256"]) and type(row.get("bytes")) is int and
                0 <= row["bytes"] <= MAX_FILE, "invalid seed receipt row")
        manifest[name] = row["sha256"]
    require(manifest == locked, "P315 seed receipt is not the exact locked P313 source")
    require(seed.get("author_head") == "3abba7a502d553f6e7e2e58b92086487e3395d7e",
            "author revision differs")
    return manifest

def patch_targets(text):
    pairs = []
    old = None
    for line in text.splitlines():
        if line.startswith("--- "):
            require(old is None, "unpaired patch old header")
            old = line[4:].split("\t", 1)[0]
        elif line.startswith("+++ "):
            require(old is not None, "unpaired patch new header")
            new = line[4:].split("\t", 1)[0]
            def stripped(v, prefix):
                if v == "/dev/null":
                    return None
                require(v.startswith(prefix + "/"), "unexpected patch header prefix")
                return relative_name(v[2:])
            a, b = stripped(old, "a"), stripped(new, "b")
            require(a is not None or b is not None, "empty patch target")
            require(a is None or b is None or a == b, "rename is not in approved patch")
            pairs.append(b if b is not None else a)
            old = None
    require(old is None and len(pairs) == len(set(pairs)) and bool(pairs),
            "missing/duplicate patch headers")
    return set(pairs)

def approved_targets(label):
    if label == "common315":
        return {NATIVE + "/py_kvcache/reactor.py", NATIVE + "/py_kvcache/vllm.py"}
    if label == "increment316":
        return {"py_kvcache/reactor.py", "py_kvcache/staging_cache.py"}
    if label == "research315":
        return {NATIVE + "/py_kvcache/vllm.py", DRIVER, *NEW_POLICY}
    raise ValueError("unknown patch phase")

def patch_command(scratch, patch, *, directory=None, check=False, reverse=False):
    argv = ["git", "-c", "core.autocrlf=false", "-c", "core.filemode=false",
            "-C", str(scratch), "apply", "--whitespace=nowarn"]
    if directory is not None:
        argv += ["--directory=" + relative_name(directory)]
    if reverse:
        argv += ["--reverse"]
    if check:
        argv += ["--check"]
    argv += [str(Path(patch).resolve())]
    return argv

def output_path(root, output):
    root = Path(root).resolve()
    raw = Path(output).absolute()
    path = raw.resolve()
    allowed = root / OUT_REL
    require(path == raw and path.is_relative_to(allowed) and path != allowed,
            "output must be a new exact subdirectory of P316 artifacts")
    require(not os.path.lexists(path), "output already exists; do not reuse or delete it")
    require(path.parent.is_dir(), "output parent must already exist")
    return path

def idle_receipt(root, path, execution_sha):
    raw = Path(path).resolve()
    require(raw.is_relative_to((Path(root)/OUT_REL).resolve()), "idle receipt outside P316 artifacts")
    rel = str(raw.relative_to(Path(root).resolve()))
    value = read_json(root, rel)
    require(set(value) == {"scope", "created_utc", "active_gpu_operation", "execution_lock_sha256"},
            "strict root-owned GPU-idle receipt required")
    require(value["scope"] == "P316_CPU_PATCH_ROUNDTRIP_GPU_IDLE" and
            value["active_gpu_operation"] is None and value["execution_lock_sha256"] == execution_sha,
            "GPU idle/execution lock evidence differs")
    stamp = dt.datetime.fromisoformat(value["created_utc"])
    require(stamp.tzinfo is not None, "UTC idle receipt time required")
    age = (dt.datetime.now(dt.timezone.utc) - stamp).total_seconds()
    require(0 <= age <= 60, "fresh root GPU-idle receipt required")
    return value

def compare_native(scratch, expected):
    native = Path(scratch)/NATIVE
    names = {str(p.relative_to(native)) for p in native.rglob("*.py")}
    require(names == set(expected), "native source set changed")
    actual = {k:hash_bytes(bounded_bytes(scratch, NATIVE + "/" + k)) for k in sorted(expected)}
    require(actual == expected, "native source byte reconstruction mismatch")
    return {"files":len(actual), "sha256_by_relative_path":actual}

def dump_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")

def disk_check(output):
    free = shutil.disk_usage(Path(output).parent).free
    require(free - RESERVE_BYTES >= MIN_FREE_BYTES, "PRIMARY free floor + scratch reserve not met")
    return {"free_bytes":free, "minimum_free_bytes":MIN_FREE_BYTES,
            "bounded_scratch_reserve_bytes":RESERVE_BYTES}

def ctor_defaults(path, class_name):
    tree = ast.parse(Path(path).read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults = {}
    for a, d in zip(fn.args.args[-len(fn.args.defaults):], fn.args.defaults):
        defaults[a.arg] = ast.literal_eval(d)
    for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        if d is not None:
            defaults[a.arg] = ast.literal_eval(d)
    require(defaults.get("dispatch_controller", "absent") is None and
            defaults.get("max_accepted_parents", "absent") is None,
            "native constructor defaults no longer off")
    return {"dispatch_controller":None, "max_accepted_parents":None}

def cpu_child(scratch, mode, output):
    """Real source import plus CPU component checks, never a GPU resource test."""
    import importlib
    import importlib.abc
    import inspect
    import threading
    import types
    scratch = Path(scratch).resolve()
    require(mode in ("common", "research"), "bounded CPU composition required")
    sys.dont_write_bytecode = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path[:0] = [str(scratch/NATIVE), str(scratch/"src")]
    blocked = []
    class NoResearch(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in ("prefix_io_control.simple_stage_policy",
                            "prefix_io_control.simple_stage_options"):
                blocked.append(fullname)
                raise ImportError("common-only composition forbids research policy")
            return None
    if mode == "common":
        require(all(not (scratch/k).exists() for k in NEW_POLICY), "research source present in common-only tree")
        sys.meta_path.insert(0, NoResearch())
    import torch
    require(not torch.cuda.is_initialized(), "CUDA was initialized before CPU qualification")
    attempts = []
    def deny_cuda(*args, **kwargs):
        attempts.append("CUDA initialization attempt")
        raise RuntimeError("CPU qualification explicitly forbids CUDA initialization")
    torch.cuda.init = deny_cuda
    torch.cuda._lazy_init = deny_cuda
    reactor = importlib.import_module("py_kvcache.reactor")
    cache_module = importlib.import_module("py_kvcache.staging_cache")
    staging = importlib.import_module("py_kvcache.staging")
    require(Path(reactor.__file__).resolve() == scratch/NATIVE/"py_kvcache/reactor.py",
            "native import escaped scratch")
    require(Path(cache_module.__file__).resolve() == scratch/NATIVE/"py_kvcache/staging_cache.py",
            "cache import escaped scratch")
    checks = []
    defaults = {}
    for class_name in ("IoReactor", "TransferCoordinator"):
        defaults[class_name] = ctor_defaults(reactor.__file__, class_name)
        checks.append("AST default None: " + class_name)
    for cap, run_id in ((0, "cpu"), (65, "cpu"), (True, "cpu"), (64, None)):
        try:
            reactor.IoReactor(config=None, file_mapper=None, layout=None,
                              max_accepted_parents=cap, progress_run_id=run_id)
        except ValueError:
            checks.append("real constructor rejects invalid/common identity: " + str((cap, run_id)))
        else:
            raise AssertionError("invalid constructor passed its early common guard")
    owner = reactor.IoReactor.__new__(reactor.IoReactor)
    owner._prefix_dispatch_controller = None
    old_time_module = reactor.time
    off_clock_calls = []
    def forbidden_clock():
        off_clock_calls.append(True)
        raise AssertionError("off stage hook read a policy clock")
    try:
        reactor.time = types.SimpleNamespace(monotonic_ns=forbidden_clock)
        require(owner._prefix_stage_decide("ssd_read", 917504, ("cpu",1)) is None,
                "off hook did not return original no-controller path")
        owner._prefix_stage_settle(None, "accepted")
    finally:
        reactor.time = old_time_module
    checks.append("actual off hook returns before clock/state/policy")

    # Invoke common admission bookkeeping on a CPU object with real Python locks.
    # This does not instantiate a GPU reactor or prove native submission/drain.
    owner._submit_lock = threading.Lock()
    owner._worker = threading.current_thread()
    owner._init_parent_admission(64)
    admission = owner.parent_admission_snapshot()
    require(admission["accepted_parents"] == 0 and admission["max_accepted_parents"] == 64,
            "direct common parent-cap bookkeeping differs")
    owner._accepted_parent_count = 1
    job = types.SimpleNamespace(accepted_parent_sequence=1, accepted_parent_retired=False)
    owner._retire_accepted_parent(job)
    require(job.accepted_parent_retired and owner.parent_admission_snapshot()["accepted_parents"] == 0,
            "actual common retirement bookkeeping differs")
    checks.append("actual direct common parent-cap/retirement CPU bookkeeping")
    from prefix_io_control.stage_accounting import StageAccounting
    accounting = StageAccounting();accounting.bind()
    accounting.accepted("ssd_read", "cpu-unit", 917504)
    require(accounting.snapshot()["outstanding_records"] == 1, "CPU acceptance record absent")
    accounting.completed("ssd_read", "cpu-unit", 917504)
    require(accounting.snapshot()["outstanding_records"] == 0, "CPU acceptance record not retired")
    checks.append("actual StageAccounting CPU acceptance/completion")

    cache = cache_module.StagingDataCache(policy="lru", capacity=2)
    cache.put(b"busy", 0);cache.put(b"clean", 1)
    resident = cache._slots[b"busy"];cache.pin(resident);cache.pin(resident)
    require(cache.pinned_slot_count == 1, "copy refs double-count distinct busy slots")
    require(cache.drain_unpinned() == [1] and cache._slots[b"busy"] is resident and
            resident.copies_inflight == 2, "CPU clean drain released a pinned cache owner")
    cache.unpin(resident)
    require(cache.pinned_slot_count == 1 and cache.drain_unpinned() == [],
            "nested ref retirement released a still-busy slot")
    cache.unpin(resident)
    require(cache.drain_unpinned() == [0] and len(cache) == 0 and cache.observation_valid,
            "last CPU ref did not permit clean drain")
    checks.append("actual cache distinct-pin/STOP clean-drain CPU lifecycle")

    # Exercise the native STOP helper with CPU staging indices only.
    stop = reactor.IoReactor.__new__(reactor.IoReactor)
    stop._submit_lock = threading.Lock();stop._worker = threading.current_thread()
    stop._init_parent_admission(64)
    for key in ("_active","_inflight","_pending_copies","_copy_ready","_preload_pending",
                "_preload_slots","_shared_cached","_ready_fds_load","_ready_fds_preload"):
        setattr(stop, key, [])
    stop._stop = True
    stop._prefix_stage_accounting = None;stop._prefix_dispatch_controller = None
    stop._prefix_dispatch_shadow = None
    stop.staging_pool = staging.StagingPool(slot_count=2)
    a = stop.staging_pool.try_reserve();b = stop.staging_pool.try_reserve()
    stop._staging_cache = cache_module.StagingDataCache(policy="lru", capacity=2)
    stop._staging_cache.put(b"busy",a.index);stop._staging_cache.put(b"clean",b.index)
    busy = stop._staging_cache._slots[b"busy"];stop._staging_cache.pin(busy)
    try:
        stop._cleanup_retained_cache_after_stop()
    except reactor.NativeDrainUnknown:
        require(stop._native_drain_unknown and not stop._parent_count_valid and
                stop._staging_cache._slots[b"busy"] is busy and busy.copies_inflight == 1 and
                stop.staging_pool.free_count == 1, "unknown CPU STOP released unresolved owner")
    else:
        raise AssertionError("orphan pinned CPU cache was falsely drained")
    checks.append("actual STOP helper unknown retains CPU owner; no physical GPU witness")

    research_off_controller_none = None
    if mode == "research":
        from prefix_io_control.simple_stage_options import model_file,parse_simple_options,build_simple_kwargs
        for name in CONTROL_FILES:
            extra = model_file(scratch/"controls"/name,"patch-roundtrip-cpu")
            kwargs = build_simple_kwargs(parse_simple_options(extra))
            require(kwargs["max_accepted_parents"] == 64 and
                    kwargs["progress_run_id"] == "patch-roundtrip-cpu",
                    "shared common bound/identity differs")
            policy = kwargs.get("dispatch_controller")
            if name == "off-control.json":
                require(policy is None, "research off instantiated a controller")
                research_off_controller_none = policy is None
            else:
                require(policy is not None and policy.config.mode ==
                        ("fixed" if name.startswith("fixed") else "pressure"),
                        "frozen policy composition differs")
            checks.append("CPU parse/build original policy: " + name)
    require(not attempts and not torch.cuda.is_initialized(), "CPU guard detected CUDA initialization")
    if mode == "common":
        require(not blocked, "default/common path attempted a research import")
    for name, module in list(sys.modules.items()):
        if name == "py_kvcache" or name.startswith("py_kvcache.") or name == "prefix_io_control" or name.startswith("prefix_io_control."):
            where = getattr(module, "__file__", None)
            if where is not None:
                base = scratch/NATIVE if name.startswith("py_kvcache") else scratch/"src"
                require(Path(where).resolve().is_relative_to(base), "CPU support import escaped scratch: " + name)
    record = dict(status="PASS_CPU_COMPOSITION_ONLY",mode=mode,checks=checks,
                  default_off=dict(constructor_defaults=defaults,actual_hook_controller_none=True,
                    policy_clock_calls=len(off_clock_calls),
                    research_off_factory_controller_none=research_off_controller_none),
                  cuda_initialized=False,cuda_initialization_attempts=attempts,
                  research_import_attempts=blocked,
                  native_constructor_resources_allocated=False,real_gpu_qualification=False,
                  native_backend_or_model_executed=False,physical_completion_proved=False,
                  scope="Scratch real source import and CPU component transitions; CPU staging indices/refcounts only")
    dump_new(output, record)
    return record

def final_machine_fields(report, final_native):
    """Derive machine closeout fields only from completed commands/checks."""
    commands=report["commands"]
    def exit_for(suffix):
        selected=[r for r in commands if r["label"].endswith(suffix)]
        if len(selected)==6 and all(r.get("returncode")==0 and "error" not in r for r in selected):
            return 0
        failed=[r.get("returncode") for r in selected if r.get("returncode") not in (None,0)]
        return failed[0] if failed else None
    check_exit=exit_for("-check");apply_exit=exit_for("-apply")
    stages={r["name"]:r for r in report["stages"]}
    full=stages.get("full316-research315",{}).get("native",{}).get("sha256_by_relative_path",{})
    files=[dict(path=P316_NATIVE+"/"+k,expected_sha256=v,actual_sha256=full.get(k))
           for k,v in sorted(final_native.items())]
    wanted={"seed-p313","common315","common316","full316-research315",
            "reverse-research-common316","reverse-increment-common315","reverse-common-seed-p313"}
    exact=(set(stages)==wanted and len(full)==31 and full==final_native)
    common=report.get("common_cpu_smoke",{})
    research=report.get("research_cpu_smoke",{})
    def cpu_ok(row,mode):
        return (row.get("status")=="PASS_CPU_COMPOSITION_ONLY" and row.get("mode")==mode and
                row.get("cuda_initialized") is False and row.get("cuda_initialization_attempts")==[] and
                row.get("real_gpu_qualification") is False and
                row.get("native_constructor_resources_allocated") is False and
                row.get("native_backend_or_model_executed") is False and
                row.get("physical_completion_proved") is False)
    common_ok=cpu_ok(common,"common") and common.get("research_import_attempts")==[]
    research_ok=cpu_ok(research,"research")
    def off_ok(row, research_mode):
        proof=row.get("default_off",{})
        return (proof.get("actual_hook_controller_none") is True and type(proof.get("policy_clock_calls")) is int and proof["policy_clock_calls"]==0 and
                proof.get("constructor_defaults")=={
                    "IoReactor":{"dispatch_controller":None,"max_accepted_parents":None},
                    "TransferCoordinator":{"dispatch_controller":None,"max_accepted_parents":None}} and
                (not research_mode or proof.get("research_off_factory_controller_none") is True))
    default_ok=common_ok and research_ok and off_ok(common,False) and off_ok(research,True)
    preservation=report.get("source_preservation",{})
    preserved=preservation.get("changed")==[] and preservation.get("unverified")==[]
    report["check_exit"]=check_exit;report["apply_exit"]=apply_exit
    report["exact_content"]=bool(exact and preserved)
    report["files"]=files
    report["individual_checks"]=dict(
        common_only=dict(passed=common_ok,receipt="common-cpu-smoke.json",scope="source import and CPU components only"),
        full=dict(passed=research_ok and full==final_native,receipt="research-cpu-smoke.json",
                  scope="full native byte reconstruction and CPU policy parse/build"),
        default_off=dict(passed=default_ok,common_proof=common.get("default_off"),
                         full_proof=research.get("default_off")),
        reverse=dict(passed=exact and check_exit==apply_exit==0,
                     final_stage="reverse-common-seed-p313"))
    passed=(report["status"]=="PASS_CPU_PATCH_ROUNDTRIP_COMPOSITIONS" and
            check_exit==apply_exit==0 and report["exact_content"] and
            common_ok and research_ok and default_ok and "storage_error" not in report)
    report["status"]="PASS_PATCH_ROUNDTRIP" if passed else "FAILED_CPU_PATCH_ROUNDTRIP"
    return report

def run(root, args):
    root = Path(root).resolve()
    require(type(args.execution_lock_sha256) is str and SHA.fullmatch(args.execution_lock_sha256),
            "execution lock digest required")
    lock_path = Path(args.execution_lock).resolve()
    require(lock_path.is_relative_to(root/OUT_REL), "execution lock outside P316 artifacts")
    lock_rel = str(lock_path.relative_to(root))
    lock = read_json(root, lock_rel, args.execution_lock_sha256)
    require(type(lock) is dict, "execution lock mapping required")
    require(args.timeout_seconds == 600, "fixed CPU wall envelope is 600 seconds")
    receipt = idle_receipt(root, args.gpu_idle_receipt, args.execution_lock_sha256)
    out = output_path(root,args.output)
    initial_disk = disk_check(out)
    boundary = read_json(root, BOUNDARY, BOUNDARY_SHA)
    source_boundary = boundary["source_boundary"]
    require(source_boundary["base_native_tree"] == P313_NATIVE and
            source_boundary["final_native_tree"] == NATIVE and
            source_boundary["native_python_file_count"] == 31, "P315 source boundary changed")
    seed = read_json(root, SEED)
    seeded = seed_manifest(seed,expected_native(lock,P313_NATIVE))
    final_native = expected_native(lock,P316_NATIVE)
    require(set(seeded) == set(final_native), "native source set differs across stages")
    common315 = dict(seeded)
    for name,row in source_boundary["virtual_common_intermediate_files"].items():
        require(name.startswith(NATIVE + "/"),"intermediate native path changed")
        common315[name[len(NATIVE)+1:]] = row["common_intermediate_sha256"]
    common316 = dict(final_native)
    common316["py_kvcache/vllm.py"] = common315["py_kvcache/vllm.py"]

    inputs = {lock_rel:args.execution_lock_sha256,BOUNDARY:BOUNDARY_SHA}
    def remember(relative, expected=None):
        data = bounded_bytes(root,relative)
        actual = hash_bytes(data)
        require(expected is None or actual == expected,"input source changed: " + relative)
        require(relative not in inputs or inputs[relative] == actual,"input lock conflicts")
        inputs[relative] = actual
        return data
    remember(SEED)
    text_copies = {}
    for name,sha in seeded.items():
        original = P313_NATIVE + "/" + name
        data = remember(original,sha)
        require(len(data) == seed["files"][name]["bytes"], "P313 seed file size differs")
        text_copies[NATIVE + "/" + name] = data
    for name,sha in final_native.items():
        remember(P316_NATIVE + "/" + name,sha)
    controls = {k:v for k,v in lock.items() if k.startswith("src/prefix_io_control/") and k.endswith(".py")}
    require(len(controls) == 21,"frozen control source count changed")
    for name,sha in controls.items():
        data = remember(name,sha)
        if name not in NEW_POLICY:
            text_copies[name] = data
    driver_spec = boundary["patch_sequence"][1]["driver"]
    require(driver_spec["path"] == DRIVER and driver_spec["base_snapshot"] == DRIVER_BEFORE,
            "driver seed identity changed")
    text_copies[DRIVER] = remember(DRIVER_BEFORE,driver_spec["base_sha256"])
    for name in CONTROL_FILES:
        key = OUT_REL + "/" + name
        require(key in lock,"frozen control configuration absent from execution lock")
        text_copies["controls/"+name] = remember(key,lock[key])
    new_policy_sha = {row["path"]:row["sha256"] for row in boundary["patch_sequence"][1]["new_sources"]}
    require(set(new_policy_sha) == set(NEW_POLICY), "new policy sources changed")
    for label,relative,sha,directory in PATCHES:
        data = remember(relative,sha)
        require(patch_targets(data.decode()) == approved_targets(label),
                "patch target set exceeds declared boundary")
    require(sum(len(v) for v in text_copies.values()) <= MAX_INPUT_BYTES,
            "bounded source-only scratch size exceeded")

    out.mkdir()
    scratch = out/"scratch";scratch.mkdir()
    for relative,data in text_copies.items():
        dest = scratch/relative_name(relative)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("xb") as stream:stream.write(data)
    runner_source = Path(__file__).read_bytes()
    (out/"runner-source.py").write_bytes(runner_source)
    deadline = time.monotonic() + args.timeout_seconds
    report = dict(status="FAILED_CPU_PATCH_ROUNDTRIP",scope="CPU source composition/reversibility; no new GPU qualification",
                  runner_sha256=hash_bytes(runner_source),execution_lock_sha256=args.execution_lock_sha256,
                  idle_receipt=receipt,initial_disk=initial_disk,scratch=str(scratch),
                  input_hashes=inputs,commands=[],stages=[],source_files_modified=False,
                  gpu_operations=0,gpu_qualification=False,unique_full_matrix_test_total_added=0)
    dump_new(out/"input-manifest.json",inputs)
    command_index = 0
    temp_dir=scratch/"temporary";cache_dir=scratch/"cache"
    temp_dir.mkdir();cache_dir.mkdir()
    env = {k:v for k,v in os.environ.items() if not k.startswith("GIT_")}
    env.update(PYTHONPATH="",PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",
               GIT_CONFIG_NOSYSTEM="1",GIT_CONFIG_GLOBAL=os.devnull,
               TMPDIR=str(temp_dir),TMP=str(temp_dir),TEMP=str(temp_dir),
               XDG_CACHE_HOME=str(cache_dir),VLLM_CACHE_ROOT=str(cache_dir/"vllm"),
               TORCH_EXTENSIONS_DIR=str(cache_dir/"torch_extensions"))
    def command(argv,label):
        nonlocal command_index
        remaining = deadline-time.monotonic()
        require(remaining>0,"CPU roundtrip wall deadline exhausted")
        command_index += 1
        stem = "command-"+str(command_index).zfill(2)+"-"+label
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        row=dict(label=label,argv=list(argv),cwd=str(scratch),started_utc=started,
                 timeout_seconds=min(120,remaining),returncode=None,
                 stdout=stem+".stdout.txt",stderr=stem+".stderr.txt")
        stdout=stderr=""
        failure=None
        try:
            completed=subprocess.run(argv,cwd=scratch,env=env,capture_output=True,text=True,
                                     timeout=row["timeout_seconds"],check=False)
            row["returncode"]=completed.returncode
            stdout,stderr=completed.stdout,completed.stderr
            if completed.returncode!=0:
                failure=RuntimeError("CPU command failed: "+label)
        except BaseException as exc:
            failure=exc
            row["error"]=type(exc).__name__+": "+str(exc)
            for key in ("stdout","stderr"):
                data=getattr(exc,key,None)
                if isinstance(data,bytes):data=data.decode("utf-8",errors="replace")
                if isinstance(data,str):
                    if key=="stdout":stdout=data
                    else:stderr=data
        finally:
            row["ended_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
            (out/(stem+".stdout.txt")).write_text(stdout)
            (out/(stem+".stderr.txt")).write_text(stderr)
            report["commands"].append(row)
            dump_new(out/(stem+".json"),row)
        if failure is not None:
            raise failure
    def patch(label,relative,sha,directory,reverse=False):
        p = root/relative
        require(hash_bytes(bounded_bytes(root,relative))==sha,"patch changed during run")
        direction = "reverse" if reverse else "forward"
        command(patch_command(scratch,p,directory=directory,check=True,reverse=reverse),label+"-"+direction+"-check")
        command(patch_command(scratch,p,directory=directory,reverse=reverse),label+"-"+direction+"-apply")
    def stage(name,expected,policy_present=False):
        row=dict(name=name,native=compare_native(scratch,expected))
        for key,sha in new_policy_sha.items():
            if policy_present:
                require(hash_bytes(bounded_bytes(scratch,key))==sha,"research source reconstruction differs")
            else:
                require(not (scratch/key).exists(),"research source remains in common-only/base tree")
        report["stages"].append(row)
        dump_new(out/(name+"-manifest.json"),row)
    def smoke(mode):
        output = out/(mode+"-cpu-smoke.json")
        command([sys.executable,"-I",str(out/"runner-source.py"),"--_cpu-child",
                 "--scratch",str(scratch),"--mode",mode,"--child-output",str(output)],mode+"-cpu-smoke")
        value = json.loads(output.read_text())
        require(value["status"]=="PASS_CPU_COMPOSITION_ONLY" and not value["cuda_initialized"] and
                value["real_gpu_qualification"] is False,"CPU composition evidence invalid")
        report[mode+"_cpu_smoke"]=value
    try:
        command(["git","-c","init.templateDir=","-C",str(scratch),"init","--quiet"],"git-init")
        stage("seed-p313",seeded)
        patch(*PATCHES[0]);stage("common315",common315)
        patch(*PATCHES[1]);stage("common316",common316)
        smoke("common")
        patch(*PATCHES[2]);stage("full316-research315",final_native,True)
        require(hash_bytes(bounded_bytes(scratch,DRIVER))==driver_spec["final_sha256"],
                "research driver reconstruction differs")
        smoke("research")
        patch(*PATCHES[2],reverse=True);stage("reverse-research-common316",common316)
        require(hash_bytes(bounded_bytes(scratch,DRIVER))==driver_spec["base_sha256"],
                "research driver reverse failed")
        patch(*PATCHES[1],reverse=True);stage("reverse-increment-common315",common315)
        patch(*PATCHES[0],reverse=True);stage("reverse-common-seed-p313",seeded)
        require(all(hash_bytes(bounded_bytes(scratch,k))==hash_bytes(v) for k,v in text_copies.items()),
                "full scratch source roundtrip mismatch")
        report["status"]="PASS_CPU_PATCH_ROUNDTRIP_COMPOSITIONS"
    except BaseException as exc:
        report.update(error=str(exc),traceback=traceback.format_exc())
    finally:
        changed=[];unverified=[]
        for key,sha in inputs.items():
            try:
                if hash_bytes(bounded_bytes(root,key))!=sha:changed.append(key)
            except BaseException as exc:
                unverified.append(dict(path=key,error=type(exc).__name__+": "+str(exc)))
        report["source_preservation"]=dict(checked_text_inputs=len(inputs),changed=changed,unverified=unverified)
        if changed or unverified:report["status"]="FAILED_CPU_PATCH_ROUNDTRIP"
        try:
            report["final_disk"]=disk_check(out)
            size=sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
            report["output_file_bytes_before_result_json"]=size
            if size>RESERVE_BYTES:
                raise ValueError("scratch/receipts exceeded 64MiB")
        except BaseException as exc:
            report.update(status="FAILED_CPU_PATCH_ROUNDTRIP",storage_error=str(exc))
        report["scope_limits"]=[
            "No GPU stream/event/tensor/AIO backend/model execution occurred in built-in checks.",
            "CUDA init deny guard prevents GPU use; no mocked GPU result qualifies a real physical operation.",
            "CPU synthetic refcounts/staging indices qualify software transitions only, not DMA or native I/O completion.",
            "Existing production path-exact guards are unmodified; separate full native suites are not rerun by this runner.",
            "Final reconstructed native source matches frozen P316 bytes; separate common-only scratch has CPU qualification only.",
            "Duplicate CPU checks across compositions do not increase the unique full-matrix test total.",
            "Research-disabled native path retains common capacity bookkeeping and STOP fixes."
        ]
        final_machine_fields(report,final_native)
        dump_new(out/"result.json",report)
    return report

def main():
    if "--_cpu-child" in sys.argv:
        parser=argparse.ArgumentParser()
        parser.add_argument("--_cpu-child",action="store_true")
        parser.add_argument("--scratch",type=Path,required=True)
        parser.add_argument("--mode",choices=("common","research"),required=True)
        parser.add_argument("--child-output",type=Path,required=True)
        a=parser.parse_args()
        cpu_child(a.scratch,a.mode,a.child_output)
        return 0
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execution-lock",type=Path,required=True)
    parser.add_argument("--execution-lock-sha256",required=True)
    parser.add_argument("--gpu-idle-receipt",type=Path,required=True)
    parser.add_argument("--timeout-seconds",type=int,default=600)
    a=parser.parse_args()
    value=run(ROOT,a)
    print(json.dumps(dict(status=value["status"],result=str(a.output/"result.json"),
                         gpu_operations=0,gpu_qualification=False),indent=2))
    return 0 if value["status"].startswith("PASS_") else 1

if __name__=="__main__":
    raise SystemExit(main())
