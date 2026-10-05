"""Guard-child adapter for the pinned original one-token native cost pilot.

Import/preflight are CPU-only. Permission, complete lock and active reservation
checks belong to the new parent launcher, not to the old G2 permission flow.
No executor/cache implementation, extra consuming call, timer or release credit.
"""
import collections
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import weakref

COMMON = "artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/run_g2_normal_model_lifecycle.py"
ACQUIRE = "experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py"
SMOKE = "experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py"
CONCURRENT = "experiments/prefix_io_v1/scripts/concurrent_pilot_contract.py"
PINNED = {
    COMMON: (46746, "82cea015522596a89a89eb1ea0245f9ea111c942b74456cd87288507aec54422"),
    ACQUIRE: (18995, "68b2c045bcc9a7de84771d556b0e01180a592280002a2d4360d1c7500c9c856e"),
    SMOKE: (21990, "7d31cce851ca2334fedbe6b66c4e2d9bfff19e8768bafaa52f68e414518d77e2"),
    CONCURRENT: (3982, "645320e10a23795435f8a7eb77b6d7c24def8b58b1c6f82c662fa069482811f8"),
}
PURPOSE = "CURRENT_CONTEXT_NATIVE_RAW_COST_PILOT_ONLY"
GPU_UUID = "GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2"
MODES = ("cold", "populate", "paired")
STAGES = ("ssd_read", "ssd_write", "h2d", "d2h")
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
METRICS_SOURCE = "third_party/work/vllm-author-p4-02-cpu/vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py"
METRICS_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics"
METRICS_REF = dict(path=METRICS_SOURCE, bytes=6195,
    sha256="5ab3c65af3c0f78b0696bcc60c68ecfe51332c6d5f6684bb2cfa0e0ea9677d73")
STATS_COMPAT_SOURCE = "artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002/gpu-actual-20261002/metrics-common-fix-cpu/compat_offloading_stats_dict.py"
STATS_COMPAT_REF = dict(path=STATS_COMPAT_SOURCE, bytes=9744,
    sha256="7204b0fb3f784f58e5c8cc396810ba52a40769a168980c7e8e0ed210e19bb007")


def require(ok, reason):
    if not ok: raise ValueError(reason)


def scalar_copy(value, depth=0):
    """Bounded JSON scalars only; no owner, tensor, Future or arbitrary repr."""
    require(depth <= 7, "bounded scalar nesting")
    if value is None or type(value) in (bool, int): return value
    if type(value) is float:
        require(abs(value) < float("inf"), "finite scalar")
        return value
    if type(value) is str:
        require(len(value) <= 4096, "bounded scalar text")
        return value
    if type(value) in (list, tuple):
        require(len(value) <= 256, "bounded scalar sequence")
        return [scalar_copy(v, depth+1) for v in value]
    require(type(value) is dict and len(value) <= 128, "bounded scalar object")
    require(all(type(k) is str and len(k) <= 128 for k in value), "scalar keys")
    return {k: scalar_copy(v, depth+1) for k, v in value.items()}


def checked_source(root, refs, relative, expected=None):
    require(type(relative) is str and relative and "\\" not in relative and not Path(relative).is_absolute()
            and all(part not in ("", ".", "..") for part in relative.split("/")), "safe source relative path")
    require(type(refs) is dict and relative in refs, "exact source ref required")
    row = refs[relative]
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"} and row["path"] == relative,
            "source ref shape")
    if expected is not None:
        require(row["bytes"] == expected[0] and row["sha256"] == expected[1], "fixed source pin differs")
    require(type(row["bytes"]) is int and 0 < row["bytes"] <= 4*1024**2 and
            type(row["sha256"]) is str and len(row["sha256"]) == 64, "bounded source pin")
    path = root / relative
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), "source path symlink")
        cursor = cursor.parent
    require(path.resolve().is_relative_to(root) and path.is_file() and path.stat().st_size == row["bytes"],
            "source path/bytes drift")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == row["sha256"], "source SHA drift")
    return path, raw


def load_source(root, refs, relative):
    path, raw = checked_source(root, refs, relative, PINNED.get(relative))
    name = "_g3_calibration_"+hashlib.sha256(str(path).encode()).hexdigest()[:20]
    require(name not in sys.modules, "fresh private adapter module required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try: exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None); raise
    return module


def preflight_runtime(root, refs):
    """Read sources/assets only; never configure caches or import backends."""
    root = Path(root).resolve(strict=True)
    for rel, expected in PINNED.items(): checked_source(root, refs, rel, expected)
    checked_source(root, refs, METRICS_SOURCE, (METRICS_REF["bytes"], METRICS_REF["sha256"]))
    checked_source(root, refs, STATS_COMPAT_SOURCE, (STATS_COMPAT_REF["bytes"], STATS_COMPAT_REF["sha256"]))
    common = load_source(root, refs, COMMON)
    try:
        # Existing pure-CPU checks verify actual CUDA13 inventory/tool bytes and
        # exact optional absence. They do not prepare overlay or install hooks.
        common.load_sdk_assets(root, refs)
        common.verify_ninja(root, refs)
        common.load_optional_probe(root, refs)
        return dict(status="CPU_RUNTIME_SOURCE_ASSETS_ONLY", source_pins_verified=True,
                    framework_imports=0, GPU_operations=0, directories_created=0,
                    calibration_qualified=False, production_qualified=False)
    finally: sys.modules.pop(common.__name__, None)


def validate_guard(witness, mode):
    require(type(mode) is str and mode in MODES and type(witness) is dict, "fixed raw mode/guard witness")
    expected = {"schema_version", "purpose", "gpu_uuid", "label", "source_lock_sha256",
        "scope_sha256", "permissions_ref", "session_id", "guard_command", "seconds_limit",
        "reserved_seconds", "authorized_scope_verified", "active_reservation_verified"}
    require(set(witness) == expected and type(witness["schema_version"]) is int and witness["schema_version"] == 1,
            "exact parent guard witness")
    require(witness["purpose"] == PURPOSE and witness["gpu_uuid"] == GPU_UUID and
            witness["label"] == "server09-g3-calibration-"+mode+"-02", "new pilot identity")
    require(witness["authorized_scope_verified"] is True and witness["active_reservation_verified"] is True,
            "parent must verify human scope and active original reservation")
    require(type(witness["seconds_limit"]) is int and witness["seconds_limit"] == 300 and
            type(witness["reserved_seconds"]) is int and witness["reserved_seconds"] == 320,
            "finite pilot budget")
    for key in ("source_lock_sha256", "scope_sha256"):
        value = witness[key]
        require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "exact evidence SHA")
    scalar_copy(witness["permissions_ref"])
    command = witness["guard_command"]
    require(type(command) is list and command and all(type(s) is str and len(s) <= 4096 for s in command), "exact active command")
    require(os.name == "posix" and type(witness["session_id"]) is int and witness["session_id"] == os.getsid(0)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID, "actual guarded child SID/UUID")


def fixed_argv(mode, output, storage, model, model_plan):
    require(type(mode) is str and mode in MODES, "raw cost mode only")
    return ["acquire_native_aio_costs.py", "--output-dir", str(output), "--storage", str(storage),
        "--model-dir", str(model), "--model-plan", str(model_plan), "--mode", mode,
        "--domain", "1024", "--sizes", "128", "--reps", "3", "--max-num-seqs", "1", "--iodepth", "4"]


def _codes(raw, filename):
    result = set()
    def walk(code):
        for value in code.co_consts:
            if type(value) is types.CodeType: result.add(value); walk(value)
    walk(compile(raw, str(filename), "exec", dont_inherit=True))
    return result


def original_method(owner, method, root, refs):
    require(method not in vars(owner), "existing instance override")
    bound = getattr(owner, method)
    require(type(bound) is types.MethodType and bound.__self__ is owner, "original bound method")
    function = bound.__func__
    require(type(function) is types.FunctionType and not hasattr(function, "__wrapped__"), "original unwrapped boundary")
    module = sys.modules.get(function.__module__)
    require(type(module) is types.ModuleType and function.__globals__ is module.__dict__, "actual original globals")
    path = Path(function.__code__.co_filename).resolve()
    relative = path.relative_to(root).as_posix()
    _, raw = checked_source(root, refs, relative)
    require(Path(module.__file__).resolve() == path and function.__code__ in _codes(raw, path), "actual original source code")
    require(type(owner) is module.__dict__.get(type(owner).__name__), "exact original class identity")
    return function


def original_module_function(module, name, root, refs):
    function=module.__dict__.get(name)
    require(type(module) is types.ModuleType and type(function) is types.FunctionType and
            function.__globals__ is module.__dict__, "original module function globals")
    path=Path(function.__code__.co_filename).resolve()
    relative=path.relative_to(root).as_posix();_,raw=checked_source(root,refs,relative)
    require(Path(module.__file__).resolve()==path and function.__code__ in _codes(raw,path),
            "original module function source identity")
    return function


def _field(owner, name):
    values = vars(owner)
    require(name in values, "missing actual field: "+name)
    return values[name]


def _count(value):
    require(type(value) in (dict, list, set, tuple, collections.deque), "actual owner collection")
    return len(value)


def post_shutdown_snapshot(handler):
    """One published snapshot plus actual fields; absent stage observer is UNKNOWN."""
    coordinator = _field(handler, "coordinator")
    reactor = _field(coordinator, "reactor")
    require(_field(handler, "is_shutdown") is True and _field(reactor, "_closed") is True,
            "original shutdown not complete")
    require(reactor._worker.is_alive() is False, "original reactor not joined")
    snapshot = scalar_copy(coordinator.inspect_snapshot())  # Sole new post-original snapshot.
    require(type(snapshot) is dict and snapshot.get("native_shutdown_read") is True and
            snapshot.get("owner_capture") is False, "actual native shutdown snapshot")
    ring = _field(reactor, "ring")
    require(_field(ring, "_closed") is True and _field(ring, "_drained") is True and
            _field(ring, "_fatal") is None and ring._worker.is_alive() is False, "actual AIO tail not closed")
    stats = _field(ring, "_stats")
    actual_aio = {key: stats[key] for key in ("accepted", "completed", "reaped")}
    for value in actual_aio.values(): require(type(value) is int and value >= 0, "actual AIO count")
    for key, attr in (("outstanding", "_ops"), ("pending", "_pending"), ("ready", "_ready"), ("unreaped", "_done")):
        actual_aio[key] = _count(_field(ring, attr))
    require(type(snapshot.get("aio")) is dict and all(snapshot["aio"].get(k) == v for k,v in actual_aio.items()), "AIO published/actual discrepancy")
    native_map = {"active_parents": "_active", "ring_ops": "_inflight", "pending_copies": "_pending_copies",
        "copy_ready": "_copy_ready", "ready_load_fds": "_ready_fds_load", "ready_preload_fds": "_ready_fds_preload"}
    native = {key: _count(_field(reactor, attr)) for key, attr in native_map.items()}
    require(type(snapshot.get("native")) is dict and snapshot["native"] == native, "native published/actual discrepancy")
    account = _field(reactor, "_prefix_stage_accounting")
    stage_rows = None
    if account is not None:
        require(type(snapshot.get("stage_accounting")) is dict, "missing actual stage snapshot")
        stage_rows = scalar_copy(_field(account, "stats"))
        require(set(stage_rows) == set(STAGES) and snapshot["stage_accounting"].get("stages") == stage_rows,
                "four-stage published/actual discrepancy")
    else: require(snapshot.get("stage_accounting") is None, "unexpected published stage counters")
    return dict(snapshot=scalar_copy(snapshot), actual_aio=actual_aio, actual_native=native,
        actual_handler_active=_count(_field(handler, "_active")), actual_stage_counters=stage_rows,
        stage_accounting_status="UNKNOWN_NOT_ENABLED" if stage_rows is None else "RAW_COUNTERS_ONLY",
        observation_failures=_field(reactor, "_observation_failures"),
        GPU_timer_qualified=False, native_release_qualified=False, production_qualified=False,
        payload_byte_identity_verified=False, cost_qualified=False, frame_io_attribution="UNKNOWN")


class ScalarTaps:
    """Only bounded copied outputs; original method calls are never retried."""
    def __init__(self, evidence_dir=None):
        self.requests=[]; self.tails=[]; self.failures=[]; self.restorations=[]
        self.engine_shutdown_calls=0; self.engine_shutdown_returned=False
        self.handler_shutdown_calls=0
        self._patches=[]; self._handlers=set(); self._cores=set()
        self.evidence_dir=evidence_dir

    def fault(self, error):
        if len(self.failures) < 8: self.failures.append(type(error).__name__)

    def observe(self, operation):
        try: operation()
        except BaseException as error: self.fault(error)

    def patch(self, owner, key, wrapper):
        values=vars(owner); present=key in values; old=values.get(key)
        setattr(owner,key,wrapper)
        self._patches.append((weakref.ref(owner),key,present,old,wrapper))

    def restore(self):
        for reference,key,present,old,wrapper in reversed(self._patches):
            owner=reference()
            if owner is None: continue
            if vars(owner).get(key) is not wrapper:
                self.restorations.append(False); continue
            if present: setattr(owner,key,old)
            else: delattr(owner,key)
            self.restorations.append(True)
        self._patches.clear()

    def install_generate(self, klass):
        original=klass.generate
        taps=self
        def generate(owner,*args,**kwargs):
            result=original(owner,*args,**kwargs)
            def copy():
                require(len(taps.requests) < 18 and type(result) is list and len(result)==1, "bounded actual generate outputs")
                output=result[0]
                prompt=list(output.prompt_token_ids); tokens=list(output.outputs[0].token_ids)
                require(len(prompt) in (1,129) and len(tokens)==1 and
                    all(type(v) is int for v in prompt+tokens), "actual fixed prompt/output IDs")
                cached=output.num_cached_tokens;request_id=output.request_id
                require(type(cached) is int and 0<=cached<=128 and type(request_id) is str and
                    0<len(request_id)<=128,"bounded actual output metadata")
                row=dict(ordinal=len(taps.requests),prompt_token_ids=prompt,
                    output_token_ids=tokens,num_cached_tokens=cached,
                    sentinel=len(prompt)==1,request_id=request_id)
                taps.requests.append(scalar_copy(row))
                if taps.evidence_dir is not None:
                    with (taps.evidence_dir/("original-generate-%02d.json"%(len(taps.requests)-1))).open("x",encoding="utf-8") as stream:
                        json.dump(taps.requests[-1],stream,allow_nan=False);stream.write("\n")
            taps.observe(copy)
            return result
        self.patch(klass,"generate",generate)

    def install_engine_shutdown(self, core, original):
        if id(core) in self._cores: return
        self._cores.add(id(core)); taps=self; reference=weakref.ref(core)
        def shutdown(*args,**kwargs):
            owner=reference(); require(owner is not None,"original core still exists")
            taps.engine_shutdown_calls+=1
            result=original(owner,*args,**kwargs)
            taps.engine_shutdown_returned=True
            return result
        self.patch(core,"shutdown",shutdown)

    def install_handler_shutdown(self, handler, original):
        if id(handler) in self._handlers: return
        require(len(self._handlers) < 1,"sole original native handler")
        self._handlers.add(id(handler)); reference=weakref.ref(handler); taps=self
        def shutdown(*args,**kwargs):
            owner=reference(); require(owner is not None,"original handler still exists")
            taps.handler_shutdown_calls+=1
            result=original(owner,*args,**kwargs)
            taps.observe(lambda: taps.tails.append(post_shutdown_snapshot(owner)))
            return result
        self.patch(handler,"shutdown",shutdown)


def execute_original_acquisition(root, mode, out, storage, refs, guard_witness):
    """Called only by the new parent's validated original-budget guard child.

    The witness is a checked caller receipt, not an authorization interface:
    caller must independently re-read actual human scope/lock/ledger/argv first.
    """
    validate_guard(guard_witness,mode)  # Before any source/module/backend/cache.
    root=Path(root).resolve(strict=True); out=Path(out).resolve(strict=True)
    storage=Path(storage).absolute()
    require(out.is_relative_to(root) and out.is_dir() and not out.is_symlink(),"actual approved details directory")
    output=out/"acquisition"; require(not output.exists(),"original acquisition must be append-new")
    preflight_runtime(root,refs)
    common=load_source(root,refs,COMMON)
    base=load_source(root,refs,SMOKE)
    original_base=dict(ROOT=base.ROOT,AUTHOR_ROOT=base.AUTHOR_ROOT,ENGINE=base.ENGINE,
        configure_runtime_environment=base.configure_runtime_environment)
    aliases={}; old_path=list(sys.path); old_argv=sys.argv; old_cwd=Path.cwd()
    old_environment=dict(os.environ); finder=None; probe=None; acquire=None; stats_compat=None; stats_handle=None
    taps=ScalarTaps(out); receipt=dict(origin="source_bound_runtime",purpose=PURPOSE,mode=mode,
        guard_witness=scalar_copy(guard_witness),framework_import_attempted=False,
        GPU_initialization_attempted=False,original_exit_code=None,common_environment={},
        source_lock_sha256=guard_witness["source_lock_sha256"],scope_sha256=guard_witness["scope_sha256"],
        process_identity=dict(pid=os.getpid(),sid=os.getsid(0)),
        cost_qualified=False,production_qualified=False,performance_claim=False,
        raw_calibration_only=True,curves=None,load_planner="off",GPU_elapsed_ns=None,
        common_stats_dictionary_fix=dict(installed=False,restored=False,
            original_metrics_ref=METRICS_REF,helper_ref=STATS_COMPAT_REF,witness=None))
    def register(name,module):
        aliases[name]=(name in sys.modules,sys.modules.get(name)); sys.modules[name]=module
    try:
        base.ROOT=root; base.AUTHOR_ROOT=root/common.AUTHOR
        base.ENGINE=dict(base.ENGINE,async_scheduling=False)
        require(base.MODEL_ID==MODEL_NAME and base.ENGINE["enable_prefix_caching"] is True,"fixed original alias/Prefix")
        register("native_gpu_prefix_smoke",base)
        sys.path[:0]=[str(root/"experiments/prefix_io_v1/scripts"),
            str(root/"third_party/work/py-kvcache-p4-02-cpu"),
            str(root/"third_party/work/prefix-io-p4-02-cpu/src")]
        storage_module=importlib.import_module("experiment_storage")
        original_permission=original_module_function(storage_module,"permission",root,refs)
        for function in ("authorized_path","preflight"):
            original_module_function(storage_module,function,root,refs)
        original_method_source=Path(storage_module.__file__).resolve().relative_to(root).as_posix()
        checked_source(root,refs,original_method_source)
        prepare=common.load_ref(root,refs[common.PREPARE])
        permit_ref=guard_witness["permissions_ref"]
        permit_path=common.checked_ref(root,permit_ref)
        effective=prepare.permission_fields(permit_path.read_text(encoding="utf-8"))
        require(effective["allow_gpu_runs"] is True and effective["approved_gpu_ids"]==[GPU_UUID]
            and Path(effective["approved_experiment_root"]).resolve()==root/"experiments/prefix_io_v1/runs",
            "actual PRIMARY effective permission")
        effective["approved_auxiliary_storage"]=None
        def permission(project=storage_module.ROOT):
            require(Path(project).resolve()==root,"original permission project identity")
            return dict(effective)
        storage_module.permission=permission
        receipt["effective_storage_permission_ref"]=scalar_copy(permit_ref)
        def configure():
            nonlocal finder,probe,stats_compat,stats_handle
            receipt["common_environment"]["caches"]=common.configure_process_caches(out,base)
            sdk=common.prepare_process_sdk(root,refs,out); receipt["common_environment"]["SDK"]=sdk
            receipt["common_environment"]["Ninja"]=common.prepare_process_ninja(root,refs,out,sdk)
            optional,absence=common.load_optional_probe(root,refs)
            receipt["optional_absence"]=absence
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
            os.environ["PYTHONHASHSEED"]="0"
            loader=common.load_ref(root,refs[common.QUALIFY])
            finder=loader.BoundAuthorFinder(root,refs);sys.meta_path.insert(0,finder)
            require(not any(n=="vllm" or n.startswith("vllm.") for n in sys.modules),"fresh original author import required")
            receipt["framework_import_attempted"]=receipt["GPU_initialization_attempted"]=True
            torch=importlib.import_module("torch"); vllm=importlib.import_module("vllm")
            utilities=importlib.import_module("vllm.utils.import_utils")
            probe=optional.install_capability_probe(utilities,finder,root,refs)
            loader.require(loader.normalize_uuid(torch.cuda.get_device_properties(0).uuid)==loader.normalize_uuid(GPU_UUID),"actual GPU UUID")
            # Common interface repair: preserve the original producer call and
            # every byte/time metric; no transfer/executor/logging change.
            checked_source(root,refs,METRICS_SOURCE,(METRICS_REF["bytes"],METRICS_REF["sha256"]))
            metrics=importlib.import_module(METRICS_MODULE)
            stats_compat=load_source(root,refs,STATS_COMPAT_SOURCE)
            stats_handle=stats_compat.install_stats_dict_compat(metrics,Path(metrics.__file__))
            receipt["common_stats_dictionary_fix"]["installed"]=True
            # Source-code binding before installing the bounded diagnostic tap.
            dummy=object.__new__(vllm.LLM)
            original_method(dummy,"generate",root,refs)
            taps.install_generate(vllm.LLM)
            receipt["author_module"]=str(Path(vllm.__file__).resolve())
        base.configure_runtime_environment=configure
        acquire=load_source(root,refs,ACQUIRE)
        original_control=acquire.worker_control
        def control(worker,*args,**kwargs):
            result=original_control(worker,*args,**kwargs)
            def install():
                # Original callback supplies the real Uni worker; never locate
                # owners through Future.done or consume get_finished ourselves.
                state=sys.modules.get("vllm.distributed.kv_transfer.kv_transfer_state")
                if state is None: return
                connector=state.__dict__.get("_KV_CONNECTOR_AGENT")
                if connector is None: return
                cw=connector.connector_worker; registry=cw.worker
                handlers=registry.handlers
                require(type(handlers) is set and len(handlers)==1,"sole native registered handler")
                handler=next(iter(handlers))
                if id(handler) in taps._handlers: return
                for owner in (worker,connector,cw,registry): original_method(owner,"shutdown",root,refs)
                mapping=registry.transfer_type_to_handler
                require(type(mapping) is dict and set(mapping)=={("GPU","SHARED_STORAGE"),("SHARED_STORAGE","GPU")}
                    and all(v is handler for v in mapping.values()),"original registered direction chain")
                method=original_method(handler,"shutdown",root,refs)
                original_method(handler.coordinator,"inspect_snapshot",root,refs)
                taps.install_handler_shutdown(handler,method)
            taps.observe(install)
            return result
        acquire.worker_control=control
        # Original main owns model alias/config/prompt/reset/generate/TTFT/drain.
        # Capture core shutdown via the real original LLM.generate boundary.
        previous_generate=taps.install_generate
        def install_generate(klass):
            original=klass.generate
            def original_with_core(owner,*args,**kwargs):
                core=owner.llm_engine.engine_core
                if id(core) not in taps._cores:
                    taps.observe(lambda:taps.install_engine_shutdown(core,original_method(core,"shutdown",root,refs)))
                return original(owner,*args,**kwargs)
            taps.patch(klass,"generate",original_with_core)
            previous_generate(klass)
        taps.install_generate=install_generate
        sys.argv=fixed_argv(mode,output,storage,root/common.MODEL,root/common.MODEL_PLAN)
        code=acquire.main()
        require(type(code) is int,"original integer exit code")
        receipt["original_exit_code"]=code
        return_code=code
    finally:
        taps.restore()
        if stats_handle is not None:
            try:
                stats_handle.detach()
                stats_witness=stats_handle.witness()
                receipt["common_stats_dictionary_fix"]["witness"]=scalar_copy(stats_witness)
                receipt["common_stats_dictionary_fix"]["restored"]=(stats_witness["installed"] is False)
            except BaseException as error:
                taps.fault(error)
                receipt["common_stats_dictionary_fix"]["restored"]=False
        if stats_compat is not None:
            sys.modules.pop(stats_compat.__name__,None)
        receipt["requests"]=scalar_copy(taps.requests)
        receipt["native_post_shutdown"]=scalar_copy(taps.tails)
        receipt["observer_fault_types"]=list(taps.failures)
        receipt["engine_shutdown_calls"]=taps.engine_shutdown_calls
        receipt["original_engine_shutdown_returned"]=taps.engine_shutdown_returned
        receipt["handler_shutdown_calls"]=taps.handler_shutdown_calls
        receipt["tap_restorations"]=list(taps.restorations)
        receipt["original_shutdown_completed"]=taps.engine_shutdown_returned and taps.engine_shutdown_calls==1
        receipt["sentinel_observations"]=[dict(rep=r["prompt_token_ids"][0]-30000,**r) for r in taps.requests if r["sentinel"]]
        receipt["frozen_config"]=None
        if (output/"frozen-config.json").is_file():
            receipt["frozen_config"]=json.loads((output/"frozen-config.json").read_text(encoding="utf-8"))
        receipt["source_unchanged"]=False
        receipt["source_preservation_scope"]="FOUR_PINNED_ADAPTER_ORIGINAL_FILES_ONLY"
        receipt["preserved_source_refs"]=[scalar_copy(refs[relative]) for relative in (ACQUIRE,COMMON,SMOKE,CONCURRENT)]
        receipt["complete_source_lock_postrun_verified"]=False
        try:
            for relative in (ACQUIRE,COMMON,SMOKE,CONCURRENT): checked_source(root,refs,relative,PINNED[relative])
            receipt["source_unchanged"]=True
        except BaseException as error: taps.fault(error)
        try:
            try: receipt["optional_probe_restored"]=True if probe is None else probe.detach()
            except BaseException as error:
                taps.fault(error);receipt["optional_probe_restored"]=False
        finally:
            if finder is not None and finder in sys.meta_path: sys.meta_path.remove(finder)
            for key,value in original_base.items(): setattr(base,key,value)
            if "storage_module" in locals() and "permission" in locals():
                if storage_module.permission is permission:
                    storage_module.permission=original_permission
                    receipt["storage_permission_restored"]=True
                else: receipt["storage_permission_restored"]=False
            for name,(present,old) in aliases.items():
                if sys.modules.get(name) is base:
                    if present: sys.modules[name]=old
                    else: sys.modules.pop(name,None)
            if acquire is not None: sys.modules.pop(acquire.__name__,None)
            sys.modules.pop(base.__name__,None);sys.modules.pop(common.__name__,None)
            sys.path[:]=old_path;sys.argv=old_argv;os.chdir(old_cwd)
            os.environ.clear();os.environ.update(old_environment)
            receipt["common_helpers_restored"]=all(taps.restorations) and receipt.get("optional_probe_restored") is True and receipt.get("storage_permission_restored") is True and receipt["common_stats_dictionary_fix"]["installed"] is True and receipt["common_stats_dictionary_fix"]["restored"] is True
            receipt["observer_fault_types"]=list(taps.failures)
            with (out/"calibration-runtime-result.json").open("x",encoding="utf-8") as stream:
                json.dump(scalar_copy(receipt),stream,indent=2,allow_nan=False);stream.write("\n")
    return dict(original_exit_code=return_code,runtime_receipt=receipt)
