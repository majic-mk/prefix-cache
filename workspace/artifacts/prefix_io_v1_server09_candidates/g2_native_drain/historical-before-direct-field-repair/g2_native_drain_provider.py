"""Independent CPU post-original-shutdown scalar drain adapter; no live hook.

Only original shutdown releases owners. This candidate never calls a Future,
join, close, synchronization or I/O submission itself. A source-bound successful
original shutdown and actual native/AIO/accounting fields must all agree.
"""
import collections
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

CONNECTOR_REF = (14795, "347fb842989e8e14c73968528396721b546a11d7312206aae2efbb3739b0f3cb")
SOURCE_REFS = {
    "third_party/work/py-kvcache-p4-02-cpu/py_kvcache/vllm.py": (33693,"901cc5b9a20245e60d2200ef190979b80adb77ca5a049624c6bc4ec171a712d1"),
    "third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py": (169906,"2b0ac1e8e68f82adb42ade03b74b49843eabe55361a00f0ac1f7a738ac423e18"),
    "third_party/work/py-kvcache-p4-02-cpu/py_kvcache/linux_aio.py": (17646,"0a987479722c7d6b520b99c9b283febb55c5145fc17c5623b8b1151c4f45a6d7"),
    "third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/stage_accounting.py": (3580,"11a736586a9278e2c373d4fc1bfe0faf9cd166b81e9d9c3f658500a51ab954aa"),
}
NATIVE_SOURCE_SHA256 = SOURCE_REFS["third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"][1]
STAGES = ("ssd_read","ssd_write","h2d","d2h")
OWNER_FIELDS=("_active","_inflight","_pending_copies","_copy_ready","_preload_pending",
    "_preload_slots","_shared_cached","_ready_fds_load","_ready_fds_preload","_store_inflight",
    "_preload_inflight_hashes","_preload_inflight_total","_data_inflight","_open_inflight",
    "_prefix_p4_controls_pending","_owner_snapshot_pending")


def require(ok,message):
    if not ok: raise ValueError(message)


def integer(value,name,minimum=0):
    require(type(value) is int and value>=minimum,"explicit actual integer required: "+name)
    return value


def boolean(value,expected,name):
    require(type(value) is bool and value is expected,"actual boolean differs: "+name)


def checked_file(path,expected):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),"pinned source absent/symlink")
    require(path.stat().st_size==expected[0],"source byte count drift")
    with path.open("rb") as handle: data=handle.read(expected[0]+1)
    require((len(data),hashlib.sha256(data).hexdigest())==expected,"source bytes/SHA drift")
    return path.resolve()


def load_connector(path):
    source=checked_file(path,CONNECTOR_REF)
    name="_g2_drain_connector_"+hashlib.sha256(str(source).encode()).hexdigest()[:20]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,source)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        try: spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name,None)
            raise
    module=sys.modules[name]
    require(Path(module.__file__).resolve()==source,"private connector module path differs")
    return module


@dataclass(frozen=True)
class CPUNativeDrainReceipt:
    run_id: str
    native_source_sha256: str
    captured_ns: int
    drain: object
    source_refs: tuple
    handler_active: int
    remaining_native_owners: tuple
    origin: str = "cpu_fixture"
    status: str = "CPU_POST_ORIGINAL_SHUTDOWN_SCALAR_DRAIN_ONLY"

    @property
    def gpu_verified(self): return False

    @property
    def production_qualified(self): return False

    @property
    def effect_verified(self): return False


class NativeDrainCPUProvider:
    """No owner reference or callback is retained, including on failure."""
    GPU_collector_verified=False
    gpu_verified=False
    production_qualified=False
    executable_native_G2=False
    runtime_hook_status="not_installed"

    def __init__(self,*,enabled=False,origin="cpu_fixture",run_id=None,source_root=None,
                 adapter_path=None,connector_path=None):
        require(type(enabled) is bool,"explicit optional drain switch")
        self.enabled=enabled
        self.valid=True
        self.status="off"
        self.last_reason="off"
        self.last_receipt=None
        self.last_drain=None
        self._tail_open=False
        if not enabled: return
        require(origin=="cpu_fixture","CPU drain candidate rejects native/GPU origin before source or owner reads")
        require(type(run_id) is str and 0<len(run_id)<=128,"bounded actual run id required")
        self.run_id=run_id
        self.root=Path(source_root).resolve(strict=True)
        siblings=Path(__file__).resolve().parent.parent
        self.connector_path=Path(connector_path or siblings/"runtime_connector/p4_runtime_scalar_connector.py")
        self.adapter_path=Path(adapter_path or siblings/"collector/p4_full_step_frame_adapter.py")
        self.connector=load_connector(self.connector_path)
        self.F=self.connector.load_frozen_adapter(self.adapter_path)
        self.paths={relative:checked_file(self.root/relative,ref) for relative,ref in SOURCE_REFS.items()}
        self.native_source_sha256=NATIVE_SOURCE_SHA256
        self._check_sources()
        self.status="cpu_source_bound_not_installed"
        self.last_reason="await_original_shutdown"

    def _check_sources(self):
        for relative,ref in SOURCE_REFS.items(): checked_file(self.root/relative,ref)
        checked_file(self.connector_path,CONNECTOR_REF)
        checked_file(self.adapter_path,(self.connector.ADAPTER_BYTES,self.connector.ADAPTER_SHA256))

    def _method(self,owner,name,qualname,relative):
        require(name not in vars(owner),"existing native instance method override is not source-bound")
        method=getattr(owner,name)
        require(type(method) is types.MethodType and method.__self__ is owner,"original bound native method required")
        function=method.__func__
        require(type(function) is types.FunctionType and not hasattr(function,"__wrapped__")
                and function.__qualname__==qualname
                and Path(function.__code__.co_filename).resolve()==self.paths[relative],
                "original native method source/qualname differs")
        return function

    def _bindings(self,handler):
        coordinator=handler.coordinator
        reactor=coordinator.reactor
        ring=reactor.ring
        vllm="third_party/work/py-kvcache-p4-02-cpu/py_kvcache/vllm.py"
        native="third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"
        aio="third_party/work/py-kvcache-p4-02-cpu/py_kvcache/linux_aio.py"
        boundaries=((handler,"shutdown","NoopSharedStorageOffloadingHandler.shutdown",vllm),
            (coordinator,"shutdown","TransferCoordinator.shutdown",native),
            (coordinator,"inspect_snapshot","TransferCoordinator.inspect_snapshot",native),
            (reactor,"shutdown","IoReactor.shutdown",native),
            (reactor,"inspect_snapshot","IoReactor.inspect_snapshot",native),
            (reactor,"_capture_owner_snapshot","IoReactor._capture_owner_snapshot",native),
            (reactor,"parent_admission_snapshot","IoReactor.parent_admission_snapshot",native),
            (ring,"close","LinuxAioRing.close",aio),(ring,"snapshot","LinuxAioRing.snapshot",aio))
        return tuple(self._method(*row) for row in boundaries)

    def invalidate(self,reason):
        self.valid=False
        self.status="CPU_DRAIN_UNKNOWN_OR_INVALID"
        self.last_reason=str(reason)[:240]
        self.last_receipt=None
        self.last_drain=None

    def observe_original_shutdown(self,handler):
        if not self.enabled: return handler.shutdown()
        self.last_receipt=self.last_drain=None
        bindings=None
        if self.valid:
            try:
                self._check_sources()
                bindings=self._bindings(handler)
            except BaseException as exc:
                NativeDrainCPUProvider.invalidate(self,"source_observer_fault:"+type(exc).__name__)
        try:
            result=handler.shutdown()  # Exactly the original call; no observer release.
        except BaseException:
            NativeDrainCPUProvider.invalidate(self,"original_handler_shutdown_failed")
            raise
        if self.valid:
            try:
                require(self._bindings(handler)==bindings,"original native boundary identity changed during shutdown")
                snapshot=handler.coordinator.inspect_snapshot()
                self._tail_open=True
                receipt=self.convert_post_shutdown(snapshot,self._copy_native_fields(handler))
                self._check_sources()
                self.last_receipt=receipt
                self.last_drain=receipt.drain
                self.status="CPU_POST_ORIGINAL_SHUTDOWN_SCALAR_DRAIN_ONLY"
                self.last_reason="original_return_and_actual_scalar_closure"
            except BaseException as exc:
                NativeDrainCPUProvider.invalidate(self,"tail_observer_fault:"+type(exc).__name__)
            finally:
                self._tail_open=False
        return result

    def _copy_native_fields(self,handler):
        reactor=handler.coordinator.reactor
        alive=reactor._worker.is_alive()
        boolean(alive,False,"original reactor worker still alive")
        boolean(handler.is_shutdown,True,"handler shutdown flag")
        boolean(reactor._closed,True,"reactor submission closed")
        boolean(reactor._native_drain_unknown,False,"reactor drain unknown")
        require(reactor._native_fatal_reason is None,"native reactor fatal reason remains")
        require(type(handler._active) is dict,"actual original handler active map required")
        pending={}
        for name in ("_active","_inflight","_pending_copies","_copy_ready","_preload_pending",
                     "_preload_slots","_shared_cached","_ready_fds_load","_ready_fds_preload",
                     "_store_inflight","_preload_inflight_hashes"):
            value=getattr(reactor,name)
            require(type(value) in (dict,list,collections.deque),"native owner collection missing: "+name)
            pending[name]=len(value)
        for name in ("_preload_inflight_total","_data_inflight","_open_inflight",
                     "_prefix_p4_controls_pending","_owner_snapshot_pending"):
            pending[name]=integer(getattr(reactor,name),name)
        return dict(worker_alive=alive,handler_active=len(handler._active),
            observation_failures=integer(reactor._observation_failures,"actual observation failures"),
            native_owners=pending)

    def convert_post_shutdown(self,snapshot,actual):
        """Shape semantics only; caller cannot turn this CPU result into GPU proof."""
        require(self.enabled and self.valid and self._tail_open,
                "source-bound successful original shutdown tail required")
        require(type(snapshot) is dict and type(actual) is dict,"copied actual native scalar dictionaries required")
        require(snapshot["run_id"]==self.run_id,"actual native run id differs")
        boolean(snapshot["native_shutdown_read"],True,"post-original native shutdown snapshot")
        boolean(snapshot["owner_capture"],False,"worker-owner snapshot before shutdown")
        boolean(snapshot["physical_drain_inferred"],False,"inferred physical drain forbidden")
        boolean(snapshot["gpu_release_credit"],False,"GPU release credit forbidden")
        captured=integer(snapshot["captured_ns"],"actual snapshot timestamp",1)
        boolean(actual["worker_alive"],False,"native worker termination")
        active=integer(actual["handler_active"],"actual handler active jobs")
        observations=integer(actual["observation_failures"],"actual native observation failures")
        require(type(actual["native_owners"]) is dict and set(actual["native_owners"])==set(OWNER_FIELDS),
                "all actual native owner fields required")
        require(all(integer(v,"actual native owner count")==0 for v in actual["native_owners"].values()),
                "native owner references remain after original shutdown")
        require(active==observations==0,"handler jobs or observation failures remain")
        accounting=snapshot["stage_accounting"]
        require(type(accounting) is dict and accounting["error"] is None,"actual stage accounting unavailable/error")
        boolean(accounting["valid"],True,"stage accounting validity")
        boolean(accounting["bound"],True,"native accounting binding")
        outstanding=integer(accounting["outstanding_records"],"actual accounting outstanding")
        require(outstanding==0,"actual accounting records remain")
        require(type(accounting["stages"]) is dict and set(accounting["stages"])==set(STAGES),"all four original stages required")
        accepted,completed,transferred,failed=[],[],[],[]
        for stage in STAGES:
            row=accounting["stages"][stage]
            require(type(row) is dict,"actual stage counters required")
            numbers={key:integer(row[key],stage+" "+key) for key in
                ("accepted_ops","accepted_bytes","completed_ops","completed_requested_bytes",
                 "transferred_bytes","failed_ops","inflight_ops","inflight_bytes")}
            require(numbers["inflight_ops"]==numbers["inflight_bytes"]==0,"unfinished actual physical stage")
            accepted.append(self.F.StageTotal(numbers["accepted_ops"],numbers["accepted_bytes"]))
            completed.append(self.F.StageTotal(numbers["completed_ops"],numbers["completed_requested_bytes"]))
            transferred.append(numbers["transferred_bytes"]);failed.append(numbers["failed_ops"])
        admission=snapshot["parent_admission"]
        require(type(admission) is dict and snapshot["admission"]==admission,"actual parent admission snapshot disagrees")
        boolean(admission["count_valid"],True,"parent acceptance count validity")
        boolean(admission["native_drain_unknown"],False,"parent drain unknown")
        require(admission["fatal_reason"] is None,"parent native fatal reason")
        parents=integer(admission["accepted_parents"],"actual accepted parents")
        require(parents==integer(admission["accepted_count_lower_bound"],"parent count lower bound")==
                integer(admission["waiting_submitters"],"waiting submitters")==0,"native parents/submitters remain")
        native=snapshot["native"]
        require(type(native) is dict,"actual native snapshot missing")
        require(all(integer(native[name],name)==0 for name in
            ("active_parents","ring_ops","pending_copies","copy_ready","ready_load_fds","ready_preload_fds")),
            "actual native snapshot is not empty")
        aio=snapshot["aio"]
        require(type(aio) is dict and aio["fatal"] is None,"actual Linux AIO snapshot unavailable/fatal")
        boolean(aio["closed"],True,"actual ring closed")
        boolean(aio["drained"],True,"actual ring drained")
        require(all(integer(aio[name],"AIO "+name)==0 for name in ("outstanding","pending","ready","unreaped")),
            "unresolved or unconsumed actual Linux AIO result")
        require(integer(aio["accepted"],"AIO acceptance")==integer(aio["completed"],"AIO completion")==
                integer(aio["reaped"],"AIO reaped"),"actual Linux AIO acceptance/completion/reap differ")
        drain=self.F.DrainEvidence(self.run_id,self.native_source_sha256,tuple(accepted),tuple(completed),
            tuple(transferred),tuple(failed),True,outstanding,True,aio["closed"],aio["drained"],aio["outstanding"],
            native["pending_copies"],active,parents,observations)
        drain.validate(self.run_id,self.native_source_sha256)
        refs=tuple((relative,*ref) for relative,ref in SOURCE_REFS.items())
        refs+=(('runtime_connector/p4_runtime_scalar_connector.py',*CONNECTOR_REF),
            ('collector/p4_full_step_frame_adapter.py',self.connector.ADAPTER_BYTES,self.connector.ADAPTER_SHA256))
        return CPUNativeDrainReceipt(self.run_id,self.native_source_sha256,captured,drain,refs,active,
            tuple(sorted(actual["native_owners"].items())))

    def require_gpu_launch(self,*args,**kwargs):
        raise ValueError("CPU drain candidate has no GPU launch or native qualification authority")
