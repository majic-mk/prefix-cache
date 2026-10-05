"""Passive G3 registration/transfer/tail observer; no executor or live installer.

CPU fixtures never establish native I/O facts. Runtime facts require the loaded
locked module graph, original functions and the actual registered handler. Raw
physical totals are copied only after original shutdown; no observer releases,
polls, waits, synchronizes, consumes a second result batch or owns a Future.
"""
import dataclasses
import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import threading
import types

AUTHOR = "third_party/work/vllm-author-p4-02-cpu/vllm/"
NATIVE = "third_party/work/py-kvcache-p4-02-cpu/py_kvcache/"
CONTROL = "third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/"
MODULE_REFS = {
    "vllm.distributed.kv_transfer.kv_transfer_state": (AUTHOR+"distributed/kv_transfer/kv_transfer_state.py",2791,"7778b57aa57a8911fe7efba7d03b4d25f4c26036fcd83a9307850a59082b8cbc"),
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector": (AUTHOR+"distributed/kv_transfer/kv_connector/v1/offloading_connector.py",9669,"02969dfa19fe6e9d7e9856e587fd115e67e614b09ce3cc50f5b009d179d05177"),
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker": (AUTHOR+"distributed/kv_transfer/kv_connector/v1/offloading/worker.py",24711,"5712de78cf9cd13ca5ec2bc0d9d349964a6d5a7986cbd9a2f46a26078176dc86"),
    "vllm.v1.kv_offload.worker.worker": (AUTHOR+"v1/kv_offload/worker/worker.py",6553,"90e8864b39d47971a89139d15c855e1efee214cf9c2c4ae3b0a6f017c309c2db"),
    "vllm.v1.kv_offload.base": (AUTHOR+"v1/kv_offload/base.py",15330,"204d682bf61d3a2c1bbbe8ca1ecfd6fe6b10eca32b2600c39118403db6aae1d2"),
    "vllm.v1.worker.gpu_worker": (AUTHOR+"v1/worker/gpu_worker.py",49164,"bcbdfba6d0846f4151551c39140b370cba33a1af6401e0a2362f5be1de3cfd3b"),
    "py_kvcache.vllm": (NATIVE+"vllm.py",33693,"901cc5b9a20245e60d2200ef190979b80adb77ca5a049624c6bc4ec171a712d1"),
    "py_kvcache.reactor": (NATIVE+"reactor.py",169906,"2b0ac1e8e68f82adb42ade03b74b49843eabe55361a00f0ac1f7a738ac423e18"),
    "py_kvcache.linux_aio": (NATIVE+"linux_aio.py",17646,"0a987479722c7d6b520b99c9b283febb55c5145fc17c5623b8b1151c4f45a6d7"),
    "py_kvcache.transfer": (NATIVE+"transfer.py",12061,"1dcc5f4370e3db694d34924038a8fdd5a2eb12cf7f04ba81072dc969f275c1ba"),
    "prefix_io_control.stage_accounting": (CONTROL+"stage_accounting.py",3580,"11a736586a9278e2c373d4fc1bfe0faf9cd166b81e9d9c3f658500a51ab954aa"),
}
JOURNAL_REF = (CONTROL+"p4_native_window_journal.py",9647,"3a9ded39846cccf88ee9aceed2e16b86d8a7bbe953adfea3ba1a302b2c5aa8c3")
DRAIN_REF = (21997,"261f8cd2438a4941d41077b20c95bfab4ca4418858bbab09aa7a3ef459353a44")
QUALIFIER_REF = ("experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py",29692,"5d9604fc07d80495a49febfcd488cb2601f74a2bef107261c8e623a4d0a1c77d")
STAGES = ("ssd_read","ssd_write","h2d","d2h")


def require(ok, message):
    if not ok: raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, "exact scalar integer: "+name)
    return value


def text(value, name, allow_empty=False):
    require(type(value) is str and len(value)<=128 and (allow_empty or value), "bounded scalar text: "+name)
    return value


def field(owner, name):
    values=vars(owner)
    require(name in values,"actual owner field missing: "+name)
    return values[name]


def checked(path, size, sha):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve()==path.absolute(),"source missing/symlink/noncanonical")
    require(0<size<=4*1024**2 and path.stat().st_size==size,"source bytes drift")
    with path.open("rb") as handle: raw=handle.read(size+1)
    require(len(raw)==size and hashlib.sha256(raw).hexdigest()==sha,"source SHA drift")
    return path.resolve(),raw


def frozen_module(path):
    path,raw=checked(path,*DRAIN_REF)
    name="_g3_frozen_drain_"+hashlib.sha256(str(path).encode()).hexdigest()[:20]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        try: spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name,None)
            raise
    module=sys.modules[name]
    require(type(module) is types.ModuleType and Path(module.__file__).resolve()==path,"frozen drain module differs")
    return module


def source_codes(raw, path):
    found={}
    def walk(code):
        for item in code.co_consts:
            if type(item) is types.CodeType:
                found.setdefault(item.co_qualname,[]).append(item)
                walk(item)
    walk(compile(raw,str(path),"exec",dont_inherit=True))
    return found


@dataclasses.dataclass(frozen=True)
class OwnerWitness:
    run_id: str
    origin: str
    registration_chain: tuple
    source_refs: tuple
    source_bound_runtime_observed: bool

    def __post_init__(self):
        text(self.run_id,"witness run")
        require(type(self.origin) is str and self.origin in ("cpu_fixture","source_bound_runtime"),"exact witness origin")
        require(type(self.source_bound_runtime_observed) is bool and
            (self.origin!="cpu_fixture" or not self.source_bound_runtime_observed),"CPU fixture cannot establish runtime owner")

    @property
    def gpu_verified(self): return False
    @property
    def production_qualified(self): return False


@dataclasses.dataclass(frozen=True)
class TransferObservation:
    job_id: int
    request_id: str
    direction: str
    block_ids: tuple
    block_hash_sha256: tuple
    storage_block_bytes: int
    accepted: bool


@dataclasses.dataclass(frozen=True)
class TransferResultObservation:
    job_id: int
    success: bool
    transfer_size: object
    transfer_type: object


@dataclasses.dataclass(frozen=True)
class NativeTailObservation:
    run_id: str
    origin: str
    source_bound_runtime_observed: bool
    captured_ns: int
    stages: tuple
    aio_counts: tuple
    remaining_owners: tuple
    task_results_complete: bool
    all_four_stages_nonzero: bool
    source_refs: tuple

    def __post_init__(self):
        text(self.run_id,"tail run")
        require(type(self.origin) is str and self.origin in ("cpu_fixture","source_bound_runtime"),"exact tail origin")
        require(type(self.source_bound_runtime_observed) is bool and
            (self.origin!="cpu_fixture" or not self.source_bound_runtime_observed),"CPU fixture cannot establish runtime tail")

    @property
    def native_io_completion_observed(self):
        # Source/owner/counter consistency alone lacks an independently bound
        # GPU device, budget guard and actual native-run witness.
        return False
    @property
    def source_bound_stage_closure_observed(self):
        return self.origin=="source_bound_runtime" and self.source_bound_runtime_observed
    @property
    def gpu_verified(self): return False
    @property
    def production_qualified(self): return False
    @property
    def cost_qualified(self): return False
    @property
    def release_credit(self): return False
    @property
    def effect_verified(self): return False
    @property
    def kv_byte_exact_verified(self): return False
    @property
    def gpu_elapsed_ns(self): return None
    @property
    def frame_io_attribution(self): return "UNKNOWN"


class NativeOwnerObserver:
    """Per-call owner arguments are ephemeral; retained state is bounded scalars.

    This does not install hooks. Call at the already-authorized original handler
    boundary; do not invoke get_finished/shutdown separately for observation.
    Runtime registration verification is a fact gate, never GPU authorization.
    """
    runtime_hook_status="not_installed"
    gpu_verified=False
    production_qualified=False
    cost_qualified=False
    effect_verified=False

    def __init__(self, *, enabled=False, origin="cpu_fixture", run_id=None,
                 source_root=None, support_root=None, drain_module_path=None,
                 connector_path=None, adapter_path=None, max_records=1024):
        require(type(enabled) is bool,"explicit observer switch")
        self.enabled=enabled;self.valid=True;self.reason="off"
        self.last_owner=None;self.last_tail=None;self.transfers=[];self.results=[]
        self._pending={};self._completed=set();self._shutdown_seen=False
        if not enabled: return
        require(type(origin) is str and origin in ("cpu_fixture","source_bound_runtime"),"explicit observation origin")
        self.origin=origin;self.run_id=text(run_id,"run_id")
        require(type(max_records) is int and 1<=max_records<=4096,"bounded observations")
        self.max_records=max_records
        self.root=Path(source_root).resolve(strict=True)
        self.support=Path(support_root or source_root).resolve(strict=True)
        sibling=Path(__file__).resolve().parent.parent
        self.drain_path=Path(drain_module_path or sibling/"g2_native_drain/g2_native_drain_provider.py").resolve(strict=True)
        self.D=frozen_module(self.drain_path)
        self.validator=self.D.NativeDrainCPUProvider(enabled=True,run_id=self.run_id,source_root=self.support,
            connector_path=connector_path,adapter_path=adapter_path)
        self.paths={};self.codes={}
        for name,ref in MODULE_REFS.items():
            base=self.support if name in ("vllm.v1.worker.gpu_worker","py_kvcache.vllm","py_kvcache.reactor","py_kvcache.linux_aio","prefix_io_control.stage_accounting") else self.root
            path,raw=checked(base/ref[0],*ref[1:])
            self.paths[name]=path;self.codes[name]=source_codes(raw,path)
        self.reason="source_ready_no_runtime_owner_observed"

    def invalidate(self, reason):
        self.valid=False;self.reason=reason[:240] if type(reason) is str else "non_scalar_observer_fault"
        self.last_owner=None;self.last_tail=None

    def _observe(self, operation):
        if not self.enabled or not self.valid: return None
        try: return operation()
        except BaseException as error:
            self.invalidate("observer_fault:"+type(error).__name__)
            return None

    def _check_sources(self):
        for name,ref in MODULE_REFS.items(): checked(self.paths[name],*ref[1:])
        checked(self.drain_path,*DRAIN_REF)
        self.validator._check_sources()

    def _module(self, name):
        module=sys.modules.get(name)
        require(type(module) is types.ModuleType and module.__name__==name,"exact loaded module missing: "+name)
        require(Path(module.__file__).resolve()==self.paths[name],"loaded module source differs: "+name)
        if self.origin=="source_bound_runtime":
            spec=module.__spec__
            require(type(spec) is importlib.machinery.ModuleSpec and spec.name==name and
                Path(spec.origin).resolve()==self.paths[name] and module.__loader__ is spec.loader,"loaded module spec/loader differs")
            loader=spec.loader
            if name.startswith("vllm."):
                qualifier=field(loader,"path")
                require(Path(qualifier).resolve()==self.paths[name],"locked author loader path differs")
                qpath,qraw=checked(self.root/QUALIFIER_REF[0],*QUALIFIER_REF[1:])
                function=type(loader).__dict__.get("get_code")
                require(type(function) is types.FunctionType and function.__qualname__=="LockedSourceLoader.get_code" and
                    Path(function.__code__.co_filename).resolve()==qpath and
                    function.__code__ in source_codes(qraw,qpath)["LockedSourceLoader.get_code"],"exact locked author loader required")
                qualifier_module=sys.modules.get(function.__module__)
                require(type(qualifier_module) is types.ModuleType and function.__globals__ is qualifier_module.__dict__ and
                    Path(qualifier_module.__file__).resolve()==qpath and
                    type(loader) is qualifier_module.__dict__.get("LockedSourceLoader"),"locked author loader globals/class differs")
                require(field(loader,"expected_sha")==MODULE_REFS[name][2],"locked author loader SHA differs")
            else:
                require(type(loader) is importlib.machinery.SourceFileLoader and loader.name==name and
                    Path(loader.path).resolve()==self.paths[name],"native source loader differs")
        else:
            require(module.__spec__ is None and module.__loader__ is None,"CPU fixture must have explicit fixture-only loader")
        return module

    def _function(self, module, qualname, function):
        require(type(function) is types.FunctionType and function.__globals__ is module.__dict__ and
            function.__qualname__==qualname and not hasattr(function,"__wrapped__") and
            Path(function.__code__.co_filename).resolve()==self.paths[module.__name__] and
            function.__code__ in self.codes[module.__name__].get(qualname,()),"original loaded code/globals differ: "+qualname)
        return function

    def _class(self, module_name, class_name, owner, methods):
        module=self._module(module_name);klass=module.__dict__.get(class_name)
        require(isinstance(klass,type) and type(owner) is klass and klass.__module__==module_name and
            klass.__qualname__==class_name,"exact original owner class differs: "+class_name)
        for method_name in methods:
            require(method_name not in vars(owner),"original instance method override: "+method_name)
            original=getattr(owner,method_name)
            require(type(original) is types.MethodType and original.__self__ is owner,"original bound method required")
            self._function(module,class_name+"."+method_name,original.__func__)
        return module

    def _verify_owner(self, worker, handler):
        self._check_sources()
        gpu=self._class("vllm.v1.worker.gpu_worker","Worker",worker,("shutdown",))
        state=self._module("vllm.distributed.kv_transfer.kv_transfer_state")
        for name in ("get_kv_transfer_group","ensure_kv_transfer_shutdown"):
            self._function(state,name,state.__dict__.get(name))
        require(gpu.__dict__.get("get_kv_transfer_group") is state.get_kv_transfer_group and
            gpu.__dict__.get("ensure_kv_transfer_shutdown") is state.ensure_kv_transfer_shutdown,"worker KV singleton functions differ")
        connector=state.__dict__.get("_KV_CONNECTOR_AGENT")
        self._class("vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector","OffloadingConnector",connector,("shutdown",))
        cw=field(connector,"connector_worker")
        self._class("vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker","OffloadingConnectorWorker",cw,("shutdown","_register_handlers"))
        registry=field(cw,"worker")
        rm=self._class("vllm.v1.kv_offload.worker.worker","OffloadingWorker",registry,("shutdown","register_handler","transfer_async","get_finished"))
        hm=self._class("py_kvcache.vllm","NoopSharedStorageOffloadingHandler",handler,("shutdown","transfer_async","get_finished","_check_native_fatal","_future_to_transfer_result"))
        spec=field(cw,"spec")
        self._class("py_kvcache.vllm","PyKvCacheOffloadingSpec",spec,("get_handlers",))
        require(field(spec,"_handler") is handler,"original spec handler differs")
        require(field(spec,"vllm_config") is field(worker,"vllm_config"),"original worker/spec config identity differs")
        owners=field(registry,"handlers");mapping=field(registry,"transfer_type_to_handler")
        require(type(owners) is set and len(owners)==1 and next(iter(owners)) is handler,"sole original registered handler required")
        require(type(mapping) is dict and set(mapping)=={("GPU","SHARED_STORAGE"),("SHARED_STORAGE","GPU")} and
            all(value is handler for value in mapping.values()),"registered transfer direction map differs")
        base=self._module("vllm.v1.kv_offload.base")
        require(hm.GPULoadStoreSpec is base.GPULoadStoreSpec and hm.TransferResult is rm.TransferResult,"source class import aliases differ")
        coordinator=field(handler,"coordinator")
        reactor_module=self._class("py_kvcache.reactor","TransferCoordinator",coordinator,("shutdown","inspect_snapshot","submit_load","submit_store","raise_if_native_fatal"))
        reactor=field(coordinator,"reactor")
        self._class("py_kvcache.reactor","IoReactor",reactor,("shutdown","inspect_snapshot","_capture_owner_snapshot","parent_admission_snapshot"))
        ring=field(reactor,"ring")
        self._class("py_kvcache.linux_aio","LinuxAioRing",ring,("close","snapshot"))
        require(hm.TransferCoordinator is reactor_module.TransferCoordinator,"handler coordinator import differs")
        account=field(reactor,"_prefix_stage_accounting")
        self._class("prefix_io_control.stage_accounting","StageAccounting",account,("snapshot","accepted","completed"))
        require(field(field(reactor,"_prefix_progress"),"run_id")==self.run_id,"actual reactor run differs")
        require(type(field(reactor,"_native_drain_unknown")) is bool and not reactor._native_drain_unknown,"native drain unknown")
        if self.origin=="source_bound_runtime":
            for thread in (field(reactor,"_worker"),field(ring,"_worker")):
                require(type(thread) is threading.Thread and "is_alive" not in vars(thread) and "join" not in vars(thread),"actual original actor Thread required")
                integer(thread.ident,"actual actor identity",1)
            owner_ident=field(account,"owner")
            require(owner_ident is None or type(owner_ident) is int and owner_ident==reactor._worker.ident,"accounting owner differs from original reactor")
            if any(field(account,"stats")[stage]["accepted_ops"] for stage in STAGES):
                require(field(account,"owner")==reactor._worker.ident,"active accounting owner differs from original reactor")
        refs=tuple((name,*ref) for name,ref in MODULE_REFS.items())
        witness=OwnerWitness(self.run_id,self.origin,("GPUWorker","KVTransferSingleton","OffloadingConnector","OffloadingConnectorWorker",
            "OffloadingWorker","NoopSharedStorageOffloadingHandler","TransferCoordinator","IoReactor","LinuxAioRing","StageAccounting"),
            refs,self.origin=="source_bound_runtime")
        self.last_owner=witness
        return witness

    def verify_owner(self,worker,handler):
        if not self.enabled: return None
        return self._observe(lambda:self._verify_owner(worker,handler))

    def _capacity(self):
        require(len(self.transfers)+len(self.results)<self.max_records,"scalar observation capacity exceeded")

    def _copy_transfer(self, handler, args, kwargs, accepted):
        require(type(accepted) is bool,"original transfer return must be explicit bool")
        self._capacity()
        require(len(args)==2 and set(kwargs)<= {"profile_tid","req_id"},"original transfer argument contract")
        job=integer(args[0],"job_id");pair=args[1]
        require(type(pair) is tuple and len(pair)==2,"original transfer spec pair")
        hm=self._module("py_kvcache.vllm");base=self._module("vllm.v1.kv_offload.base")
        src,dst=pair
        store=type(src) is base.GPULoadStoreSpec and type(dst) is hm.SharedStorageLoadStoreSpec
        load=type(src) is hm.SharedStorageLoadStoreSpec and type(dst) is base.GPULoadStoreSpec
        require(store or load,"original exact KV spec classes required")
        gpu=src if store else dst;shared=dst if store else src
        blocks=field(gpu,"block_ids")
        if self.origin=="cpu_fixture": require(type(blocks) is list,"CPU fixture block list")
        else:
            numpy=sys.modules.get("numpy")
            require(type(numpy) is types.ModuleType and type(blocks) is numpy.ndarray and blocks.ndim==1 and
                blocks.size<=4096,"original bounded CPU block array required")
            blocks=blocks.tolist()
        require(type(blocks) is list and len(blocks)<=4096,"bounded original block IDs")
        ids=tuple(integer(value,"GPU block ID") for value in blocks)
        hashes=field(shared,"block_hashes")
        require(type(hashes) is list and len(hashes)<=4096 and all(type(value) is bytes and 0<len(value)<=256 for value in hashes),"bounded original hashes")
        layout=field(field(handler,"coordinator"),"layout")
        self._class("py_kvcache.transfer","ParsedKvLayout",layout,())
        sizes=field(layout,"bytes_per_kernel_block");factor=integer(field(layout,"storage_block_size_factor"),"storage block factor",1)
        require(type(sizes) is list and 0<len(sizes)<=128,"actual KV layout sizes")
        nbytes=factor*sum(integer(size,"KV kernel block bytes",1) for size in sizes)
        row=TransferObservation(job,text(kwargs.get("req_id",""),"request ID",True),"store" if store else "load",ids,
            tuple(hashlib.sha256(value).hexdigest() for value in hashes),nbytes,accepted)
        if accepted:
            require(job not in self._pending and job not in self._completed,"reused accepted job ID lacks a new owner generation")
            self._pending[job]=row
        self.transfers.append(row)

    def observe_transfer(self,worker,handler,*args,**kwargs):
        if not self.enabled: return handler.transfer_async(*args,**kwargs)
        self.verify_owner(worker,handler)
        try: result=handler.transfer_async(*args,**kwargs)
        except BaseException:
            self.invalidate("original_transfer_failed")
            raise
        self._observe(lambda:(self._verify_owner(worker,handler),self._copy_transfer(handler,args,kwargs,result)))
        return result

    def _copy_results(self, values):
        require(type(values) is list and len(values)<=self.max_records,"original bounded result batch")
        module=self._module("vllm.v1.kv_offload.worker.worker")
        copied=[]
        for value in values:
            self._capacity()
            require(type(value) is module.TransferResult,"exact original TransferResult required")
            job=integer(field(value,"job_id"),"result job_id")
            success=field(value,"success");require(type(success) is bool,"explicit original result success")
            size=field(value,"transfer_size");require(size is None or type(size) is int and size>=0,"actual transfer size")
            direction=field(value,"transfer_type")
            require(direction is None or type(direction) is tuple and len(direction)==2 and
                all(type(item) is str for item in direction) and
                direction in (("GPU","SHARED_STORAGE"),("SHARED_STORAGE","GPU")),"actual transfer direction")
            require(job in self._pending and job not in self._completed,"result without uniquely observed accepted task")
            copied.append(TransferResultObservation(job,success,size,direction))
            self._pending.pop(job);self._completed.add(job)
            self.results.append(copied[-1])

    def observe_get_finished(self,worker,handler):
        if not self.enabled: return handler.get_finished()
        self.verify_owner(worker,handler)
        try: result=handler.get_finished()  # Sole original consuming call.
        except BaseException:
            self.invalidate("original_get_finished_failed")
            raise
        self._observe(lambda:(self._verify_owner(worker,handler),self._copy_results(result)))
        return result

    def observe_shutdown(self,worker,handler):
        if not self.enabled: return handler.shutdown()
        self.verify_owner(worker,handler)
        if self._shutdown_seen: self.invalidate("second_shutdown_boundary")
        self._shutdown_seen=True
        try: result=handler.shutdown()  # Only original native release/join/close.
        except BaseException:
            self.invalidate("original_handler_shutdown_failed")
            raise
        def tail():
            witness=self._verify_owner(worker,handler)
            snapshot=handler.coordinator.inspect_snapshot()  # One real post-original copy.
            actual=self.validator._copy_native_fields(handler)
            self.validator._tail_open=True
            try: cpu=self.validator.convert_post_shutdown(snapshot,actual)
            finally: self.validator._tail_open=False
            self._check_sources()
            stages=tuple((name,accepted.ops,accepted.nbytes,completed.ops,completed.nbytes,transferred,failed)
                for name,accepted,completed,transferred,failed in zip(STAGES,cpu.drain.accepted,cpu.drain.completed,
                    cpu.drain.transferred_bytes,cpu.drain.failed_ops))
            self.last_tail=NativeTailObservation(self.run_id,self.origin,witness.source_bound_runtime_observed,
                cpu.captured_ns,stages,tuple((key,actual["aio"][key]) for key in ("accepted","completed","reaped")),
                cpu.remaining_native_owners,not self._pending,all(row[1]>0 and row[2]>0 for row in stages),witness.source_refs)
            self.reason="CPU_SHAPE_ONLY" if self.origin=="cpu_fixture" else "SOURCE_BOUND_NATIVE_TAIL_FACTS_ONLY"
        self._observe(tail)
        return result
