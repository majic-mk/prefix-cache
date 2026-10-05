"""Bounded P4 off/shadow native qualification on the existing author executor.

CPU dry-run/check-launch paths use only the standard library. Actual execution
requires a separate explicit GPU scope record and the existing GPU budget guard.
This new driver reuses the P316 tensor/store/restore/shared qualification sequence;
it adds only new source bindings and the optional P4 shadow bridge.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.machinery
import importlib.abc
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
ART = "artifacts/prefix_io_v1/server08-p4-02-cpu"
NATIVE = "third_party/work/py-kvcache-p4-02-cpu"
CONTROL = "third_party/work/prefix-io-p4-02-cpu/src"
AUTHOR = "third_party/work/vllm-author-p4-02-cpu"
BINARY = "third_party/work/vllm-author-build/vllm"
SCRIPT = "experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py"
SOURCE_TEMPLATE = "experiments/prefix_io_v1/scripts/qualify_simple_stage_gpu_p316.py"
GPU_SCOPE = ART + "/GPU_STAGE_AUTHORIZATION.json"
SOURCE_LOCK = ART + "/gpu-next-day/final-source-lock.json"
STORAGE_RESERVE = 128 * 1024**2
TIME_LIMIT = 180
FLOOR = 8 * 1024**3
BINARY_NAMES = ("_C.abi3.so","_C_stable_libtorch.abi3.so")

def require(ok, message):
    if not ok:
        raise ValueError(message)

def new_json(path, value):
    with path.open("x") as h:
        json.dump(value,h,indent=2,allow_nan=False);h.write("\n")

def safe(root, relative):
    require(type(relative) is str and relative and "\\" not in relative,
            "POSIX project-relative path required")
    p=Path(relative)
    require(not p.is_absolute() and all(c not in ("",".","..") for c in relative.split("/")),
            "path escapes project")
    result=root/p
    cursor=result
    while cursor!=root:
        require(not cursor.is_symlink(),"source/scope symlink rejected")
        cursor=cursor.parent
    return result

def sha_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for raw in iter(lambda:f.read(1024**2),b""):
            h.update(raw)
    return h.hexdigest()

def source_refs(root, relative):
    if relative is None:
        return None
    path=safe(root,relative)
    raw=json.loads(path.read_text())
    require(type(raw) is dict and type(raw.get("files")) is list and raw["files"],"source files lock")
    refs={}
    for row in raw["files"]:
        require(type(row) is dict and set(row)=={"path","bytes","sha256"},"exact source ref")
        p=safe(root,row["path"])
        require(row["path"] not in refs and p.stat().st_size==row["bytes"] and sha_file(p)==row["sha256"],
                "source changed/duplicate")
        refs[row["path"]]=row
    required={NATIVE+"/py_kvcache/reactor.py",NATIVE+"/py_kvcache/vllm.py",
              AUTHOR+"/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
              AUTHOR+"/vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py",
              AUTHOR+"/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py"}
    from prepare_p4_gpu_next_day import required_source_paths
    required.update(required_source_paths(root))
    require(required<=set(refs),"all actual new Python/native/control sources and adapters required")
    require(any(p.startswith(CONTROL+"/prefix_io_control/p4_") for p in refs),
            "actual P4-02 control source refs required")
    for name in BINARY_NAMES:
        require(BINARY+"/"+name in refs,"precise old compiled binary ref required")
    return refs

def preview(root, mode, name, source_lock=None):
    require(mode in ("off","shadow"),"off/shadow G1 scope")
    require(type(name) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}",name),
            "safe append-new run name")
    lock=source_refs(root,source_lock) if source_lock is not None else None
    out=root/"experiments/prefix_io_v1/runs"/name/"details"
    env={"PYTHONPATH":":".join(str(root/p) for p in (CONTROL,NATIVE,AUTHOR,
                                                       "experiments/prefix_io_v1/scripts")),
         "PYTHONDONTWRITEBYTECODE":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}
    child=[".venv/bin/python",SCRIPT,"--mode",mode,"--name",name,"--execute",
           "--scope-record",GPU_SCOPE,"--source-lock",source_lock or SOURCE_LOCK]
    guard=[".venv/bin/python","experiments/prefix_io_v1/scripts/run_gpu_stage.py",
           "--label",name,"--seconds",str(TIME_LIMIT),"--",*child]
    return dict(schema_version=1,status="P4_NATIVE_GPU_DRY_RUN_BLOCKED",mode=mode,name=name,
        new_gpu_runs=0,gpu_initialized=False,gpu_availability_probed=False,budget_reserved_seconds=0,
        child_argv=child,guard_argv=guard,environment_preview=env,
        shell_preview=" ".join(shlex.quote(k+"="+v) for k,v in env.items())+" "+shlex.join(guard),
        source_template=SOURCE_TEMPLATE,source_lock_pending=lock is None,
        source_lock=source_lock,output=str(out),ordinary_policy_controller=False,
        staging_budget_bytes=16*1024**2,staging_iodepth=1,
        new_storage_reserve_bytes=STORAGE_RESERVE,
        seconds_limit=TIME_LIMIT,termination_reserve_seconds=20,
        real_cases_prepared=["66-file store/full parent","age store",
                             "SSD exact restore","two shared-preload consumers"],
        actual_GPU_cases_run=0,zero_ordinary_quota_not_covered_by_off_shadow=True,
        new_executor=False,requires_explicit_scope_restore=True,
        CPU_mock_is_GPU_qualification=False,full_P4_complete=False,effect_verified=False,
        new_author_source=AUTHOR,compiled_binary_fallback=BINARY,
        binary_compatibility_GPU_unverified=True)

def execution_gates(root,args):
    # All scope/source/guard/storage checks precede any torch/vLLM/native import.
    scope_path=safe(root,args.scope_record)
    if not scope_path.is_file():
        return None,"BLOCKED_CPU_ONLY_NO_GPU_SCOPE_RESTORE"
    scope=json.loads(scope_path.read_text())
    require(scope.get("status")=="USER_AUTHORIZED_P4_GPU_NATIVE_QUALIFICATION",
            "explicit new native GPU scope required")
    require(type(scope.get("schema_version")) is int and scope["schema_version"]==1,
            "GPU scope schema")
    for key in ("allow_gpu_initialization","allow_gpu_runs"):
        require(type(scope.get(key)) is bool and scope[key] is True,"explicit GPU scope boolean")
    require(scope.get("allowed_modes")==["off","shadow"],"bounded native G1 modes")
    require(scope.get("new_executor") is False and scope.get("allow_model_downloads") is False,
            "native qualification/download scope")
    permission_path=safe(root,"experiments/prefix_io_v1/configs/permissions.yaml")
    raw=permission_path.read_bytes()
    require(scope.get("base_permissions")==dict(
        path="experiments/prefix_io_v1/configs/permissions.yaml",bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest()),"scope/permissions binding")
    from prepare_p4_gpu_next_day import permission_fields
    permit=permission_fields(raw.decode())
    require(permit["allow_gpu_runs"] and scope.get("gpu_uuid")==permit["approved_gpu_ids"][0],
            "same approved GPU scope")
    require(scope.get("source_lock")==args.source_lock,"scope binds final source-lock path")
    refs=source_refs(root,args.source_lock)
    require(scope.get("source_lock_sha256")==sha_file(safe(root,args.source_lock)),
            "scope binds final source-lock bytes")
    ledger=json.loads(safe(root,"experiments/prefix_io_v1/gpu-budget-ledger.json").read_text())
    active=ledger.get("active_reservation")
    require(type(active) is dict and active.get("label")==args.name and
            active.get("gpu_uuid")==scope["gpu_uuid"] and active.get("session_id")==os.getsid(0),
            "actual existing guard reservation/session required")
    require(active.get("seconds_limit")==TIME_LIMIT and
            active.get("reserved_seconds")==TIME_LIMIT+20,"bounded guard time reserve")
    require(ledger["gpu_wall_seconds"]+active["reserved_seconds"]<=permit["max_gpu_hours"]*3600,
            "GPU budget no longer fits")
    expected=[".venv/bin/python",SCRIPT,"--mode",args.mode,"--name",args.name,"--execute",
              "--scope-record",args.scope_record,"--source-lock",args.source_lock]
    require(active.get("command")==expected,"exact guarded child argv required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES")==scope["gpu_uuid"],
            "actual guard UUID environment")
    for key in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"):
        require(os.environ.get(key)=="1","offline guarded native qualification")
    out=root/"experiments/prefix_io_v1/runs"/args.name/"details"
    require(not out.exists(),"append-new native output")
    return dict(scope=scope,source_refs=refs,out=out),"READY_FOR_AUTHORIZED_NATIVE_GPU"

class LockedSourceLoader(importlib.machinery.SourceFileLoader):
    """Read individually pinned source bytes directly; do not execute cached pyc."""
    def __init__(self,fullname,path,expected_sha):
        super().__init__(fullname,path);self.expected_sha=expected_sha
    def get_code(self,fullname):
        raw=Path(self.path).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==self.expected_sha,"changed author Python source")
        return compile(raw,self.path,"exec",dont_inherit=True)

class BoundAuthorFinder(importlib.abc.MetaPathFinder):
    """Select new Python sources; permit only individually locked old binaries."""
    def __init__(self,root,refs):
        self.root=root;self.refs=refs
    def find_spec(self,fullname,path=None,target=None):
        if fullname!="vllm" and not fullname.startswith("vllm."):
            return None
        paths=path if path is not None else [str(self.root/AUTHOR)]
        spec=importlib.machinery.PathFinder.find_spec(fullname,paths)
        if spec is None:
            raise ModuleNotFoundError(fullname+" not present in locked author/binary paths",name=fullname)
        if spec.origin is None:
            require(all(Path(p).resolve().is_relative_to(self.root/AUTHOR)
                        for p in spec.submodule_search_locations or ()),
                    "old author namespace fallback forbidden")
            return spec
        require(spec.origin not in ("built-in","frozen"),"unexpected vLLM builtin source")
        origin=Path(spec.origin).resolve()
        if origin.suffix in (".py",".pyc"):
            require(origin.suffix==".py" and origin.is_relative_to(self.root/AUTHOR),
                    "old vLLM Python/bytecode fallback forbidden")
            rel=origin.relative_to(self.root).as_posix()
            require(rel in self.refs and sha_file(origin)==self.refs[rel]["sha256"],
                    "unlocked author Python source")
            spec.loader=LockedSourceLoader(fullname,str(origin),self.refs[rel]["sha256"])
        elif origin.suffix==".so":
            rel=origin.relative_to(self.root).as_posix()
            require(rel in self.refs and rel.startswith(BINARY+"/") and
                    sha_file(origin)==self.refs[rel]["sha256"],"unlocked binary fallback")
        return spec

def normalize_uuid(value):
    if type(value) is bytes:
        require(len(value)==16,"GPU UUID byte length")
        import uuid
        value=str(uuid.UUID(bytes=value))
    return str(value).removeprefix("GPU-").lower()

def qualify(out,mode,root,refs):
    # This is the original native execution sequence, never a replacement executor.
    finder=BoundAuthorFinder(root,refs);sys.meta_path.insert(0,finder)
    import vllm
    require(Path(vllm.__file__).resolve().is_relative_to(root/AUTHOR),"new author Python root")
    vllm.__path__.append(str(root/BINARY))
    import torch
    import py_kvcache.reactor as native
    from py_kvcache.reactor import TransferCoordinator
    from py_kvcache.vllm import NoopSharedStorageOffloadingHandler,SharedStorageLoadStoreSpec
    from py_kvcache.fs_config import SharedFileConfig
    from py_kvcache.file_mapper import FileMapper
    from py_kvcache.transfer import ParsedKvLayout
    from prefix_io_control.stage_accounting import StageAccounting
    from prefix_io_control.p4_bridge import make_native_bridge
    from prefix_io_control.p4_types import P4Config
    import prefix_io_control.p4_bridge as control
    from vllm.v1.kv_offload.base import CanonicalKVCaches,CanonicalKVCacheTensor,CanonicalKVCacheRef,GPULoadStoreSpec
    require(Path(native.__file__).resolve().is_relative_to(root/NATIVE),"actual new native module")
    require(Path(control.__file__).resolve().is_relative_to(root/CONTROL),"actual new P4 control module")
    require(torch.cuda.device_count()==1,"one approved device")
    actual=str(torch.cuda.get_device_properties(0).uuid)
    expected=os.environ["CUDA_VISIBLE_DEVICES"]
    require(normalize_uuid(torch.cuda.get_device_properties(0).uuid)==normalize_uuid(expected),
            "actual CUDA device UUID differs")
    torch.manual_seed(316)
    tensors=[torch.randint(-128,128,(132,32768),device="cuda",dtype=torch.int8) for _ in range(4)]
    original=[t[:66].cpu().clone() for t in tensors]
    caches=CanonicalKVCaches([CanonicalKVCacheTensor(t,page_size_bytes=32768) for t in tensors],
        [[CanonicalKVCacheRef(tensor_idx=i,page_size_bytes=32768) for i in range(4)]])
    layout=ParsedKvLayout.from_canonical_kv_caches(gpu_block_size=16,storage_block_size=16,kv_caches=caches)
    cases=[];handlers=[]
    def create(phase):
        path=out/mode/"files";path.mkdir(parents=True,exist_ok=True)
        mapper=FileMapper(root_dir=str(path),model_name="p402-native",gpu_block_size=16,
            gpu_blocks_per_file=1,tp_size=1,pp_size=1,pcp_size=1,rank=0,dtype="bfloat16")
        bridge=make_native_bridge(mode+"-"+phase,P4Config(mode,200_000_000,200_000_000))
        coordinator=TransferCoordinator(config=SharedFileConfig(root_dir=str(path),iodepth=1,
            staging_mem=16/1024,enable_preload=True,preload_share_staging=True,io_backend="linux_aio"),
            file_mapper=mapper,layout=layout,storage_block_tokens=16,
            progress_run_id=mode+"-"+phase,max_accepted_parents=2,
            stage_accounting=StageAccounting(),p4_bridge=bridge)
        h=NoopSharedStorageOffloadingHandler(coordinator=coordinator);handlers.append(h)
        require(coordinator.reactor.staging_buffer.is_pinned(),"actual pinned staging")
        require(coordinator.reactor._prefix_dispatch_controller is None,"off/shadow changed ordinary allowance")
        return h,mapper
    def close(h,phase):
        h.shutdown();r=h.coordinator.reactor
        require(not r._worker.is_alive() and not r._active and not r._inflight and not r._pending_copies,
                "actual native owners not drained")
        aio=r.ring.snapshot();a=r._prefix_stage_accounting.snapshot()
        require(aio["closed"] and aio["drained"] and aio["outstanding"]==0,"actual Linux-AIO drain")
        require(a["valid"] and a["outstanding_records"]==0,"stage ledger drain")
        require(all(x["failed_ops"]==0 and x["inflight_bytes"]==0 and
                    x["accepted_bytes"]==x["transferred_bytes"] for x in a["stages"].values()),
                "actual physical stage balance")
        admission=r.parent_admission_snapshot()
        require(admission["accepted_parents"]==0 and admission["peak_accepted_parents"]<=2,
                "common admission/full parent retirement")
        require(r.actual_staging_bytes<=r.staging_budget_bytes,"actual staging budget")
        bridge=r._prefix_p4_bridge
        view=bridge.snapshot(native_shutdown=True) if bridge is not None else None
        require(view is None or view["valid"],"optional P4 shadow fault")
        require(view is None or view["gpu_release_credit"] is None,"unsupported GPU release credit")
        row=dict(mode=mode,phase=phase,accounting=a,admission=admission,aio=aio,p4=view,
                 actual_staging_bytes=r.actual_staging_bytes,staging_budget_bytes=r.staging_budget_bytes)
        cases.append(row);return row
    try:
        hashes=[hashlib.sha256((mode+"-"+str(i)).encode()).digest() for i in range(66)]
        h,mapper=create("store")
        for jid,first,count in ((1,0,64),(2,64,1),(3,65,1)):
            disk=SharedStorageLoadStoreSpec(hashes[first:first+count])
            require(h.transfer_async(jid,(GPULoadStoreSpec(list(range(first,first+count)),[count],[0]),disk),
                                     req_id=str(jid)),"native store rejected")
        futures=[h._active[j][0] for j in (1,2,3)];h.wait({1,2,3})
        require([f.result() for f in futures]==[64*layout.storage_block_bytes,
                layout.storage_block_bytes,layout.storage_block_bytes],"store full parent byte result")
        close(h,"store")
        h,_=create("age_store")
        require(h.transfer_async(6,(GPULoadStoreSpec([0],[1],[0]),
            SharedStorageLoadStoreSpec([hashlib.sha256((mode+"-age").encode()).digest()])),req_id="age"),
            "age store rejected")
        require(h._active[6][0].result(timeout=5)==layout.storage_block_bytes,"independent native progress")
        close(h,"age_store")
        require(all(Path(mapper.get_file_name(x)).stat().st_size==layout.storage_block_bytes for x in hashes),
                "original file publish")
        for phase in ("restore","shared"):
            for t in tensors:t.zero_()
            torch.cuda.synchronize()
            h,_=create(phase);disk=SharedStorageLoadStoreSpec(hashes)
            if phase=="restore":
                require(h.transfer_async(4,(disk,GPULoadStoreSpec(list(range(66)),[66],[0])),req_id=phase),
                        "native restore rejected")
                futures=[h._active[4][0]];h.wait({4})
            else:
                for name in ("A","B"):require(h.preload_async(name,disk,req_id=name),"original preload")
                for jid,name,offset in ((4,"A",0),(5,"B",66)):
                    require(h.load_from_preload_async(jid,name,disk,
                        GPULoadStoreSpec(list(range(offset,offset+66)),[66],[0]),req_id=name),"shared consumer")
                futures=[h._active[j][0] for j in (4,5)];h.wait({4,5})
            require(all(f.result()==66*layout.storage_block_bytes for f in futures),"full restore parent")
            torch.cuda.synchronize()
            require(all(torch.equal(t[:66].cpu(),s) for t,s in zip(tensors,original)),"exact original KV")
            if phase=="shared":
                require(all(torch.equal(t[66:].cpu(),s) for t,s in zip(tensors,original)),"exact shared destination")
            row=close(h,phase)
            require(row["accounting"]["stages"]["ssd_read"]["accepted_bytes"]==66*layout.storage_block_bytes,
                    "actual shared SSD read bytes")
            require(row["accounting"]["stages"]["h2d"]["accepted_bytes"]==
                    (2 if phase=="shared" else 1)*66*layout.storage_block_bytes,"actual original fusion bytes")
        return dict(status="PASS_REAL_GPU_P4_NATIVE_"+mode.upper(),cases=cases,exact_content=True,
            real_cuda=True,real_linux_aio=True,gpu_uuid=expected,modules=dict(
                native=str(native.__file__),control=str(control.__file__),author=str(vllm.__file__)),
            existing_author_executor=True,new_executor=False,source_template=SOURCE_TEMPLATE,
            full_P4_complete=False,effect_verified=False,model_loaded=False,
            zero_ordinary_quota_not_covered_by_off_shadow=True)
    finally:
        for h in handlers:
            if h.coordinator.reactor._worker.is_alive():h.shutdown()
        sys.meta_path.remove(finder)

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",type=Path,default=ROOT)
    ap.add_argument("--mode",choices=("off","shadow"),required=True)
    ap.add_argument("--name",required=True)
    ap.add_argument("--scope-record",default=GPU_SCOPE)
    ap.add_argument("--source-lock")
    actions=ap.add_mutually_exclusive_group()
    actions.add_argument("--dry-run",action="store_true")
    actions.add_argument("--check-launch",action="store_true")
    actions.add_argument("--execute",action="store_true")
    ap.add_argument("--receipt",type=Path)
    args=ap.parse_args(argv);root=args.project.resolve()
    result=preview(root,args.mode,args.name,args.source_lock)
    if args.check_launch or not (args.execute or args.dry_run):
        result["status"]="BLOCKED_CPU_ONLY_NO_GPU_LAUNCH";code=78
    elif args.dry_run:
        code=0
    else:
        gates,reason=execution_gates(root,args)
        if gates is None:
            result["status"]=reason;code=78
        else:
            from experiment_storage import preflight
            contract=preflight(gates["out"],STORAGE_RESERVE,project=root)
            gates["out"].mkdir(parents=True,exist_ok=False)
            start=time.monotonic()
            try:
                result=qualify(gates["out"],args.mode,root,gates["source_refs"])
                code=0
            except BaseException as e:
                result=dict(status="FAIL_REAL_GPU_P4_NATIVE",error=repr(e),
                    traceback=traceback.format_exc(),effect_verified=False,full_P4_complete=False)
                code=1
            torch_module=sys.modules.get("torch")
            initialized=bool(torch_module is not None and torch_module.cuda.is_initialized())
            result.update(seconds=time.monotonic()-start,storage_preflight=contract,
                          new_gpu_runs=1,gpu_initialization_attempted=True,gpu_initialized=initialized)
            source_refs(root,args.source_lock)
            new_json(gates["out"]/"result.json",result)
    if args.receipt:
        relative=args.receipt.relative_to(root).as_posix() if args.receipt.is_absolute() else args.receipt.as_posix()
        require(relative.startswith(ART+"/gpu-next-day/"),"bounded CPU receipt path")
        destination=safe(root,relative);destination.parent.mkdir(parents=True,exist_ok=True)
        new_json(destination,result)
    print(json.dumps(result,indent=2))
    return code

if __name__=="__main__":
    raise SystemExit(main())
