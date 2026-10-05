"""Owned P3 calibration probe: one native I/O handler, original worker execution.
All tensors belong to this probe; no production KV page or allocator is accessed.
Imports are CUDA-free until install, and release follows real native shutdown.
"""
from __future__ import annotations
import copy, hashlib, time
from pathlib import Path
from prefix_io_control.stage_accounting import StageAccounting, STAGES

def require(ok, message):
    if not ok: raise RuntimeError(message)

def is_pure_decode(scheduler_output, request_ids):
    counts=getattr(scheduler_output,"num_scheduled_tokens",None)
    new=getattr(scheduler_output,"scheduled_new_reqs",None)
    return isinstance(counts,dict) and set(counts)==set(request_ids) and bool(counts) and all(
        type(n) is int and n==1 for n in counts.values()) and new is not None and not new

def bind_native_request(scheduler_output, frontend_id, prompt_token_ids):
    new=getattr(scheduler_output,"scheduled_new_reqs",None)
    require(type(new) is list and len(new)==1,"exactly one new native request must bind")
    row=new[0];rid=getattr(row,"req_id",None)
    suffix=rid[len(frontend_id)+1:] if type(rid) is str and rid.startswith(frontend_id+"-") else ""
    require(type(rid) is str and len(suffix)==8 and all(c in "0123456789abcdef" for c in suffix),
            "native randomized request ID does not match armed frontend request")
    require(getattr(row,"prompt_token_ids",None)==prompt_token_ids,"native prefix differs from frozen window")
    counts=getattr(scheduler_output,"num_scheduled_tokens",None)
    require(type(counts) is dict and set(counts)=={rid},"unexpected native scheduler cohort")
    return rid

def expected_stage_bytes(anchor, units, pulses, quantum=917504):
    n=units*pulses*quantum
    return {s:(n if s in {
        "none":(), "warm_h2d":("h2d",), "d2h_write":("d2h","ssd_write"),
        "cold_ssd_h2d":("ssd_read","h2d"), "joint":STAGES
    }[anchor] else 0) for s in STAGES}

class TimedAccounting(StageAccounting):
    def __init__(self):
        super().__init__(4096);self.starts={};self.spans=[]
    def accepted(self,stage,key,nbytes):
        super().accepted(stage,key,nbytes)
        if self.valid:self.starts[(stage,key)]=(time.monotonic_ns(),nbytes)
    def completed(self,stage,key,result=None):
        super().completed(stage,key,result)
        if self.valid:
            started,nbytes=self.starts.pop((stage,key))
            require(len(self.spans)<4096,"bounded stage spans exhausted")
            self.spans.append(dict(stage=stage,start_ns=started,end_ns=time.monotonic_ns(),
                bytes=nbytes,result=nbytes if result is None else result))
    def export(self,include_spans=False):
        out=dict(accounting=self.snapshot(),span_count=len(self.spans))
        if include_spans:out["stage_spans"]=[dict(x) for x in self.spans]
        return out

class ProbeState:
    def __init__(self,worker,args):
        from run_decode_interference_calibration import validate_spec, check_inputs, TOTAL_KV, STAGING
        self.spec=validate_spec(args["spec"]);check_inputs(self.spec)
        self.worker=worker;self.out=Path(args["output"]).resolve()
        from native_gpu_prefix_smoke import ROOT
        require(self.out.is_relative_to(ROOT/"experiments/prefix_io_v1/runs")
                and self.out.is_dir(),"unsafe owned output")
        import torch
        from py_kvcache import reactor
        require(Path(reactor.__file__).resolve().is_relative_to(ROOT/self.spec["native_worktree"]),
                "native import differs from frozen worktree")
        require(torch.cuda.device_count()==1,"single GPU required")
        runner=worker.model_runner;cfg=runner.kv_cache_config
        require(len(cfg.kv_cache_groups)==1 and len(runner.kv_caches)==28,"actual model layout unsupported")
        unique={}
        for t in runner.kv_caches:
            require(t.is_cuda and t.dtype==torch.bfloat16,"actual model KV dtype/device")
            storage=t.untyped_storage();unique[storage.data_ptr()]=storage.nbytes()
        self.model_bytes=sum(unique.values())
        require(self.model_bytes==sum(t.size for t in cfg.kv_cache_tensors)
            and 0<self.model_bytes<=self.spec["engine"]["kv_cache_memory_bytes"],"actual model KV budget")
        require(worker.vllm_config.kv_transfer_config is None,"model external connector must be disabled")
        self.tensors=[torch.zeros((32,16384),dtype=torch.bfloat16,device="cuda") for i in range(28)]
        for i,t in enumerate(self.tensors):
            for page in range(16):t[page].fill_((i+1)/32+page/1024)
        owned=sum(t.untyped_storage().nbytes() for t in self.tensors)
        require(owned==self.spec["owned"]["owned_bytes"] and self.model_bytes+owned<=TOTAL_KV,
                "actual combined KV exceeds 2 GiB")
        require(not set(unique).intersection(t.untyped_storage().data_ptr() for t in self.tensors),
                "owned and production storage overlaps")
        from vllm.v1.kv_offload.base import CanonicalKVCaches,CanonicalKVCacheTensor,CanonicalKVCacheRef
        from py_kvcache.transfer import ParsedKvLayout
        canonical=CanonicalKVCaches(
            [CanonicalKVCacheTensor(t.view(torch.int8),32768) for t in self.tensors],
            [[CanonicalKVCacheRef(i,32768) for i in range(28)]])
        self.layout=ParsedKvLayout.from_canonical_kv_caches(gpu_block_size=16,
            storage_block_size=16,kv_caches=canonical)
        require(self.layout.storage_block_bytes==917504 and
                self.layout.bytes_per_kernel_block==[32768]*28,"native storage quantum differs")
        # Setup-only completion; no CUDA synchronization in a measurement window.
        torch.cuda.current_stream().synchronize()
        self.allocation=dict(actual_model_kv_bytes=self.model_bytes,actual_owned_bytes=owned,
            combined_gpu_kv_bytes=self.model_bytes+owned,combined_budget_bytes=TOTAL_KV,
            staging_budget_bytes=STAGING,canonical_operand_count=28,canonical_page_bytes=32768,
            storage_unit_bytes=917504,production_kv_touched=False,
            engine_conditional=True,active_decode_requests=1)
        self.original_execute=worker.execute_model;self.handler=None;self.active=False
        self.jobs=[];self.next_job=0;self.finished_windows=0;self.closed_handlers=[]
        def execute(scheduler_output,*a,**kw):
            if self.active and not self.request_ids:
                self.request_ids={bind_native_request(scheduler_output,self.frontend_id,self.spec["prompt_token_ids"])}
            pure=self.active and is_pure_decode(scheduler_output,self.request_ids)
            if pure:
                self.decode_ordinal+=1
                p=self.spec["pulse"]
                offset=self.decode_ordinal-p["start_decode_step"]
                if offset>=0 and offset%p["interval_decode_steps"]==0 and self.opportunities<p["max_pulses"]:
                    self.opportunities+=1;self.maybe_issue()
            start=time.monotonic_ns()
            try:return self.original_execute(scheduler_output,*a,**kw)
            finally:
                if pure:
                    self.decode_steps.append(dict(ordinal=self.decode_ordinal,
                        start_ns=start,end_ns=time.monotonic_ns()))
        self.execute=execute;worker.execute_model=execute

    def handler_new(self,folder):
        from py_kvcache.file_mapper import FileMapper
        from py_kvcache.fs_config import SharedFileConfig
        from py_kvcache.reactor import TransferCoordinator
        from py_kvcache.vllm import NoopSharedStorageOffloadingHandler
        self.accounting=TimedAccounting();self.latest=None;self.sink_error=None;self.next_snapshot_ns=0
        io=dict(self.spec["io"])
        cfg=SharedFileConfig(root_dir=str(folder),**io)
        self.mapper=FileMapper(root_dir=str(folder),model_name="owned-synthetic",
            gpu_block_size=16,gpu_blocks_per_file=1,tp_size=1,pp_size=1,pcp_size=1,rank=0,dtype="bfloat16")
        def sink(r):
            try:
                now=time.monotonic_ns()
                if now<self.next_snapshot_ns:return
                self.next_snapshot_ns=now+2_000_000
                snap=r.ring.snapshot()
                zero=(r._incoming.empty() and not r._active and not r._inflight and
                      not r._pending_copies and not r._copy_ready and r._data_inflight==0 and
                      r._open_inflight==0 and not r._ready_fds_load and not r._ready_fds_preload and
                      not r._preload_pending and r._preload_inflight_total==0 and
                      not r._preload_waiters and snap["outstanding"]==0 and
                      snap["pending"]==0 and snap["ready"]==0 and snap["unreaped"]==0 and
                      not snap["fatal"] and self.accounting.valid and
                      not self.accounting.records)
                self.latest=dict(at_ns=now,idle=zero,stage=self.accounting.export(),aio=snap,
                    observation_failures=r._observation_failures,staging_bytes=r.actual_staging_bytes)
            except BaseException as exc:
                self.sink_error=repr(exc);raise
        coordinator=TransferCoordinator(config=cfg,file_mapper=self.mapper,layout=self.layout,
            storage_block_tokens=16,progress_run_id=self.spec["session_id"]+"-w"+str(self.index),
            observation_sink=sink,stage_accounting=self.accounting)
        self.handler=NoopSharedStorageOffloadingHandler(coordinator=coordinator)
        require(coordinator.reactor.actual_staging_bytes<=1024**3,"native staging budget")
        require(coordinator.reactor.staging_buffer.is_pinned(),"native staging must be pinned")
        self.jobs=[];self.last_submit_ns=time.monotonic_ns()
        return self.handler

    def idle(self):
        require(self.sink_error is None,"owner observation failed: "+str(self.sink_error))
        r=self.handler.coordinator.reactor
        r.raise_if_native_fatal() if hasattr(r,"raise_if_native_fatal") else None
        if any(not j["future"].done() for j in self.jobs):return False
        for j in self.jobs:
            require(j["future"].result()==j["expected_bytes"],"native parent transfer mismatch")
        snap=self.latest
        return snap is not None and snap["at_ns"]>self.last_submit_ns and snap["idle"]

    def wait_idle(self):
        started=time.monotonic()
        self.handler.wait({j["job_id"] for j in self.jobs})
        while not self.idle():
            require(time.monotonic()-started<30,"native drain observation deadline")
            time.sleep(.001)
        require(self.latest["observation_failures"]==0,"native observer disabled")
        return copy.deepcopy(self.latest)

    def hashes(self,label,pulse,units):
        return [hashlib.sha256((self.spec["session_id"]+"/"+str(self.index)+"/"+label+
            "/"+str(pulse)+"/"+str(i)).encode()).digest() for i in range(units)]

    def transfer(self,direction,hashes,phase):
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from py_kvcache.vllm import SharedStorageLoadStoreSpec
        n=len(hashes);require(1<=n<=16,"parent unit limit")
        gpu=GPULoadStoreSpec(list(range(0 if direction=="store" else 16,
                                       n if direction=="store" else 16+n)),[n],[0])
        disk=SharedStorageLoadStoreSpec(hashes)
        jid=self.next_job;self.next_job+=1
        require(self.handler.transfer_async(jid,(gpu,disk) if direction=="store" else (disk,gpu),
            req_id=self.spec["session_id"],profile_tid="owned-"+phase),"original handler rejected")
        future=self.handler._active[jid][0]
        self.last_submit_ns=time.monotonic_ns()
        self.jobs.append(dict(job_id=jid,direction=direction,phase=phase,future=future,
            expected_bytes=n*917504,hashes=[h.hex() for h in hashes]))
        return jid

    def collect(self):
        results=self.handler.get_finished()
        require(all(r.success for r in results),"native transfer result failed")
        self.jobs=[]

    def shutdown_handler(self):
        h=self.handler
        if h is None:return None
        wait_error=None
        try:self.wait_idle()
        except BaseException as exc:wait_error=exc
        h.shutdown()
        r=h.coordinator.reactor;snap=r.ring.snapshot()
        require(not r._worker.is_alive() and snap["closed"] and snap["drained"] and
            snap["outstanding"]==0 and not r._pending_copies and not r._active
            and not r._inflight,"native shutdown/drain unproven")
        evidence=dict(worker_joined=True,aio=snap,actual_staging_bytes=r.actual_staging_bytes,
            observation_failures=r._observation_failures,wait_error=None if wait_error is None else repr(wait_error))
        self.closed_handlers.append(evidence);self.handler=None;self.jobs=[]
        if wait_error is not None:raise wait_error
        return evidence

    def verify_target(self,units):
        import torch
        # Full native idle established first. This setup/tail check is outside timed decode.
        require(all(torch.equal(t[:units],t[16:16+units]) for t in self.tensors),
                "owned native round trip byte mismatch")
        return True

    def prepare(self,args):
        require(not self.active and self.handler is None,"previous window not fully released")
        self.index=args["index"];self.anchor=args["anchor"];self.units=args["units"]
        require(type(self.index) is int and self.index==self.finished_windows,"window order")
        require(self.anchor in ("none","warm_h2d","d2h_write","cold_ssd_h2d","joint")
            and type(self.units) is int and self.units in (1,8),"outside sparse domain")
        directory=self.out/"windows"/("w"+str(self.index))
        directory.mkdir(parents=True,exist_ok=False);self.directory=directory
        self.handler_new(directory/"files")
        sets=12 if self.anchor in ("cold_ssd_h2d","joint") else 1
        self.read_hashes=[self.hashes("read",p,self.units) for p in range(sets)]
        for hashes in self.read_hashes:
            self.transfer("store",hashes,"setup-store");self.wait_idle();self.collect()
            require(all(Path(self.mapper.get_file_name(h)).stat().st_size==917504 for h in hashes),
                    "published fixture size differs")
        self.setup_store=self.latest["stage"]
        self.setup_shutdown=self.shutdown_handler()
        self.handler_new(directory/"files")
        if self.anchor=="warm_h2d":
            self.transfer("load",self.read_hashes[0],"setup-warm")
            self.wait_idle();self.verify_target(self.units);self.collect()
        self.setup_read=copy.deepcopy(self.latest["stage"]) if self.latest else None
        self.wait_idle()
        self.prepared=True
        return dict(index=self.index,setup_store=self.setup_store,setup_read=self.setup_read,
            setup_handler_shutdown=self.setup_shutdown,fixture_sets=sets,
            cache_state="original retained lru SSD load" if self.anchor=="warm_h2d" else "fresh original coordinator")

    def begin(self,args):
        require(self.prepared and not self.active and args["index"]==self.index,"begin without setup")
        require(args["anchor"] in (self.anchor,"none") and args["units"]==self.units,"anchor mismatch")
        require(args["frontend_request_ids"]==["cal-w"+str(self.index)+"-0"],
            "single frozen frontend request required")
        self.baseline=self.wait_idle()["stage"]
        self.frontend_id=args["frontend_request_ids"][0];self.request_ids=set();self.mode=args["anchor"]
        self.decode_steps=[];self.decode_ordinal=0;self.opportunities=0;self.issued=0
        self.skipped=0;self.pulses=[];self.native_parents=[];self.active=True;self.prepared=False
        return dict(armed=True,mode=self.mode,schedule=self.spec["pulse"])

    def maybe_issue(self):
        if self.mode=="none":return
        if not self.idle():
            self.skipped+=1;return
        self.collect()
        p=self.issued
        pulse=dict(pulse=p,decode_ordinal=self.decode_ordinal,job_ids=[],hashes={})
        if self.mode in ("d2h_write","joint"):
            hashes=self.hashes("measurement-write",p,self.units)
            pulse["job_ids"].append(self.transfer("store",hashes,"measurement"))
            pulse["hashes"]["write"]=[h.hex() for h in hashes]
        if self.mode in ("warm_h2d","cold_ssd_h2d","joint"):
            hashes=self.read_hashes[0 if self.mode=="warm_h2d" else p]
            pulse["job_ids"].append(self.transfer("load",hashes,"measurement"))
            pulse["hashes"]["read"]=[h.hex() for h in hashes]
        require(not set(pulse["hashes"].get("read",())).intersection(pulse["hashes"].get("write",())),
                "joint source/destination hash collision")
        self.native_parents.extend({k:v for k,v in j.items() if k!="future"} for j in self.jobs)
        self.pulses.append(pulse);self.issued+=1

    def end(self):
        require(self.active,"end outside measurement");self.active=False
        snap=self.wait_idle();export=snap["stage"]
        start=self.baseline["span_count"];end=export["span_count"]
        delta={}
        for s in STAGES:
            now=export["accounting"]["stages"][s];old=self.baseline["accounting"]["stages"][s]
            delta[s]={k:now[k]-old[k] for k in ("accepted_ops","accepted_bytes","completed_ops",
                "completed_requested_bytes","transferred_bytes","failed_ops")}
        expected=expected_stage_bytes(self.mode,self.units,self.issued)
        require(export["accounting"]["valid"] and export["accounting"]["outstanding_records"]==0,
                "stage evidence incomplete")
        require(all(delta[s]["accepted_bytes"]==expected[s] and
            delta[s]["completed_requested_bytes"]==expected[s] and
            delta[s]["transferred_bytes"]==expected[s] and delta[s]["failed_ops"]==0
            for s in STAGES),"actual stages differ from sparse anchor")
        require(self.decode_ordinal>=120,"continuous 128-output decode not observed")
        require(self.mode=="none" or self.issued>0,"no measured native I/O submission")
        bytecheck=True
        if self.mode in ("warm_h2d","cold_ssd_h2d","joint"):bytecheck=self.verify_target(self.units)
        self.collect()
        # Restore through the same original load API, after measurement counters freeze.
        # This proves completed write content without direct payload reads.
        if self.mode in ("d2h_write","joint"):
            hashes=[bytes.fromhex(h) for h in self.pulses[-1]["hashes"]["write"]]
            self.transfer("load",hashes,"postmeasurement-write-verify")
            self.wait_idle();bytecheck=self.verify_target(self.units);self.collect()
        closure=self.shutdown_handler();self.finished_windows+=1
        # Native owner is now terminated; reading append-only spans is safe and bounded.
        spans=self.accounting.export(include_spans=True)["stage_spans"][start:end]
        result=dict(index=self.index,anchor=self.mode,units=self.units,decode_steps=self.decode_steps,
            bound_native_request_ids=sorted(self.request_ids),frontend_request_id=self.frontend_id,
            opportunities=self.opportunities,issued_pulses=self.issued,skipped_busy_opportunities=self.skipped,
            pulses=self.pulses,native_parents=self.native_parents,stage_spans=spans,stage_delta=delta,
            expected_stage_bytes=expected,full_native_idle=snap,shutdown=closure,synthetic_roundtrip_exact=bytecheck,
            setup_store=self.setup_store,setup_read=self.setup_read,
            native_wait_in_decode=False,measurement_global_cuda_sync=False,
            observation_cadence_min_ns=2_000_000,observation_scope="bounded owner stats/span_count; spans exported only after native shutdown",
            output_scope="frontend records full original model outputs independently")
        from native_gpu_prefix_smoke import write_new_json
        write_new_json(self.directory/"probe.json",result)
        return result

    def finish(self):
        self.active=False
        if getattr(self.worker,"execute_model") is self.execute:self.worker.execute_model=self.original_execute
        closure=self.shutdown_handler() if self.handler is not None else None
        # Storage remains owned until native shutdown; failed/unknown drain keeps this state alive.
        failed_window_evidence=dict(index=getattr(self,"index",None),mode=getattr(self,"mode",None),
            decode_steps=getattr(self,"decode_steps",[]),pulses=getattr(self,"pulses",[]),
            latest_native_owner_snapshot=self.latest if hasattr(self,"latest") else None)
        self.tensors=[];self.layout=None
        return dict(restored_original_execute=True,windows=self.finished_windows,
            final_handler_shutdown=closure,all_handler_shutdowns=self.closed_handlers,
            last_window_evidence=failed_window_evidence,
            owned_released_after_native_shutdown=True)

def worker_probe(worker,action,args=None):
    if action=="install":
        require(not hasattr(worker,"_owned_decode_interference"),"probe already installed")
        state=ProbeState(worker,args);worker._owned_decode_interference=state
        return state.allocation
    require(hasattr(worker,"_owned_decode_interference"),"probe not installed")
    state=worker._owned_decode_interference
    if action=="prepare":return state.prepare(args)
    if action=="begin":return state.begin(args)
    if action=="end":return state.end()
    if action=="finish":
        result=state.finish();del worker._owned_decode_interference;return result
    raise ValueError("unknown calibration probe action")
