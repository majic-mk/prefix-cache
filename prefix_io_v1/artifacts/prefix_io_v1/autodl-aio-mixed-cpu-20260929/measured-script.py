"""CPU-only author file-store mixed-load screen; no cache/model simulator."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import threading
import time

import numpy as np
import torch
from py_kvcache.fs_config import SharedFileConfig
from py_kvcache.liburing_file import DirectIoFileStore
from py_kvcache.linux_aio import LinuxAioRing

def cpu_guard():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only run requires empty CUDA_VISIBLE_DEVICES")
    assert not torch.cuda.is_initialized()
    def deny(*a, **k):
        raise RuntimeError("CUDA initialization forbidden in CPU mixed-load screen")
    torch.cuda._lazy_init = deny

def percentile(values, q):
    return sorted(values)[max(0, math.ceil(len(values)*q)-1)] if values else None

def buffer(size, byte):
    owner=C.create_string_buffer(size+4096)
    address=(C.addressof(owner)+4095)&~4095
    C.memset(address,byte,size)
    view=np.ctypeslib.as_array((C.c_ubyte*size).from_address(address))
    return owner,view

def run_case(path, size, depth, repeat, *, jobs=None, sync=False, metadata_workers=2):
    path.mkdir()
    store=DirectIoFileStore(SharedFileConfig(str(path),io_backend="linux_aio",
                             iodepth=depth,sync_on_store=sync),payload_size=size)
    jobs=jobs or max(depth*2,min(256,math.ceil(16*1024**2/size)))
    # Author reactor gives the backend extra room for open/close completions.
    ring_depth=max(depth+depth+16,16)
    before_fds=len(os.listdir("/proc/self/fd"))
    slots=[]
    seed_bytes=0
    for slot in range(depth):
        owner,view=buffer(store.io_size,slot%200+23)
        seed=path/("seed-"+str(slot))
        fd,temp=store.open_temp_write(str(seed))
        assert os.pwrite(fd,view,0)==store.io_size
        os.fsync(fd)
        store.finish_write(fd=fd,temp_path=temp,final_path=str(seed))
        seed_bytes+=store.io_size
        slots.append(dict(owner=owner,view=view,seed=str(seed),byte=slot%200+23,fd=None))
    ring=LinuxAioRing(ring_depth,metadata_workers=metadata_workers)
    # Timing includes author metadata calls and Python full-byte validation.
    pending={};active={};next_job=0;token=0;completed=0
    elapsed_us=[];stage_us={k:[] for k in ("open","read","write","close","publish")}
    counts={k:0 for k in ("open","read","write","close")}
    empty_submits=0;polls=0;peak_fds=before_fds
    writes=[];start=time.perf_counter();cpu_start=time.process_time()
    deadline=time.monotonic()+30
    def enqueue(kind, lane):
        nonlocal token
        token+=1
        st=slots[lane];t=time.perf_counter_ns()
        if kind=="open":
            st["path_owner"]=store.queue_open_read(ring,user_data=token,path=st["seed"])
        elif kind in ("read","write"):
            fn=store.queue_read if kind=="read" else store.queue_write
            fn(ring,user_data=token,fd=st["fd"],io_array=st["view"])
        else:
            store.queue_close(ring,user_data=token,fd=st["fd"])
            # Ownership of this close belongs to accepted request until completion.
        pending[token]=(kind,lane,t)
        counts[kind]+=1
    try:
        while completed<jobs:
            for lane,st in enumerate(slots):
                if lane in active or next_job>=jobs:
                    continue
                job=next_job;next_job+=1
                active[lane]=(job,time.perf_counter_ns())
                if job%2:
                    C.memset(st["view"].ctypes.data,st["byte"],store.io_size)
                    final=str(path/("store-"+str(job)))
                    st["fd"],st["temp"]=store.open_temp_write(final)
                    st["final"]=final
                    enqueue("write",lane)
                else:
                    C.memset(st["view"].ctypes.data,0,store.io_size)
                    enqueue("open",lane)
            if ring.pending_submit==0:empty_submits+=1
            ring.submit_pending()
            done=ring.poll_all();polls+=1
            for tok,result in done:
                kind,lane,t=pending.pop(tok);st=slots[lane]
                stage_us[kind].append((time.perf_counter_ns()-t)/1000)
                if kind=="open":
                    assert result>=0,(kind,result)
                    st["fd"]=result;enqueue("read",lane)
                elif kind=="read":
                    assert result==store.io_size,(kind,result)
                    assert st["view"].tobytes()==bytes([st["byte"]])*store.io_size
                    enqueue("close",lane)
                elif kind=="write":
                    assert result==store.io_size,(kind,result)
                    a=time.perf_counter_ns()
                    assert store.finish_write(fd=st["fd"],temp_path=st["temp"],
                                              final_path=st["final"])
                    stage_us["publish"].append((time.perf_counter_ns()-a)/1000)
                    st["fd"]=None
                    writes.append((st["final"],st["byte"]))
                else:
                    assert result==0,(kind,result)
                    st["fd"]=None
                if kind in ("write","close"):
                    _,beg=active.pop(lane)
                    elapsed_us.append((time.perf_counter_ns()-beg)/1000);completed+=1
            peak_fds=max(peak_fds,len(os.listdir("/proc/self/fd")))
            if time.monotonic()>deadline:raise TimeoutError((pending,ring.snapshot()))
            if not done:time.sleep(0)  # matches author's pump yield, not a new scheduler
        wall=time.perf_counter()-start;cpu=time.process_time()-cpu_start
        assert not pending and not active
        snapshot=ring.snapshot()
        assert snapshot["outstanding"]==0
        assert snapshot["accepted"]==snapshot["completed"]==snapshot["reaped"]
    finally:
        # Keep all buffers alive until kernel I/O and accepted closes are drained.
        ring.close()
        leftovers=ring.poll_all()
        # Any undelivered successful opens were cancelled/closed by the backend.
        closed_tokens={tok for tok,res in leftovers if pending.get(tok,("",))[0]=="close" and res==0}
        for tok in closed_tokens:slots[pending[tok][1]]["fd"]=None
        for st in slots:
            if st["fd"] is not None:
                os.close(st["fd"]);st["fd"]=None
    # Validate every newly published file outside the timing window.
    for final,byte in writes:
        assert Path(final).read_bytes()==bytes([byte])*store.io_size
    assert not list(path.glob("*.tmp-*"))
    assert len(os.listdir("/proc/self/fd"))==before_fds
    assert not [t for t in threading.enumerate() if t.name.startswith("pykvcache-aio-")]
    assert not torch.cuda.is_initialized()
    return dict(block_bytes=size,io_bytes=store.io_size,concurrency=depth,
                ring_depth=ring_depth,metadata_workers=metadata_workers,repeat=repeat,
                sync_on_store=sync,jobs=jobs,logical_bytes=jobs*store.io_size,
                setup_write_bytes=seed_bytes,verification_read_bytes=len(writes)*store.io_size,
                operation_counts=counts,elapsed_seconds=wall,cpu_seconds=cpu,
                MiB_per_second=jobs*store.io_size/1024**2/wall,
                job_latency_us_p50=percentile(elapsed_us,.5),
                job_latency_us_p95=percentile(elapsed_us,.95),
                stage_us_p95={k:percentile(v,.95) for k,v in stage_us.items()},
                pump_iterations=polls,empty_submit_calls=empty_submits,
                peak_fd_delta=peak_fds-before_fds,fd_leak=0,worker_leak=0,
                snapshot=snapshot)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True)
    ap.add_argument("--repeats",type=int,default=2)
    args=ap.parse_args()
    cpu_guard()
    out=Path(args.output);out.mkdir(exist_ok=False)
    plan_path=Path("artifacts/prefix_io_v1/new-server-04/calibration-preparation/candidate-plan.json")
    plan=json.loads(plan_path.read_text())
    bpt=plan["identity"]["kv_bytes_per_token"]
    assert bpt==57344
    rows=[]
    for rep in range(args.repeats):
        cases=[(s,d) for s in [4096,bpt*16,bpt*64] for d in [1,4,16]]
        if rep%2:cases.reverse()
        for size,depth in cases:
            row=run_case(out/f"r{rep}-s{size}-d{depth}",size,depth,rep)
            rows.append(row)
            print(json.dumps({k:row[k] for k in ("block_bytes","concurrency","repeat",
                  "MiB_per_second","empty_submit_calls")}),flush=True)
    result=dict(scope="CPU real O_DIRECT + author open/publish/close; synthetic contents; 50/50 store/load; warm seeds and new stores; includes full byte validation and Python pump overhead; no GPU/reactor end-to-end or cold-SSD claim",
       gpu_operations=0,cuda_initialized=torch.cuda.is_initialized(),model_loaded=False,
       layout_source=str(plan_path),layout_source_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
       storage_factors=[1,4],factors_are_candidates_not_frozen_runtime=True,
       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       total_logical_bytes=sum(x["logical_bytes"] for x in rows),
       total_setup_write_bytes=sum(x["setup_write_bytes"] for x in rows),
       total_verification_read_bytes=sum(x["verification_read_bytes"] for x in rows),
       peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,rows=rows)
    (out/"results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}))
if __name__=="__main__":main()
