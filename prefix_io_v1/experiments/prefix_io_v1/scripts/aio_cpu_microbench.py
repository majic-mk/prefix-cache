"""Bounded CPU-only warm-file O_DIRECT screening; not a model benchmark."""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import resource
import time

import torch
from py_kvcache.linux_aio import LinuxAioRing, _KernelAio, _Request

class DirectReference:
    """Data-only synchronous *submission* control, not a candidate cache backend."""
    def __init__(self, depth):
        self.kernel=_KernelAio(depth); self.pending=[];self.active={}
        self.calls=0;self.max_ns=0;self.max_inflight=0
    def queue_rw(self, *, user_data,fd,ptr,nbytes,write):
        self.pending.append(_Request(user_data,"write" if write else "read",
                                     fd=fd,ptr=ptr,nbytes=nbytes))
    def submit_pending(self):
        while self.pending:
            start=time.perf_counter_ns()
            n=self.kernel.submit(self.pending)
            self.max_ns=max(self.max_ns,time.perf_counter_ns()-start);self.calls+=1
            if n<=0:raise RuntimeError("direct reference could not submit")
            for op in self.pending[:n]:self.active[op.user_data]=op
            self.pending=self.pending[n:]
            self.max_inflight=max(self.max_inflight,len(self.active))
    def poll_all(self):
        events=self.kernel.poll()
        for token,_ in events:del self.active[token]
        return events
    def snapshot(self):
        return dict(max_kernel_inflight=self.max_inflight,submit_calls=self.calls,
                    max_submit_ns=self.max_ns,outstanding=len(self.active))
    def close(self):
        self.kernel.close()

def pct(values, q):
    return sorted(values)[max(0, math.ceil(len(values)*q)-1)]

def aligned(n, value):
    owner = C.create_string_buffer(n+4096)
    address = (C.addressof(owner)+4095)&~4095
    C.memset(address,value,n)
    return owner,C.c_void_p(address)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    out=Path(args.output);out.mkdir(exist_ok=False)
    rows=[]
    for size in [4096,65536,1048576]:
        for depth in [1,4,16]:
            path=out/f"block-{size}-depth-{depth}"
            path.mkdir()
            fds=[os.open(path/str(i),os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_DIRECT,0o600) for i in range(depth)]
            owners=[];write_ptrs=[];read_ptrs=[]
            for i in range(depth):
                w,pw=aligned(size,i+21);b,pb=aligned(size,0)
                owners.extend([w,b]);write_ptrs.append(pw);read_ptrs.append(pb)
            # Equalize extent allocation before either mode is timed.
            for i, fd in enumerate(fds):
                view=memoryview((C.c_ubyte*size).from_address(write_ptrs[i].value))
                assert os.pwrite(fd,view,0)==size
                os.fsync(fd)
            batches=max(1,math.ceil((8*1024*1024)/(size*depth)))
            try:
                for repeat in range(2):
                    modes=["adapter","direct_reference"] if repeat==0 else ["direct_reference","adapter"]
                    for mode in modes:
                        ring=LinuxAioRing(depth) if mode=="adapter" else DirectReference(depth)
                        try:
                            for write in [True,False]:
                                latency=[];caller_submit=[]
                                start=time.perf_counter();cpu=time.process_time()
                                for batch in range(batches):
                                    stamps={}
                                    for i in range(depth):
                                        token=batch*depth+i
                                        stamps[token]=time.perf_counter_ns()
                                        ring.queue_rw(user_data=token,fd=fds[i],
                                                      ptr=write_ptrs[i] if write else read_ptrs[i],
                                                      nbytes=size,write=write)
                                    a=time.perf_counter_ns();ring.submit_pending()
                                    caller_submit.append((time.perf_counter_ns()-a)/1000)
                                    received=set();deadline=time.monotonic()+10
                                    while len(received)<depth:
                                        completions=ring.poll_all()
                                        now=time.perf_counter_ns()
                                        for token,result in completions:
                                            assert token in stamps and token not in received
                                            assert result==size
                                            received.add(token)
                                            latency.append((now-stamps[token])/1000)
                                        assert time.monotonic()<deadline,"I/O completion timed out"
                                        if len(received)<depth:time.sleep(.00005)
                                elapsed=time.perf_counter()-start;cpu_used=time.process_time()-cpu
                                if not write:
                                    for i in range(depth):
                                        assert C.string_at(write_ptrs[i],size)==C.string_at(read_ptrs[i],size)
                                row=dict(mode=mode,block_bytes=size,depth=depth,repeat=repeat,
                                         operation="write" if write else "read",
                                         logical_bytes=batches*depth*size,operations=batches*depth,
                                         elapsed_seconds=elapsed,cpu_seconds=cpu_used,
                                         cpu_core_equivalents=cpu_used/elapsed,
                                         MiB_per_second=(batches*depth*size)/(1024**2)/elapsed,
                                         latency_us_p50=pct(latency,.5),latency_us_p95=pct(latency,.95),
                                         caller_submit_us_p95=pct(caller_submit,.95),
                                         caller_submit_us_max=max(caller_submit),
                                         ring_snapshot=ring.snapshot())
                                rows.append(row)
                                print(json.dumps({k:row[k] for k in ["mode","block_bytes","depth","repeat","operation","MiB_per_second"]}),flush=True)
                        finally:
                            ring.close()
            finally:
                for fd in fds:os.close(fd)
    assert not torch.cuda.is_initialized()
    result=dict(scope="CPU-only warm-file O_DIRECT; fixed offsets and pre-opened files; includes Python polling; direct_reference only measures data-path submission overhead; no io_uring comparison, GPU or production KV",
                target_bytes_per_case=8*1024*1024,
                total_logical_bytes=sum(x["logical_bytes"] for x in rows),
                preconditioning_write_bytes=sum(s*d for s in [4096,65536,1048576] for d in [1,4,16]),
                gpu_operations=0,model_loaded=False,cuda_initialized=False,
                peak_process_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                rows=rows)
    (out/"results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}))
if __name__=="__main__":main()
