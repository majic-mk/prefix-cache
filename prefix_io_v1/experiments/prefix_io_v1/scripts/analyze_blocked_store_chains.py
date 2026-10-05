"""Offline diagnosis of matched native store fences, never a scheduling policy.

CUDA trace ts is the host enqueue time; dur is CUDA event elapsed time. It
does not place device execution on a common clock or expose device queue order.
"""
import argparse,json,math,statistics
from pathlib import Path
from analyze_flush_diagnostic import analyze

def require(ok,message):
    if not ok:raise ValueError(message)

def finite_number(x):
    return type(x) in (int,float) and math.isfinite(x)

def chain(trace,parent,wait):
    """Only a complete, uniquely matched single-file parent can be decomposed."""
    jid=parent["job_id"];req=parent["req_id"]
    events=trace["traceEvents"]
    def select(name,pid=None):
        return [e for e in events if e.get("ph")=="X" and e.get("name")==name
                and e.get("args",{}).get("job_id")==jid
                and e["args"].get("req_id")==req
                and (pid is None or e.get("pid")==pid)]
    transfers=select("py_kvcache.transfer")
    require(len(transfers)==1,"ambiguous/missing parent transfer")
    transfer=transfers[0];args=transfer["args"]
    require(args.get("direction")=="gpu_to_storage" and args.get("success") is True,"not a completed store")
    require(args.get("num_files")==1 and args.get("num_blocks")==1,"partial/multi-file trace unsupported")
    require(type(args.get("num_bytes")) is int and args["num_bytes"]>0,"invalid transfer bytes")
    copies=select("py_kvcache.cuda_staging",transfer["pid"])
    writes=select("py_kvcache.file_write",transfer["pid"])
    require(len(copies)==len(writes)==1,"incomplete/duplicate stage trace")
    copy=copies[0];write=writes[0]
    require(copy["args"].get("direction")=="gpu_to_storage","wrong copy direction")
    require(copy["args"].get("sample")==0 and copy["args"].get("batch_size")==1,"partial/fused copy unsupported")
    require(write["args"].get("file_index")==0 and write["args"].get("num_files")==1,"partial file trace")
    require(all(e["args"].get("num_bytes")==args["num_bytes"] for e in (copy,write)),"byte mismatch")
    require(type(trace["start_ts_ns"]) is int and trace["start_ts_ns"]>=0,"invalid trace origin")
    require(all(finite_number(e.get(k)) for e in (transfer,copy,write) for k in ("ts","dur")),"nonfinite timing")
    require(all(e["dur"]>=0 for e in (transfer,copy,write)),"negative duration")
    origin=trace["start_ts_ns"]/1e9
    begin=lambda e:origin+e["ts"]/1e6
    end=lambda e:begin(e)+e["dur"]/1e6
    require(begin(transfer)<=begin(copy)<=begin(write)<=end(write)<=end(transfer)+1e-6,"invalid host stage order")
    require(finite_number(wait["start"]) and finite_number(wait["seconds"]) and wait["seconds"]>0,"invalid wait")
    require(begin(transfer)<=wait["start"]<=end(transfer)<=wait["start"]+wait["seconds"]+0.002,"transfer not bounded by wait")
    return dict(job_id=jid,req_id=req,pid=transfer["pid"],bytes=args["num_bytes"],
        parent_age_at_wait_seconds=wait["start"]-begin(transfer),
        copy_host_enqueue_from_wait_seconds=begin(copy)-wait["start"],
        copy_cuda_event_elapsed_seconds=copy["dur"]/1e6,
        write_start_from_wait_seconds=begin(write)-wait["start"],
        write_host_interval_seconds=write["dur"]/1e6,
        transfer_complete_from_wait_seconds=end(transfer)-wait["start"],
        immediately_reusable_bytes=None,
        timing_scope="host enqueue timestamp plus CUDA event duration; no device start/completion timestamp")

def fence(case,trace,records):
    result=dict(qualified=False,reason=None,parents=[],immediately_reusable_bytes=None,
                ready_before_dispatch_known=False,causal_speedup_estimate=None)
    try:
        wait=case["wait"];batch=case["batch"]
        require(case.get("unique_match") is True and batch is not None,"ambiguous flush match")
        require(not wait["ids_truncated"] and not batch["parents_truncated"],"truncated flush")
        parents={}
        for cause in batch["causes"]:
            require(cause["reason"]=="restore_destination" and not cause["parents_truncated"],"unsupported/partial cause")
            for p in cause["parents"]:
                jid=p["job_id"]
                require(jid not in parents or parents[jid]==p,"inconsistent duplicate parent")
                parents[jid]=p
        require(set(parents)==set(wait["ids"])==set(batch["parent_ids"]),"incomplete parent closure")
        for jid,p in sorted(parents.items()):
            require(p["is_store"] is True and p["pending_worker_acks"]==1,"unsupported parent state")
            require(p["source_sample_known"] and not p["source_truncated"],"partial owner witness")
            blocks=p["source_blocks"]
            require(p["source_count"]==len(blocks)==1,"non-single-block witness")
            b=blocks[0]
            require(b["owner_known"] and b["same_generation"] is False
                    and type(b["generation"]) is int and type(b["source_generation"]) is int
                    and b["generation"]!=b["source_generation"],"allocation generation reuse not witnessed")
            require(b["active_refs"]>0 and not b["free_queue_linked"] and not b["is_null"]
                    and not b["parents_truncated"] and jid in b["protecting_jobs"],"not a protected active destination")
            retired=[r for r in records if r["kind"]=="native_parent_retired" and r["job_id"]==jid and r["req_id"]==p["req_id"]]
            require(len(retired)==1 and not retired[0]["source_truncated"],"missing/ambiguous retirement")
            rb=retired[0]["source_blocks"]
            require(len(rb)==1 and rb[0]["block"]==b["block"] and rb[0]["owner_known"]
                    and rb[0]["generation"]==b["generation"] and rb[0]["source_generation"]==b["source_generation"]
                    and rb[0]["active_refs"]>0 and not rb[0]["free_queue_linked"]
                    and not rb[0]["parents_truncated"] and jid not in rb[0]["protecting_jobs"],"retirement witness mismatch")
            row=chain(trace,p,wait)
            row.update(block=b["block"],source_generation=b["source_generation"],
                       destination_generation=b["generation"],retired_active_refs=rb[0]["active_refs"])
            result["parents"].append(row)
        result.update(qualified=True,reason="destination reuse fence, not new free GPU capacity",
            wait_seconds=wait["seconds"],
            before_first_copy_enqueue_seconds=max(0,min(p["copy_host_enqueue_from_wait_seconds"] for p in result["parents"])),
            remaining_after_first_enqueue_seconds=wait["seconds"]-max(0,min(p["copy_host_enqueue_from_wait_seconds"] for p in result["parents"])),
            protected_parent_bytes=sum(p["bytes"] for p in result["parents"]))
    except (ValueError,KeyError,TypeError) as e:
        result["reason"]=str(e)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--plan",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();plan=json.loads(a.plan.read_text())
    require([r["depth"] for r in plan["runs"]]==[8,4,4,8],"expected frozen ABBA screen")
    require(all(r["probe"] for r in plan["runs"]),"all arms need same passive probe")
    r=analyze(a.plan)
    require(r["status"]=="PASSED_BOUNDED_FLUSH_DIAGNOSTIC" and len(r["runs"])==4,"incomplete/numeric failure")
    for row,entry in zip(r["runs"],plan["runs"]):
        folder=Path(entry["command"][entry["command"].index("--output")+1])
        native=json.loads((folder/"result.json").read_text());trace=json.loads((folder/"native-cohort.trace.json").read_text())
        probe=native["probe"]["flush_diagnostic"]
        require(not any(probe[k] for k in ("dropped_records","dropped_jobs","dropped_causes","errors","faulted")),"incomplete observer")
        row["store_fence_chains"]=[fence(case,trace,probe["records"]) for case in row["causal_waits"]]
        row["wait_fraction"]=row["counters"]["pending_flush_wait_seconds"]/row["cohort_seconds"]
    r["by_depth"]={str(depth):dict(repeats=2,flush_positive_runs=sum(bool(x["causal_waits"]) for x in r["runs"] if x["depth"]==depth),
        median_cohort_seconds=statistics.median(x["cohort_seconds"] for x in r["runs"] if x["depth"]==depth)) for depth in (4,8)}
    r["repeatability_gate_met"]=all(x["causal_waits"] and all(c["unique_match"] for c in x["causal_waits"]) for x in r["runs"] if x["depth"]==8)
    r["complete_fence_chain_witnesses"]=sum(c["qualified"] for x in r["runs"] for c in x["store_fence_chains"])
    r["interpretation"]="Four instrumented native baseline runs in frozen ABBA order. Bounded development recurrence only; no off/on overhead pair, formal goodput, research-strategy benefit, queue-order causality or universal recurrence claim."
    r["timing_warning"]="Wait fractions are descriptive, not an upper bound on joint-strategy gains. Time before copy host enqueue lacks reason/ready-state samples. CUDA durations do not define device start/end on the host timeline."
    with a.out.open("x") as f:json.dump(r,f,indent=2)
    print(json.dumps({k:r[k] for k in ("status","actual_requests","exact_reference_outputs","actual_token_events","repeatability_gate_met","complete_fence_chain_witnesses","by_depth")},indent=2))
if __name__=="__main__":main()
