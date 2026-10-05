"""Export measured native acquisition data into the author's v2 schema."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from py_kvcache.break_even import load_curves
from py_kvcache.cost_model import build_cost_tables
from py_kvcache._pchip import Pchip
def check(x,message):
    if not x:raise ValueError(message)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def export(cold,populate,restore,out):
    paths=[Path(p) for p in (cold,populate,restore)]
    reports=[json.loads(p.read_text()) for p in paths]
    configs=[json.loads((p.parent/"frozen-config.json").read_text()) for p in paths]
    check(all(r["status"].startswith("PASSED_NATIVE_") for r in reports),"failed acquisition")
    for c in configs[1:]:
        for key in ["model","model_alias","alias_target","gpu_uuid","sizes","reps","sampling","pythonhashseed"]:
            check(c[key]==configs[0][key],"identity/measurement mismatch: "+key)
        for key in ["dtype","kv_cache_dtype","block_size","kv_cache_memory_bytes","max_model_len","max_num_batched_tokens","attention_config","prefix_caching_hash_algo"]:
            check(c["engine"][key]==configs[0]["engine"][key],"engine mismatch: "+key)
    check(reports[2]["mode"]=="paired","paired SSD/staging acquisition required")
    allrows=reports[0]["rows"]+[r for r in reports[1]["rows"] if r["kind"]=="store"]+reports[2]["rows"]
    sizes=configs[0]["sizes"];reps=configs[0]["reps"]
    for n in sizes:
        for rep in range(reps):
            rows=[r for r in allrows if r["prefix_tokens"]==n and r["rep"]==rep]
            check({r["kind"] for r in rows}=={"f","store","g_mem","g_ssd"} and len(rows)==4,"incomplete or duplicate pair")
            check(all(r["prompt_token_ids"]==rows[0]["prompt_token_ids"] and r["output_token_ids"]==rows[0]["output_token_ids"] for r in rows),"token mismatch")
            for r in rows:
                check(len(r["prompt_token_ids"])==n+1,"wrong token axis")
                check(not r["metrics"]["is_corrupted"],"corrupted output")
                if r["kind"] in ("g_mem","g_ssd"):
                    t=r["trace"];check(t["transfer_success"] and t["h2d_events"]>0,"unproven load")
                    check(sum(x["num_bytes"] for x in t["load_transfers"])==n*57344,"transfer bytes differ")
                    b=t["foreground_logical_read_bytes"]+t["preload_actual_read_bytes"]
                    check(b==(0 if r["kind"]=="g_mem" else n*57344),"wrong path")
                    check(r["num_cached_tokens"]==n,"wrong prefix match")
                    for h in r["drain"]["handlers"]:
                        check(h["io_size"]==h["storage_block_bytes"]==917504 and h["pinned"] and h["staging_bytes"]<=h["staging_budget"],"layout/pinned budget mismatch")
    summary={};curves={}
    for kind in ("f","g_ssd","g_mem"):
        summary[kind]={}
        for n in sizes:
            values=[r["metrics"]["first_token_latency"] for r in allrows if r["kind"]==kind and r["prefix_tokens"]==n and not r["warmup"]]
            check(len(values)==reps-1>=3 and all(math.isfinite(v) and v>0 for v in values),"invalid sample")
            summary[kind][str(n)]=dict(n=len(values),median_s=statistics.median(values),min_s=min(values),max_s=max(values),samples_s=values)
        knots={n:d["median_s"] for n,d in summary[kind].items()}
        curves[kind]=dict(floor=0.0 if kind=="f" else min(knots.values()),knots=knots)
    for n in sizes:
        check(curves["g_ssd"]["knots"][str(n)]>=curves["g_mem"]["knots"][str(n)],"negative paired storage service: "+str(n))
    interp={k:Pchip([int(x) for x in v["knots"]],list(v["knots"].values()),floor=v["floor"]) for k,v in curves.items()}
    limit=max(sizes)
    def threshold(kind):
        blocks=list(range(16,limit+1,16))
        profitable=[n for n in blocks if all(interp["f"](m)>interp[kind](m) for m in blocks if m>=n)]
        return min(profitable) if profitable else limit+16
    payload=dict(schema_version=2,model_name=configs[0]["model_alias"],kv_dtype="auto",
        kv_bytes_per_token=57344,break_even_ssd_tokens=threshold("g_ssd"),
        break_even_mem_tokens=threshold("g_mem"),safety_margin_tokens=0,curves=curves,
        golden=[dict(tokens=n,**{k:fn(n) for k,fn in interp.items()}) for n in sorted(set(sizes+[0,32,96,384,768]))],
        provenance=dict(method="native first-token latency; paired SSD/staging measurements and separate identical cold requests; median of measured trials, warmup retained/excluded; no outlier filtering",
            axis="N reusable prefix; identical N+1 token request with one output token",
            max_supported_model_len=limit,min_nonzero_measured_tokens=min(sizes),
            floors="Original emitter convention: f(0)=0, g floors=min measured median; not independently measured zero-token costs",
            golden_scope="original PCHIP numerical consistency only",
            threshold_method="first block with f>g for every following block in measured domain; max+block denotes none, never zero",
            pythonhashseed="0",gpu_uuid=configs[0]["gpu_uuid"],model=configs[0]["model"],
            sources=[dict(path=str(p.resolve()),sha256=digest(p),config_sha256=digest(p.parent/"frozen-config.json")) for p in paths],
            engine=configs[1]["engine"],predictive_validation="pending planned holdout",formal_performance_claim=False))
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    target=out/"curves-v2.json";target.write_text(json.dumps(payload,indent=2))
    data=load_curves(str(target),model_name=payload["model_name"],kv_dtype="auto")
    tables=build_cost_tables(data,block_tokens=16,max_model_len=limit)
    check(tables.max_blocks==limit//16,"table coverage")
    (out/"sample-summary.json").write_text(json.dumps(summary,indent=2))
    (out/"validation.json").write_text(json.dumps(dict(token_pairs_checked=len(sizes)*reps,
        byte_proven_loads=2*len(sizes)*reps,table_max_blocks=tables.max_blocks,
        original_parser_and_golden_pass=True,curves_sha256=digest(target)),indent=2))
    print(json.dumps({"thresholds":{k:payload[k] for k in ("break_even_ssd_tokens","break_even_mem_tokens")},"summary":summary},indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("cold","populate","restore","out"):p.add_argument("--"+k,required=True)
    a=p.parse_args();export(a.cold,a.populate,a.restore,a.out)
