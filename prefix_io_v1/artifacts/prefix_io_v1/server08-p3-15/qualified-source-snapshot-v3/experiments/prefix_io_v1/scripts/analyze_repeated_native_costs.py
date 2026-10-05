"""Validate repeated native cost points without exporting or changing runtime curves."""
import argparse, hashlib, json, math, statistics
from pathlib import Path

def require(value, message):
    if not value:
        raise ValueError(message)

def check_configs(configs):
    first = configs[0]
    keys = ("model", "model_alias", "alias_target", "gpu_uuid", "sizes", "reps",
            "sampling", "pythonhashseed", "acquisition_domain")
    for c in configs:
        require(all(c[k] == first[k] for k in keys), "measurement identity mismatch")
        require(not c.get("native_hot_diagnostic", False), "diagnostic is not cost data")
        require(c["sampling"].get("logprobs") is None, "logprobs are not cost data")
        require({k:v for k,v in c["engine"].items() if k!="kv_transfer_config"} ==
                {k:v for k,v in first["engine"].items() if k!="kv_transfer_config"},
                "engine mismatch")
    external = [c["engine"]["kv_transfer_config"] for c in configs
                if c["engine"].get("kv_transfer_config") is not None]
    require(external and all(v == external[0] for v in external), "connector mismatch")

def check_drains(report):
    require(report.get("engine_shutdown") == "completed", "engine not shut down")
    drains = [report.get("final_drain", {})] + [r.get("drain", {}) for r in report["rows"]]
    for d in drains:
        for h in d.get("handlers", []):
            require(h["pinned"] and 0 < h["staging_bytes"] <= h["staging_budget"],
                    "staging budget/pinning")
            require(h["io_size"] == h["storage_block_bytes"] == 917504, "layout")
            a = h["aio"]
            require(a["accepted"] == a["completed"] == a["reaped"] and
                    not any(a[k] for k in ("outstanding","pending","ready","unreaped")) and
                    a["fatal"] is None, "physical I/O not settled")

def check_pair(cold, paired, populate, n, reps):
    for r, mode in ((cold,"cold"),(paired,"paired"),(populate,"populate")):
        require(r["mode"] == mode and r["status"] == "PASSED_NATIVE_"+mode.upper()+"_ACQUISITION",
                "failed/wrong acquisition")
        check_drains(r)
    rows = cold["rows"] + paired["rows"] + [r for r in populate["rows"] if r["kind"]=="store"]
    require(len(rows) == reps*4, "unexpected row count")
    ssd_bytes = 0
    samples = {k:[] for k in ("f","g_ssd","g_mem")}
    for rep in range(reps):
        group = [r for r in rows if r["prefix_tokens"]==n and r["rep"]==rep]
        require(len(group)==4 and {r["kind"] for r in group}=={"f","g_ssd","g_mem","store"},
                "duplicate/missing pair")
        require(all(r["prompt_token_ids"] == group[0]["prompt_token_ids"] and
                    r["output_token_ids"] == group[0]["output_token_ids"] for r in group),
                "token mismatch including warmup")
        for r in group:
            require(len(r["prompt_token_ids"])==n+1 and len(r["output_token_ids"])==1,
                    "token axis")
            require(r["warmup"] == (rep==0), "warmup classification")
            require(not r["metrics"]["is_corrupted"], "corrupted result")
            v = r["metrics"]["first_token_latency"]
            require(math.isfinite(v) and v>0, "invalid latency")
            if r["kind"]=="f":
                require(r["num_cached_tokens"]==0, "cold cache hit")
            if r["kind"] in ("g_ssd","g_mem"):
                t = r["trace"]
                actual = t["foreground_logical_read_bytes"]+t["preload_actual_read_bytes"]
                require(r["num_cached_tokens"]==n and t["transfer_success"] and
                        t["h2d_events"]>0 and sum(x["num_bytes"] for x in t["load_transfers"])==n*57344,
                        "unproven complete load")
                require(actual==(n*57344 if r["kind"]=="g_ssd" else 0), "wrong medium bytes")
                require(bool(r["drain"].get("handlers")), "missing physical drain")
                if r["kind"]=="g_ssd": ssd_bytes += actual
            if r["kind"] in samples and not r["warmup"]:
                samples[r["kind"]].append(v)
    require(reps>=3 and all(len(v)==reps-1 for v in samples.values()), "sample coverage")
    return dict(samples_s=samples, medians_s={k:statistics.median(v) for k,v in samples.items()},
                checked_prompt_pairs=reps, ssd_read_bytes=ssd_bytes)

def analyze(root, plan_path):
    plan = json.loads(plan_path.read_text())
    sources = []
    configs = []
    def load(label):
        folder = root/"experiments/prefix_io_v1/runs"/label/"details"
        p = folder/"result.json"; c = folder/"frozen-config.json"
        sources.append(dict(label=label,result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                            config_sha256=hashlib.sha256(c.read_bytes()).hexdigest()))
        configs.append(json.loads(c.read_text()))
        return json.loads(p.read_text())
    populate = load(plan["discovery"][1])
    sessions = []
    labels = []
    for split,key in (("fit","fit_sessions"),("temporal_validation","validation_sessions")):
        for entry in plan[key]:
            labels.extend(entry.values())
            row = check_pair(load(entry["cold"]),load(entry["paired"]),populate,
                             plan["prefix_tokens"],plan["reps"])
            row.update(split=split,labels=entry)
            row["ssd_relative_reduction"] = 1-row["medians_s"]["g_ssd"]/row["medians_s"]["f"]
            sessions.append(row)
    require(len(labels)==len(set(labels)), "reused session")
    check_configs(configs)
    fit = [s for s in sessions if s["split"]=="fit"]
    predicted = {k:statistics.median([s["medians_s"][k] for s in fit])
                 for k in ("f","g_ssd","g_mem")}
    for s in sessions:
        if s["split"]=="temporal_validation":
            s["relative_prediction_error"] = {k:(s["medians_s"][k]-v)/v
                                              for k,v in predicted.items()}
    return dict(status="PASSED_REPEATED_POINT_CHECKS",plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                sessions=sessions,fit_session_median_s=predicted,sources=sources,
                runtime_curve_exported=False,formal_performance_claim=False,
                full_P3_complete=False,limitations=plan["limitations"])

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(Path.cwd(),args.plan)
    with args.output.open("x") as f: json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="sources"},indent=2))
