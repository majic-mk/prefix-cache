"""Held-out content checks against a frozen curve; never refits or installs it."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from experiment_storage import details_path
from heldout_manifest import load_manifest
from analyze_cached_references import analyze as analyze_cached
from analyze_repeated_native_costs import check_drains,require
from export_uniform_calibration_candidate import verify_measurement
from export_native_aio_costs import Pchip

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# Capacity-specific checks supplement every original numerical/medium/drain check.
from concurrent_capacity_contract_p316 import (
    DELTA, CANDIDATE_SHA256, POINT_TOKENS, POINT_REPS, validate_declaration,
    connector_delta as expected_delta, expected_connector_variant,
    expected_variant_engine)
ACQUIRER = "acquire_native_aio_costs_capacity_p316.py"

def _domain_plan(plan, requested=None, *, cost=False):
    domain = validate_declaration(plan, requested)
    require(plan.get("engine_delta") == DELTA and
            type(plan["engine_delta"].get("max_num_seqs")) is int,
            "capacity plan requires exact C2 engine")
    require(plan.get("connector_delta") == expected_delta(domain["capacity_domain_id"]) and
            type(plan["connector_delta"].get("iodepth")) is int and
            type(plan["connector_delta"].get("preload_lookahead_requests")) is int and
            type(plan["connector_delta"].get("staging_mem")) is float,
            "unregistered capacity connector change")
    if cost:
        require(type(plan.get("maximum_absolute_relative_median_error")) is float and
                plan["maximum_absolute_relative_median_error"] == .25 and
                plan.get("used_for_fit") is False and plan.get("formal_evaluation") is False,
                "original cost gate standard changed")
        require(plan.get("candidate_sha256") == CANDIDATE_SHA256,
                "frozen curve candidate changed")
        require([job.get("mode") for job in plan["jobs"]] == ["cold", "paired", "paired", "cold"],
                "two fresh cold/paired engine sessions required")
    else:
        require(plan.get("domain") == 16384 and type(plan.get("domain")) is int and
                plan.get("sizes") == [POINT_TOKENS] and plan.get("reps") == POINT_REPS and
                type(plan.get("reps")) is int and plan.get("kv_budget_bytes") == 2147483648 and
                type(plan.get("kv_budget_bytes")) is int and
                plan.get("staging_budget_bytes") == domain["staging_bytes"] and
                type(plan.get("staging_budget_bytes")) is int and
                plan.get("cached_output_tolerance") == 0 and
                type(plan.get("cached_output_tolerance")) is int and
                plan.get("selected_output_tolerance") == 0 and
                type(plan.get("selected_output_tolerance")) is int,
                "original point/reference budget or tolerance changed")
        require(len(plan["external_groups"]) == 1 and
                plan["external_groups"][0]["sizes"] == [POINT_TOKENS],
                "one complete cached reference domain required")
    return domain

def _expected_source_hashes(root):
    folder = Path(root) / "experiments/prefix_io_v1/scripts"
    return dict(acquisition=digest(folder / ACQUIRER),
                contract=digest(folder / "concurrent_capacity_contract_p316.py"))

def _option(command, key, default=None):
    hits = [i for i, item in enumerate(command) if item == key]
    require(len(hits) <= 1, "duplicate GPU command option: " + key)
    if not hits:
        return default
    require(hits[0] + 1 < len(command), "missing GPU command option value: " + key)
    return command[hits[0] + 1]

def _phase_receipt(root, label, plan, config, report):
    folder = details_path(root, label, plan)
    path = folder.parent / "result.json"
    receipt = json.loads(path.read_text())
    require(receipt.get("label") == label and receipt.get("gpu_job_attempted") is True and
            receipt.get("exit") == 0 and type(receipt.get("exit")) is int and
            receipt.get("child_exit") == 0 and type(receipt.get("child_exit")) is int and
            receipt.get("timed_out") is False and receipt.get("error") is None and
            receipt.get("interrupted_signal") is None and receipt.get("session_drained") is True and
            receipt.get("session_members_after_cleanup") == [] and
            receipt.get("gpu_uuid") == config["gpu_uuid"],
            "actual guarded GPU phase did not complete/drain")
    command = receipt.get("command")
    require(type(command) is list and all(type(item) is str for item in command) and
            sum(Path(item).name == ACQUIRER for item in command) == 1,
            "GPU receipt is not NEW capacity acquisition")
    require(_option(command, "--capacity-domain") == plan["capacity_domain_id"],
            "GPU receipt capacity domain mismatch")
    mode = "cold" if config["engine"]["kv_transfer_config"] is None else "paired"
    require(_option(command, "--mode") == mode == report["mode"] and
            _option(command, "--domain", "16384") == "16384" and
            _option(command, "--max-num-seqs", "2") == "2" and
            _option(command, "--iodepth", "8") == "8" and
            _option(command, "--sizes", "16256") == "16256" and
            _option(command, "--reps", "3") == "3",
            "GPU receipt point/config mismatch")
    require(("--native-hot-diagnostic" in command) is report["native_hot_diagnostic"] and
            ("--cached-reference-logprobs" in command) is report["cached_reference_logprobs"],
            "diagnostic flags differ from actual GPU command")
    return dict(label=label, path=str(path.relative_to(root)), sha256=digest(path))

def _cached_geometry(report, domain):
    geometry = dict(slot_count=domain["slot_count"],
        cache_preload_ceiling_slots=domain["cache_preload_ceiling_slots"],
        staging_cache_ceiling_slots=domain["cache_preload_ceiling_slots"],
        io_depth=8, staging_budget_bytes=domain["staging_bytes"], io_quantum_bytes=917504)
    drains = [report.get("final_drain", {})] + [row.get("drain", {}) for row in report["rows"]]
    for drain in drains:
        handlers = drain.get("handlers", [])
        require(type(handlers) is list and len(handlers) == 1,
                "actual native cached handler geometry missing")
        for handler in handlers:
            actual = handler.get("capacity_geometry")
            require(actual == geometry and all(type(actual.get(k)) is int for k in geometry),
                    "actual native capacity geometry mismatch")
            require(handler["staging_budget"] == domain["staging_bytes"],
                    "actual staging budget differs")
            owner = handler.get("owner_snapshot", {})
            require(owner.get("owner_capture") is True and
                    owner.get("native_shutdown_read") is False and
                    owner.get("physical_drain_inferred") is False and
                    owner.get("gpu_release_credit") is False,
                    "actual reactor owner snapshot missing")
            admission = owner.get("admission", {})
            require(admission.get("count_valid") is True and
                    type(admission.get("accepted_parents")) is int and
                    admission["accepted_parents"] == 0 and
                    admission.get("native_drain_unknown") is False,
                    "parent physical drain unknown")
            capacity = owner.get("staging_capacity", {})
            require(capacity.get("valid") is True and
                    capacity.get("observation_valid") is True and
                    capacity.get("error") is None and
                    capacity.get("physical_release_credit") is False and
                    capacity.get("slot_count") == domain["slot_count"],
                    "unknown native capacity observation")

def _phase(root, label, plan, config, report, domain):
    require(validate_declaration(config) == domain and validate_declaration(report) == domain,
            "phase/config capacity declaration differs")
    expected_sources = _expected_source_hashes(root)
    require(config.get("capacity_source_hashes") == report.get("capacity_source_hashes") ==
            expected_sources, "capacity acquisition/contract source changed")
    require(config["sizes"] == [POINT_TOKENS] and config["reps"] == POINT_REPS and
            config["acquisition_domain"] == 16384 and config["engine"]["max_num_seqs"] == 2 and
            config["engine"]["kv_cache_memory_bytes"] == 2147483648,
            "actual phase point/budget mismatch")
    if config["engine"]["kv_transfer_config"] is not None:
        _cached_geometry(report, domain)
    return _phase_receipt(root, label, plan, config, report)



def reference(root,plan_path,*,capacity_domain_id=None):
    root=Path(root);plan_path=Path(plan_path)
    plan=json.loads(plan_path.read_text())
    domain=_domain_plan(plan,capacity_domain_id)
    receipts=[]
    rows,identity=load_manifest(root/plan["prompt_manifest"],plan["sizes"],plan["reps"])
    require(identity["sha256"]==plan["prompt_manifest_sha256"],"manifest changed")
    result=analyze_cached(root,plan_path)
    for label in [plan["native_label"],*[g["label"] for g in plan["external_groups"]]]:
        folder=details_path(root,label,plan)
        config=json.loads((folder/"frozen-config.json").read_text())
        report=json.loads((folder/"result.json").read_text())
        receipts.append(_phase(root,label,plan,config,report,domain))
        for row in report["rows"]:
            if row["kind"] == "g_mem":
                require(row["trace"]["src_cache"] == POINT_TOKENS // 16,
                        "whole-prefix original staging path not proven")
        require(config["prompt_manifest"]==report["prompt_manifest"]==identity,"manifest identity mismatch")
        require(digest(folder/"prompt-manifest.json")==identity["sha256"],"run manifest changed")
        for row in report["rows"]:
            require(row["prompt_token_ids"]==rows[(row["prefix_tokens"],row["rep"])],"run prompt mismatch")
    result.update(independent_content_validation=True,manifest=identity,
        validation_domain=plan["sizes"],formal_evaluation=False,
        limitations=("Three independent document families at the frozen length; one warmup, two measured. No full-vocabulary or unmeasured-length generalization claim." if "run_details" in plan else
                     "Three independent document families at one length; one warmup, two measured. No full-vocabulary or long-prefix generalization claim."))
    result.update(capacity_domain_id=domain["capacity_domain_id"],capacity_domain=domain,
                  guarded_gpu_receipts=receipts,capacity_source_hashes=_expected_source_hashes(root))
    return result

def prediction_check(predicted,samples,tolerance):
    require(isinstance(predicted,(int,float)) and math.isfinite(predicted) and predicted>0,"bad prediction")
    require(len(samples)>=4 and all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in samples),"sample coverage")
    require(isinstance(tolerance,(int,float)) and 0<tolerance<1,"bad tolerance")
    observed=statistics.median(samples);error=(observed-predicted)/predicted
    return dict(prediction_s=predicted,observed_median_s=observed,
        relative_error=error,absolute_relative_error=abs(error),passed=abs(error)<=tolerance,
        n=len(samples),samples_s=samples,min_s=min(samples),max_s=max(samples))

def costs(root,plan_path,*,capacity_domain_id=None):
    root=Path(root);plan_path=Path(plan_path)
    plan=json.loads(plan_path.read_text())
    domain=_domain_plan(plan,capacity_domain_id,cost=True)
    receipts=[]
    curve_path=root/plan["candidate"];require(digest(curve_path)==plan["candidate_sha256"],"curve changed")
    curve=json.loads(curve_path.read_text())
    require(curve["provenance"]["candidate_only"] and plan["used_for_fit"] is False and plan["formal_evaluation"] is False,
            "validation cannot silently fit/qualify")
    rp=json.loads((root/plan["reference_plan"]).read_text())
    require(_domain_plan(rp,domain["capacity_domain_id"])==domain,"reference capacity domain changed")
    rr=reference(root,root/plan["reference_plan"],capacity_domain_id=domain["capacity_domain_id"])
    require(rr["status"]=="PASSED_EXACT_CACHED_REFERENCE" and rr==json.loads((root/plan["reference_result"]).read_text()),"numeric gate failed/changed")
    folder=details_path(root,rp["native_label"],rp)
    nc=json.loads((folder/"frozen-config.json").read_text())
    nr=json.loads((folder/"result.json").read_text())
    refs={(r["prefix_tokens"],r["rep"],r["kind"]):r for r in nr["rows"]}
    ec=json.loads((details_path(root,rp["external_groups"][0]["label"],rp)/"frozen-config.json").read_text())
    engine_without_connector=lambda e:{k:v for k,v in e.items() if k!="kv_transfer_config"}
    expected_engine=engine_without_connector(curve["provenance"]["engine"])
    expected_engine=expected_variant_engine(expected_engine,DELTA)
    require(engine_without_connector(nc["engine"])==expected_engine,"candidate budget/config mismatch")
    require(nc["model"]==curve["provenance"]["model"] and nc["gpu_uuid"]==curve["provenance"]["gpu_uuid"],"candidate identity mismatch")
    extra=ec["engine"]["kv_transfer_config"]["kv_connector_extra_config"].copy()
    expected=curve["provenance"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"].copy()
    extra.pop("shared_storage_path");expected.pop("shared_storage_path")
    expected=expected_connector_variant(expected,domain["capacity_domain_id"])
    require(extra==expected,"candidate storage mechanism/config mismatch")
    samples={k:{n:[] for n in rp["sizes"]} for k in ("f","g_ssd","g_mem")}
    session_counts={k:0 for k in samples};sources=[];seen={rp["native_label"],*[g["label"] for g in rp["external_groups"]]};checked_rows=0;session_medians=[]
    for job in plan["jobs"]:
        label=job["label"];require(label not in seen,"duplicate engine session");seen.add(label)
        f=details_path(root,label,plan)
        c=json.loads((f/"frozen-config.json").read_text());report=json.loads((f/"result.json").read_text())
        receipts.append(_phase(root,label,plan,c,report,domain))
        require(report["status"]=="PASSED_NATIVE_"+job["mode"].upper()+"_ACQUISITION","acquisition failed")
        require(not report["native_hot_diagnostic"] and not report["cached_reference_logprobs"],"diagnostic cost forbidden")
        require(c["sampling"]=={k:v for k,v in nc["sampling"].items() if k!="logprobs"},"sampling mismatch")
        for key in ("model","model_alias","alias_target","gpu_uuid","reps","pythonhashseed","acquisition_domain","sizes","prompt_manifest"):
            require(c[key]==nc[key],"identity mismatch: "+key)
        require(digest(f/"prompt-manifest.json")==plan["prompt_manifest_sha256"],"measurement manifest changed")
        require(engine_without_connector(c["engine"])==engine_without_connector(nc["engine"]),"engine mismatch")
        if job["mode"]=="cold":
            kinds={"f"};require(c["engine"]["kv_transfer_config"] is None,"cold connector active")
        else:
            require(job["mode"]=="paired" and c["engine"]["kv_transfer_config"]==ec["engine"]["kv_transfer_config"],"connector mismatch")
            kinds={"g_ssd","g_mem"}
        check_drains(report)
        for row in report["rows"]:
            if row["kind"] == "g_mem":
                require(row["trace"]["src_cache"] == POINT_TOKENS // 16,
                        "whole-prefix original staging path not proven")
        require(len(report["rows"])==len(rp["sizes"])*rp["reps"]*len(kinds),"row coverage")
        local={k:{n:[] for n in rp["sizes"]} for k in kinds};keys=set()
        for row in report["rows"]:
            n,rep,kind=row["prefix_tokens"],row["rep"],row["kind"];key=(n,rep,kind)
            require(n in rp["sizes"] and 0<=rep<rp["reps"] and kind in kinds and key not in keys,"unexpected/duplicate sample")
            keys.add(key)
            verify_measurement(row,refs[(n,rep,"f" if kind=="f" else "gpu_hot")],n);checked_rows+=1
            if not row["warmup"]:
                samples[kind][n].append(row["metrics"]["first_token_latency"])
                local[kind][n].append(row["metrics"]["first_token_latency"])
        for k in kinds:session_counts[k]+=1
        session_medians.append(dict(label=label,median_s={k:{str(n):statistics.median(v) for n,v in d.items()} for k,d in local.items()}))
        sources.append(dict(label=label,result_sha256=digest(f/"result.json"),config_sha256=digest(f/"frozen-config.json")))
    require(all(v==2 for v in session_counts.values()),"two fresh engine sessions per path required")
    checks={}
    for kind,points in samples.items():
        c=curve["curves"][kind];knots=c["knots"];keys=sorted(int(k) for k in knots)
        interpolator=Pchip(keys,[knots[str(k)] for k in keys],floor=c["floor"])
        checks[kind]={}
        for n,values in points.items():
            require(keys[0]<=n<=keys[-1],"outside measured domain")
            checks[kind][str(n)]=prediction_check(interpolator(n),values,plan["maximum_absolute_relative_median_error"])
    passed=all(c["passed"] for d in checks.values() for c in d.values())
    return dict(status="PASSED_HELDOUT_POINT_PREDICTION" if passed else "FAILED_HELDOUT_POINT_PREDICTION",
        checks=checks,checked_rows=checked_rows,session_medians=session_medians,sources=sources,
        candidate_sha256=digest(curve_path),plan_sha256=digest(plan_path),
        capacity_domain_id=domain["capacity_domain_id"],capacity_domain=domain,
        guarded_gpu_receipts=rr["guarded_gpu_receipts"]+receipts,
        capacity_source_hashes=_expected_source_hashes(root),
        independent_content_validation=True,full_domain_validated=False,
        runtime_curve_updated=False,formal_performance_claim=False,
        limitations="Two measured content families and two engine sessions per path at the frozen point only. No confidence interval or full-domain qualification.")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--capacity-domain",choices=("cap960-l1","cap1024-l2"),required=True)
    p.add_argument("--reference-only",action="store_true");a=p.parse_args()
    result=reference(Path.cwd(),a.plan,capacity_domain_id=a.capacity_domain) if a.reference_only else costs(Path.cwd(),a.plan,capacity_domain_id=a.capacity_domain)
    with a.out.open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ("comparisons","sources","cold_hot")},indent=2))
    raise SystemExit(0 if result["status"].startswith("PASSED") else 1)
