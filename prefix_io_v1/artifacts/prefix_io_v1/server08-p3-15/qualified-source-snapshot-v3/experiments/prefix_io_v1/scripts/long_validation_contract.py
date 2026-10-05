"""Frozen independent long diagnostic contract; not a production policy grant."""
import math
def require(ok,message):
    if not ok:raise ValueError(message)

def validate(manifest,candidate,qualification,prompts,gpu_uuid):
    require(manifest.get("independent_validation_only") is True and
        manifest.get("formal_goodput_enabled") is False and manifest.get("slo") is None,"independent diagnostic only")
    require(qualification.get("status")=="PASSED_LONG_DIAGNOSTIC_GATE" and
        qualification.get("runtime_scope")=="same-budget independent long validation only","qualification missing")
    require(qualification["validated_prefix_tokens"]==16256,"unvalidated prefix length")
    p=candidate["provenance"];require(p["candidate_only"] and p["gpu_uuid"]==gpu_uuid,"candidate/GPU mismatch")
    engine={k:v for k,v in p["engine"].items() if k!="kv_transfer_config"}
    require(manifest["engine"]==engine,"engine mismatch")
    require(engine["kv_cache_memory_bytes"]==2147483648 and
        engine["max_model_len"]==engine["max_num_batched_tokens"]==16400 and engine["max_num_seqs"]==1,"frozen budgets")
    require(manifest["staging_budget_bytes"]==1073741824 and manifest["output_tokens"]==128,"staging/output budget")
    require(manifest["new_policy"]=="off" and manifest["original_load_planner"]=="on","original admission and strategy off required")
    require(prompts["partition"]=="validation" and prompts["sizes"]==[16256] and prompts["reps"]==3,"prompt partition")
    families={r["family"]:r["token_ids"] for r in prompts["prompts"]}
    rows=manifest["requests"];require(1<=len(rows)<=8,"bounded cohort")
    seen=set();times=[]
    for row in rows:
        rid=row["request_id"];tokens=row["prompt_token_ids"];family=row["validation_family"];t=row["scheduled_time"]
        require(type(rid) is int and rid not in seen,"duplicate request")
        require(family in families and tokens==families[family] and len(tokens)==16257 and
            len(tokens)+128<=engine["max_model_len"],"prompt/context mismatch")
        require(type(t) in (int,float) and math.isfinite(t) and 0<=t<30,"invalid arrival")
        seen.add(rid);times.append(t)
    require(times==sorted(times),"arrival order")
    return engine
