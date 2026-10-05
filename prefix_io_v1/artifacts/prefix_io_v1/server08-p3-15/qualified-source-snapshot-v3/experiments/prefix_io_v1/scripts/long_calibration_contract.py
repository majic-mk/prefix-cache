"""Guards for the calibration-family integration diagnostic only."""
import math

def require(value,message):
    if not value:raise ValueError(message)

def validate(manifest,candidate,gpu_uuid):
    require(manifest.get("calibration_integration_only") is True and
            manifest.get("formal_goodput_enabled") is False and manifest.get("slo") is None,
            "calibration-only contract")
    p=candidate["provenance"]
    require(p.get("candidate_only") is True and p.get("allowed_consumer")==
            "same-budget calibration integration diagnostic only","wrong curve qualification")
    require(p["gpu_uuid"]==gpu_uuid,"GPU mismatch")
    reference={k:v for k,v in p["engine"].items() if k!="kv_transfer_config"}
    require(manifest["engine"]==reference,"engine mismatch")
    require(reference["kv_cache_memory_bytes"]==2147483648 and
            reference["max_model_len"]==reference["max_num_batched_tokens"]==16400 and
            reference["max_num_seqs"]==1,"frozen engine budget mismatch")
    require(manifest["staging_budget_bytes"]==1073741824 and manifest["output_tokens"]==128,"frozen staging/output budget")
    rows=manifest["requests"]
    require(1<=len(rows)<=8,"bounded diagnostic cohort")
    times=[r["scheduled_time"] for r in rows]
    require(all(math.isfinite(t) for t in times) and times==sorted(times) and times[0]>=0 and times[-1]<30,"arrival schedule")
    for r in rows:
        tokens=r["prompt_token_ids"]
        require(len(tokens)==16257 and len(tokens)+128<=reference["max_model_len"],"context overflow")
        require(p["minimum_supported_prefix_tokens"]<=16256<=p["max_supported_model_len"],"curve domain")
        rep=r["calibration_family"]
        require(type(rep) is int and rep in (0,1,2),"unknown calibration family")
        expected=[2000+16384+rep*1500]+[1000+(i%500) for i in range(16255)]+[777]
        require(tokens==expected,"calibration-family identity mismatch")
    return reference
