"""Bounded two-sequence P3 diagnostic; no research policy permission."""
import math

DELTA = {"max_num_seqs": 2}
PATTERNS = {
    "all_hit": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
    "mixed_readwrite": [0, 3, 1, 0, 4, 2, 3, 1, 4, 2],
}
def require(ok, message):
    if not ok:
        raise ValueError(message)

def acquisition_delta(domain, max_num_seqs):
    require(type(max_num_seqs) is int and max_num_seqs in (1, 2), "unbounded concurrency")
    require(max_num_seqs == 1 or domain == 16384, "concurrency variant only for long domain")
    return {} if max_num_seqs == 1 else dict(DELTA)

def expected_variant_engine(original, delta):
    require(delta == DELTA and type(delta.get("max_num_seqs")) is int, "only frozen concurrency change allowed")
    require(original["max_num_seqs"] == 1, "original engine must have one sequence")
    require(original["kv_cache_memory_bytes"] == 2147483648 and
            original["max_model_len"] == original["max_num_batched_tokens"] == 16400, "original budget differs")
    return dict(original, **DELTA)

def validate(manifest, candidate, gpu_uuid):
    require(manifest["partition"] == "development" and manifest["formal_goodput"] is False and
            manifest["slo"] is None, "development only")
    require(manifest["policy"] == "shadow" and manifest["load_planner"] == "on" and
            manifest["artificial_io_delay"] is False and manifest["cache_resets"] == 0, "native pipeline required")
    require(manifest["gpu_uuid"] == candidate["provenance"]["gpu_uuid"] == gpu_uuid, "GPU mismatch")
    original = {k:v for k,v in candidate["provenance"]["engine"].items() if k != "kv_transfer_config"}
    require(manifest["engine"] == expected_variant_engine(original, DELTA), "engine changed")
    require(manifest["staging_bytes"] == 1073741824 and manifest["output_tokens"] == 128, "memory/output budget changed")
    families = manifest["families"]
    require(len(families) == 5 and len({f["name"] for f in families}) == 5, "family coverage")
    require(len({tuple(f["tokens"][:16]) for f in families}) == 5, "prefix families overlap")
    for i, f in enumerate(families):
        require(len(f["tokens"]) == 16257 and all(type(t) is int and 0 <= t < 152064 for t in f["tokens"]), "token domain")
        require(f["initial_ssd_present"] is (i < 3), "initial cache declaration differs")
    profile = manifest["profile"]
    require(profile in PATTERNS, "unknown profile")
    rows = manifest["requests"]
    require(len(rows) == 10 and [r["family_index"] for r in rows] == PATTERNS[profile], "frozen workload changed")
    require([r["request_id"] for r in rows] == list(range(10)), "request IDs")
    times = [r["scheduled_time"] for r in rows]
    require(all(type(t) in (int, float) and math.isfinite(t) and 0 <= t < 30 for t in times) and
            times == sorted(times), "arrival domain/order")
    require(manifest["arrival_seed"] == 1704 and manifest["arrival_rate"] == 1.0, "arrival settings changed")
    return dict(manifest["engine"])

def required_new_bytes(profile):
    require(profile in PATTERNS, "unknown profile")
    return (3 * 1024**3) if profile == "mixed_readwrite" else (512 * 1024**2)
