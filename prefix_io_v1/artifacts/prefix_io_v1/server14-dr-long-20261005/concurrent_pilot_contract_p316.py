"""P316 bounded C2 replay and explicitly registered true GPU-hot control."""
import hashlib, json
from pathlib import Path
import math

DELTA = {"max_num_seqs": 2}
PATTERNS = {
    "all_hit": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
    "gpu_hot": [0] * 10,
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
    depth = manifest.get("io_depth", 4)
    acquisition_io_depth(16384, 2, depth)
    require(depth == 4 or (manifest.get("baseline_tuning") is True and manifest["profile"] in PATTERNS),
            "unregistered fixed I/O tuning")
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

def acquisition_io_depth(domain, max_num_seqs, depth):
    require(type(depth) is int and depth in (2,4,8), "unbounded I/O depth")
    require(depth == 4 or (domain == 16384 and max_num_seqs == 2), "I/O tuning only in frozen two-sequence scope")
    return depth

def expected_connector_variant(original, delta):
    require(set(delta) == {"iodepth"} and type(delta["iodepth"]) is int and delta["iodepth"] in (2,8),
            "only the two frozen I/O-depth baselines allowed")
    require(original["iodepth"] == 4, "original I/O depth differs")
    return dict(original, **delta)


GPU_HOT_PREFIX = 16256
GPU_HOT_WRITE_UNITS = 128
NATIVE_IO_QUANTUM = 917504
SUPPLEMENT_SCOPE = "P316_GPU_HOT_SUPPLEMENTAL_MANIFEST_ONLY"

def _read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate supplemental key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_supplement(root, path, manifest_path, candidate_path, qualification_path, permit, gpu_uuid):
    """Additional workload registration; this never creates a GPU permission."""
    root = Path(root).resolve()
    raw = _read_json(path)
    required = {"schema_version", "scope", "gpu_uuid", "manifest_sha256",
                "candidate_sha256", "qualification_sha256", "parent_manifest", "reference_results"}
    require(type(raw) is dict and set(raw) == required, "strict GPU-hot registration required")
    require(type(raw["schema_version"]) is int and raw["schema_version"] == 1 and
            raw["scope"] == SUPPLEMENT_SCOPE, "wrong supplemental scope")
    require(raw["gpu_uuid"] == gpu_uuid, "supplemental GPU differs")
    for field, file_path in (("manifest_sha256", manifest_path), ("candidate_sha256", candidate_path),
                             ("qualification_sha256", qualification_path)):
        require(raw[field] == _sha(file_path), "supplemental input changed: " + field)
    files = {}
    for key in ("parent_manifest", "reference_results"):
        row = raw[key]
        require(type(row) is dict and set(row) == {"path", "sha256"}, "strict supplemental file binding")
        p = (root / row["path"]).resolve()
        require(p.is_relative_to(root) and p.is_file() and row["sha256"] == _sha(p),
                "unsafe or changed supplemental file")
        files[key] = p
    parent = _read_json(files["parent_manifest"])
    current = _read_json(manifest_path)
    require(current["profile"] == "gpu_hot", "supplement only admits GPU-hot profile")
    require(parent["profile"] in ("all_hit", "mixed_readwrite"), "unknown qualified parent")
    require(permit["manifests"][parent["profile"]]["sha256"] == _sha(files["parent_manifest"]),
            "parent manifest not in the original qualified permit")
    require({k:v for k,v in current.items() if k not in ("profile","requests","scope")} ==
            {k:v for k,v in parent.items() if k not in ("profile","requests","scope")},
            "GPU-hot supplement changed engine/source/family/arrival contract")
    require(type(current["scope"]) is str and 0 < len(current["scope"]) <= 256, "bounded supplement scope")
    require(len(current["requests"]) == len(parent["requests"]) == 10, "supplement request count")
    for current_row, parent_row in zip(current["requests"], parent["requests"]):
        require(set(current_row) == set(parent_row) and current_row["family_index"] == 0 and
                {k:v for k,v in current_row.items() if k != "family_index"} ==
                {k:v for k,v in parent_row.items() if k != "family_index"},
                "supplement changed frozen arrival or request identity")
    reference = _read_json(files["reference_results"])
    matches = [r for r in reference["rows"] if r["family"] == current["families"][0]["name"]
               and r["kind"] == "gpu_hot"]
    require(len(matches) == 1, "one frozen GPU-hot golden row required")
    golden = matches[0]
    require(golden["prompt_token_ids"] == current["families"][0]["tokens"] and
            type(golden["output_tokens"]) is list and len(golden["output_tokens"]) == 128 and
            all(type(t) is int and 0 <= t < 152064 for t in golden["output_tokens"]),
            "golden tokens differ or incomplete")
    return dict(registration_sha256=_sha(path), registration=raw,
                reference_path=str(files["reference_results"]),
                golden_output_tokens=list(golden["output_tokens"]),
                cached_prefix_tokens=GPU_HOT_PREFIX, ssd_read_bytes=0,
                ssd_write_max_bytes=GPU_HOT_WRITE_UNITS*NATIVE_IO_QUANTUM,
                storage_io_quantum=NATIVE_IO_QUANTUM)

def validate_gpu_hot_measurement(rows, read_bytes, write_bytes, quantum, golden):
    require(type(read_bytes) is int and read_bytes == 0, "GPU-hot control issued SSD reads")
    require(type(write_bytes) is int and 0 <= write_bytes <= GPU_HOT_WRITE_UNITS*NATIVE_IO_QUANTUM,
            "GPU-hot paid generation writes exceed frozen cap")
    require(type(quantum) is int and quantum == NATIVE_IO_QUANTUM, "GPU-hot I/O unit changed")
    require(len(rows) == 10, "GPU-hot cohort incomplete")
    for row in rows:
        require(row["num_cached_tokens"] == GPU_HOT_PREFIX, "GPU-hot prefix not retained")
        require(row["output_tokens"] == golden and len(row["output_tokens"]) == 128,
                "GPU-hot full output differs from golden")


def validate_optional_sampling(mode, profile, simple_extra, native_cpu_probe=False):
    """Only the registered common-off GPU-hot control may suppress sampling."""
    require(type(mode) is str and mode in ("on","off"), "unknown optional sampling mode")
    require(type(native_cpu_probe) is bool, "CPU probe flag must be boolean")
    require(not native_cpu_probe or type(simple_extra) is dict,
            "CPU hook timing requires the common simple-stage probe")
    if mode == "off":
        require(profile == "gpu_hot" and type(simple_extra) is dict and
                simple_extra.get("prefix_io_stage_policy") == {"mode":"off"},
                "sampling-off is limited to registered GPU-hot common-off control")
    return mode

def validate_pressure_capacity(policy_mode, sampling, live):
    if policy_mode != "pressure":
        return
    require(sampling == "on", "pressure comparison must retain observations")
    require(bool(live), "pressure capacity witness missing")
    for snapshot in live:
        capacity = snapshot["staging_capacity"]
        require(capacity["valid"] is True and capacity["observation_valid"] is True and
                capacity["error"] is None and capacity["physical_release_credit"] is False,
                "pressure clean capacity unknown or invalid")
        # preload_cached_slots includes the shared cache; do not add shared twice.
        keys = ("cache_slots","preload_cached_slots")
        require(all(type(capacity[k]) is int and capacity[k]>=0 for k in keys) and
                sum(capacity[k] for k in keys)>64, "pressure >64-registry witness missing")
