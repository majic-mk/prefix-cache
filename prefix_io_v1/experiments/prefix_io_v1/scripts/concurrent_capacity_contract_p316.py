"""Exact finite P3 capacity domains; CPU definitions confer no GPU permission."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from concurrent_pilot_contract import DELTA, expected_variant_engine, validate as validate_baseline

ROOT = Path(__file__).resolve().parents[3]
BASELINE_MANIFEST = "artifacts/prefix_io_v1/server08-p3-14/d8/mixed_readwrite-manifest.json"
BASELINE_MANIFEST_SHA256 = "1128f0b4c5481837f87c689e29f1b0f92fc80d29d8d13573fe08714bef58c197"
CANDIDATE = "artifacts/prefix_io_v1/server08-p3-14/calibration-candidate/curves-v2.json"
CANDIDATE_SHA256 = "fd05683418e851d5130ba2ab646e9e33cbb0ea490584787dca3d019abff6d451"
DOMAIN_IDS = ("cap960-l1", "cap1024-l2")
QUANTUM = 917504
POINT_TOKENS = 16256
POINT_REPS = 3

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def capacity_domain(domain_id):
    require(type(domain_id) is str and domain_id in DOMAIN_IDS, "unregistered capacity domain")
    mib, lookahead = (960, 1) if domain_id == "cap960-l1" else (1024, 2)
    staging_bytes = mib * 1024**2
    slots = (staging_bytes - 4095) // QUANTUM
    require(slots - 8 >= POINT_TOKENS // 16, "whole-prefix g_mem capacity impossible")
    return dict(capacity_domain_id=domain_id, staging_bytes=staging_bytes,
                staging_mem_gib=mib / 1024, preload_lookahead_requests=lookahead,
                io_depth=8, slot_count=slots, cache_preload_ceiling_slots=slots - 8)

def validate_declaration(data, domain_id=None):
    require(isinstance(data, dict), "capacity declaration missing")
    declared = data.get("capacity_domain_id")
    domain = capacity_domain(declared if domain_id is None else domain_id)
    require(declared == domain["capacity_domain_id"], "capacity domain mismatch")
    value = data.get("capacity_domain")
    require(type(value) is dict and value == domain and
            all(type(value.get(k)) is type(v) for k, v in domain.items()),
            "capacity geometry/declaration mismatch")
    return domain

def connector_delta(domain_id):
    domain = capacity_domain(domain_id)
    return dict(iodepth=8, staging_mem=domain["staging_mem_gib"],
                preload_lookahead_requests=domain["preload_lookahead_requests"])

def expected_connector_variant(original, domain_id):
    require(type(original) is dict, "original connector missing")
    require(type(original.get("iodepth")) is int and original["iodepth"] == 4 and
            type(original.get("staging_mem")) in (int, float) and
            not isinstance(original["staging_mem"], bool) and original["staging_mem"] == 1 and
            type(original.get("preload_lookahead_requests")) is int and
            original["preload_lookahead_requests"] == 1,
            "original capacity/I/O baseline differs")
    return dict(original, **connector_delta(domain_id))

def load_baseline(baseline_manifest_path=None, *, root=None):
    project = ROOT if root is None else Path(root)
    path = project / BASELINE_MANIFEST if baseline_manifest_path is None else Path(baseline_manifest_path)
    require(digest(path) == BASELINE_MANIFEST_SHA256, "frozen baseline manifest changed")
    baseline = json.loads(path.read_text())
    require(baseline["profile"] == "mixed_readwrite" and baseline["io_depth"] == 8 and
            baseline["staging_bytes"] == 1073741824, "wrong baseline capacity/profile")
    return baseline

def make_manifest(domain_id, baseline_manifest_path=None, *, root=None):
    baseline = load_baseline(baseline_manifest_path, root=root)
    domain = capacity_domain(domain_id)
    result = copy.deepcopy(baseline)
    result.update(schema_version=2, capacity_domain_id=domain_id,
                  capacity_domain=domain, staging_bytes=domain["staging_bytes"],
                  preload_lookahead_requests=domain["preload_lookahead_requests"],
                  baseline_manifest=BASELINE_MANIFEST,
                  baseline_manifest_sha256=BASELINE_MANIFEST_SHA256)
    return result

def validate(manifest, candidate, gpu_uuid, *, root=None):
    require(type(manifest) is dict and type(manifest.get("schema_version")) is int and
            manifest["schema_version"] == 2, "capacity manifest schema required")
    domain = validate_declaration(manifest)
    baseline = load_baseline(root=root)
    validate_baseline(baseline, candidate, gpu_uuid)
    require(manifest == make_manifest(domain["capacity_domain_id"], root=root),
            "frozen workload or capacity manifest changed")
    require(type(manifest["staging_bytes"]) is int and
            type(manifest["preload_lookahead_requests"]) is int,
            "typed capacity fields required")
    return dict(manifest["engine"])

def required_new_bytes(profile):
    require(profile == "mixed_readwrite", "capacity domain only permits frozen mixed profile")
    return 3 * 1024**3

def validate_acquisition_args(args):
    domain = capacity_domain(args.capacity_domain)
    require(args.domain == 16384 and args.max_num_seqs == 2 and args.iodepth == 8,
            "capacity acquisition requires C2/d8/long domain")
    require(args.sizes == str(POINT_TOKENS) and args.reps == POINT_REPS,
            "capacity point/repetition count changed")
    require(args.mode in ("cold", "paired") and args.prompt_manifest is not None and
            args.curves is None, "original heldout native point only")
    require(not args.cached_reference_logprobs or
            (args.mode == "paired" and not args.native_hot_diagnostic and not args.diagnostic_logprobs),
            "cached reference requires paired mode only")
    require(not args.diagnostic_logprobs or args.native_hot_diagnostic,
            "logprob diagnostic requires native hot reference")
    require(not args.native_hot_diagnostic or args.mode == "cold",
            "native hot diagnostic must have no external connector")
    return domain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-domain", choices=DOMAIN_IDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = make_manifest(args.capacity_domain)
    with args.output.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status="CPU_CAPACITY_MANIFEST_DEFINED_NOT_QUALIFIED",
                         path=str(args.output), sha256=digest(args.output),
                         capacity_domain=manifest["capacity_domain"])))

if __name__ == "__main__":
    main()
