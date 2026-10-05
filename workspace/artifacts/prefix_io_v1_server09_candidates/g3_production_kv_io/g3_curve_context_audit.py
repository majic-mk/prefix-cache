"""Read-only CPU context audit; never prices, edits or qualifies a GPU curve.

Run the pinned original LoadPlanner separately. A matching metadata audit is
still insufficient without the underlying real calibration/content evidence.
"""
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def inspect_curve_context(path, ref, context):
    """Audit exact existing bytes against a planned current environment.

    Caller metadata does not establish actual GPU availability or measurements.
    Return factual discrepancies and missing evidence, never runtime authority.
    """
    require(type(context) is dict and set(context) == {
        "gpu_uuid", "model_name", "kv_dtype", "kv_bytes_per_token", "source_lock_sha256"},
        "explicit complete expected context required")
    for name in ("gpu_uuid", "model_name", "kv_dtype", "source_lock_sha256"):
        require(type(context[name]) is str and 0 < len(context[name]) <= 4096,
                "bounded context string required: " + name)
    require(context["gpu_uuid"].startswith("GPU-"), "specific expected GPU UUID required")
    require(type(context["kv_bytes_per_token"]) is int and context["kv_bytes_per_token"] > 0,
            "positive exact density required")
    require(len(context["source_lock_sha256"]) == 64 and
            all(c in "0123456789abcdef" for c in context["source_lock_sha256"]), "source lock SHA required")
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular unchanged curve file required")
    require(type(ref) is dict and {"bytes", "sha256"} <= set(ref), "curve byte reference required")
    require(type(ref["bytes"]) is int and 0 < ref["bytes"] <= 524288, "bounded curve bytes required")
    require(path.stat().st_size == ref["bytes"], "curve size differs")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "curve SHA differs")
    data = parse_json(raw)
    require(type(data) is dict and type(data.get("schema_version")) is int and
            data["schema_version"] == 2, "original v2 curve required")
    require(type(data.get("curves")) is dict and set(data["curves"]) == {"f", "g_ssd", "g_mem"},
            "all three original curves required")
    provenance = data.get("provenance")
    require(provenance is None or type(provenance) is dict, "object provenance required")
    provenance = provenance or {}
    reasons = []
    for key in ("model_name", "kv_dtype", "kv_bytes_per_token"):
        if type(data.get(key)) is not type(context[key]) or data.get(key) != context[key]:
            reasons.append(key + "_mismatch")
    if provenance.get("gpu_uuid") != context["gpu_uuid"]:
        reasons.append("gpu_uuid_mismatch_or_missing")
    if provenance.get("source_lock_sha256") != context["source_lock_sha256"]:
        reasons.append("current_source_lock_binding_missing_or_mismatch")
    if provenance.get("candidate_only") is not False:
        reasons.append("candidate_only_or_qualification_unknown")
    if provenance.get("independent_content_validation") is not True:
        reasons.append("independent_content_validation_missing_or_false")
    allowed_consumer = provenance.get("allowed_consumer")
    if allowed_consumer is not None:
        reasons.append("restricted_consumer_requires_separate_evidence_review")
    ranges = {}
    for name, curve in data["curves"].items():
        require(type(curve) is dict and type(curve.get("knots")) is dict and curve["knots"],
                "nonempty original curve knots required")
        points = []
        for key, value in curve["knots"].items():
            require(type(key) is str and key.isascii() and key.isdecimal() and int(key) > 0 and
                    str(int(key)) == key, "canonical positive knot token required")
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    "finite nonnegative measured cost required")
            points.append(int(key))
        ranges[name] = [min(points), max(points)]
    for name, key in (("g_ssd", "break_even_ssd_tokens"), ("g_mem", "break_even_mem_tokens")):
        threshold = data.get(key)
        require(type(threshold) is int and threshold >= 0, "exact original threshold required")
        if threshold > min(ranges[name][1], ranges["f"][1]):
            reasons.append(name + "_threshold_above_measured_support")
    # Re-read before returning: this audit neither changes bytes nor author policy.
    require(path.read_bytes() == raw, "curve changed during read-only audit")
    return dict(status="CPU_EXISTING_CURVE_BLOCKED" if reasons else "CPU_METADATA_MATCH_ONLY_UNQUALIFIED",
                curve_ref={k: ref[k] for k in ("bytes", "sha256")},
                model_name=data.get("model_name"), gpu_uuid=provenance.get("gpu_uuid"),
                curve_support=ranges, rejection_reasons=reasons,
                independent_runtime_evidence_review="NOT_PERFORMED", original_LoadPlanner_executed=False,
                curve_edited=False, GPU_operations=0, gpu_authorized=False,
                production_qualified=False, SSD_restore_verified=False, performance_claim=False)
