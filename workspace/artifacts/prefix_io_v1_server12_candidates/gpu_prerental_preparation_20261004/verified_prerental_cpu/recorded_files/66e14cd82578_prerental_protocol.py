"""CPU-only prospective trace, budget, and investment gates.

This module does not import a model, native backend, network client, or GPU guard.
It replays selected unchanged author ASTs and writes append-only evidence files.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "natural_trace_workload_v1"
ORIGINAL_GPU_LIMIT_SECONDS = 28800
REMAINING_SNAPSHOT_SECONDS = 4121.104761094321
INVESTMENT_GAIN_TARGET_FRACTION = 0.10
LOW_CONTENTION_REGRESSION_TARGET_FRACTION = 0.02
AUTHOR_TRACE_SHA256 = "83713db1b011a954616896ec0ccfa05dc766805e80ed8defdef0da65100ff99e"
AUTHOR_COMMON_SHA256 = "a331d1595b815c3c8fc63d3e2f236264c2f181ece7a9de51f50beb4703338df5"
ORIGINAL_P3_LOW_CONTENTION_SHA256 = "3ef3b7a65d87cb8f4d1cb88c885137ec29e1eebdfebe0aedf420e803d9d0a5d9"
ORIGINAL_P3_INDEX_SHA256 = "ab21d8b18acf2328d8918f45db3d1a94823133bee0ed58251553209a610931fe"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def positive_int(value: Any, field: str) -> int:
    require(type(value) is int and value > 0, f"{field} must be positive integer")
    return value


def sha256_value(value: Any, field: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"{field} must be lowercase SHA-256 hex")
    return value


def verify_closed_ref(ref: dict, field: str) -> dict:
    require(isinstance(ref, dict), f"{field} ref required")
    sha256_value(ref.get("sha256"), field)
    path = Path(ref.get("path", ""))
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), f"{field} must refer to actual regular file")
    require(type(ref.get("bytes")) is int and ref["bytes"] >= 0 and path.stat().st_size == ref["bytes"], f"{field} file size mismatch")
    require(file_sha(path) == ref["sha256"], f"{field} actual byte SHA mismatch")
    return dict(ref)


def validate_service_SLO(value: Any) -> dict | None:
    if value is None:
        return None
    require(isinstance(value, dict) and set(value) == {"schema", "origin", "TTFT_ns", "request_ITL_P95_ns"}, "strict service SLO fields")
    require(value["schema"] == "independent_service_SLO_v1" and value["origin"] == "independent_requirement_before_development", "service SLO origin")
    positive_int(value["TTFT_ns"], "TTFT SLO")
    positive_int(value["request_ITL_P95_ns"], "request ITL P95 SLO")
    return value


def clock_scope(value: Any) -> dict:
    require(isinstance(value, dict) and set(value) == {"host_id", "boot_id", "clock"}, "same host/boot clock scope required")
    require(isinstance(value["host_id"], str) and bool(value["host_id"]), "host id required")
    require(isinstance(value["boot_id"], str) and re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", value["boot_id"]) is not None, "boot id required")
    require(value["clock"] == "CLOCK_MONOTONIC", "host monotonic clock required")
    return value


def append_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")


def author_cpu_namespace(trace_source: Path, common_source: Path) -> dict[str, Any]:
    """Extract exact pure parser/schedule ASTs; importing the author script launches no clients."""
    require(file_sha(trace_source) == AUTHOR_TRACE_SHA256, "author trace source drift")
    require(file_sha(common_source) == AUTHOR_COMMON_SHA256, "author RequestSpec source drift")
    wanted = {"prompt_from_sharegpt_item", "prompt_from_json_item", "load_trace_prompts", "build_global_specs"}
    trace_tree = ast.parse(trace_source.read_text(encoding="utf-8"))
    common_tree = ast.parse(common_source.read_text(encoding="utf-8"))
    nodes = [n for n in common_tree.body if (isinstance(n, ast.ClassDef) and n.name == "RequestSpec") or
             (isinstance(n, ast.FunctionDef) and n.name == "generate_request_schedule")]
    require(len(nodes) == 2, "missing original RequestSpec or request schedule")
    functions = [n for n in trace_tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    require({n.name for n in functions} == wanted, "missing original author functions")
    namespace = {"json": json, "random": random, "math": math, "Path": Path, "Any": Any, "dataclass": dataclass}
    module = ast.Module(body=[*nodes, *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "unchanged_author_cpu_ast", "exec"), namespace)
    return namespace


def inspect_trace(dataset: Path, trace_source: Path, common_source: Path, expected_sha256: str | None = None) -> dict:
    """Read all prompts in author order. Inspection never injects, caps, filters by performance, or writes the dataset."""
    require(dataset.is_file() and not dataset.is_symlink(), "dataset must be existing regular file")
    before = file_sha(dataset)
    if expected_sha256 is not None:
        require(before == expected_sha256, "dataset source drift")
    namespace = author_cpu_namespace(trace_source, common_source)
    prompts = namespace["load_trace_prompts"](str(dataset), None)
    require(file_sha(dataset) == before, "dataset mutated while inspecting")
    rows = [{"request_id": index, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
             "utf8_bytes": len(prompt.encode("utf-8"))} for index, prompt in enumerate(prompts)]
    return {"schema": "author_trace_inspection_v1", "dataset": str(dataset), "dataset_sha256": before,
            "dataset_bytes": dataset.stat().st_size, "author_trace_sha256": AUTHOR_TRACE_SHA256,
            "author_common_sha256": AUTHOR_COMMON_SHA256, "accepted_prompt_count": len(prompts),
            "ordered_prompt_digest": digest(rows), "records": rows,
            "author_max_prompts": None, "original_data_bytes_unchanged": True,
            "duplicate_exact_prompts": len(prompts) - len({r["prompt_sha256"] for r in rows}),
            "prefix_family_proof_available": False, "natural_recorded_arrival_timestamps_preserved": False,
            "gpu_operations": 0, "effect_qualified": False}


def freeze_trace(dataset: Path, trace_source: Path, common_source: Path, declaration: dict, tokenizer_receipt: dict) -> dict:
    """Freeze every author-accepted prompt, with externally verified tokenizer/family evidence.

    The declaration chooses contiguous partition cuts before GPU outcomes. Explicit
    tokenizer receipts are mandatory; text equality alone does not establish shared
    prefix-family closure. No family/request is silently dropped or reordered.
    """
    require(declaration.get("schema") == "natural_trace_declaration_v1", "trace declaration schema")
    require(declaration.get("origin") == "prospective_before_any_new_gpu_outcome", "posthoc trace declaration")
    require(declaration.get("prefix_injection") is False and declaration.get("per_request_cache_reset") is False,
            "artificial injection or cache reset")
    require(declaration.get("artificial_throttle") is False, "artificial throttle")
    require(declaration.get("initial_cache_state") == "fresh_equal_namespace_preserved_through_whole_partition", "initial cache contract")
    inspection = inspect_trace(dataset, trace_source, common_source, declaration.get("dataset_sha256"))
    count = inspection["accepted_prompt_count"]
    cuts = declaration.get("partition_counts")
    require(isinstance(cuts, dict) and set(cuts) == {"calibration", "development", "evaluation"}, "partition counts")
    for name, value in cuts.items():
        positive_int(value, name)
    require(sum(cuts.values()) == count, "all author accepted requests must be retained")
    output_len = positive_int(declaration.get("output_tokens"), "output_tokens")
    require(output_len >= 128, "full qualification and effect output contract requires at least 128 tokens")
    max_model_len = positive_int(declaration.get("max_model_len"), "max_model_len")
    concurrency = positive_int(declaration.get("max_concurrency"), "max_concurrency")
    seed = declaration.get("schedule_seed")
    require(type(seed) is int and seed >= 0, "schedule seed")
    rate = declaration.get("arrival_rate")
    require(rate is None or (type(rate) in (int, float) and math.isfinite(rate) and rate > 0), "arrival rate")
    require(tokenizer_receipt.get("schema") == "cpu_actual_tokenizer_family_receipt_v1", "tokenizer receipt schema")
    require(tokenizer_receipt.get("synthetic_fixture") is False, "fixture cannot qualify workload")
    require(tokenizer_receipt.get("dataset_sha256") == inspection["dataset_sha256"], "tokenizer dataset mismatch")
    require(tokenizer_receipt.get("ordered_prompt_digest") == inspection["ordered_prompt_digest"], "tokenizer prompt order drift")
    require(tokenizer_receipt.get("model_manifest_sha256") == declaration.get("model_manifest_sha256"), "tokenizer model mismatch")
    require(isinstance(tokenizer_receipt.get("tokenizer_source_refs"), list) and tokenizer_receipt["tokenizer_source_refs"], "tokenizer source refs required")
    for ref in tokenizer_receipt["tokenizer_source_refs"]:
        verify_closed_ref(ref, "actual tokenizer source")
    tokenizer_result_ref = verify_closed_ref(tokenizer_receipt.get("tokenization_result_ref"), "actual CPU tokenizer result")
    tokenizer_result = json.loads(Path(tokenizer_result_ref["path"]).read_text(encoding="utf-8"))
    require(tokenizer_result.get("schema") == "actual_cpu_tokenizer_result_v1" and tokenizer_result.get("exit_code") == 0
            and tokenizer_result.get("synthetic_fixture") is False, "actual CPU tokenizer result required")
    require(tokenizer_result.get("dataset_sha256") == inspection["dataset_sha256"] and tokenizer_result.get("ordered_prompt_digest") == inspection["ordered_prompt_digest"], "actual tokenizer input closure")
    require(tokenizer_receipt.get("family_proof") == "all_sessions_documents_and_public_prefix_families_closed_before_partitioning", "family closure proof required")
    tokenized = tokenizer_receipt.get("records")
    require(isinstance(tokenized, list) and len(tokenized) == count, "tokenizer request completeness")
    require(tokenizer_result.get("tokenized_records_sha256") == digest(tokenized), "actual tokenizer result record byte closure")
    namespace = author_cpu_namespace(trace_source, common_source)
    schedule = namespace["build_global_specs"](count, rate, random.Random(seed))
    records = []
    family_partitions: dict[str, str] = {}
    prompt_partitions: dict[str, str] = {}
    start_dev = cuts["calibration"]
    start_eval = start_dev + cuts["development"]
    for index, (item, prompt, spec) in enumerate(zip(tokenized, inspection["records"], schedule)):
        require(item.get("request_id") == index and item.get("prompt_sha256") == prompt["prompt_sha256"], "tokenized prompt identity/order drift")
        ids = item.get("prompt_token_ids")
        require(isinstance(ids, list) and ids and all(type(t) is int and t >= 0 for t in ids), "actual token ids required")
        require(len(ids) + output_len <= max_model_len, "model context overflow")
        family = item.get("prefix_family")
        require(isinstance(family, str) and family, "verified prefix family required")
        split = "calibration" if index < start_dev else "development" if index < start_eval else "evaluation"
        require(family_partitions.setdefault(family, split) == split, "prefix family leaks across partitions")
        require(prompt_partitions.setdefault(prompt["prompt_sha256"], split) == split, "identical prompt leaks across partitions")
        scheduled_ns = math.ceil(spec.scheduled_time * 1_000_000_000)
        records.append({"request_id": index, "prompt_sha256": prompt["prompt_sha256"], "prompt_token_ids": ids,
                        "scheduled_ns": scheduled_ns, "max_tokens": output_len, "min_tokens": output_len,
                        "seed": seed, "split": split, "prefix_family": family})
    result = {"schema": SCHEMA, "origin": "frozen_actual_existing_trace_no_new_gpu_outcomes", "dataset_sha256": inspection["dataset_sha256"],
              "ordered_prompt_digest": inspection["ordered_prompt_digest"], "author_trace_sha256": AUTHOR_TRACE_SHA256,
              "author_common_sha256": AUTHOR_COMMON_SHA256, "tokenizer_receipt_digest": digest(tokenizer_receipt),
              "declaration_digest": digest(declaration), "model_manifest_sha256": declaration["model_manifest_sha256"],
              "records": records, "partition_counts": cuts, "max_concurrency": concurrency,
              "arrival_rate": rate, "schedule_seed": seed, "schedule_origin": "unchanged_author_build_global_specs",
              "recorded_natural_arrival_claim": False, "no_prefix_injection": True, "no_per_request_cache_reset": True,
              "no_request_drops_or_reorder": True, "initial_cache_state": declaration["initial_cache_state"],
              "cost_domain_covered": False, "gpu_effect_qualified": False, "gpu_operations": 0}
    result["workload_sha256"] = digest(result)
    return result


def freeze_qualification_trace(dataset: Path, trace_source: Path, common_source: Path, declaration: dict) -> dict:
    """Freeze a prospectively bounded author prompt prefix for off qualification.

    No token/family truth is asserted before the actual model tokenizer is used.
    This weaker manifest is permanently unsuitable for fitting or effect claims.
    """
    require(declaration.get("schema") == "natural_off_qualification_declaration_v1", "qualification declaration schema")
    require(declaration.get("origin") == "prospective_before_any_new_gpu_outcome", "posthoc qualification trace")
    require(declaration.get("prefix_injection") is False and declaration.get("artificial_throttle") is False,
            "qualification must preserve natural prompt content")
    require(declaration.get("per_request_cache_reset") is False, "per request reset")
    require(declaration.get("initial_cache_state") == "fresh_equal_namespace_preserved_through_whole_partition", "initial cache contract")
    inspection = inspect_trace(dataset, trace_source, common_source, declaration.get("dataset_sha256"))
    limit = positive_int(declaration.get("max_prompts"), "qualification prompt limit")
    require(limit <= 32, "bounded qualification limit is 32")
    count = min(limit, inspection["accepted_prompt_count"])
    require(count > 0, "no natural prompts in existing source")
    output_len = positive_int(declaration.get("output_tokens"), "output tokens")
    require(output_len >= 128, "full qualification output contract requires at least 128 tokens")
    concurrency = positive_int(declaration.get("max_concurrency"), "concurrency")
    require(concurrency <= 8, "bounded qualification concurrency is 8")
    seed = declaration.get("schedule_seed")
    require(type(seed) is int and seed >= 0, "schedule seed")
    rate = declaration.get("arrival_rate")
    require(rate is None or (type(rate) in (int, float) and math.isfinite(rate) and rate > 0), "arrival rate")
    namespace = author_cpu_namespace(trace_source, common_source)
    prompts = namespace["load_trace_prompts"](str(dataset), count)
    specs = namespace["build_global_specs"](count, rate, random.Random(seed))
    rows = [{"request_id": index, "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
             "scheduled_time_s": spec.scheduled_time, "scheduled_ns": math.ceil(spec.scheduled_time * 1_000_000_000),
             "max_tokens": output_len, "min_tokens": output_len, "seed": seed, "split": "qualification",
             "prefix_family": None, "prompt_token_ids": None} for index, (prompt, spec) in enumerate(zip(prompts, specs))]
    result = {"schema": "natural_off_qualification_workload_v1", "origin": "existing_trace_first_N_original_author_order",
              "dataset_sha256": inspection["dataset_sha256"], "dataset_accepted_prompt_count": inspection["accepted_prompt_count"],
              "selected_prompt_count": count, "unselected_tail_count": inspection["accepted_prompt_count"] - count,
              "prospective_declaration_digest": digest(declaration), "records": rows,
              "author_trace_sha256": AUTHOR_TRACE_SHA256, "author_common_sha256": AUTHOR_COMMON_SHA256,
              "schedule_origin": "unchanged_author_build_global_specs", "recorded_natural_arrival_claim": False,
              "max_concurrency": concurrency, "arrival_rate": rate, "schedule_seed": seed,
              "model_manifest_sha256": declaration.get("model_manifest_sha256"),
              "initial_cache_state": declaration["initial_cache_state"], "no_prefix_injection": True,
              "no_per_request_cache_reset": True, "selected_requests_not_reordered_or_dropped": True,
              "fit_or_evaluation_input_allowed": False, "prefix_family_closure_proved": False,
              "actual_prompt_ids_capture_required": True, "SLO": None, "gpu_effect_qualified": False, "gpu_operations": 0}
    result["workload_sha256"] = digest(result)
    return result


def freeze_original_p3_qualification(dataset: Path, index: Path, source_refs: Path,
                                     trace_source: Path, common_source: Path, model_manifest_sha256: str) -> dict:
    """Bind the existing original 12-request P3 artifact, never generate a new trace."""
    sha256_value(model_manifest_sha256, "model manifest")
    require(file_sha(dataset) == ORIGINAL_P3_LOW_CONTENTION_SHA256, "original P3 request bytes drift")
    require(file_sha(index) == ORIGINAL_P3_INDEX_SHA256, "original P3 index bytes drift")
    ref_rows = json.loads(source_refs.read_text(encoding="utf-8"))["files"]
    dataset_refs = [ref for ref in ref_rows if ref.get("sha256") == ORIGINAL_P3_LOW_CONTENTION_SHA256]
    index_refs = [ref for ref in ref_rows if ref.get("sha256") == ORIGINAL_P3_INDEX_SHA256]
    require(len(dataset_refs) == 1 and len(index_refs) == 1, "actual original P3 source refs required")
    dataset_ref, index_ref = dataset_refs[0], index_refs[0]
    require(dataset.stat().st_size == dataset_ref.get("bytes") and index.stat().st_size == index_ref.get("bytes"), "original P3 source sizes")
    indexed = json.loads(index.read_text(encoding="utf-8"))["low_contention"]
    require(indexed["sha256"] == ORIGINAL_P3_LOW_CONTENTION_SHA256 and indexed["requests"] == 12, "P3 index binding")
    require(str(dataset_ref["path"]).endswith(indexed["path"]), "P3 canonical source path binding")
    data = json.loads(dataset.read_text(encoding="utf-8"))
    require(data.get("profile") == "low_contention" and data.get("seed") == 0 and data.get("output_tokens") == 128, "original P3 profile")
    require(data.get("scope") == "controlled synthetic-token development replay; no artificial I/O delay, not a production QA dataset", "controlled provenance")
    rows = data.get("requests")
    require(isinstance(rows, list) and len(rows) == 12, "original P3 complete request stream")
    namespace = author_cpu_namespace(trace_source, common_source)
    specs = namespace["generate_request_schedule"](12, 512, 512, 0.75, 0.90, 0.90, 0.5, random.Random(0))
    records = []
    for i, (row, spec) in enumerate(zip(rows, specs)):
        expected_spec = dict(vars(spec))
        observed_spec = {key: value for key, value in row.items() if key != "prompt_token_ids"}
        require(observed_spec == expected_spec, "original P3 request spec or arrival drift")
        tokens = row.get("prompt_token_ids")
        require(isinstance(tokens, list) and len(tokens) == 512 and all(type(token) is int and token >= 0 for token in tokens), "original P3 exact token geometry")
        if row["reuse_source_id"] is not None:
            length = row["reuse_prefix_len"]
            require(tokens[:length] == rows[row["reuse_source_id"]]["prompt_token_ids"][:length], "original P3 reuse identity")
        records.append({"request_id": row["request_id"], "prompt": None, "prompt_token_ids": tokens,
                        "prompt_sha256": digest(tokens), "scheduled_time_s": row["scheduled_time"],
                        "scheduled_ns": math.ceil(row["scheduled_time"] * 1_000_000_000), "max_tokens": 128,
                        "min_tokens": 128, "seed": data["seed"], "split": "qualification", "prefix_family": None,
                        "original_request_spec": observed_spec})
    result = {"schema": "controlled_original_p3_qualification_workload_v1", "workload_kind": "controlled_original_P3_mechanism",
              "origin": "existing_original_P3_low_contention_bytes_no_new_token_generation", "source_manifest_ref": dataset_ref,
              "source_index_ref": index_ref, "model_manifest_sha256": model_manifest_sha256,
              "records": records, "selected_prompt_count": 12, "unselected_tail_count": 0, "max_concurrency": data["engine"]["max_num_seqs"],
              "arrival_rate": 0.5, "schedule_seed": 0, "schedule_origin": "unchanged_author_generate_request_schedule",
              "author_trace_sha256": AUTHOR_TRACE_SHA256, "author_common_sha256": AUTHOR_COMMON_SHA256,
              "initial_cache_state": "fresh_equal_namespace_preserved_through_whole_partition", "no_prefix_injection": True,
              "no_per_request_cache_reset": True, "no_artificial_throttle": True, "no_extra_warmup": True,
              "natural_trace_bound": False, "recorded_natural_arrival_claim": False, "natural_problem_claim_allowed": False,
              "fit_or_evaluation_input_allowed": False, "gpu_effect_qualified": False, "formal_goodput_allowed": False,
              "cost_domain_covered": False, "SLO": None, "gpu_operations": 0,
              "qualification_input_is_original_prior_development_not_new_heldout_data": True}
    result["workload_sha256"] = digest(result)
    return result


def union_ns(intervals: list[list[int]]) -> int:
    require(intervals and all(isinstance(i, list) and len(i) == 2 and type(i[0]) is int and type(i[1]) is int and 0 <= i[0] < i[1] for i in intervals), "invalid measured control intervals")
    ordered = sorted(intervals)
    total, start, end = 0, *ordered[0]
    for left, right in ordered[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    return total + end - start


def freeze_development_budget(declaration: dict, reserve_receipt: dict) -> dict:
    require(declaration.get("schema") == "independent_deadline_declaration_v1", "deadline declaration schema")
    require(declaration.get("origin") == "independent_requirement_before_development", "deadline must be independent of observed A/B/on outcomes")
    deadline = positive_int(declaration.get("full_control_window_deadline_ns"), "independent deadline")
    authority_ref = verify_closed_ref(declaration.get("authority_ref"), "independent authority evidence")
    service_SLO = validate_service_SLO(declaration.get("service_SLO"))
    declaration_clock = clock_scope(declaration.get("clock_scope"))
    authority = json.loads(Path(authority_ref["path"]).read_text(encoding="utf-8"))
    require(authority.get("schema") == "independent_deadline_authority_v1" and authority.get("origin") == "independent_requirement_before_development",
            "authority must be independent prospective requirement")
    require(authority.get("full_control_window_deadline_ns") == deadline and authority.get("service_SLO") == service_SLO,
            "deadline authority byte closure mismatch")
    declared_ns = positive_int(declaration.get("declared_monotonic_ns"), "deadline declaration time")
    require(reserve_receipt.get("schema") == "actual_development_control_reserve_v1", "reserve receipt schema")
    require(reserve_receipt.get("actual_native_gpu_run") is True and reserve_receipt.get("synthetic_fixture") is False, "reserve needs actual development evidence")
    require(reserve_receipt.get("deadline_declaration_sha256") == digest(declaration), "deadline provenance drift")
    require(clock_scope(reserve_receipt.get("clock_scope")) == declaration_clock, "deadline and reserve must be same host and boot")
    native_result_ref = verify_closed_ref(reserve_receipt.get("native_result_ref"), "actual native development result")
    native_result = json.loads(Path(native_result_ref["path"]).read_text(encoding="utf-8"))
    require(native_result.get("schema") == "actual_development_reserve_source_v1" and native_result.get("actual_native_gpu_run") is True
            and native_result.get("synthetic_fixture") is False and native_result.get("phase") == "development", "native development source contract")
    require(native_result.get("control_intervals_sha256") == digest(reserve_receipt.get("per_step_control_intervals")), "reserve actual intervals byte closure")
    require(reserve_receipt.get("first_development_monotonic_ns", 0) > declared_ns, "deadline declared after development")
    require(reserve_receipt.get("phase") == "development" and reserve_receipt.get("first_on_has_run") is False, "must freeze before on/evaluation")
    require(reserve_receipt.get("clock") == "monotonic_ns_host_control_intervals", "mixed CUDA and host clocks")
    require(reserve_receipt.get("all_control_categories") == ["scheduler", "sampling", "output", "controller"], "missing full control reserve categories")
    intervals = reserve_receipt.get("per_step_control_intervals")
    require(isinstance(intervals, list) and intervals, "missing measured reserve intervals")
    observed = max(union_ns(step) for step in intervals)
    uncertainty = reserve_receipt.get("reserve_uncertainty_ns")
    require(type(uncertainty) is int and uncertainty >= 0, "reserve uncertainty")
    reserve = observed + uncertainty
    require(reserve < deadline, "no independent ordinary budget remains")
    return {"schema": "frozen_independent_development_budget_v1", "full_control_window_deadline_ns": deadline,
            "non_gpu_reserve_ns": reserve, "internal_step_budget_ns": deadline - reserve,
            "deadline_declaration_sha256": digest(declaration), "reserve_receipt_sha256": digest(reserve_receipt),
            "service_SLO": service_SLO, "formal_goodput_allowed": False,
            "authority_ref_verified": authority_ref, "native_result_ref_verified": native_result_ref,
            "clock_scope": declaration_clock, "CPU_math_and_byte_validation_only": True,
            "native_qualification_verifier_required_separately": True,
            "derived_from_A_max": False, "evaluation_used_to_fit": False,
            "cost_domain_coverage_required_separately": True, "on_lifecycle_required_separately": True}


def budget_plan(remaining_seconds: float, jobs: list[dict]) -> dict:
    require(type(remaining_seconds) in (int, float) and math.isfinite(remaining_seconds) and 0 <= remaining_seconds <= ORIGINAL_GPU_LIMIT_SECONDS, "remaining original budget")
    require(jobs and len({j["id"] for j in jobs}) == len(jobs), "job ids required and unique")
    total = 0
    for job in jobs:
        total += positive_int(job.get("execution_seconds"), "execution seconds") + positive_int(job.get("cleanup_seconds"), "cleanup seconds")
    return {"schema": "gpu_prerental_budget_plan_v1", "original_limit_seconds": ORIGINAL_GPU_LIMIT_SECONDS,
            "remaining_snapshot_seconds": remaining_seconds, "maximum_reserved_seconds": total,
            "remaining_after_maximum_seconds": remaining_seconds - total, "fits_snapshot": total <= remaining_seconds,
            "jobs": jobs, "attempt_consumed_on_failure": True, "retry_slots": 0,
            "live_original_guard_ledger_recheck_required": True, "actual_reservation_performed": False,
            "GPU_operations": 0, "complete_effect_result_promised": False}


def rental_decision(*, workload_bound: bool, strong_runner_cpu_ready: bool, source_lock_verified: bool,
                    guarded_off_max_seconds: int, remaining_seconds: float,
                    independent_budget_frozen: bool = False, strong_cost_covered: bool = False,
                    actual_on_qualified: bool = False, old_planner_off_cost_requested: bool = False,
                    workload_kind: str = "existing_natural_trace") -> dict:
    positive_int(guarded_off_max_seconds, "off max reservation")
    require(type(remaining_seconds) in (int, float) and math.isfinite(remaining_seconds) and 0 <= remaining_seconds <= ORIGINAL_GPU_LIMIT_SECONDS,
            "remaining original budget must be finite number from 0 to 28800")
    require(all(type(value) is bool for value in (workload_bound, strong_runner_cpu_ready, source_lock_verified,
                independent_budget_frozen, strong_cost_covered, actual_on_qualified, old_planner_off_cost_requested)), "gate fields must be exact bool")
    require(workload_kind in {"existing_natural_trace", "controlled_original_P3_mechanism"}, "qualification workload provenance")
    blocked = []
    if not workload_bound:
        blocked.append("actual_qualification_workload_source_unbound")
    if not strong_runner_cpu_ready:
        blocked.append("strong_U_original_execution_driver_not_CPU_ready")
    if not source_lock_verified:
        blocked.append("new_strong_domain_source_lock_not_verified")
    if guarded_off_max_seconds > remaining_seconds:
        blocked.append("original_remaining_GPU_budget_insufficient")
    if old_planner_off_cost_requested:
        blocked.append("planner_off_cal01_does_not_cover_strong_U_domain")
    status = "BLOCK_RENTAL_UNTIL_CPU_INPUTS_BOUND" if blocked else "BOUNDED_STRONG_U_OFF_QUALIFICATION_HAS_DIAGNOSTIC_VALUE"
    return {"schema": "gpu_prerental_decision_v1", "status": status, "prerental_blockers": blocked,
            "first_allowed_GPU_purpose": None if blocked else "strong_U_planner_on_complete_output_lifecycle_mechanism_qualification",
            "qualification_workload_kind": workload_kind,
            "natural_problem_claim_allowed": workload_kind == "existing_natural_trace" and not blocked,
            "ordinary_interference_or_effect_qualified": False,
            "first_job_max_reserved_seconds": guarded_off_max_seconds, "remaining_snapshot_seconds": remaining_seconds,
            "later_effect_gate_missing": [name for name, value in (("independent_development_budget", independent_budget_frozen),
                ("planner_on_exact_condition_cost_coverage", strong_cost_covered), ("actual_on_output_and_native_drain", actual_on_qualified)) if not value],
            "benefit_proved": False, "method_permanently_infeasible_proved": False,
            "investment_gain_target_fraction": INVESTMENT_GAIN_TARGET_FRACTION,
            "low_contention_regression_target_fraction": LOW_CONTENTION_REGRESSION_TARGET_FRACTION,
            "targets_are_not_promised_results": True, "GPU_operations": 0,
            "stop_if": ["no_normal_legal_ordinary_candidate", "no_reproducible_natural_problem", "strong_simple_baseline_has_same_effect",
                        "requires_artificial_throttle_or_scope_expansion", "original_budget_exhausted_or_native_drain_unsafe"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    freeze = commands.add_parser("freeze-trace")
    qualification = commands.add_parser("freeze-qualification")
    controlled = commands.add_parser("freeze-original-p3")
    controlled.add_argument("--dataset", type=Path, required=True)
    controlled.add_argument("--index", type=Path, required=True)
    controlled.add_argument("--source-refs", type=Path, required=True)
    controlled.add_argument("--author-trace", type=Path, required=True)
    controlled.add_argument("--author-common", type=Path, required=True)
    controlled.add_argument("--model-manifest-sha256", required=True)
    controlled.add_argument("--output", type=Path, required=True)
    for sub in (inspect, freeze, qualification):
        sub.add_argument("--dataset", type=Path, required=True)
        sub.add_argument("--author-trace", type=Path, required=True)
        sub.add_argument("--author-common", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
    inspect.add_argument("--expected-sha256")
    freeze.add_argument("--declaration", type=Path, required=True)
    freeze.add_argument("--tokenizer-receipt", type=Path, required=True)
    qualification.add_argument("--declaration", type=Path, required=True)
    development = commands.add_parser("freeze-budget")
    development.add_argument("--declaration", type=Path, required=True)
    development.add_argument("--reserve-receipt", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_trace(args.dataset, args.author_trace, args.author_common, args.expected_sha256)
    elif args.command == "freeze-trace":
        result = freeze_trace(args.dataset, args.author_trace, args.author_common,
                              json.loads(args.declaration.read_text(encoding="utf-8")), json.loads(args.tokenizer_receipt.read_text(encoding="utf-8")))
    elif args.command == "freeze-qualification":
        result = freeze_qualification_trace(args.dataset, args.author_trace, args.author_common,
                                           json.loads(args.declaration.read_text(encoding="utf-8")))
    elif args.command == "freeze-original-p3":
        result = freeze_original_p3_qualification(args.dataset, args.index, args.source_refs,
                                                  args.author_trace, args.author_common, args.model_manifest_sha256)
    else:
        result = freeze_development_budget(json.loads(args.declaration.read_text(encoding="utf-8")), json.loads(args.reserve_receipt.read_text(encoding="utf-8")))
    append_json(args.output, result)
    print(json.dumps({"status": "PASS_CPU_PROSPECTIVE_EVIDENCE", "output": str(args.output), "GPU_operations": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
