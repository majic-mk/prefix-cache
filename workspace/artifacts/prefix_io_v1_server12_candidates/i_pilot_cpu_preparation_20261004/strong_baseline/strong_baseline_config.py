"""CPU-only U/I configuration bridge for the retained native strong baseline.

No engine, GPU, author cluster launcher or storage operation is imported or run.
The caller must separately pass prospective workload/cost/SLO/guard gates.
"""
from __future__ import annotations
from copy import deepcopy
import json
import math

P4 = "prefix_io_p4_policy"
PARENT = "prefix_io_parent_admission"


def require(value, reason):
    if not value:
        raise ValueError("STRONG_BASELINE_REJECTED: " + reason)


def integer(value, name, lower=1, upper=None):
    require(type(value) is int and value >= lower and (upper is None or value <= upper), name)
    return value


def identity(value):
    require(type(value) is str and value.strip() == value and 0 < len(value) <= 128, "run identity")
    return value


def path_text(value, name):
    require(type(value) is str and value and value.startswith("/") and "\\" not in value and
            not any(p in (".", "..", "") for p in value[1:].split("/")), name + " is an absolute local server path")
    return value


def validate_common(engine, sampling, extra):
    require(type(engine) is type(sampling) is type(extra) is dict, "exact common dictionaries")
    for key in ("enable_prefix_caching", "enable_chunked_prefill"):
        require(engine.get(key) is True, "retain original " + key)
    require(engine.get("tensor_parallel_size") == 1 and type(engine["tensor_parallel_size"]) is int,
            "single original model group")
    require(engine.get("kv_transfer_config") is None, "one connector bridge only")
    for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes", "block_size"):
        integer(engine.get(key), key)
    require(engine["block_size"] == 16, "current exact Prefix block geometry")
    require(sampling.get("ignore_eos") is True and type(sampling.get("temperature")) in (int, float)
            and sampling.get("temperature") == 0,
            "fixed full deterministic outputs")
    require(type(sampling.get("min_tokens")) is int and sampling.get("min_tokens") == sampling.get("max_tokens")
            and sampling["min_tokens"] >= 128, "complete sustained decode rather than one-output calibration")
    require(extra.get("spec_name") == "PyKvCacheOffloadingSpec" and extra.get("spec_module_path") == "py_kvcache.vllm",
            "original py-kvcache specification")
    require(extra.get("load_planner") == "on", "original LoadPlanner on for formal U/I")
    require(extra.get("enable_preload") is True and extra.get("preload_share_staging") is True,
            "retain preload and shared staging")
    integer(extra.get("preload_lookahead_requests"), "original scheduler preload lookahead")
    integer(extra.get("iodepth"), "same native I/O depth")
    require(extra.get("io_backend") in ("linux_aio", "io_uring"), "original common backend")
    memory = extra.get("staging_mem")
    require(type(memory) in (int, float) and math.isfinite(memory) and memory > 0, "bounded original staging budget")
    require(extra.get("staging_cache") in ("lru", "arc"), "retain the chosen native shared staging cache")
    path_text(extra.get("prefix_cache_break_even_path"), "original break-even curve input")
    require(P4 not in extra and "shared_storage_path" not in extra and PARENT not in extra,
            "only builder binds policy, namespace and common parent identity")
    denied = ("prefix_io_stage_policy", "prefix_io_store_order", "prefix_io_start_budget")
    require(not any(k in extra for k in denied), "no old research strategy combined with I")
    require(not any(k in extra for k in ("fusion", "disable_fusion", "serial_pipeline", "disable_pipeline")),
            "do not override native fusion or asynchronous pipeline")


def build_runtime_pair(engine_common, sampling_common, connector_common, *, storage_paths, run_ids,
                       i_policy, max_accepted_parents=8):
    validate_common(engine_common, sampling_common, connector_common)
    require(type(storage_paths) is type(run_ids) is dict and set(storage_paths) == set(run_ids) == {"U", "I"},
            "both explicit native arms")
    require(storage_paths["U"] != storage_paths["I"], "separate namespace with equivalent initial state")
    require(run_ids["U"] != run_ids["I"], "distinct journal run identities")
    integer(max_accepted_parents, "common native parent cap", upper=64)
    require(max_accepted_parents in (1, 2, 4, 8, 16, 32, 64), "existing common parent-cap domain")
    require(type(i_policy) is dict and i_policy.get("mode") == "interference" and
            i_policy.get("run_id") == run_ids["I"], "same-run I-only policy")
    require(set(i_policy) == {"schema_version", "run_id", "mode", "sample_max_age_ns", "max_wait_ns",
                             "internal_step_budget_ns", "candidate_batches", "fixed_stage_policy", "cost_table"},
            "strict original P4 policy fields")
    require(i_policy.get("fixed_stage_policy") is None, "I-only preserves native U continuation")
    pair = {}
    for arm in ("U", "I"):
        run = identity(run_ids[arm])
        extra = deepcopy(connector_common)
        extra["shared_storage_path"] = path_text(storage_paths[arm], "arm namespace")
        extra[PARENT] = dict(schema_version=1, run_id=run, max_accepted_parents=max_accepted_parents)
        if "prefix_io_observation_run_id" in extra:
            extra["prefix_io_observation_run_id"] = run
        extra[P4] = {"mode": "off"} if arm == "U" else deepcopy(i_policy)
        engine = deepcopy(engine_common)
        engine["kv_transfer_config"] = dict(kv_connector="OffloadingConnector", kv_role="kv_both",
                                            kv_connector_extra_config=extra)
        pair[arm] = dict(engine=engine, sampling=deepcopy(sampling_common))
    validate_runtime_pair(pair)
    return pair


def validate_runtime_pair(pair):
    require(type(pair) is dict and set(pair) == {"U", "I"}, "two arms only")
    normalized, namespaces, run_names = [], [], []
    for arm in ("U", "I"):
        require(type(pair[arm]) is dict and set(pair[arm]) == {"engine", "sampling"}, "arm data only")
        item = deepcopy(pair[arm])
        transfer = item["engine"].get("kv_transfer_config")
        require(type(transfer) is dict and set(transfer) == {"kv_connector", "kv_role", "kv_connector_extra_config"}
                and transfer["kv_connector"] == "OffloadingConnector" and transfer["kv_role"] == "kv_both",
                "original transfer route")
        extra = transfer["kv_connector_extra_config"]
        require(type(extra) is dict, "connector mapping")
        policy = extra.pop(P4)
        require(policy == {"mode": "off"} if arm == "U" else type(policy) is dict and policy.get("mode") == "interference",
                "off/native U versus I-only")
        namespaces.append(path_text(extra.pop("shared_storage_path"), "explicit namespace"))
        parent = extra.pop(PARENT)
        require(type(parent) is dict and set(parent) == {"schema_version", "run_id", "max_accepted_parents"}
                and type(parent["schema_version"]) is int and parent["schema_version"] == 1, "strict original parent fields")
        run = identity(parent.pop("run_id"))
        run_names.append(run)
        integer(parent.get("max_accepted_parents"), "common parent cap", upper=64)
        require(parent["max_accepted_parents"] in (1, 2, 4, 8, 16, 32, 64), "native parent cap domain")
        if arm == "I":
            require(policy.get("run_id") == run and policy.get("fixed_stage_policy") is None, "I common identity")
        if "prefix_io_observation_run_id" in extra:
            require(extra.pop("prefix_io_observation_run_id") == run, "common observer run identity")
        transfer["kv_connector_extra_config"] = extra
        engine = deepcopy(item["engine"])
        engine["kv_transfer_config"] = None
        validate_common(engine, item["sampling"], extra)
        normalized.append((item, parent))
    require(normalized[0] == normalized[1], "all original engine/sampling/planner/resource settings must match")
    require(namespaces[0] != namespaces[1] and run_names[0] != run_names[1], "isolated namespaces and journal identities")
    return dict(common_runtime_equal=True, policy_changes_only="off versus interference",
                independent_workload_deadline_gate_required=True, new_strong_domain_cost_gate_required=True,
                gpu_effect_qualified=False)


def author_trace_arguments(*, shared_storage_path, dataset_path, max_prompts, arrival_rate,
                           max_concurrency, seed, output_len=128):
    """Arguments consumed by the SHA-bound original author parser.

    No cluster is launched. Prefix injection and storage wiping are omitted.
    The local dataset and arrival protocol are separately frozen by the caller.
    """
    path_text(shared_storage_path, "author shared storage")
    path_text(dataset_path, "existing trace dataset")
    integer(max_prompts, "bounded trace requests", upper=32)
    integer(max_concurrency, "bounded original concurrency", upper=8)
    integer(seed, "trace seed", lower=0)
    integer(output_len, "full decode output", lower=128)
    require(type(arrival_rate) in (float, int) and math.isfinite(arrival_rate) and arrival_rate > 0,
            "prospectively declared positive arrival rate")
    return ["--shared-storage-path", shared_storage_path, "--dataset-path", dataset_path,
            "--max-prompts", str(max_prompts), "--arrival-rate", str(arrival_rate),
            "--max-concurrency", str(max_concurrency), "--seed", str(seed),
            "--output-len", str(output_len), "--completions", "--no-launch"]


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="CPU configuration preparation; cannot authorize GPU effects")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.input.is_symlink() and args.input.is_file() and args.input.stat().st_size <= 1024**2,
            "bounded local input")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    document = json.loads(args.input.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda value: require(False, "nonfinite JSON"))
    require(type(document) is dict and set(document) == {"engine_common", "sampling_common", "connector_common",
                                                        "storage_paths", "run_ids", "i_policy", "max_accepted_parents"},
            "explicit configuration input fields")
    pair = build_runtime_pair(**document)
    record = dict(schema="strong_native_u_i_cpu_configuration_v1", configurations=pair,
                  contract=validate_runtime_pair(pair), gpu_effect_qualified=False,
                  native_execution_verified=False, cost_receipt=None,
                  required_external_gates=["independent_development_deadline", "normal_frozen_request_stream",
                                           "new_strong_domain_calibration_and_holdout", "on_gpu_lifecycle", "original_guard"])
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(status="CPU_CONFIG_CREATED_EFFECT_BLOCKED", gpu_jobs=0)))


if __name__ == "__main__":
    main()
