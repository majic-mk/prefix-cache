"""CPU replay of one actual, controlled A/B pair for the bounded I experiment.

This issues no production qualification, SLO claim, GPU work or native I/O.
The original evidence validators retain authority over CUDA and native I/O.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import time

VALIDATION_PATH = "artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/calibration_v2/native_conditional_cost.py"
VALIDATION_REF = dict(path=VALIDATION_PATH, bytes=41753,
    sha256="675be4f91821863e23cc82128aadc040d8a9513c3aaf9b407d6c3fccd978ccb8")
MATERIAL_FIELDS = {"gpu_uuid", "common_runtime_domain_sha256", "source_lock_ref", "model_manifest_ref",
    "kv_layout_ref", "native_source_ref", "collector_source_ref", "cuda_event_source_ref", "kernel_mode",
    "context_length", "a_ns", "b_ns", "budget_ns", "stage", "physical_bytes", "operations",
    "existing_io", "batch", "active_decode", "prefill_tokens"}
SOURCE_FIELDS = ("source_lock_ref", "model_manifest_ref", "kv_layout_ref", "native_source_ref",
                 "collector_source_ref", "cuda_event_source_ref")
_BOUND_DRIVER = None


def require(ok, message):
    if not ok:
        raise ValueError("ACTUAL_DEVELOPMENT_PAIR_REJECTED: " + message)


def bind_driver(driver):
    """Bind the already source-verified original CPU driver, once per module."""
    global _BOUND_DRIVER
    require(driver is not None and all(callable(getattr(driver, k, None)) for k in ("read", "check_ref", "load")) and
        type(getattr(driver, "MAX_GPU_SECONDS", None)) is int, "original CPU replay driver")
    require(_BOUND_DRIVER is None or _BOUND_DRIVER is driver, "replay driver cannot be replaced")
    _BOUND_DRIVER = driver
    return validate_actual_development_pair


def _model_identity(root, result, material, documents, driver):
    identity = result.get("model_identity")
    manifest = documents[material["model_manifest_ref"]["path"]]
    require(type(identity) is dict and type(manifest) is dict and
        identity.get("manifest_sha256") == identity.get("plan_sha256") == material["model_manifest_ref"]["sha256"] and
        all(identity.get(k) == manifest[k] for k in ("model_id", "revision", "provenance")),
        "actual original complete model source")
    expected = {r["path"]: {k: r[k] for k in ("bytes", "hash_algorithm", "hash")} for r in manifest["files"]}
    require(len(expected) == len(manifest["files"]) and expected, "complete unique original model manifest")
    require(identity.get("historically_verified_local_files") == expected and
        identity.get("validation_basis") == "actual_Uoff03_verified_model_hashes_plus_current_source_plan_and_leaf_stat" and
        identity.get("current_whole_model_rehash_performed") is False,
        "inherited full model hashes, never a CPU receipt or repeated weight hash")
    preceding = identity.get("actual_preceding_U_result_ref")
    require(type(preceding) is dict and preceding["path"] in documents,
        "actual preceding original model result required")
    historical = documents[preceding["path"]].get("model_identity", {})
    require(historical.get("verified_local_files") == expected and
        historical.get("manifest_sha256") == historical.get("plan_sha256") == material["model_manifest_ref"]["sha256"],
        "actual preceding full model identity join")
    directory = Path(identity["model_directory"]).resolve(strict=True)
    require(directory.is_relative_to(Path(root).resolve(strict=True)), "actual model project directory")
    stats = identity.get("current_model_leaf_stats")
    require(type(stats) is dict and set(stats) == set(expected), "complete actual model leaf metadata")
    for name, record in expected.items():
        path = directory / name
        require(Path(name).name == name and not path.is_symlink() and path.is_file(), "original model leaf")
        stat = path.stat()
        require(record["hash_algorithm"] == "sha256" and stat.st_size == record["bytes"] and
            stats[name] == [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns],
            "original complete model leaf changed after actual run")


def _drained_before_after(result):
    for field in ("native_before", "native_after"):
        row = result.get(field)
        require(type(row) is dict and type(row.get("owner_snapshot")) is dict,
            "actual original before/after drain")
        snapshot = row["owner_snapshot"]
        native, aio, accounting = snapshot.get("native"), snapshot.get("aio"), snapshot.get("stage_accounting")
        require(snapshot.get("owner_capture") is True and type(native) is dict and native and
            all(type(x) is int and x == 0 for x in native.values()) and type(aio) is dict and
            all(type(aio.get(k)) is int and aio[k] == 0 for k in ("outstanding", "pending", "ready", "unreaped")),
            "original before/after native/AIO work remains")
        require(type(accounting) is dict and accounting.get("valid") is True and
            all(type(r[k]) is int and r[k] == 0 for r in accounting["stages"].values()
                for k in ("inflight_ops", "inflight_bytes", "failed_ops")), "original stage drain invalid")


def validate_actual_development_pair(root, refs, binding_ref, *, driver=None,
        expected_gpu_uuid, expected_common_runtime_domain_sha256):
    """Return exact material only after replaying two complete real GPU records.

    `refs` includes the frozen evidence references as well as source rows.
    `evidence_refs` is a list, not authority granted by origin/status text.
    Exactly one ledger snapshot joins both completed original guard records.
    """
    driver = _BOUND_DRIVER if driver is None else driver
    require(driver is not None, "original replay driver not bound")
    root = Path(root).resolve(strict=True)
    require(type(refs) is dict and refs.get(binding_ref["path"]) == binding_ref, "frozen actual binding reference")
    binding = driver.read(driver.check_ref(root, binding_ref))
    require(type(binding) is dict and set(binding) == {"schema", "material", "evidence_refs"} and
        binding["schema"] == "bounded_I_actual_AB_cost_input_v1", "exact bounded A/B binding schema")
    material, evidence = binding["material"], binding["evidence_refs"]
    require(type(material) is dict and set(material) == MATERIAL_FIELDS, "exact development material")
    require(type(evidence) is list and 1 <= len(evidence) <= 64 and
        len({r["path"] for r in evidence}) == len(evidence), "unique actual evidence references")
    used = {r["path"]: r for r in evidence}
    for field in SOURCE_FIELDS:
        row = material[field]
        require(type(row) is dict and used.get(row["path"]) == row, "material source absent from evidence: " + field)
    require(used.get(VALIDATION_PATH) == refs.get(VALIDATION_PATH) == VALIDATION_REF,
        "pinned original CUDA/I/O/lifecycle validator")
    documents = {}
    for row in evidence:
        require(refs.get(row["path"]) == row, "evidence absent from frozen references")
        path = driver.check_ref(root, row)
        if path.suffix == ".json":
            documents[row["path"]] = driver.read(path)
    records = [d for d in documents.values() if type(d) is dict and
        d.get("schema") == "bounded_original_model_result_v1" and d.get("mode") in ("A", "B")]
    require(len(records) == 2 and {r["mode"] for r in records} == {"A", "B"}, "exact actual A and B results")
    require(all(r.get("origin") == "native_gpu_recording" for r in records), "CPU or simulated origin cannot open gate")
    ledgers = [d for d in documents.values() if type(d) is dict and
        "gpu_wall_seconds" in d and type(d.get("events")) is list and "active_reservation" in d]
    require(len(ledgers) == 1, "one actual completed original budget ledger snapshot")
    ledger = ledgers[0]
    require(ledger["active_reservation"] is None and type(ledger["gpu_wall_seconds"]) in (int, float) and
        math.isfinite(ledger["gpu_wall_seconds"]) and 0 <= ledger["gpu_wall_seconds"] <= driver.MAX_GPU_SECONDS,
        "original completed GPU budget within authorization")
    validation = driver.load(root, VALIDATION_REF, "_bounded_actual_AB_validation_" + str(time.monotonic_ns()))
    lock = documents[material["source_lock_ref"]["path"]]
    require(type(lock) is dict and type(lock.get("files")) is list, "complete inherited source lock metadata")
    locked = {r["path"]: r for r in lock["files"]}
    require(len(locked) == len(lock["files"]) and all(locked.get(material[f]["path"]) == material[f]
        for f in SOURCE_FIELDS if f != "source_lock_ref"), "actual complete source/model/geometry closure")
    geometry = documents[material["kv_layout_ref"]["path"]]
    configs = [r for r in documents[material["model_manifest_ref"]["path"]]["files"] if r["path"] == "config.json"]
    require(type(geometry) is dict and geometry.get("physical_block_bytes") == 917504 and
        geometry.get("tokens_per_block") == 16 and geometry.get("tensor_parallel_size") == 1 and
        geometry.get("dtype_bytes") == 2 and geometry.get("dtype") == "bfloat16" and
        geometry.get("num_hidden_layers") == 28 and geometry.get("num_key_value_heads") == 4 and
        geometry.get("head_dim") == 128 and len(configs) == 1 and
        geometry.get("model_config_sha256") == configs[0]["hash"], "original single-file KV layout")
    arms = {}
    for result in records:
        mode = result["mode"]
        require(result.get("status") == "COMPLETE_REQUIRES_ORIGINAL_GUARD_CLOSURE_AND_PERFORMANCE_ANALYSIS" and
            result.get("error") is None and result.get("original_engine_shutdown_returned") is True and
            result.get("native_tail_drained") is True and result.get("optional_probe_restored") is True,
            "actual complete original model, observer restoration and shutdown")
        config_ref = result["config_ref"]
        require(used.get(config_ref["path"]) == config_ref, "actual frozen arm configuration")
        config = documents[config_ref["path"]]
        rid = config["run_id"]
        require(config["mode"] == mode and result["run_id"] == rid and
            result["gpu_uuid"] == config["gpu_uuid"] == expected_gpu_uuid == material["gpu_uuid"] and
            result["common_runtime_domain_sha256"] == config["common_runtime_domain_sha256"] ==
                expected_common_runtime_domain_sha256 == material["common_runtime_domain_sha256"] and
            result["source_lock_ref"] == config["source_lock_ref"] == material["source_lock_ref"] and
            result["kv_layout_ref"] == config["kv_layout_ref"] == material["kv_layout_ref"] and
            result["collector_ref"] == config["collector_ref"] == material["collector_source_ref"] and
            result["native_source_ref"] == material["native_source_ref"], "actual arm identities and common runtime domain")
        guards = [d for d in documents.values() if type(d) is dict and
            d.get("label") == rid and "reservation_id" in d and "session_drained" in d]
        require(len(guards) == 1, "one actual completed original arm guard")
        guard = guards[0]
        runner_ref = config["runner_ref"]
        require(used.get(runner_ref["path"]) == locked.get(runner_ref["path"]) == runner_ref,
            "actual frozen thin consumer source")
        validation.validate_guard(guard, gpu_uuid=expected_gpu_uuid, job_id=rid, wrapper_path=runner_ref["path"])
        matches = [e for e in ledger["events"] if e.get("reservation_id") == guard["reservation_id"]]
        require(matches == [guard], "unique actual completed guard equals original ledger event")
        active = result.get("guard_start")
        require(type(active) is dict and active.get("id") == guard["reservation_id"] and
            active.get("label") == rid and active.get("gpu_uuid") == expected_gpu_uuid and
            active.get("session_id") == active.get("process_group") == result.get("subprocess_sid") == guard["session_id"] and
            active.get("command") == guard["command"] and "--config" in guard["command"] and
            config_ref["path"] in guard["command"], "actual child/config/native OS session join")
        engine = result.get("engine_config")
        require(type(engine) is dict and engine.get("enforce_eager") is True and
            engine.get("max_num_seqs") == 1 and type(engine["max_num_seqs"]) is int,
            "actual original eager single-sequence executor")
        _model_identity(root, result, material, documents, driver)
        loaded = result.get("loaded_native_source_binding")
        require(type(loaded) is list and any(r.get("module") == "py_kvcache.reactor" and
            r.get("source_ref") == material["native_source_ref"] for r in loaded), "actual native reactor source binding")
        _drained_before_after(result)
        frontend = result["frontend"]
        output = frontend["output"]
        rows = frontend["original_scheduler_hash_observations"]
        require(type(rows) is list and rows and all(r.get("request_id") == output["native_request_id"] and
            r.get("prompt_token_ids") == rows[0]["prompt_token_ids"] for r in rows), "actual original foreground token/hash source")
        prompt = rows[0]["prompt_token_ids"]
        inputs_ref = config["inputs_ref"]
        require(used.get(inputs_ref["path"]) == inputs_ref and
            prompt == documents[inputs_ref["path"]]["records"][9]["prompt_token_ids"], "same pre-registered exact original input")
        require(type(prompt) is list and len(prompt) > 0 and all(type(t) is int and t >= 0 for t in prompt), "actual frozen prompt tokens")
        capture = result["capture"]
        frames = validation.validate_capture(capture, run_id=rid, request_id=output["native_request_id"],
            output_ids=output["output_token_ids"], prompt_tokens=len(prompt), measured_offset=16,
            warmup_offsets=[1], cached_tokens=output["num_cached_tokens"])
        measured = frames[16]
        prepared = capture["frames"][16]["prepared"]
        require([prepared["batch"], prepared["active_decode"], prepared["prefill_tokens"]] == [1, 1, 0],
            "actual selected single decode frame")
        drain = validation.original_post_shutdown_drain(result["post_shutdown"], result["native_tail_assertions"],
            run_id=rid, native_source_sha256=material["native_source_ref"]["sha256"])
        journal = result["journal"]
        require(journal.get("run_id") == rid and journal.get("source_sha256") == material["native_source_ref"]["sha256"],
            "actual full native journal source/run join")
        validation.validate_io(journal, drain, capture=capture, frames=frames, run_id=rid,
            native_source_sha256=material["native_source_ref"]["sha256"], arm="baseline" if mode == "A" else "action",
            measured_offset=16, physical_bytes=917504, operations=1, independent_payload=result["independent_payload"])
        arms[mode] = dict(config=config, result=result, guard=guard, prompt=prompt,
            output=output["output_token_ids"], context=prepared["context_length"], ns=measured["gpu_elapsed_ns"])
    a, b = arms["A"], arms["B"]
    require(a["prompt"] == b["prompt"] and a["output"] == b["output"] and a["context"] == b["context"] and
        a["result"]["model_identity"] == b["result"]["model_identity"] and
        a["config"]["inputs_ref"] == b["config"]["inputs_ref"] and a["config"]["sampling"] == b["config"]["sampling"] and
        a["config"]["engine"] == b["config"]["engine"] and a["config"]["out"] != b["config"]["out"] and
        a["guard"]["session_id"] != b["guard"]["session_id"] and
        a["guard"]["started_unix"] + a["guard"]["elapsed_seconds"] <= b["guard"]["started_unix"],
        "actual sequential same-input complete128 pair, same engine/model and fresh namespaces")
    actual = {field: deepcopy(material[field]) for field in SOURCE_FIELDS}
    actual.update(gpu_uuid=expected_gpu_uuid, common_runtime_domain_sha256=expected_common_runtime_domain_sha256,
        kernel_mode="eager", context_length=a["context"], a_ns=a["ns"], b_ns=b["ns"], budget_ns=a["ns"],
        stage="ssd_read", physical_bytes=917504, operations=1, existing_io=[dict(ops=0, bytes=0) for _ in range(4)],
        batch=1, active_decode=1, prefill_tokens=0)
    require(actual == material, "declared material differs from actual prepared frame/CUDA elapsed evidence")
    for row in [binding_ref] + evidence:
        driver.check_ref(root, row)
    return actual
