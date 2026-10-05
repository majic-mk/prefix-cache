"""One bounded development U/I run using the existing author engine and I component.

A/B are explicitly controlled unit-cost measurements. U/I never inject I/O.
All execution, cache ownership and draining remain the existing original APIs.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

P = "artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004"
D = "artifacts/prefix_io_v1/server14-bounded-u-i-20261005"
DRIVER = P + "/runner/strong_trace_runner_v7.py"


def bootstrap(root):
    path = root / DRIVER
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "0d916dedd52c2aaec3822adff6560987d443425619f96d57a027a60be10fe9dd":
        raise ValueError("original thin driver source changed")
    spec = importlib.util.spec_from_file_location("_bounded_original_driver", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(path), "exec"), vars(module))
    return module


def records_from(inputs):
    return [dict(r, seed=0, min_tokens=128) for r in inputs["records"]]


def observe_native_returns(module):
    """Bounded value counters on the actual original dispatch return, both U/I."""
    original = module.IoReactor._prefix_stage_decide
    counts = dict(calls=0, original_returns=0, native_deferrals=0, other_dispatches=0,
                  stages={})
    def observed(owner, stage, *args, **kwargs):
        value = original(owner, stage, *args, **kwargs)
        counts["calls"] += 1
        counts["stages"][stage] = counts["stages"].get(stage, 0) + 1
        if value is None:
            counts["original_returns"] += 1
        elif isinstance(value.decision, module._P4Deferred):
            counts["native_deferrals"] += 1
        else:
            counts["other_dispatches"] += 1
        return value
    module.IoReactor._prefix_stage_decide = observed
    return original, observed, counts


def execute(root, config, refs, R):
    ledger = R.read(root / R.LEDGER)
    guard = ledger.get("active_reservation")
    R.require(type(guard) is dict and guard["label"] == config["run_id"] and
              guard["gpu_uuid"] == config["gpu_uuid"] and guard["session_id"] == os.getsid(0) and
              guard["seconds_limit"] == 300, "actual original same-session GPU reservation")
    out = R.safe(root, config["out"])
    R.require(not out.exists() and shutil.disk_usage(root).free >= 8 * 1024**3 + 1024**3,
              "fresh private run and protected space floor")
    out.mkdir(parents=True)
    storage = out / "storage"
    storage.mkdir()
    mode = config["mode"]
    inputs = R.read(R.check_ref(root, config["inputs_ref"]))
    R.require(mode in ("A", "B", "U", "I") and len(inputs["records"]) == 12 and
              inputs["frontend_max_concurrency"] == 3 and inputs["engine_max_num_seqs"] == 1,
              "only the frozen bounded development comparison")
    result = dict(schema="bounded_original_model_result_v1", origin="native_gpu_recording",
        status="FAILED", mode=mode, run_id=config["run_id"], gpu_uuid=config["gpu_uuid"],
        config_ref=config["config_ref"], source_lock_ref=config["source_lock_ref"],
        common_runtime_domain_sha256=config["common_runtime_domain_sha256"],
        collector_ref=config["collector_ref"], native_source_ref=refs[R.NATIVE + "/reactor.py"],
        kv_layout_ref=config["kv_layout_ref"], controlled_unit_cost_only=mode in ("A", "B"),
        synthetic_IO_injected=mode == "B", production_qualified=False,
        error=None, guard_start=deepcopy(guard), subprocess_sid=os.getsid(0),
        model_manifest_ref=refs[R.MODEL_PLAN], cuda_event_source_ref=config["cuda_event_source_ref"],
        original_engine_shutdown_returned=False, native_tail_drained=False,
        OS_session_drain_owned_by_original_outer_guard=True)
    llm = capture = handler = probe = finder = startup = native = journal = original_factory = factory = None
    reactor_module = return_original = return_wrapper = return_counts = None
    owner_original = owner_wrapper = None
    previous_env, previous_path, previous_cwd = dict(os.environ), list(sys.path), Path.cwd()
    try:
        common = R.load(root, refs[R.COMMON], "_bounded_common")
        base = R.load(root, refs[R.SMOKE], "_bounded_smoke")
        base.ROOT, base.AUTHOR_ROOT = root, root / R.AUTHOR
        sys.modules["native_gpu_prefix_smoke"] = base
        tail = R.load(root, refs[R.TAIL], "_bounded_tail")
        runtime = R.load(root, refs[P + "/runner/native_runtime_v7.py"], "_bounded_runtime_functions")
        acquire = R.load(root, refs[R.DRAIN], "_bounded_original_drain")
        raw = R.load(root, refs[R.PREVIOUS + "/calibration_v2/run_native_cost_experiment.py"], "_bounded_old_unit")
        sys.path[:0] = [str(root / D / "control_source"),
                       str(root / R.OVERLAY / "third_party/work/py-kvcache-p4-02-cpu"),
                       str(root / R.OVERLAY / "third_party/work/prefix-io-p4-02-cpu/src"),
                       str(root / R.CANDIDATE), str(root / "experiments/prefix_io_v1/scripts")]
        result["runtime_cache_environment"] = common.configure_process_caches(out, base)
        sdk = R.load(root, config["site_sdk_ref"], "_bounded_existing_site_sdk")
        evidence = sdk.prepare_site_sdk(root, refs, out, migration_ref=config["site_sdk_migration_ref"])
        result["sdk_environment"] = evidence
        result["ninja_environment"] = common.prepare_process_ninja(root, refs, out, evidence)
        optional, absence = common.load_optional_probe(root, refs)
        result["optional_absence"] = absence
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["PYTHONHASHSEED"] = "0"
        R.require(os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1", "offline guard")
        model_helper = R.load(root, config["inherited_model_helper_ref"], "_bounded_existing_model_identity")
        model_gates = dict(refs=refs, plan=dict(model_manifest_ref=refs[R.MODEL_PLAN]),
                           prior_U=R.read(R.check_ref(root, config["prior_model_U_ref"])))
        model_dir, identity, stats = model_helper.inherited_model_binding(root, model_gates,
            root / R.MODEL, root / R.MODEL_PLAN, R=R)
        result["model_identity"] = identity
        qualifier = R.load(root, refs[R.QUALIFIER], "_bounded_author_finder")
        finder = qualifier.BoundAuthorFinder(root, refs)
        R.require(not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules), "fresh original author import")
        sys.meta_path.insert(0, finder)
        os.chdir(out)
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        import vllm.utils.import_utils as import_utils
        probe = optional.install_capability_probe(import_utils, finder, root, refs)
        R.require(qualifier.normalize_uuid(torch.cuda.get_device_properties(0).uuid) == config["gpu_uuid"], "actual current GPU")
        R.require(Path(vllm.__file__).resolve().is_relative_to(root / R.AUTHOR), "actual author executor")
        from prefix_io_control.p4_native_window_journal import NativeWindowJournal
        native = importlib.import_module("py_kvcache.vllm")
        journal = NativeWindowJournal(config["run_id"], refs[R.NATIVE + "/reactor.py"]["sha256"], enabled=True, max_events=4096)
        sparse = raw.SparseOwnerSink(journal)
        original_factory = native.TransferCoordinator
        def observed_factory(**kwargs):
            R.require(kwargs.get("progress_run_id") == config["run_id"] and kwargs.get("max_accepted_parents") == 8,
                      "original same-run parent limit")
            prior = kwargs.get("observation_sink")
            def sink(owner):
                if prior is not None:
                    prior(owner)
                sparse(owner)
            kwargs.update(stage_accounting=journal, observation_sink=sink)
            return original_factory(**kwargs)
        factory = observed_factory
        native.TransferCoordinator = factory
        engine = deepcopy(config["engine"])
        extra = engine["kv_transfer_config"]["kv_connector_extra_config"]
        extra["shared_storage_path"] = str(storage)
        cost_table = binding = None
        if mode == "I":
            from prefix_io_control.p4_cost_table import load_gpu_development_table
            cost_validator = R.load(root, config["cost_validator_ref"], "_bounded_actual_cost_replay")
            R.validate_actual_development_pair = cost_validator.bind_driver(R)
            cost_table = load_gpu_development_table(root, refs, config["cost_input_ref"], driver=R,
                expected_gpu_uuid=config["gpu_uuid"],
                expected_common_runtime_domain_sha256=config["common_runtime_domain_sha256"])
            extra["prefix_io_p4_policy"]["internal_step_budget_ns"] = cost_table.development_input.budget_ns
            binding = R.load(root, config["finite_binding_ref"], "_bounded_existing_finite_inputs")
            start = R.load(root, config["finite_startup_ref"], "_bounded_existing_finite_startup")
            startup = start.FiniteStartup(native_module=native, reactor_module=importlib.import_module("py_kvcache.reactor"),
                table=cost_table, issuer=None, run_id=config["run_id"], budget_ns=cost_table.development_input.budget_ns,
                root=root, runtime_refs=refs, driver=R, binding_module=binding,
                validate_drained=runtime.validate_drained_snapshot, observation_only=False,
                reserve_covered_cell_signatures=None)
        if mode in ("U", "I"):
            reactor_module = importlib.import_module("py_kvcache.reactor")
            return_original, return_wrapper, return_counts = observe_native_returns(reactor_module)
        alias = out / base.MODEL_ID
        alias.parent.mkdir(parents=True, exist_ok=False)
        alias.symlink_to(model_dir, target_is_directory=True)
        result["engine_config"] = engine
        if mode in ("A", "B"):
            asset = config["unit_asset"]
            R.check_ref(root, asset["ref"])
            target = storage / asset["relative"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with (root / asset["ref"]["path"]).open("rb") as inp, target.open("xb") as dst:
                shutil.copyfileobj(inp, dst)
        llm = LLM(model=base.MODEL_ID, **engine)
        owners = []
        def worker_control(owner):
            owners.append(runtime.capture_original_worker(owner))
            return {"captured": True}
        value = llm.collective_rpc(worker_control, timeout=10)
        R.require(len(value) == len(owners) == 1, "actual single Uni worker")
        worker, handler = owners[0]
        result["loaded_native_source_binding"] = runtime.verify_loaded_native(root, refs, R)
        result["original_planner_identity"] = runtime.original_planner_identity(llm, handler, root, refs, R)
        R.require(handler.coordinator.reactor._prefix_p4_bridge is None if mode != "I" else
                  handler.coordinator.reactor._prefix_p4_bridge.policy.table is cost_table, "requested actual U/I native branch")
        result["native_before"] = runtime.drain_original(llm, acquire, handler)
        collector = R.load(root, config["collector_ref"], "_bounded_same_full_step_collector")
        collector_refs = dict(refs)
        collector_refs["third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"] = refs[R.NATIVE + "/reactor.py"]
        selected = records_from(inputs)
        if mode in ("A", "B"):
            prompt = inputs["records"][9]["prompt_token_ids"]
            sampling = SamplingParams(**config["sampling"])
            hashes = [config["unit_asset"]["block_hash"]]
            owner_original, owner_wrapper = raw.install_owner_snapshot_observation(
                handler.coordinator.reactor, hashes, journal)
            mapper = handler.coordinator.reactor.file_mapper
            result["actual_native_file_mapping"] = str(Path(mapper.get_file_name(bytes.fromhex(hashes[0]))))
            R.require(Path(result["actual_native_file_mapping"]).absolute() ==
                      (storage / config["unit_asset"]["relative"]).absolute(), "original SSD file mapping")
            for tokens, number in ((prompt, 1), (prompt, 128), ([32001], 1)):
                llm.generate([{"prompt_token_ids": tokens.copy()}],
                    SamplingParams(**dict(config["sampling"], max_tokens=number, min_tokens=number)), use_tqdm=False)
                runtime.drain_original(llm, acquire, handler)
            result["native_before"] = runtime.drain_original(llm, acquire, handler)
            result["cold_input_membership"] = raw.require_cold_inputs(result["native_before"]["owner_snapshot"], hashes)
            source_spec = native.SharedStorageLoadStoreSpec([bytes.fromhex(config["unit_asset"]["block_hash"])])
            def action(offset, ordinal):
                R.require(offset == 16, "frozen unit measurement offset")
                accepted = handler.preload_async(config["run_id"] + "-unit", source_spec,
                    profile_tid="bounded_unit_cost", req_id=config["run_id"])
                R.require(accepted is True, "original single unit accepted")
                return dict(accepted=True, stage="ssd_read", physical_bytes=917504,
                    hashes=[config["unit_asset"]["block_hash"]], h2d_requested=False)
            capture = collector.install(worker, common=common, root=root, refs=collector_refs, run_id=config["run_id"],
                event_class=torch.cuda.Event, selected_offsets=(16,), action=action if mode == "B" else None,
                origin="native_gpu_recording", max_steps=128)
            common.PROMPT = prompt.copy()
            result["frontend"] = raw.frontend_capture(common, llm.llm_engine, sampling, config["run_id"],
                [config["unit_asset"]["block_hash"]])
            result["independent_payload"] = None if mode == "A" else dict(ssd_only=True,
                cache_miss_before_submit=True,physical_bytes=917504,operations=1,
                preload_key_sha256=[config["unit_asset"]["block_hash"]],
                request_prefix_key_sha256=result["frontend"]["original_scheduler_hash_observations"][0]["block_hashes"][:48])
        else:
            capture = collector.install(worker, common=common, root=root, refs=collector_refs, run_id=config["run_id"],
                event_class=torch.cuda.Event, selected_offsets=(), action=None, origin="native_gpu_recording", max_steps=4096)
            if startup is not None:
                identity_live = binding.verify_running_identity(worker, handler, capture, cost_table, issuer=None,
                    root=root, runtime_refs=refs, common_runtime_domain_sha256=config["common_runtime_domain_sha256"],
                    model_identity=identity, calibration_lock_ref=R.read(R.check_ref(root,
                        config["cost_input_ref"]))["material"]["source_lock_ref"])
                result["I_owner_install"] = startup.install_after_original_drain(capture, identity_live)
            measured_start = time.monotonic_ns()
            result["frontend"] = R.drive_original_engine(llm.llm_engine,
                lambda row: SamplingParams(**dict(config["sampling"], seed=row["seed"])), selected,
                max_concurrency=3, deadline_seconds=120,
                on_progress=lambda rows, steps: capture.observer.resolve_ready())
            result["stream_call_start_ns"] = measured_start
            R.require(result["frontend"]["actual_request_stream_completed"] is True, "all frozen requests complete")
        result["capture"] = capture.export()
        if mode in ("A", "B"):
            R.require(result["capture"]["valid"] is True, "actual complete same-source cost capture")
        capture.detach()
        capture = None
        result["native_after"] = runtime.drain_original(llm, acquire, handler)
        result["after_stream_drain_finished_ns"] = time.monotonic_ns()
        if startup is not None:
            result["I_decisions"] = startup.binding.evidence()
            result["I_bridge_evidence"] = handler.coordinator.reactor._prefix_p4_bridge.snapshot()
        model_helper.inherited_model_binding(root, model_gates, model_dir, root / R.MODEL_PLAN, R=R, previous_stats=stats)
        result["status"] = "COMPLETE_REQUIRES_ORIGINAL_GUARD_CLOSURE_AND_PERFORMANCE_ANALYSIS"
    except Exception as exc:
        result["error"] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(limit=18))
    finally:
        if capture is not None:
            try:
                result["failed_capture"] = capture.export()
                capture.detach()
            except Exception as exc:
                result["capture_cleanup_error"] = str(exc)
        if llm is not None:
            try:
                result["final_before_shutdown"] = runtime.drain_original(llm, acquire, handler)
            except Exception as exc:
                result["status"] = "FAILED_DRAIN"
                result["drain_error"] = str(exc)
            try:
                llm.llm_engine.engine_core.shutdown(timeout=15)
                result["original_engine_shutdown_returned"] = True
                result["post_shutdown"] = tail.post_shutdown_snapshot(handler)
                result["native_tail_drained"] = runtime.verify_original_tail(result["post_shutdown"])
                reactor = handler.coordinator.reactor
                result["native_tail_assertions"] = dict(handler_shutdown=handler.is_shutdown,
                    worker_alive=reactor._worker.is_alive(), aio_worker_alive=reactor.ring._worker.is_alive(),
                    reactor_closed=reactor._closed)
            except Exception as exc:
                result["status"] = "FAILED_SHUTDOWN"
                result["shutdown_error"] = str(exc)
        if return_wrapper is not None:
            result["actual_native_dispatch_returns"] = deepcopy(return_counts)
            if reactor_module.IoReactor._prefix_stage_decide is return_wrapper:
                reactor_module.IoReactor._prefix_stage_decide = return_original
                result["native_return_counter_restored"] = True
            else:
                result["native_return_counter_restored"] = False
                result["status"] = "FAILED_RETURN_COUNTER_RESTORE"
        if owner_wrapper is not None:
            reactor = handler.coordinator.reactor
            if reactor._capture_owner_snapshot is owner_wrapper:
                del reactor._capture_owner_snapshot
                result["owner_snapshot_restored"] = True
            else:
                result["owner_snapshot_restored"] = False
                result["status"] = "FAILED_MEMBERSHIP_RESTORE"
        if startup is not None:
            try:
                result["I_decisions"] = startup.binding.evidence() if startup.binding is not None else None
                result["I_startup_restored"] = startup.detach_after_original_shutdown()
                R.require(result["I_startup_restored"], "I original owner and startup restored")
            except Exception as exc:
                result["status"] = "FAILED_I_RESTORE"
                result["I_restore_error"] = str(exc)
        if probe is not None:
            try:
                result["optional_probe_restored"] = probe.detach()
            except Exception as exc:
                result["optional_probe_restored"] = False
                result["probe_restore_error"] = str(exc)
                result["status"] = "FAILED_PROBE_RESTORE"
        if native is not None and factory is not None:
            if native.TransferCoordinator is factory:
                native.TransferCoordinator = original_factory
                result["native_factory_restored"] = True
            else:
                result["native_factory_restored"] = False
                result["status"] = "FAILED_FACTORY_RESTORE"
        if journal is not None:
            events, frames, valid = journal.published()
            result["journal"] = dict(run_id=config["run_id"], source_sha256=refs[R.NATIVE + "/reactor.py"]["sha256"],
                valid=valid, lost=journal.lost, events=[asdict(x) for x in events], frames=[asdict(x) for x in frames])
        if finder is not None and finder in sys.meta_path:
            sys.meta_path.remove(finder)
        sys.modules.pop("native_gpu_prefix_smoke", None)
        sys.path[:] = previous_path
        os.chdir(previous_cwd)
        os.environ.clear()
        os.environ.update(previous_env)
        R.new_json(out / "bounded-model-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute requires the existing original GPU guard")
    root = args.project.resolve(strict=True)
    R = bootstrap(root)
    config_path = R.safe(root, args.config)
    config = R.read(config_path)
    R.require(config["config_ref"] == R.ref(root, args.config) if "config_ref" in config else True, "config bytes")
    config["config_ref"] = R.ref(root, args.config)
    refs = R.source_rows(root, config["source_lock_ref"], full=False)
    for key in ("runner_ref", "inputs_ref", "collector_ref", "site_sdk_ref", "site_sdk_migration_ref", "kv_layout_ref"):
        R.check_ref(root, config[key])
    R.require(Path(__file__).resolve() == R.check_ref(root, config["runner_ref"]), "actual frozen consumer")
    result = execute(root, config, refs, R)
    print(json.dumps({"status": result["status"], "run_id": config["run_id"], "error": result.get("error")}, ensure_ascii=False))
    return 0 if result["status"].startswith("COMPLETE_") and result["original_engine_shutdown_returned"] and result["native_tail_drained"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
