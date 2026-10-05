"""Private current-device adapter around the unchanged P316 request evaluator.

The original runner owns arrivals, engine stepping, token timelines, hardlink
initialization and output checks. This file only binds the already used common
runtime, supplies the two declared control configurations and verifies closure.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

P = "artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004"
DRIVER = P + "/runner/strong_trace_runner_v7.py"
DRIVER_SHA = "0d916dedd52c2aaec3822adff6560987d443425619f96d57a027a60be10fe9dd"


def bootstrap(root):
    path = root / DRIVER
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DRIVER_SHA:
        raise ValueError("existing driver bytes changed")
    spec = importlib.util.spec_from_file_location("_dr_existing_driver", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(path), "exec"), vars(module))
    return module


def control_engine(original, controls, arm):
    """Only append the declared native controls; retain the original executor."""
    engine = deepcopy(original)
    required = dict(dtype="bfloat16", kv_cache_memory_bytes=2147483648,
                    max_num_seqs=2, max_model_len=16400,
                    max_num_batched_tokens=16400)
    if any(engine.get(k) != value for k, value in required.items()):
        raise ValueError("original long engine contract changed")
    extra = engine["kv_transfer_config"]["kv_connector_extra_config"]
    if (extra.get("iodepth") != 8 or extra.get("load_planner") != "on" or
            extra.get("staging_mem") != 1 or extra.get("enable_preload") is not True or
            extra.get("preload_share_staging") is not True):
        raise ValueError("original native admission/staging/I/O configuration changed")
    allowed = {"prefix_io_parent_admission", "prefix_io_p4_policy"}
    if type(controls) is not dict or not set(controls).issubset(allowed):
        raise ValueError("P4 owns its nested fixed policy; no parallel top-level stage key")
    policy = controls.get("prefix_io_p4_policy", {"mode": "off"})
    if type(policy) is not dict:
        raise ValueError("exact P4 policy mapping required")
    stage = policy.get("fixed_stage_policy") or {"mode": "off"}
    if arm == "U":
        if policy.get("mode") != "off" or stage.get("mode") != "off":
            raise ValueError("U must disable all new research controls")
    elif arm in ("CAL", "F8+D_R"):
        if policy.get("mode") != "dependency_only" or stage.get("mode") != "fixed":
            raise ValueError("current restore ordering requires the declared fixed controller")
    else:
        raise ValueError("only U, independent calibration, and F8+D_R")
    extra.update(deepcopy(controls))
    return engine


def validate_original_identity(validate, manifest, candidate, current_gpu,
                               historical_gpu, migration):
    """Replay the old contract unchanged, with a disclosed device migration.

    This does not turn historical unloaded-point qualification into current
    device prediction qualification. The original cost-admission curve remains
    the frozen curve in both arms.
    """
    if (migration.get("historical_gpu_uuid") != historical_gpu or
            migration.get("current_gpu_uuid") != current_gpu or
            migration.get("scope") != "P316_CURRENT_DEVICE_EXPLORATORY_REPLAY"):
        raise ValueError("explicit old/current device migration binding missing")
    if (manifest["gpu_uuid"] != historical_gpu or
            candidate["provenance"]["gpu_uuid"] != historical_gpu):
        raise ValueError("historical frozen workload/curve identity differs")
    return validate(manifest, candidate, historical_gpu)


class KernelSubmissionTap:
    """Scalar receipt of successful original Linux io_submit returns, both arms."""
    def __init__(self, *, limit=16384):
        if type(limit) is not int or not 1 <= limit <= 16384:
            raise ValueError("bounded submission receipt limit")
        self.limit = limit
        self.records = []
        self.overflow = False
        self.fault = None
        self.original = self.wrapper = self.kernel = None

    def install(self, kernel):
        if self.original is not None:
            raise ValueError("one original kernel tap")
        original = kernel.submit
        def accepted(batch):
            count = original(batch)
            # Diagnostics never change the actual count or original exception.
            try:
                if type(count) is not int or not 0 <= count <= len(batch):
                    raise ValueError("original syscall count outside batch")
                if count:
                    at_ns = time.monotonic_ns()
                    for request in batch[:count]:
                        if len(self.records) >= self.limit:
                            self.overflow = True
                            break
                        fields = (request.user_data, request.fd, request.nbytes)
                        if any(type(value) is not int or value < 0 for value in fields) or request.kind not in ("read", "write"):
                            raise ValueError("native kernel request scalar receipt invalid")
                        self.records.append(dict(sequence=len(self.records)+1, at_ns=at_ns,
                            user_data=request.user_data, kind=request.kind,
                            fd=request.fd, nbytes=request.nbytes,
                            boundary="original_linux_io_submit_successful_prefix"))
            except BaseException as exc:
                if self.fault is None:
                    self.fault = type(exc).__name__ + ": " + str(exc)
            return count
        self.kernel, self.original, self.wrapper = kernel, original, accepted
        kernel.submit = accepted

    def snapshot(self):
        return dict(valid=self.fault is None and not self.overflow, overflow=self.overflow,
            fault=self.fault, successful_operation_count=len(self.records),
            records=tuple(self.records), scope="actual kernel submission order; not completion or resource-release receipt")

    def restore_after_shutdown(self, *, joined):
        if self.original is None:
            return True
        if joined is not True or self.kernel.submit is not self.wrapper:
            return False
        self.kernel.submit = self.original
        self.original = self.wrapper = self.kernel = None
        return True


class DetailsPathReplay:
    """One creation replay for the original main's freshly owned details path.

    The real SDK requires this directory before importing the author executor.
    The unchanged P316 main later creates the same directory. The narrow Path
    type is installed only in that module; it cannot admit another output or an
    already existing historical run.
    """
    def __init__(self, details, *, run_id, session_id, revalidate_guard):
        self.details = Path(details).resolve()
        if self.details.exists() or self.details.is_symlink():
            raise FileExistsError("existing output is never adopted")
        if type(run_id) is not str or not run_id or type(session_id) is not int or session_id < 1:
            raise ValueError("actual run/session identity required")
        self.revalidate_guard = revalidate_guard
        self.revalidate_guard()
        self.details.mkdir(parents=True, exist_ok=False)
        self.marker = self.details / "current-adapter-owned-details.json"
        self.row = dict(run_id=run_id, session_id=session_id,
            subprocess_pid=os.getpid(), creation_nonce=uuid.uuid4().hex,
            details=str(self.details), creation_monotonic_ns=time.monotonic_ns())
        self.marker_bytes = (json.dumps(self.row, sort_keys=True) + "\n").encode("utf-8")
        with self.marker.open("xb") as target:
            target.write(self.marker_bytes)
        self.consumed = False

    def path_type(self):
        replay = self
        concrete = type(Path())
        class OriginalMainPath(concrete):
            def mkdir(self, mode=0o777, parents=False, exist_ok=False):
                expected = (Path(self).resolve() == replay.details and mode == 0o777 and
                            parents is True and exist_ok is False and not replay.consumed)
                if not expected:
                    return super().mkdir(mode=mode, parents=parents, exist_ok=exist_ok)
                replay.revalidate_guard()
                if (not replay.details.is_dir() or replay.details.is_symlink() or
                        replay.marker.is_symlink() or replay.marker.read_bytes() != replay.marker_bytes or
                        any((replay.details / name).exists() for name in
                            ("result.json", "frozen-config.json", "driver-source.py", "probe-source.py", "storage"))):
                    raise ValueError("only the newly owned unconsumed original output can replay creation")
                replay.consumed = True
                return None
        return OriginalMainPath


class InprocessOffloadStatsSerialization:
    """Supply the IPC-equivalent plain scalar entry to the existing loggers.

    The author producer emits a dataclass while its unchanged reducers consume
    dictionaries. The in-process executor skips IPC serialization. Only the one
    exact newly appended producer object is converted; measured values and the
    original producer call/return are retained.
    """
    def __init__(self, stats_type, operation_type):
        self.stats_type, self.operation_type = stats_type, operation_type
        self.original = stats_type.record_transfer
        self.conversions = 0
        self.original_calls = 0
        adapter = self
        def record_transfer(owner, num_bytes, time, transfer_type):
            src, dst = transfer_type
            key = src + "_to_" + dst
            if type(owner.data) is not dict:
                raise ValueError("original transfer stats exact mapping required")
            prior = owner.data.get(key)
            if prior is not None and type(prior) is not list:
                raise ValueError("original transfer stats list required")
            before = 0 if prior is None else len(prior)
            adapter.original_calls += 1
            result = adapter.original(owner, num_bytes, time, transfer_type)
            appended = owner.data[key]
            if (type(appended) is not list or len(appended) != before + 1 or
                    (prior is not None and appended is not prior) or
                    type(appended[-1]) is not adapter.operation_type):
                raise ValueError("only the one exact original newly appended transfer measurement can serialize")
            actual = appended[-1]
            appended[-1] = dict(op_size=actual.op_size, op_time=actual.op_time)
            adapter.conversions += 1
            return result
        self.wrapper = record_transfer
        stats_type.record_transfer = record_transfer

    def snapshot(self):
        return dict(scope="common in-process transport compatibility for original stats",
            conversions=self.conversions, extra_measurements=0, measured_values_changed=False,
            original_producer_calls=self.original_calls, log_stats_disabled_by_adapter=False)

    def restore_after_shutdown(self, *, ended):
        if ended is not True or self.stats_type.record_transfer is not self.wrapper:
            return False
        self.stats_type.record_transfer = self.original
        return True


def bounded_restore_metadata(value):
    """Copy the existing bounded producer's plain values only, once at closure."""
    remaining = [2_000_000]
    def copy(item, depth):
        remaining[0] -= 1
        if remaining[0] < 0 or depth > 16:
            raise ValueError("bounded restore metadata nesting/node count")
        if item is None or type(item) in (bool, int):
            return item
        if type(item) is float and abs(item) < float("inf"):
            return item
        if type(item) is str and len(item) <= 4096:
            return item
        if type(item) in (tuple, list) and len(item) <= 16384:
            return [copy(part, depth+1) for part in item]
        if type(item) is dict and len(item) <= 128 and all(
                type(key) is str and len(key) <= 128 for key in item):
            return {key:copy(part, depth+1) for key, part in item.items()}
        raise ValueError("restore evidence must remain existing bounded primitive values")
    return copy(value, 0)


def projected_original_tail(tail, handler):
    """Keep every original real-owner closure check, project optional metadata.

    Only the private tail module's serializer is adapted while its unchanged
    checker runs. No owner method, native counter, shutdown flag or source file
    is replaced. Full optional metadata is preserved separately.
    """
    original = tail.scalar_copy
    full = {}
    keys = ("native_shutdown_read", "owner_capture", "aio", "native", "stage_accounting")
    def terminal_copy(value, depth=0):
        if (depth == 0 and type(value) is dict and
                value.get("native_shutdown_read") is True and value.get("owner_capture") is False and
                all(key in value for key in keys)):
            if not full:
                full.update(bounded_restore_metadata({key:value.get(key)
                    for key in ("captured_ns", "run_id", "p4", "restore_order_journal")}))
            return original({key:value[key] for key in keys}, depth)
        return original(value, depth)
    tail.scalar_copy = terminal_copy
    try:
        actual = tail.post_shutdown_snapshot(handler)
        return actual, full
    finally:
        tail.scalar_copy = original


def execute(root, config, refs, R):
    arm = config["arm"]
    out = R.safe(root, config["out"])
    active = R.read(root / R.LEDGER).get("active_reservation")
    R.require(type(active) is dict and active["label"] == config["run_id"] and
              active["gpu_uuid"] == config["gpu_uuid"] and
              active["session_id"] == os.getsid(0) and
              active["seconds_limit"] == 300, "actual original bounded GPU guard")
    R.require(not out.exists(), "new isolated original-runner output required")
    R.require(out == root / "experiments/prefix_io_v1/runs" / config["run_id"] / "details",
              "actual original guarded label details path")
    previous_env, previous_path, previous_cwd = dict(os.environ), list(sys.path), Path.cwd()
    evidence = dict(schema="dr_p316_current_device_adapter_v5", arm=arm,
        run_id=config["run_id"], gpu_uuid=config["gpu_uuid"],
        common_runtime_domain_sha256=config["common_runtime_domain_sha256"],
        runtime_refs=config["runtime_refs"], subprocess_sid=os.getsid(0),
        subprocess_pid=os.getpid(), guard_start=deepcopy(active),
        config_ref=config["config_ref"], runner_ref=config["original_runner_ref"],
        adapter_ref=config["runner_ref"], device_migration=config["device_migration"],
        historical_qualification_remains_historical=True,
        production_prediction_qualified=False, engine_executor_replaced=False,
        original_engine_shutdown_returned=False, native_tail_drained=False)
    llm = handler = worker = finder = probe = bridge_original = bridge_wrapper = None
    reactor_original = reactor_wrapper = order_journal = reactor_module = None
    kernel_tap = KernelSubmissionTap()
    details_replay = None
    stats_serialization = None
    base = main_module = native = optional = qualifier = runtime = tail = None
    old_llm = old_write = old_model = old_environment = None
    selected_engine = None
    original_probe = None
    try:
        common = R.load(root, refs[R.COMMON], "_dr_existing_common")
        base = R.load(root, refs[R.SMOKE], "_dr_existing_smoke")
        base.ROOT, base.AUTHOR_ROOT = root, root / R.AUTHOR
        sys.modules["native_gpu_prefix_smoke"] = base
        runtime = R.load(root, refs[P + "/runner/native_runtime_v7.py"], "_dr_existing_runtime")
        tail = R.load(root, refs[R.TAIL], "_dr_existing_tail")
        # Packages are complete private byte-for-byte source copies, except the
        # explicitly source-locked D_R files shared by U/CAL/method.
        sys.path[:0] = [str(R.safe(root, p)) for p in config["runtime_import_roots"]]
        sys.path.insert(len(config["runtime_import_roots"]), str(root / "experiments/prefix_io_v1/scripts"))
        def revalidate_details_guard():
            fresh_guard = R.read(root / R.LEDGER).get("active_reservation")
            R.require(type(fresh_guard) is dict and fresh_guard["label"] == config["run_id"] and
                      fresh_guard["gpu_uuid"] == config["gpu_uuid"] and
                      fresh_guard["session_id"] == os.getsid(0) == active["session_id"] and
                      fresh_guard["seconds_limit"] == 300, "actual same-session details ownership")
        details_replay = DetailsPathReplay(out, run_id=config["run_id"],
            session_id=os.getsid(0), revalidate_guard=revalidate_details_guard)
        evidence["fresh_owned_details"] = deepcopy(details_replay.row)
        setup = out
        evidence["runtime_cache_environment"] = common.configure_process_caches(setup, base)
        sdk = R.load(root, config["site_sdk_ref"], "_dr_existing_site_sdk")
        sdk_evidence = sdk.prepare_site_sdk(root, refs, setup,
            migration_ref=config["site_sdk_migration_ref"])
        evidence["sdk_environment"] = sdk_evidence
        evidence["ninja_environment"] = common.prepare_process_ninja(root, refs, setup, sdk_evidence)
        optional, evidence["optional_absence"] = common.load_optional_probe(root, refs)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["PYTHONHASHSEED"] = "0"
        R.require(os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
                  "existing guard offline environment")
        model_helper = R.load(root, config["inherited_model_helper_ref"], "_dr_existing_model_identity")
        model_gates = dict(refs=refs, plan=dict(model_manifest_ref=refs[R.MODEL_PLAN]),
            prior_U=R.read(R.check_ref(root, config["prior_model_U_ref"])))
        model_dir, identity, stats = model_helper.inherited_model_binding(root, model_gates,
            root / R.MODEL, root / R.MODEL_PLAN, R=R)
        evidence["model_identity"] = identity
        qualifier = R.load(root, refs[R.QUALIFIER], "_dr_existing_author_finder")
        finder = qualifier.BoundAuthorFinder(root, refs)
        R.require(not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules), "fresh author import")
        sys.meta_path.insert(0, finder)
        import torch
        import vllm
        import vllm.utils.import_utils as import_utils
        probe = optional.install_capability_probe(import_utils, finder, root, refs)
        R.require(qualifier.normalize_uuid(torch.cuda.get_device_properties(0).uuid) ==
                  qualifier.normalize_uuid(config["gpu_uuid"]), "actual current device")
        R.require(Path(vllm.__file__).resolve().is_relative_to(root / R.AUTHOR), "actual author executor")
        author_metrics = importlib.import_module(
            "vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics")
        metrics_ref = refs[R.AUTHOR + "/vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py"]
        R.require(Path(author_metrics.__file__).resolve() == R.check_ref(root, metrics_ref),
                  "actual frozen author transfer stats producer and logger")
        stats_serialization = InprocessOffloadStatsSerialization(
            author_metrics.OffloadingConnectorStats, author_metrics.OffloadingOperationMetrics)
        evidence["common_stats_serialization_source_ref"] = metrics_ref
        native = importlib.import_module("py_kvcache.vllm")
        reactor_module = importlib.import_module("py_kvcache.reactor")
        from prefix_io_control.p4_restore_order_journal import RestoreOrderJournal
        order_journal = RestoreOrderJournal(config["run_id"])
        reactor_original = reactor_module.IoReactor.__init__
        def observed_reactor(owner, *args, **kwargs):
            owner._prefix_restore_order_journal = order_journal
            reactor_original(owner, *args, **kwargs)
        reactor_wrapper = observed_reactor
        reactor_module.IoReactor.__init__ = reactor_wrapper
        if arm in ("CAL", "F8+D_R"):
            helper = importlib.import_module("prefix_io_control.p4_restore_forecast")
            R.require(Path(helper.__file__).resolve() == R.check_ref(root, config["dr_helper_ref"]),
                      "actual frozen restore calibration helper")
            calibration = None
            if arm == "F8+D_R":
                try:
                    calibration = helper.load_restore_calibration(root, refs,
                        config["dr_calibration_result_ref"], config["dr_calibration_guard_ref"],
                        expected_gpu_uuid=config["gpu_uuid"],
                        expected_common_runtime_domain_sha256=config["common_runtime_domain_sha256"],
                        expected_runtime_refs=config["runtime_refs"], now_ns=time.monotonic_ns(),
                        max_age_ns=config["history_max_age_ns"])
                except helper.RestoreCalibrationCoverageMissing as exc:
                    # Only verified closed actual calibration with insufficient
                    # actionable cells reaches this typed path. Device/source/
                    # lifecycle/corrupt-data failures remain hard failures.
                    evidence["calibration_coverage_missing"] = dict(reason=exc.reason,
                        calibration_run_id=exc.run_id, sample_count=exc.sample_count,
                        exact_cell_counts=exc.exact_cell_counts)
                    evidence["forecast_disabled"] = True
                else:
                    evidence["forecast_disabled"] = False
            else:
                evidence["forecast_disabled"] = True
            from prefix_io_control.p4_bridge import NativeP4Bridge
            bridge_original = NativeP4Bridge.__init__
            def configure_actual_bridge(owner, *args, **kwargs):
                bridge_original(owner, *args, **kwargs)
                owner.configure_restore_development(calibration,
                    history_max_age_ns=config["history_max_age_ns"],
                    order_journal=order_journal)
            bridge_wrapper = configure_actual_bridge
            NativeP4Bridge.__init__ = bridge_wrapper
        scripts = root / "experiments/prefix_io_v1/scripts"
        p3 = R.load(root, R.ref(root, "experiments/prefix_io_v1/scripts/run_p3_native_pilot.py"),
                    "run_p3_native_pilot")
        original_probe = p3.worker_probe
        # Keeping the original helper import names preserves its worker_trace
        # and hardlink/source-content checks. The only runtime probe adaptation
        # is the current native owner's safe inspect path below.
        main_module = R.load(root, config["original_runner_ref"], "_dr_original_p316")
        main_module.Path = details_replay.path_type()
        original_validate = main_module.validate
        def migrated_validate(manifest, candidate, current_gpu):
            return validate_original_identity(original_validate, manifest, candidate,
                current_gpu, config["device_migration"]["historical_gpu_uuid"],
                config["device_migration"])
        main_module.validate = migrated_validate
        # Source registration describes the original immutable SSD corpus; its
        # historical GPU field is validated as historical, after current GPU
        # validation above. No source, threshold or qualification field changes.
        registration = importlib.import_module("storage_source_registration")
        old_registration = registration.validate_registration
        def migrated_registration(project, path, source, output, current_gpu):
            R.require(current_gpu == config["gpu_uuid"], "current source replay device")
            answer = old_registration(project, path, source, output,
                config["device_migration"]["historical_gpu_uuid"])
            answer["current_device_replay"] = config["gpu_uuid"]
            answer["historical_gpu_qualification_only"] = True
            return answer
        registration.validate_registration = migrated_registration
        old_model = base.validate_local_model
        def inherited_model(path, plan):
            R.require(Path(path).resolve() == model_dir.resolve() and
                      Path(plan).resolve() == (root / R.MODEL_PLAN).resolve(), "same inherited model")
            return model_dir, identity
        base.validate_local_model = inherited_model
        old_environment = base.configure_runtime_environment
        # Process-private caches/SDK/ninja have already been bound by the tested
        # common setup; avoid the old shared cache destination replacing them.
        base.configure_runtime_environment = lambda: None
        old_llm = vllm.LLM
        def original_llm(*args, **kwargs):
            nonlocal llm, handler, worker
            revised = control_engine(kwargs, config["engine_controls"], arm)
            if selected_engine is None or {k:v for k,v in revised.items() if k != "model"} != selected_engine:
                raise ValueError("actual LLM configuration differs from saved frozen engine")
            llm = old_llm(*args, **revised)
            owners = []
            def capture(owner):
                owners.append(runtime.capture_original_worker(owner))
                return dict(captured=True)
            value = llm.collective_rpc(capture, timeout=10)
            R.require(len(value) == len(owners) == 1, "sole original worker")
            worker, handler = owners[0]
            evidence["loaded_native_source_binding"] = runtime.verify_loaded_native(root, refs, R)
            evidence["original_planner_identity"] = runtime.original_planner_identity(llm, handler, root, refs, R)
            reactor = handler.coordinator.reactor
            R.require(reactor.ring._kernel is not None, "actual original Linux AIO kernel backend")
            kernel_tap.install(reactor.ring._kernel)
            if arm == "U":
                R.require(reactor._prefix_p4_bridge is None and reactor._prefix_dispatch_controller is None,
                          "actual U controls off")
            else:
                R.require(reactor._prefix_p4_bridge.mode == "dependency_only" and
                          reactor._prefix_dispatch_controller.config.mode == "fixed" and
                          reactor._prefix_dispatch_controller.config.byte_quantum == reactor.file_store.io_size,
                          "actual original fixed controller and restore-only bridge")
            return llm
        vllm.LLM = original_llm
        def current_probe(owner, action, limits=None):
            clean = dict(limits) if limits else None
            if clean is not None:
                for key in ("native_cpu_probe", "native_cpu_phase", "native_observation"):
                    clean.pop(key, None)
            answer = original_probe(owner, action, clean)
            _, actual_handler = runtime.capture_original_worker(owner)
            reactor = actual_handler.coordinator.reactor
            deadline = time.monotonic() + 10
            while True:
                snapshot = reactor.inspect_snapshot(timeout=2)
                if action not in ("drain", "finish"):
                    break
                try:
                    runtime.validate_drained_snapshot(snapshot)
                    break
                except ValueError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.005)
            answer["current_native_snapshots"] = [deepcopy(snapshot)]
            journal_value = order_journal.snapshot()
            answer["restore_order_journal_summary"] = {
                key:deepcopy(value) for key, value in journal_value.items()
                if key not in ("actual_order_changes", "actual_accepted_read_submission_order")}
            return answer
        main_module.worker_probe = current_probe
        old_write = base.write_new_json
        def saved_original_record(path, value):
            nonlocal selected_engine
            value = deepcopy(value)
            if Path(path).name == "frozen-config.json":
                value["engine"] = control_engine(value["engine"], config["engine_controls"], arm)
                selected_engine = deepcopy(value["engine"])
                value["current_device_adapter"] = deepcopy(evidence)
            elif Path(path).name == "result.json":
                # Save the unmodified original report before adapter evidence
                # can fail. In particular, preserve original cohort deadline,
                # rows and unfinished token events independently of tail checks.
                original_report = out / "original-runner-report.json"
                old_write(original_report, deepcopy(value))
                evidence["original_runner_report_ref"] = R.ref(root, original_report.relative_to(root).as_posix())
                # This callback is reached after original runner finally invoked
                # the actual engine shutdown. Read bridge state only from owner
                # snapshots captured while that owner was still alive.
                if llm is not None and handler is not None and value.get("engine_shutdown") == "completed":
                    evidence["original_engine_shutdown_returned"] = True
                    try:
                        actual_tail, full_restore = projected_original_tail(tail, handler)
                        restore_path = out / "original-native-shutdown-restore-evidence.json"
                        old_write(restore_path, full_restore)
                        evidence["original_native_shutdown_restore_ref"] = R.ref(root,
                            restore_path.relative_to(root).as_posix())
                        evidence["post_shutdown"] = actual_tail
                        evidence["native_tail_drained"] = runtime.verify_original_tail(actual_tail)
                        reactor = handler.coordinator.reactor
                        evidence["native_tail_assertions"] = dict(handler_shutdown=handler.is_shutdown,
                            worker_alive=reactor._worker.is_alive(), aio_worker_alive=reactor.ring._worker.is_alive(),
                            reactor_closed=reactor._closed)
                        R.require(handler.is_shutdown and not reactor._worker.is_alive() and
                                  not reactor.ring._worker.is_alive() and reactor._closed,
                                  "actual original shutdown and workers ended")
                    except BaseException as exc:
                        value["status"] = "FAILED_ADAPTER_TAIL_EVIDENCE"
                        value["adapter_tail_error"] = dict(type=type(exc).__name__, message=str(exc),
                            traceback=traceback.format_exc(limit=8))
                        evidence["adapter_tail_error"] = deepcopy(value["adapter_tail_error"])
                model_helper.inherited_model_binding(root, model_gates, model_dir,
                    root / R.MODEL_PLAN, R=R, previous_stats=stats)
                actual_metadata = {k:deepcopy(evidence[k]) for k in
                    ("gpu_uuid", "common_runtime_domain_sha256", "runtime_refs", "run_id",
                     "subprocess_sid", "subprocess_pid", "original_engine_shutdown_returned", "native_tail_drained")}
                value.update(actual_metadata)
                if type(value.get("probe")) is dict:
                    value["probe"].update(deepcopy(actual_metadata))
                value["restore_order_journal"] = order_journal.snapshot()
                value["kernel_actual_submission_order"] = kernel_tap.snapshot()
                value["common_stats_serialization"] = stats_serialization.snapshot()
                value["actual_stage_mode"] = (config["engine_controls"].get("prefix_io_p4_policy", {}).get("fixed_stage_policy") or {}).get("mode", "off")
                value["actual_p4_mode"] = config["engine_controls"].get("prefix_io_p4_policy", {}).get("mode", "off")
                value["actual_ordinary_quota_installed"] = arm != "U"
                value["original_runner_policy_fields_scope"] = "historical runner fields; current actual controls separately recorded"
                value["current_device_adapter"] = deepcopy(evidence)
                value["method_component"] = "U" if arm == "U" else ("F8+D_R forecast-disabled independent calibration" if arm == "CAL" else "F8+D_R restore-parent closure ordering")
                value["production_prediction_qualified"] = False
            return old_write(path, value)
        base.write_new_json = saved_original_record
        argv = [str(scripts / "run_concurrent_pilot_p316.py"),
            "--output", str(out), "--model-dir", str(model_dir),
            "--model-plan", str(root / R.MODEL_PLAN),
            "--curves", str(R.check_ref(root, config["curves_ref"])),
            "--manifest", str(R.check_ref(root, config["manifest_ref"])),
            "--qualification", str(R.check_ref(root, config["qualification_ref"])),
            "--storage-source", str(R.safe(root, config["storage_source"])),
            "--storage-registration", str(R.check_ref(root, config["storage_registration_ref"]))]
        evidence["original_runner_argv"] = argv
        previous_argv = sys.argv
        sys.argv = argv
        try:
            code = main_module.main()
        finally:
            sys.argv = previous_argv
        if code == 0 and not (evidence["original_engine_shutdown_returned"] and
                              evidence["native_tail_drained"] and not evidence.get("adapter_tail_error")):
            return 1
        return code
    except BaseException as exc:
        evidence["adapter_error"] = dict(type=type(exc).__name__, message=str(exc),
                                         traceback=traceback.format_exc(limit=18))
        raise
    finally:
        # If the original LLM constructor returned but an adapter binding check
        # failed before returning it to the P316 runner, that runner has no LLM
        # handle. Close the actual original handle here using the same existing
        # drain/shutdown functions; the outer guard remains the OS backstop.
        if llm is not None and not evidence["original_engine_shutdown_returned"]:
            try:
                acquire = R.load(root, refs[R.DRAIN], "_dr_error_original_drain")
                if handler is None:
                    owners = []
                    def capture_for_cleanup(owner):
                        owners.append(runtime.capture_original_worker(owner))
                        return dict(captured=True)
                    llm.collective_rpc(capture_for_cleanup, timeout=10)
                    R.require(len(owners) == 1, "actual sole cleanup worker")
                    worker, handler = owners[0]
                evidence["error_path_original_drain"] = runtime.drain_original(llm, acquire, handler)
            except BaseException as exc:
                evidence["error_path_drain_error"] = type(exc).__name__ + ": " + str(exc)
            finally:
                try:
                    llm.llm_engine.engine_core.shutdown(timeout=15)
                    evidence["original_engine_shutdown_returned"] = True
                    if handler is not None:
                        error_tail, error_restore = projected_original_tail(tail, handler)
                        evidence["error_path_post_shutdown"] = error_tail
                        evidence["error_path_restore_metadata"] = error_restore
                        evidence["native_tail_drained"] = runtime.verify_original_tail(
                            evidence["error_path_post_shutdown"])
                except BaseException as exc:
                    evidence["error_path_shutdown_error"] = type(exc).__name__ + ": " + str(exc)
        if bridge_wrapper is not None:
            from prefix_io_control.p4_bridge import NativeP4Bridge
            if NativeP4Bridge.__init__ is bridge_wrapper:
                NativeP4Bridge.__init__ = bridge_original
        if reactor_wrapper is not None and reactor_module.IoReactor.__init__ is reactor_wrapper:
            reactor_module.IoReactor.__init__ = reactor_original
        if order_journal is not None:
            evidence["restore_order_journal"] = order_journal.snapshot()
        if details_replay is not None:
            evidence["original_details_creation_replay_consumed"] = details_replay.consumed
        if stats_serialization is not None:
            evidence["common_stats_serialization"] = stats_serialization.snapshot()
            evidence["common_stats_serialization_restored"] = stats_serialization.restore_after_shutdown(
                ended=llm is None or evidence["original_engine_shutdown_returned"] is True)
        evidence["kernel_actual_submission_order"] = kernel_tap.snapshot()
        if handler is not None:
            native_owner = handler.coordinator.reactor
            evidence["kernel_submission_tap_restored"] = kernel_tap.restore_after_shutdown(
                joined=(not native_owner._worker.is_alive() and not native_owner.ring._worker.is_alive()))
        else:
            evidence["kernel_submission_tap_restored"] = kernel_tap.restore_after_shutdown(joined=True)
        if base is not None:
            if old_write is not None:
                base.write_new_json = old_write
            if old_model is not None:
                base.validate_local_model = old_model
            if old_environment is not None:
                base.configure_runtime_environment = old_environment
        if old_llm is not None:
            vllm.LLM = old_llm
        if probe is not None:
            evidence["optional_probe_restored"] = probe.detach()
        if finder is not None and finder in sys.meta_path:
            sys.meta_path.remove(finder)
        if "registration" in locals():
            registration.validate_registration = old_registration
        if out.parent.is_dir():
            R.new_json(out.parent / "current-device-adapter-evidence.json", evidence)
        sys.path[:] = previous_path
        os.chdir(previous_cwd)
        os.environ.clear()
        os.environ.update(previous_env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("execution requires the existing original GPU guard")
    root = args.project.resolve(strict=True)
    R = bootstrap(root)
    config = R.read(R.safe(root, args.config))
    config["config_ref"] = R.ref(root, args.config)
    refs = R.source_rows(root, config["source_lock_ref"], full=False)
    if "native_source_root" in config:
        R.require(R.safe(root, config["native_source_root"]).name == "py_kvcache", "current native package source root")
        R.NATIVE = config["native_source_root"]
    for key in ("runner_ref", "original_runner_ref", "manifest_ref", "curves_ref",
                "qualification_ref", "storage_registration_ref", "site_sdk_ref",
                "site_sdk_migration_ref", "inherited_model_helper_ref", "prior_model_U_ref"):
        R.check_ref(root, config[key])
    R.require(Path(__file__).resolve() == R.check_ref(root, config["runner_ref"]), "frozen actual adapter")
    return execute(root, config, refs, R)


if __name__ == "__main__":
    raise SystemExit(main())
