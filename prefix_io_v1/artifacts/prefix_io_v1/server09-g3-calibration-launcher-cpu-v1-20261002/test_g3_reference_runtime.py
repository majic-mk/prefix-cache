"""Pinned original diagnostic main replay with CPU fixtures, never GPU evidence."""
import argparse
import ast
import contextlib
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--source-root", type=Path)
parser.add_argument("--delegate-path", type=Path)
ARGS, REST = parser.parse_known_args()
SOURCE_ROOT = ARGS.source_root or PROJECT / "artifacts/prefix_io_v1_server08_primary_qualification/contents"
DELEGATE_PATH = ARGS.delegate_path or HERE / "g3_calibration_runtime_metrics_v2.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = load(HERE / "g3_reference_runtime.py", "_reference_cpu_contract")
D = load(DELEGATE_PATH, "_reference_original_delegate_contract")


def witness(mode="cold"):
    return dict(schema_version=1, purpose=R.PURPOSE, gpu_uuid=R.GPU_UUID, label=R.JOB_NAMES[mode],
                source_lock_sha256="a"*64, scope_sha256="b"*64,
                permissions_ref=dict(path="effective-permit", bytes=1, sha256="c"*64), session_id=88,
                guard_command=["python", "--execute", "--mode", mode], seconds_limit=300, reserved_seconds=320,
                authorized_scope_verified=True, active_reservation_verified=True)


def fixture_rows(mode):
    kinds = ("f", "gpu_hot") if mode == "cold" else ("g_ssd", "g_mem")
    rows, requests = [], []
    for ordinal in range(6):
        kind, rep = kinds[ordinal % 2], ordinal // 2
        prompt = [2128+rep*1500]+[1000+i%500 for i in range(127)]+[777]
        item = dict(request_id="cpu-%d" % ordinal, prompt_token_ids=prompt,
                    output_token_ids=[42], num_cached_tokens=0 if kind == "f" else 128)
        values = [{str(t): dict(logprob=-0.25-rank, rank=rank) for rank, t in enumerate(range(42, 47), 1)}]
        rows.append(dict(item, kind=kind, prefix_tokens=128, rep=rep, warmup=rep == 0, output_logprobs=values))
        requests.append(dict(item, ordinal=ordinal, sentinel=False))
    report = dict(mode=mode, status="PASSED_NATIVE_"+mode.upper()+"_ACQUISITION", rows=rows,
                  native_hot_diagnostic=mode == "cold", cached_reference_logprobs=mode == "paired")
    return report, requests


@contextlib.contextmanager
def canonical_candidate():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        sources = {}
        for relative, original in ((R.SOURCE, Path(R.__file__)), (R.DELEGATE, DELEGATE_PATH)):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = original.read_bytes()
            path.write_bytes(raw)
            sources[relative] = dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        module = load(root / R.SOURCE, "_reference_canonical_cpu_fixture")
        try:
            yield root, module, sources
        finally:
            sys.modules.pop(module.__name__, None)


class ReferenceRuntimeContracts(unittest.TestCase):
    def test_stdlib_import_without_write_or_backend(self):
        tree = ast.parse(Path(R.__file__).read_bytes())
        names = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
        names.update(node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom))
        self.assertFalse(names & {"torch", "vllm", "py_kvcache"})
        with mock.patch.object(Path, "mkdir", side_effect=AssertionError("import wrote directory")):
            exec(compile(tree, "stdlib-import", "exec"), {"__name__": "stdlib_cpu_fixture"})

    def test_exact_original_diagnostic_argv_and_fixed_axis(self):
        for mode in R.MODES:
            argv = R.fixed_argv(mode, "out", "storage", "model", "plan")
            for key, value in (("--mode", mode), ("--domain", "1024"), ("--sizes", "128"), ("--reps", "3"),
                               ("--max-num-seqs", "1"), ("--iodepth", "4")):
                self.assertEqual(argv[argv.index(key)+1], value)
            self.assertEqual(argv.count("--native-hot-diagnostic"), int(mode == "cold"))
            self.assertEqual(argv.count("--diagnostic-logprobs"), int(mode == "cold"))
            self.assertEqual(argv.count("--cached-reference-logprobs"), int(mode == "paired"))
            self.assertNotIn("--curves", argv)
        with self.assertRaises(ValueError):
            R.fixed_argv("populate", 1, 2, 3, 4)

    def test_new_guard_old_authority_rejected_before_any_source(self):
        for value in ({}, dict(witness(), purpose=D.PURPOSE),
                      dict(witness(), label="server09-g3-calibration-cold-02")):
            with mock.patch.object(R, "_load_delegate", side_effect=AssertionError("source read")), \
                 mock.patch.object(Path, "resolve", side_effect=AssertionError("filesystem read")):
                with self.assertRaises(ValueError):
                    R.execute_original_acquisition(PROJECT, "cold", PROJECT, PROJECT, {}, value)

    def test_guard_actual_sid_uuid_and_reservation_required(self):
        with mock.patch.object(R.os, "name", "posix"), mock.patch.object(R.os, "getsid", return_value=88, create=True), \
             mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES=R.GPU_UUID):
            R.validate_guard(witness(), "cold")
            for key, value in (("session_id", 89), ("active_reservation_verified", False), ("reserved_seconds", 300),
                               ("gpu_uuid", "GPU-other"), ("scope_sha256", "D"*64)):
                with self.assertRaises(ValueError):
                    R.validate_guard(dict(witness(), **{key: value}), "cold")

    def test_canonical_pinned_delegate_and_source_drift(self):
        with canonical_candidate() as (root, module, refs):
            loaded = module._load_delegate(root, refs)
            self.assertEqual(loaded.PINNED[loaded.ACQUIRE], (18995, "68b2c045bcc9a7de84771d556b0e01180a592280002a2d4360d1c7500c9c856e"))
            self.assertTrue(module._unload(loaded))
            self.assertNotIn(loaded.__name__, sys.modules)
            (root / R.DELEGATE).write_bytes(DELEGATE_PATH.read_bytes()+b"\n")
            with self.assertRaises(ValueError):
                module._load_delegate(root, refs)

    def test_cpu_preflight_once_restores_private_module_without_write(self):
        with canonical_candidate() as (root, module, refs):
            delegate = module._load_delegate(root, refs)
            called = []
            def preflight(*args):
                called.append(args)
                return dict(status="CPU_FIXTURE", framework_imports=0, GPU_operations=0)
            with mock.patch.object(module, "_load_delegate", return_value=delegate), \
                 mock.patch.object(delegate, "preflight_runtime", side_effect=preflight), \
                 mock.patch.object(Path, "mkdir", side_effect=AssertionError("preflight mkdir")):
                result = module.preflight_runtime(root, refs)
            self.assertEqual(len(called), 1)
            self.assertEqual(result["purpose"], R.PURPOSE)
            self.assertFalse(result["new_GPU_authorization"])
            self.assertFalse(result["numerical_reference_verified"])
            self.assertNotIn(delegate.__name__, sys.modules)

    def test_exact_storage_lifecycle_and_no_old_job_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for mode in R.MODES:
                out = root / "experiments/prefix_io_v1/runs" / R.JOB_NAMES[mode] / "details"
                out.mkdir(parents=True)
                storage = root / R.STORAGE[mode]
                if mode == "paired":
                    storage.mkdir(parents=True)
                self.assertEqual(R._paths(root, mode, out, storage), (out, storage))
                wrong = root / "experiments/prefix_io_v1/runs/server09-g3-calibration-cold-02/details"
                with self.assertRaises(ValueError):
                    R._paths(root, mode, wrong, storage)
            self.assertFalse((root / R.STORAGE["cold"]).exists())

    def test_complete_diagnostic_values_and_output_alignment_fail_closed(self):
        report, requests = fixture_rows("cold")
        copied = R.reference_rows(report, dict(requests=requests), "cold")
        self.assertEqual(len(copied), 6)
        self.assertEqual(copied[0]["output_logprobs"], report["rows"][0]["output_logprobs"])
        requests[0]["output_token_ids"] = [99]
        with self.assertRaises(ValueError):
            R.reference_rows(report, dict(requests=requests), "cold")
        report, requests = fixture_rows("cold")
        report["rows"][0]["output_logprobs"] = [{}]
        with self.assertRaises(ValueError):
            R.reference_rows(report, dict(requests=requests), "cold")

    def test_execute_original_once_preserves_every_closure_field_and_original_receipt(self):
        with canonical_candidate() as (root, module, refs):
            out = root / "experiments/prefix_io_v1/runs" / R.JOB_NAMES["cold"] / "details"
            out.mkdir(parents=True)
            delegate = module._load_delegate(root, refs)
            original = {key: getattr(delegate, key) for key in ("PURPOSE", "MODES", "validate_guard", "fixed_argv")}
            report, requests = fixture_rows("cold")
            receipt = dict(origin="cpu_fixture", purpose=R.PURPOSE, mode="cold", source_lock_sha256="a"*64,
                           scope_sha256="b"*64, process_identity=dict(pid=1, sid=88), original_exit_code=0,
                           engine_shutdown_calls=1, handler_shutdown_calls=0, original_shutdown_completed=True,
                           source_unchanged=True, source_preservation_scope="FOUR_PINNED_ADAPTER_ORIGINAL_FILES_ONLY",
                           common_helpers_restored=True, requests=requests, frozen_config=dict(sampling=dict(logprobs=5)),
                           observer_fault_types=[], raw_calibration_only=True, cost_qualified=False,
                           production_qualified=False, performance_claim=False)
            calls = []
            def execute(*args):
                calls.append(args)
                self.assertIs(delegate.validate_guard, module.validate_guard)
                self.assertIs(delegate.fixed_argv, module.fixed_argv)
                acquisition = out / "acquisition"
                acquisition.mkdir()
                (acquisition / "result.json").write_text(json.dumps(report))
                (out / R.LEGACY_RECEIPT_FILE).write_text(json.dumps(receipt))
                return dict(original_exit_code=0, runtime_receipt=receipt)
            with mock.patch.object(module, "validate_guard"), mock.patch.object(module, "_load_delegate", return_value=delegate), \
                 mock.patch.object(delegate, "execute_original_acquisition", side_effect=execute):
                result = module.execute_original_acquisition(root, "cold", out, root / R.STORAGE["cold"], refs, witness())
            self.assertEqual(len(calls), 1)
            new = result["runtime_receipt"]
            for key, value in receipt.items():
                if key != "raw_calibration_only":
                    self.assertEqual(new[key], value, key)
            self.assertEqual(json.loads((out / R.LEGACY_RECEIPT_FILE).read_text()), receipt)
            self.assertEqual(json.loads((out / R.RECEIPT_FILE).read_text()), new)
            self.assertEqual(new["original_delegate_receipt_ref"]["sha256"],
                             hashlib.sha256((out / R.LEGACY_RECEIPT_FILE).read_bytes()).hexdigest())
            self.assertTrue(new["reference_delegate_overrides_restored"])
            self.assertTrue(set(new)-set(receipt) <= set(new["reference_metadata_fields"]))
            self.assertFalse(new["GPU_qualified"])
            self.assertFalse(new["numerical_reference_verified"])
            self.assertEqual(new["origin"], "cpu_fixture")
            for key, value in original.items():
                self.assertIs(getattr(delegate, key), value)
            self.assertNotIn(delegate.__name__, sys.modules)

    def test_original_error_same_object_once_and_all_delegate_overrides_restored(self):
        with canonical_candidate() as (root, module, refs):
            out = root / "experiments/prefix_io_v1/runs" / R.JOB_NAMES["cold"] / "details"
            out.mkdir(parents=True)
            delegate = module._load_delegate(root, refs)
            original = {key: getattr(delegate, key) for key in ("PURPOSE", "MODES", "validate_guard", "fixed_argv")}
            error = KeyboardInterrupt()
            with mock.patch.object(module, "validate_guard"), mock.patch.object(module, "_load_delegate", return_value=delegate), \
                 mock.patch.object(delegate, "execute_original_acquisition", side_effect=error) as call:
                with self.assertRaises(KeyboardInterrupt) as caught:
                    module.execute_original_acquisition(root, "cold", out, root / R.STORAGE["cold"], refs, witness())
            self.assertIs(caught.exception, error)
            call.assert_called_once()
            for key, value in original.items():
                self.assertIs(getattr(delegate, key), value)
            self.assertNotIn(delegate.__name__, sys.modules)
            self.assertFalse((out / R.RECEIPT_FILE).exists())


class OriginalDiagnosticReplay(unittest.TestCase):
    def test_pinned_original_cold_hot_and_ssd_memory_twelve_request_diagnostics(self):
        path = SOURCE_ROOT / D.ACQUIRE
        raw = path.read_bytes()
        self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), D.PINNED[D.ACQUIRE])
        nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name in ("prompt", "trace_summary", "main")]
        self.assertEqual(len(nodes), 3)
        total = 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "model").mkdir()
            for mode in R.MODES:
                out, storage = root / mode, root / (mode+"-storage")
                if mode == "paired":
                    storage.mkdir()
                taps = D.ScalarTaps()
                profile, calls, returned = {}, [], []
                @dataclasses.dataclass
                class Metrics:
                    first_token_latency: float = 0.125
                    is_corrupted: bool = False
                class Sampling:
                    def __init__(self, **values):
                        self.values = values
                class LLM:
                    def __init__(self, model, **config):
                        self.gpu, self.memory, self.counter = set(), set(), 0
                        self.llm_engine = types.SimpleNamespace(engine_core=types.SimpleNamespace(shutdown=lambda **kw: calls.append("shutdown")))
                    def reset_prefix_cache(self, **kwargs):
                        calls.append(("reset", kwargs))
                        self.gpu.clear()
                        return True
                    def collective_rpc(self, method, timeout, args):
                        return [method(self, *args)]
                    def generate(self, inputs, sampling, use_tqdm):
                        self.counter += 1
                        tokens = inputs[0]["prompt_token_ids"]
                        key, n = tuple(tokens), len(tokens)-1
                        cached = mode == "paired" or key in self.gpu
                        memory = mode == "paired" and key in self.memory
                        self.gpu.add(key)
                        self.memory.add(key)
                        rid = "cpu-original-%d" % self.counter
                        profile.update(rid=rid, n=n, cached=cached, memory=memory)
                        self_test.assertEqual(sampling.values["logprobs"], 5)
                        values = [{token: types.SimpleNamespace(logprob=-0.25-rank, rank=rank)
                                   for rank, token in enumerate(range(42, 47), 1)}]
                        output = [types.SimpleNamespace(request_id=rid, prompt_token_ids=tokens,
                            num_cached_tokens=n if cached else 0, outputs=[types.SimpleNamespace(token_ids=[42], logprobs=values)], metrics=Metrics())]
                        returned.append(output)
                        calls.append(("generate", rid))
                        return output
                self_test = self
                original_generate = LLM.generate
                taps.install_generate(LLM)
                def control(worker, action, path=None):
                    if action == "begin":
                        profile["path"] = path
                        return dict(profile_active=True)
                    if action == "end":
                        events = []
                        if mode == "paired":
                            n, rid, memory = profile["n"], profile["rid"], profile["memory"]
                            events = [dict(name="py_kvcache.transfer", args=dict(req_id=rid, direction="storage_to_gpu", success=True,
                                num_bytes=n*57344, src_cache=n//16 if memory else 0, src_file=0 if memory else n//16, src_preload=0)),
                                dict(name="py_kvcache.cuda_staging", args=dict(req_id=rid, direction="storage_to_gpu"))]
                            if not memory:
                                events.append(dict(name="py_kvcache.file_read", args=dict(req_id=rid, num_bytes=n*57344)))
                        Path(path).write_text(json.dumps(dict(traceEvents=events)))
                    return dict(handlers=[dict(staging_bytes=128*1024**2, staging_budget=128*1024**2, pinned=True,
                        io_size=917504, storage_block_bytes=917504,
                        aio=dict(accepted=0, completed=0, reaped=0, outstanding=0, pending=0, ready=0, unreaped=0, fatal=None))])
                base = types.SimpleNamespace(MODEL_ID=D.MODEL_NAME, AUTHOR_ROOT=root, ENGINE=dict(kv_transfer_config=None), SAMPLING={},
                    configure_runtime_environment=lambda: None, validate_local_model=lambda *a: (root / "model", {}),
                    write_new_json=lambda path, value: path.write_text(json.dumps(value)), require=D.require)
                vllm, torch = types.ModuleType("vllm"), types.ModuleType("torch")
                vllm.__file__, vllm.LLM, vllm.SamplingParams = str(root / "vllm.py"), LLM, Sampling
                torch.cuda = types.SimpleNamespace(mem_get_info=lambda: (32*1024**3, 32*1024**3))
                storage_module = types.ModuleType("experiment_storage")
                storage_module.authorized_path, storage_module.preflight = lambda p: None, lambda p, n: {}
                heldout = types.ModuleType("heldout_manifest")
                heldout.disk_requirement = lambda *args: dict(required_free_bytes=0, floor_bytes=0)
                concurrent = types.ModuleType("concurrent_pilot_contract")
                concurrent.acquisition_delta, concurrent.acquisition_io_depth = lambda *args: {}, lambda *args: 4
                namespace = dict(__name__="original_cpu_diagnostic", __file__=str(path), argparse=argparse, dataclasses=dataclasses,
                    hashlib=hashlib, json=json, os=os, re=re, Path=Path, time=time, traceback=traceback, base=base, worker_control=control)
                exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec", dont_inherit=True), namespace)
                old_cwd = Path.cwd()
                try:
                    with mock.patch.dict(sys.modules, {"vllm": vllm, "torch": torch, "experiment_storage": storage_module,
                        "heldout_manifest": heldout, "concurrent_pilot_contract": concurrent}), \
                         mock.patch.dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES=R.GPU_UUID), \
                         mock.patch.object(sys, "argv", R.fixed_argv(mode, out, storage, root / "model", root / "plan")), \
                         mock.patch.object(Path, "symlink_to") as alias, contextlib.redirect_stdout(io.StringIO()):
                        code = namespace["main"]()
                    alias.assert_called_once_with(root / "model", target_is_directory=True)
                finally:
                    os.chdir(old_cwd)
                    taps.restore()
                self.assertEqual(code, 0)
                self.assertIs(LLM.generate, original_generate)
                report, config = json.loads((out / "result.json").read_text()), json.loads((out / "frozen-config.json").read_text())
                receipt = dict(requests=taps.requests)
                rows = R.reference_rows(report, receipt, mode)
                self.assertEqual(len(rows), 6)
                self.assertEqual(len(returned), 6)
                self.assertEqual(calls.count("shutdown"), 1)
                self.assertEqual(report["engine_shutdown"], "completed")
                self.assertEqual(config["sampling"]["logprobs"], 5)
                self.assertEqual(config["model_alias"], D.MODEL_NAME)
                self.assertEqual([r["output_token_ids"] for r in rows], [[42]]*6)
                self.assertEqual([r["num_cached_tokens"] for r in rows], [0, 128]*3 if mode == "cold" else [128]*6)
                if mode == "cold":
                    self.assertFalse(storage.exists())
                else:
                    self.assertEqual(config["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["load_planner"], "off")
                    self.assertEqual([r["trace"]["foreground_logical_read_bytes"] for r in report["rows"]], [128*57344, 0]*3)
                self.assertEqual(taps.failures, [])
                total += len(taps.requests)
        self.assertEqual(total, 12)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *REST])
