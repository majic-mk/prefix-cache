"""CPU API/source boundaries only. Connector/event/worker objects are SYNTHETIC.

This executes the frozen C5 collector.install API with origin=cpu_fixture. It
does not load real vLLM, issue native receipts, run GPU, or measure on costs.
"""
import ast
from copy import deepcopy
from hashlib import sha256
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PREPARATION = Path(os.environ.get("C5_NATIVE_PREPARATION_ROOT", str(BASE / "notification_v5_native_cost_preparation_cpu"))).resolve()
CANDIDATE = Path(os.environ.get("C5_NATIVE_PROTOCOL_CANDIDATE_ROOT", str(BASE / "p4_single_file_candidate_v5_cpu"))).resolve()
NATIVE = Path(os.environ.get("C5_NATIVE_PROTOCOL_BASELINE_ROOT", str(BASE / "native_cost_v6"))).resolve()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tree(path): return ast.parse(path.read_bytes(), filename=str(path))


def functions(path):
    return {node.name: node for node in tree(path).body if isinstance(node, ast.FunctionDef)}


def ast_same(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def constants(path):
    values = {}
    def evaluate(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.Name): return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add): return evaluate(node.left) + evaluate(node.right)
        if isinstance(node, (ast.Tuple, ast.List)):
            result = [evaluate(x) for x in node.elts]
            return tuple(result) if isinstance(node, ast.Tuple) else result
        raise ValueError("nonliteral source constant")
    for node in tree(path).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try: values[node.targets[0].id] = evaluate(node.value)
            except (KeyError, ValueError, TypeError): pass
    return values


class SyntheticEvent:
    def __init__(self, **kwargs): self.records = self.queries = self.elapsed = 0
    def record(self): self.records += 1; return "original-record-result"
    def query(self): self.queries += 1; return True
    def elapsed_time(self, other): self.elapsed += 1; return 0.001


class SyntheticRunner:
    def __init__(self): self.calls = 0
    def _prepare_inputs(self, value): self.calls += 1; return ("original-prepare", value)


class SyntheticObserver:
    def __init__(self, enabled=True):
        self.enabled = enabled; self.scalar = NS(enabled=enabled)
        self.detaches = 0
    def detach(self): self.detaches += 1; self.enabled = False


class SyntheticScalar:
    METHODS = ("execute_model", "_prepare_inputs", "sample_tokens")
    @staticmethod
    def SourceRef(*values): return tuple(values)
    @staticmethod
    def MethodBinding(*values): return NS(refs=values[0], methods=values[1], origin=values[2])


def api_fixture(root, *, enabled=True):
    collector = load(CANDIDATE / "native_full_step_collector.py", "_protocol_frozen_C5_collector")
    observer = SyntheticObserver(enabled); calls = {}
    class WorkerModule:
        FROZEN = {"scalar": ("synthetic-bytes", "synthetic-sha")}
        @staticmethod
        def load_pinned(*args): calls["scalar_load"] = args; return SyntheticScalar
        @staticmethod
        def connect_worker_observation(worker, **kwargs):
            calls["worker"] = worker; calls["connect"] = kwargs; return observer
    class Common:
        WORKER = "synthetic-worker.py"; SCALAR = "synthetic-scalar.py"
        FRAME = "synthetic-frame.py"; CONTEXT = "synthetic-context.py"; RUNNER = "synthetic-runner.py"
        @staticmethod
        def safe(base, name): return Path(base) / name
        @staticmethod
        def load_ref(base, ref): calls["worker_ref"] = ref; return WorkerModule
    worker = NS(model_runner=SyntheticRunner())
    refs = {name: dict(path=name, bytes=1, sha256="a" * 64) for name in
            (Common.WORKER, Common.CONTEXT, Common.RUNNER)}
    refs["third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"] = dict(
        path="new-canonical-C5/source/py_kvcache/reactor.py", bytes=1, sha256="b" * 64)
    return collector, worker, Common, refs, observer, calls


class CommonCollectorBoundaryTests(unittest.TestCase):
    def test_actual_collector_install_keeps_common_RID_and_uninstalled_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            C, worker, common, refs, observer, calls = api_fixture(directory)
            capture = C.install(worker, common=common, root=directory, refs=refs, run_id="LABEL-p0-A",
                event_class=SyntheticEvent, selected_offsets=(16,), action=None, origin="cpu_fixture")
            try:
                self.assertEqual(capture.run_id, "LABEL-p0-A")
                self.assertEqual(calls["connect"]["run_id"], "LABEL-p0-A")
                self.assertEqual(calls["connect"]["native_source_sha256"], "b" * 64)
                self.assertEqual(calls["connect"]["max_pending"], 128)
                self.assertEqual(calls["connect"]["max_steps"], 128)
                self.assertEqual(calls["connect"]["binding"].origin, "cpu_fixture")
                self.assertFalse(capture.event_source["native_class_qualified"])
                self.assertIsNone(capture._wait_reactor_ref)
                self.assertIsNone(capture._wait_registration)
            finally: capture.detach()
            self.assertEqual(observer.detaches, 1)

    def test_actual_prepare_wrapper_delegates_once_and_detach_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            C, worker, common, refs, observer, calls = api_fixture(directory)
            capture = C.install(worker, common=common, root=directory, refs=refs, run_id="RID",
                event_class=SyntheticEvent, origin="cpu_fixture")
            observations = []; capture.after_prepare = lambda: observations.append("synthetic-boundary")
            self.assertEqual(worker.model_runner._prepare_inputs(7), ("original-prepare", 7))
            self.assertEqual(worker.model_runner.calls, 1); self.assertEqual(len(observations), 1)
            capture.detach()
            self.assertEqual(worker.model_runner._prepare_inputs(8), ("original-prepare", 8))
            self.assertEqual(worker.model_runner.calls, 2); self.assertEqual(len(observations), 1)
            self.assertIs(worker.model_runner._prepare_inputs.__func__, SyntheticRunner._prepare_inputs)
            self.assertIsNone(capture.original_prepare)

    def test_install_failure_detaches_observer_without_replacing_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            C, worker, common, refs, observer, calls = api_fixture(directory, enabled=False)
            with self.assertRaisesRegex(ValueError, "observer rejected"):
                C.install(worker, common=common, root=directory, refs=refs, run_id="RID",
                    event_class=SyntheticEvent, origin="cpu_fixture")
            self.assertEqual(observer.detaches, 1)
            self.assertIs(worker.model_runner._prepare_inputs.__func__, SyntheticRunner._prepare_inputs)

    def test_cpu_fixture_cannot_impersonate_torch_native_event_class(self):
        C = load(CANDIDATE / "native_full_step_collector.py", "_protocol_C5_event_source")
        with self.assertRaisesRegex(ValueError, "original installed torch"):
            C.event_class_source(SyntheticEvent, "native_gpu_recording")

    def test_common_event_proxy_preserves_original_calls_without_notify(self):
        C = load(CANDIDATE / "native_full_step_collector.py", "_protocol_C5_event_calls")
        clock_value = [1000]
        def clock(): clock_value[0] += 10; return clock_value[0]
        start, end = C.EventProxy(SyntheticEvent(), clock), C.EventProxy(SyntheticEvent(), clock)
        self.assertEqual(start.record(), "original-record-result"); self.assertTrue(start.query())
        self.assertEqual(end.record(), "original-record-result"); self.assertTrue(end.query())
        self.assertEqual(start.elapsed_time(end), 0.001)
        self.assertEqual((start.raw.records, start.raw.queries, start.raw.elapsed), (1, 1, 1))
        self.assertEqual((end.raw.records, end.raw.queries, end.raw.elapsed), (1, 1, 0))
        self.assertIsNone(start._wait_capture_ref); self.assertIsNone(end._wait_capture_ref)

    def test_detach_does_not_overwrite_later_prepare_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            C, worker, common, refs, observer, calls = api_fixture(directory)
            capture = C.install(worker, common=common, root=directory, refs=refs, run_id="RID",
                event_class=SyntheticEvent, origin="cpu_fixture")
            later = lambda x:("later", x); worker.model_runner._prepare_inputs = later
            capture.detach(); self.assertIs(worker.model_runner._prepare_inputs, later)

    def source_view(self):
        source = PREPARATION / "run_native_cost_experiment.py"
        values = constants(source); original = functions(source)
        namespace = dict(values)
        nodes = [deepcopy(original[name]) for name in ("require", "collector_source_view")]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"), namespace)
        return values, namespace["collector_source_view"]

    def test_actual_six_window_source_view_preserves_canonical_ref_without_mutation(self):
        values, view = self.source_view()
        row = dict(path=values["REACTOR"], bytes=185744, sha256="b" * 64)
        legacy = values["LEGACY_COLLECTOR_REACTOR_KEY"]
        old = dict(path=legacy, bytes=1, sha256="a" * 64)
        rows = [row, old]; refs = {value["path"]: value for value in rows}; before = deepcopy(refs)
        result = view(refs)
        self.assertEqual(refs, before); self.assertIsNot(result, refs)
        self.assertIs(result[legacy], row); self.assertIs(result[values["REACTOR"]], row)
        self.assertEqual(result[legacy]["path"], values["REACTOR"])
        self.assertEqual(rows, [row, old])

    def test_source_view_rejects_legacy_only_or_mislabelled_actual_row(self):
        values, view = self.source_view(); legacy = values["LEGACY_COLLECTOR_REACTOR_KEY"]
        with self.assertRaises(KeyError): view({legacy: dict(path=legacy)})
        with self.assertRaisesRegex(ValueError, "actual collector native"):
            view({values["REACTOR"]: dict(path=legacy)})

    def test_six_window_collector_and_journal_keep_different_namespaces(self):
        source = PREPARATION / "run_native_cost_experiment.py"
        calls = [n for n in ast.walk(tree(source)) if isinstance(n, ast.Call)]
        install = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "install" and
                   isinstance(n.func.value, ast.Name) and n.func.value.id == "collector"]
        self.assertEqual(len(install), 1)
        self.assertEqual(next(k.value.id for k in install[0].keywords if k.arg == "run_id"), "rid")
        source_view = next(k.value for k in install[0].keywords if k.arg == "refs")
        self.assertEqual(source_view.func.id, "collector_source_view")
        journals = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "NativeWindowJournal"]
        self.assertEqual(len(journals), 1); self.assertEqual(journals[0].args[0].id, "LABEL")
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in
            ("attach_single_file_wait", "prepare_notification_capture") for n in ast.walk(tree(source))))

    def test_common_original_bridge_none_parent8_and_finite_constants(self):
        source = PREPARATION / "run_native_cost_experiment.py"; values = constants(source)
        self.assertIsNone(values["GPU_UUID"])
        self.assertEqual(values["MAX_ACCEPTED_PARENTS"], 8)
        self.assertEqual(values["ORDER"], (("A", "B"), ("B", "A"), ("A", "B")))
        self.assertEqual((values["PROMPT_TOKENS"], values["FILE_BYTES"], values["OPERATION_COUNT"]), (129, 917504, 1))
        self.assertEqual(values["PROMPT_FIRST"], (18100, 19100, 20100)); self.assertEqual(values["SEEDS"], (1829, 1830, 1831))
        bridge_args = [k.value for n in ast.walk(tree(source)) if isinstance(n, ast.Call) for k in n.keywords if k.arg == "p4_bridge"]
        self.assertEqual(len(bridge_args), 1); self.assertIsNone(bridge_args[0].value)

    def test_original_128_frame_and_numerical_algorithms_unchanged(self):
        old = functions(NATIVE / "native_conditional_cost.py")
        new = functions(PREPARATION / "native_conditional_cost.py")
        for name in ("original_estimator", "validate_capture", "validate_io", "original_post_shutdown_drain", "analyze_paired"):
            with self.subTest(name=name): self.assertTrue(ast_same(old[name], new[name]))

    def test_real_adapter_128_source_continuity_checks_retained(self):
        old = functions(NATIVE / "run_native_cost_experiment.py")["execute_window"]
        new = functions(PREPARATION / "run_native_cost_experiment.py")["execute_window"]
        def checks(fn):
            return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "require"
                and any(isinstance(value, ast.Constant) and type(value.value) is str and
                    ("same correctly bound scalar adapter" in value.value or "before all frames" in value.value)
                    for value in ast.walk(n))]
        left, right = checks(old), checks(new)
        self.assertEqual(len(left), 2); self.assertEqual(len(right), 2)
        for a, b in zip(left, right): self.assertTrue(ast_same(a, b))

    def test_common_candidate_uses_exact_frozen_C5_collector_and_reactor_bytes(self):
        paths = (
            ("native_full_step_collector.py", "bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf"),
            ("source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py",
             "a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47"),
        )
        for relative, expected in paths:
            with self.subTest(source=relative):
                inherited = (PREPARATION / "common_candidate" / relative).read_bytes()
                frozen = (CANDIDATE / relative).read_bytes()
                self.assertEqual(inherited, frozen)
                self.assertEqual(sha256(inherited).hexdigest(), expected)

    def source_binding_fixture(self):
        """Execute the real source-binding function with a synthetic ref reader.

        The reader returns references computed from the two actual C5 files;
        no installed worker, CUDA Event, process receipt, or issuer is used.
        """
        values = constants(PREPARATION / "run_native_cost_experiment.py")
        overlay = values["OVERLAY"]
        source_rows = {}
        for name, local in (
            (values["REACTOR"], PREPARATION / "common_candidate/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"),
            (values["CANDIDATE"] + "/native_full_step_collector.py", PREPARATION / "common_candidate/native_full_step_collector.py"),
        ):
            raw = local.read_bytes()
            source_rows[name] = dict(path=name, bytes=len(raw), sha256=sha256(raw).hexdigest())
        native = source_rows[values["REACTOR"]]
        collector = source_rows[values["CANDIDATE"] + "/native_full_step_collector.py"]
        plan = dict(cpu_preparation_only=True, evidence_origin="synthetic_cpu_contract",
            job_id=values["LABEL"], common_overlay_relative=overlay, native_source_ref=native,
            collector_source_ref=collector, native_source_sha256=native["sha256"],
            common_owner_parameters=dict(max_accepted_parents=8, bridge_is_none=True))
        child = dict(native_source_binding=dict(actual_source_ref=native, original_run_code_verified=True,
            max_accepted_parents=8, bridge_is_none=True,
            loaded_native_modules=[dict(module="py_kvcache.reactor", source_ref=native)]),
            collector_native_source_binding=dict(legacy_lookup_key=values["LEGACY_COLLECTOR_REACTOR_KEY"],
                actual_source_ref=native, adapter_native_source_sha256_before=native["sha256"],
                adapter_native_source_sha256_after=native["sha256"], same_adapter_all_frames=True,
                full_frame_count=128))
        def require(condition, message):
            if not condition: raise ValueError(message)
        namespace = dict(require=require, ref=lambda root, path: source_rows[path])
        node = deepcopy(functions(PREPARATION / "prepare_and_verify_native_cost.py")["validate_common_source_binding"])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            "<actual-C5-source-binding-synthetic-reader>", "exec"), namespace)
        return namespace["validate_common_source_binding"], plan, child

    def test_actual_common_source_binding_accepts_only_new_C5_source_group(self):
        validate, plan, child = self.source_binding_fixture()
        before = deepcopy((plan, child))
        self.assertIsNone(validate(None, plan, child))
        self.assertEqual((plan, child), before)

    def test_actual_common_source_binding_rejects_source_or_owner_substitution(self):
        validate, plan, child = self.source_binding_fixture()
        substitutions = (
            ("old-job", lambda p, c: p.update(job_id="server11-native-cost-v6")),
            ("missing-cpu-origin", lambda p, c: p.update(evidence_origin="native_gpu_recording")),
            ("old-reactor-hash", lambda p, c: p["native_source_ref"].update(sha256="a" * 64)),
            ("old-collector-path", lambda p, c: p["collector_source_ref"].update(path="old/native_full_step_collector.py")),
            ("wrong-owner-limit", lambda p, c: p["common_owner_parameters"].update(max_accepted_parents=16)),
            ("installed-on-bridge", lambda p, c: c["native_source_binding"].update(bridge_is_none=False)),
            ("unverified-original-method", lambda p, c: c["native_source_binding"].update(original_run_code_verified=False)),
            ("reactor-module-absent", lambda p, c: c["native_source_binding"]["loaded_native_modules"][0].update(module="prefix_io_control.other")),
        )
        for name, mutate in substitutions:
            with self.subTest(substitution=name):
                changed_plan, changed_child = deepcopy((plan, child)); mutate(changed_plan, changed_child)
                with self.assertRaises(ValueError): validate(None, changed_plan, changed_child)

    def test_actual_common_source_binding_rejects_partial_or_rebound_128_frames(self):
        validate, plan, child = self.source_binding_fixture()
        for field, value in (("full_frame_count", 127), ("full_frame_count", True),
            ("same_adapter_all_frames", False), ("adapter_native_source_sha256_before", "a" * 64),
            ("adapter_native_source_sha256_after", "a" * 64), ("legacy_lookup_key", "changed-key")):
            with self.subTest(field=field, value=value):
                changed = deepcopy(child); changed["collector_native_source_binding"][field] = value
                with self.assertRaises(ValueError): validate(None, plan, changed)

    def test_serializer_preserves_original_external_native_and_journal_RIDs(self):
        def selected(path, function):
            return {n.func.attr: n for n in ast.walk(functions(path)[function])
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
                isinstance(n.func.value, ast.Name) and n.func.value.id == "C" and
                n.func.attr in ("validate_capture", "original_post_shutdown_drain", "validate_io")}
        old = selected(NATIVE / "prepare_and_verify_native_cost.py", "verify_raw_window")
        new = selected(PREPARATION / "prepare_and_verify_native_cost.py", "verify_raw_window")
        self.assertEqual(set(new), {"validate_capture", "original_post_shutdown_drain", "validate_io"})
        self.assertEqual(set(new), set(old))
        for name in new:
            with self.subTest(call=name): self.assertTrue(ast_same(old[name], new[name]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
