"""CPU interface/source fixtures only; no privately qualified table is minted."""
from __future__ import annotations
import ast
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import weakref
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


S = load("_cpu_finite_preconstruction", "finite_startup.py")
A = load("_cpu_closed_activation_request", "activation_request.py")
W = load("_cpu_strong_raw_window_wrapper", "strong_native_cost_runner.py")
H = load("_cpu_host_small_boundary_binding", "host_boundary_binding.py")
F = load("_cpu_finite_measured_subset_binding", "finite_current_binding.py")


def original_file(name):
    local = HERE.parents[1] / "i_pilot_cpu_preparation_20261004/i_bridge/frozen" / name
    if local.exists():
        return local
    return HERE.parents[1] / "server12-i-pilot-cpu-preparation-20261004/i_bridge/frozen" / name


class ThinActivationCPU(unittest.TestCase):
    def test_effect_reserve_subset_cannot_cover_other_issued_cells(self):
        cells = (("gpu", "domain", "model", "layout", 1, 1, 0, 527, 917504),
                 ("gpu", "domain", "model", "layout", 1, 1, 0, 528, 917504))
        allowed = F.covered_subset(cells, (cells[0],), observation_only=False)
        self.assertEqual(allowed, frozenset((cells[0],)))
        self.assertNotIn(cells[1], allowed)
        with self.assertRaisesRegex(ValueError, "real native-reserve covered subset"):
            F.covered_subset(cells, None, observation_only=False)
        with self.assertRaisesRegex(ValueError, "real native-reserve covered subset"):
            F.covered_subset(cells, (cells[0], cells[0]), observation_only=False)
        self.assertEqual(F.covered_subset(cells, None, observation_only=True), frozenset(cells))

    def test_real_preview_wrapper_records_matching_attempt_ids_with_CPU_fixture_only(self):
        path = HERE.parent / "activation/control_observation/host_control_observer.py"
        spec = importlib.util.spec_from_file_location("_cpu_host_preview_matching", path)
        host = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = host
        spec.loader.exec_module(host)
        observer = host.HostControlObserver("cpu-fixture-preview", _fixture_clock=host.FixtureClock([10, 20]))
        observer._boundaries["qualified_finite_controller_preview"] = dict(category="controller")
        binding = F.CurrentFiniteBinding.__new__(F.CurrentFiniteBinding)
        runner = NS(_profile_step=1)
        binding.identity = NS(_runner_ref=lambda: runner)
        binding.host_observer, binding.preview_windows, binding.preview_log_lost = observer, [], 0
        binding._preview_impl = lambda *args, **kwargs: (kwargs["attempt"].update(reason="CPU_fixture_original_result") or "original_result")
        self.assertEqual(binding._preview(NS(), NS(), NS(), now_ns=11), "original_result")
        report = observer.export()
        windows = binding.preview_windows
        self.assertEqual(windows[0]["attempt_id"], 1)
        self.assertNotIn("attempt", windows[0])
        previous, present = sys.modules.get("host_control_observer"), "host_control_observer" in sys.modules
        try:
            sys.modules["host_control_observer"] = host
            join_path = path.with_name("reserve_join.py")
            join_spec = importlib.util.spec_from_file_location("_cpu_preview_shape_verifier", join_path)
            join = importlib.util.module_from_spec(join_spec)
            join_spec.loader.exec_module(join)
        finally:
            if present:
                sys.modules["host_control_observer"] = previous
            else:
                sys.modules.pop("host_control_observer", None)
        result = join.match_preview_interval_shape(report, windows, clock_scope=report["clock_scope"])
        self.assertEqual(result["associated_attempts"], 1)
        self.assertFalse(result["actual_native_gpu_run"])
        self.assertFalse(report["production_qualified"])

    def test_optional_host_ordinal_failure_calls_original_once_and_preserves_result_exception(self):
        R = W.driver_module()
        class Holder:
            pass
        holder = Holder()
        value = H.HostBoundaryBinding.__new__(H.HostBoundaryBinding)
        value.observer = NS(_lock=__import__("threading").Lock(), _fail=lambda reason: None)
        value._runner, value.bindings, value.failure = weakref.ref(holder), [], None
        calls = []
        class Target:
            def small(self, arg):
                calls.append(arg)
                if arg == "raise":
                    raise RuntimeError("original failure unchanged")
                return "actual return unchanged"
        target = Target()
        value.observer.bind = lambda *args, **kwargs: None
        path = Path(__file__).resolve().relative_to(ROOT).as_posix()
        refs = {path: R.ref(ROOT, path)}
        value.attach(target, "small", "scheduler", "cpu_small_boundary", False, ROOT, refs, R)
        self.assertEqual(target.small("ok"), "actual return unchanged")
        with self.assertRaisesRegex(RuntimeError, "original failure unchanged"):
            target.small("raise")
        self.assertEqual(calls, ["ok", "raise"])
        value.ordinal = lambda after: (_ for _ in ()).throw(ValueError("actual getter failed"))
        self.assertEqual(target.small("getter-failure"), "actual return unchanged")
        self.assertEqual(calls, ["ok", "raise", "getter-failure"])
        self.assertTrue(value.detach())

    def test_actual_raw_shape_preserves_external_run_and_normalizes_internal_frame_request(self):
        plan = dict(gpu_uuid="GPU-cpu", common_runtime_domain_sha256="a" * 64,
            source_lock_ref=dict(path="source-lock.json", bytes=1, sha256="b" * 64),
            model_manifest_ref=dict(path="model.json", bytes=1, sha256="c" * 64),
            kv_layout_ref=dict(path="geometry.json", bytes=1, sha256="d" * 64), cells=[W.cell_descriptor()])
        rows = []
        for index in range(6):
            external = "actual-fixture-window" + str(index)
            internal = external + "-actual-returned-native-suffix"
            rows.append(dict(request_id=external, condition="A" if index in (0, 3, 4) else "B",
                capture=dict(run_id=external), frontend=dict(output=dict(request_id=external,
                    native_request_id=internal, output_token_ids=list(range(128))))))
        report = dict(status="PASS_NATIVE_SIX_PROCESS_RAW_CAPTURE_REQUIRES_VALIDATION", windows=rows)
        converted = W.normalize_actual(ROOT, dict(plan=plan, config=dict(plan_ref=dict(path="plan.json", bytes=1, sha256="e" * 64))), report)
        for original, normalized in zip(rows, converted["cells"][0]["windows"]):
            self.assertEqual(normalized["run_id"], original["request_id"])
            self.assertEqual(normalized["request_id"], original["frontend"]["output"]["native_request_id"])
            self.assertEqual(normalized["capture"], original["capture"])
        self.assertFalse(converted["production_qualified"])
        self.assertEqual(converted["source_verification_refs"], {})

    def test_development_shadow_has_closed_inputs_without_reserve_cycle_or_dummy_cells(self):
        R = W.driver_module()
        value = dict(phase="development", mode="shadow", arm="I", activation_ref=dict(path="future-actual.json"))
        self.assertIsNone(R.check_phase(value, gap=None))
        value.pop("activation_ref")
        with self.assertRaisesRegex(ValueError, "closed real finite"):
            R.check_phase(value, gap=None)
        source = (HERE / "activation_request.py").read_text()
        self.assertIn("verify_existing_native_reserve", source)
        self.assertIn('"native_qualification_verifier_required_separately") is False', source)

    def test_actual_host_observer_capacity36_intervals_under64_is_still_CPU_fixture(self):
        path = HERE.parent / "activation/control_observation/host_control_observer.py"
        spec = importlib.util.spec_from_file_location("_cpu_host_actual_capacity", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        clock = module.FixtureClock(list(range(1, 100)))
        observer = module.HostControlObserver("cpu-fixture-capacity", max_intervals_per_step=64, _fixture_clock=clock)
        observer._boundaries["fixture"] = dict(category="controller")
        for _ in range(36):
            with observer.observe("controller", 7, "fixture"):
                pass
        report = observer.export()
        self.assertEqual(len(report["per_step"][0]["intervals"]), 36)
        self.assertEqual(report["failure_count"], 0)
        self.assertFalse(report["production_qualified"])
    def test_private_mock_never_reaches_runtime_patch_or_owner(self):
        with self.assertRaisesRegex(ValueError, "real private GPU qualification"):
            S.FiniteStartup(native_module=NS(), reactor_module=NS(), table=NS(production_qualified=False),
                issuer=NS(qualified_identity=lambda value: None), run_id="cpu", budget_ns=1000,
                root=ROOT, runtime_refs={}, driver=NS(), binding_module=NS(), validate_drained=lambda value: value)

    def test_only_original_ordinary_preload_if_changes_not_lifecycle(self):
        source = original_file("reactor.py").read_bytes()
        tree, proof = S.ordinary_method_source(source)
        self.assertEqual(proof["modified_if_conditions"], 1)
        self.assertFalse(proof["production_batch_activation"])
        original = ast.parse(source)
        old_cls = next(node for node in original.body if isinstance(node, ast.ClassDef) and node.name == "IoReactor")
        old = next(node for node in old_cls.body if isinstance(node, ast.FunctionDef) and node.name == "_prefix_stage_decide")
        old_tests = [node for node in ast.walk(old) if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp)
                     and ast.unparse(node.test).startswith("p4.policy.single_file is not None and type(ready) is _ReadyFd")]
        new = tree.body[0]
        new_tests = [node for node in ast.walk(new) if isinstance(node, ast.If) and "_prefix_private_finite_enabled" in ast.unparse(node.test)]
        self.assertEqual(len(new_tests), 1)
        new_tests[0].test = old_tests[0].test
        self.assertEqual(ast.dump(old, include_attributes=False), ast.dump(new, include_attributes=False))

    def test_real_ready_fields_keep_immutable_age_sequence_and_hash(self):
        # Exercise the actual original method AST with a clearly marked CPU
        # value fixture. This fixture cannot enter FiniteStartup's private gate.
        package_path = HERE.parent / "activation/source"
        sys.path.insert(0, str(package_path))
        try:
            from prefix_io_control.dispatch_budget import Amount, STAGES
            from prefix_io_control.p4_types import IssuePreview
            class Ready:
                pass
            class Info:
                pass
            actuals = []
            ready, info = Ready(), Info()
            block = b"a" * 32
            ready.job, ready.file_index, ready.sequence, ready.open_start_ns = None, -1, 42, 800
            ready.preload_hash, ready.preload_info = block, info
            info.block_hash, info.total_files, info.file_index = block, 32, 7
            info.preload_id, info.req_id = "actual-native-preload-fixture", "cpu-fixture-request"
            bridge = NS(run_id="cpu-fixture", policy=NS(single_file=None, config=NS(sample_max_age_ns=1000, max_wait_ns=10000)),
                        _epoch=1, single_file_observation_reuses=0, fail=lambda reason: self.fail(reason))
            bridge.preview_issue = lambda work, snapshot, now_ns: (actuals.append(work) or IssuePreview("native_fallback", "cpu_fixture"))
            snapshot = NS(snapshot_epoch=1, native_state=None)
            accounting = NS(valid=True, _owner=lambda: None,
                stats={stage: dict(inflight_ops=0, inflight_bytes=0) for stage in STAGES})
            owner = NS(_prefix_dispatch_controller=None, _prefix_p4_bridge=bridge,
                _prefix_stage_accounting=accounting, _prefix_clean_reclaimable_bytes=lambda: 0,
                accepted_parent_count=lambda: 0, _prefix_p4_collect=lambda: NS(snapshot=snapshot),
                _stop=False, file_store=NS(io_size=917504), layout=NS(bytes_per_kernel_block=(917504,)))
            # dataclasses.replace needs the original immutable snapshot type;
            # use a tiny dataclass only for the pure value fixture publication.
            from dataclasses import make_dataclass
            FixtureSnapshot = make_dataclass("FixtureSnapshot", ["snapshot_epoch", "native_state"], frozen=True)
            owner._prefix_p4_collect = lambda: NS(snapshot=FixtureSnapshot(1, None))
            namespace = dict(time=NS(monotonic_ns=lambda: 1000), _ReadyFd=Ready, _PreloadInfo=Info,
                _prefix_private_finite_enabled=lambda reactor: True, _StageDispatch=lambda *args: args,
                _P4Deferred=lambda: "cpu_fixture", __name__="cpu_original_ast")
            tree, _ = S.ordinary_method_source(original_file("reactor.py").read_bytes())
            exec(compile(tree, "cpu_original_reactor_value_fixture", "exec", dont_inherit=True), namespace)
            method = namespace["_prefix_stage_decide"]
            self.assertIsNone(method(owner, "ssd_read", 917504, ("ordinary", 7), ready=ready, reserved_bytes=917504))
            self.assertEqual(len(actuals), 1)
            self.assertEqual(actuals[0].created_ns, 800)
            self.assertEqual(actuals[0].parent_id, 0)
            self.assertIn("42", str(actuals[0].work_id))
            info.block_hash = b"b" * 32
            self.assertIsNone(method(owner, "ssd_read", 917504, ("ordinary", 7), ready=ready, reserved_bytes=917504))
            self.assertEqual(len(actuals), 1)
        finally:
            sys.path.pop(0)

    def test_original_six_window_source_patches_are_explicit_and_strong(self):
        old = HERE.parents[1] / "i_pilot_cpu_preparation_20261004/calibration_v2/run_native_cost_experiment.py"
        if not old.exists():
            old = HERE.parents[1] / "server12-i-pilot-cpu-preparation-20261004/calibration_v2/run_native_cost_experiment.py"
        raw = old.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), W.OLD_SHA)
        tree, proof = W.source_patch(raw)
        source = ast.unparse(tree)
        self.assertIn("load_planner='on'", source)
        self.assertIn("CACHED_TOKENS", source)
        self.assertIn("'--project', str(root)", source)
        self.assertIn("max_accepted_parents=MAX_ACCEPTED_PARENTS", source)
        self.assertEqual(proof["original_cache_hit_rule"], "16*floor((prompt_tokens-1)/16)")
        self.assertFalse(proof["estimator_modified"])
        self.assertFalse(proof["validation_used_to_fit"])

    def test_new_cell_preregisters512_context527_and_disjoint_heldout(self):
        descriptor = W.cell_descriptor()
        self.assertEqual((descriptor["prompt_tokens"], descriptor["cached_prompt_tokens"], descriptor["measured_offset"]), (512, 496, 16))
        self.assertEqual([row["split"] for row in descriptor["entries"]], ["calibration", "calibration", "validation"])
        self.assertEqual([row["arm_order"] for row in descriptor["entries"]], ["AB", "BA", "AB"])
        self.assertEqual(len({row["prefix_family_sha256"] for row in descriptor["entries"]}), 3)
        self.assertEqual(W.MAX_SECONDS, 900)
        self.assertEqual(W.RESERVE, 512 * 1024**2)
        self.assertEqual(descriptor["existing_io"], [dict(ops=0, bytes=0)] * 4)

    def test_cpu_import_and_help_do_not_import_framework_or_backend(self):
        code = """import importlib.abc,importlib.util,sys,pathlib
class Trap(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','vllm','py_kvcache','ctypes','numpy'): raise RuntimeError('forbidden '+name)
sys.meta_path.insert(0,Trap())
root=pathlib.Path(sys.argv[1])
for index,name in enumerate(('strong_native_cost_runner.py','finite_startup.py','activation_request.py')):
 spec=importlib.util.spec_from_file_location('_cpu_pure_'+str(index),root/name)
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
print('PASS_CPU_STANDARD_LIBRARY_ONLY')
"""
        result = subprocess.run([sys.executable, "-B", "-I", "-S", "-c", code, str(HERE)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS_CPU_STANDARD_LIBRARY_ONLY", result.stdout)
        help_result = subprocess.run([sys.executable, "-B", "-I", "-S", str(HERE / "strong_native_cost_runner.py"), "--help"],
                                     capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--prepare-plan", help_result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
