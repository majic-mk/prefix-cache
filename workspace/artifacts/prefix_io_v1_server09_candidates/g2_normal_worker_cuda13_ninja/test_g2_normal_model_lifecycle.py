"""Source-bound stdlib gates and original frontend lifecycle CPU contracts."""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, default=Path(__file__).parent.parent / "g2_source_readonly")
args, rest = parser.parse_known_args(); sys.argv = [sys.argv[0], *rest]
path = Path(__file__).with_name("run_g2_normal_model_lifecycle.py")
spec = importlib.util.spec_from_file_location("g2_lifecycle_under_test", path)
M = importlib.util.module_from_spec(spec); sys.modules[spec.name] = M
spec.loader.exec_module(M)


class GateContracts(unittest.TestCase):
    def test_supported_cache_paths_are_process_local_before_any_backend(self):
        base = NS(RUNTIME_CACHE_SUBDIRS={"VLLM_CACHE_ROOT": "vllm", "TRITON_CACHE_DIR": "triton",
             "TORCHINDUCTOR_CACHE_DIR": "inductor", "XDG_CACHE_HOME": "xdg", "HF_HOME": "huggingface"})
        with tempfile.TemporaryDirectory() as folder, patch.dict(M.os.environ):
            out = Path(folder)
            paths = M.configure_process_caches(out, base)
            self.assertEqual(set(paths), set(base.RUNTIME_CACHE_SUBDIRS) | set(M.EXTRA_CACHE_DIRS) |
                             {"TMPDIR", "VLLM_RPC_BASE_PATH"})
            for key, value in paths.items():
                self.assertTrue(Path(value).is_relative_to(out))
                self.assertTrue(Path(value).is_dir())
                self.assertEqual(M.os.environ[key], value)
            self.assertEqual(paths["FLASHINFER_WORKSPACE_BASE"], str(out / "runtime-cache/flashinfer-workspace"))
            self.assertNotIn("FLASHINFER_WORKSPACE_DIR", paths)

    def test_actual_flashinfer_and_torch_cache_consumers_are_source_bound_without_imports(self):
        trees = {}
        for relative, size, sha in M.RUNTIME_CACHE_SOURCE_REFS:
            raw = (args.source_root / relative).read_bytes()
            self.assertEqual(len(raw), size); self.assertEqual(hashlib.sha256(raw).hexdigest(), sha)
            trees[relative] = ast.parse(raw)
        flash = trees[M.RUNTIME_CACHE_SOURCE_REFS[0][0]]
        selected = [n for n in flash.body if isinstance(n, ast.AnnAssign) and
                    isinstance(n.target, ast.Name) and n.target.id in ("FLASHINFER_BASE_DIR", "FLASHINFER_CACHE_DIR")]
        namespace = {"os": M.os, "pathlib": __import__("pathlib")}
        with tempfile.TemporaryDirectory() as folder, patch.dict(M.os.environ, FLASHINFER_WORKSPACE_BASE=folder):
            module = ast.Module(body=selected, type_ignores=[]); ast.fix_missing_locations(module)
            exec(compile(module, "cpu_exact_flashinfer_cache_expressions", "exec"), namespace)
            self.assertEqual(namespace["FLASHINFER_CACHE_DIR"], Path(folder) / ".cache/flashinfer")
        torch_source = trees[M.RUNTIME_CACHE_SOURCE_REFS[2][0]]
        function = next(n for n in torch_source.body if isinstance(n, ast.FunctionDef) and n.name == "_get_build_directory")
        self.assertIn("os.environ.get('TORCH_EXTENSIONS_DIR')", ast.unparse(function))

    def test_gpu_import_attempt_flags_precede_first_framework_import(self):
        tree = ast.parse(Path(M.__file__).read_bytes())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute_guarded")
        imports = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Import) and
                   any(alias.name in ("torch", "vllm") for alias in n.names)]
        attempts = {}
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and n.value.value is True:
                for target in n.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                        if target.slice.value in ("framework_import_attempted", "gpu_initialization_attempted"):
                            attempts[target.slice.value] = n.lineno
        self.assertEqual(set(attempts), {"framework_import_attempted", "gpu_initialization_attempted"})
        self.assertTrue(all(line < min(imports) for line in attempts.values()))

    def test_completed_frontend_phase_persists_before_export_or_reconciliation_failure(self):
        tree = ast.parse(Path(M.__file__).read_bytes())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute_guarded")
        append = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call) and
                  isinstance(n.func, ast.Attribute) and n.func.attr == "append" and
                  isinstance(n.func.value, ast.Subscript) and isinstance(n.func.value.slice, ast.Constant) and
                  n.func.value.slice.value == "phases"]
        reconcile = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call) and
                     isinstance(n.func, ast.Name) and n.func.id == "verify_frame_outputs"]
        export = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call) and
                  isinstance(n.func, ast.Attribute) and n.func.attr == "collective_rpc" and
                  n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == "export_worker_observation"]
        self.assertEqual(len(append), 1); self.assertEqual(len(reconcile), 1)
        self.assertEqual(len(export), 1)
        self.assertLess(append[0], export[0]); self.assertLess(append[0], reconcile[0])
        # Replay this exact driver slice with an ordinary export failure. No
        # framework/model/GPU is imported; the already-completed original IDs
        # must be durably append-new before that failure can replace evidence.
        loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and
                    isinstance(n.target, ast.Name) and n.target.id == "phase")
        begin = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == "frontend" for t in n.targets))
        end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == "exported" for t in n.targets))
        body = copy.deepcopy(loop.body[begin:end + 1])
        replay = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        frontend = dict(output_token_ids=list(range(128)), native_request_id="native-id")
        export_error = RuntimeError("export-query")
        calls = []
        def original(*a): calls.append(a); return frontend
        def failed_export(*a, **kw): raise export_error
        with tempfile.TemporaryDirectory() as folder:
            result = dict(phases=[])
            namespace = dict(result=result, phase="cold", rid="front", out=Path(folder),
                install=[{}], run_original_request=original, new_json=M.new_json,
                SamplingParams=lambda **kw: object(), SAMPLING=M.SAMPLING,
                llm=NS(llm_engine=object(), collective_rpc=failed_export),
                export_worker_observation=M.export_worker_observation, args=NS(mode="shadow"))
            with self.assertRaises(RuntimeError) as caught:
                exec(compile(replay, str(M.__file__), "exec"), namespace)
            self.assertIs(caught.exception, export_error)
            self.assertEqual(len(calls), 1)
            self.assertIs(result["phases"][0]["frontend"], frontend)
            self.assertEqual(json.loads((Path(folder) / "cold-frontend.json").read_text()), frontend)

    def test_template_has_no_gpu_authority(self):
        template = M.scope_template()
        self.assertEqual(template["status"], "NOT_AUTHORIZED_CPU_TEMPLATE")
        self.assertIs(template["allow_gpu_runs"], False)
        self.assertEqual(template["maximum_total_planned_reserve_seconds"], 640)
        self.assertEqual(template["qualification_context"]["native_io"], "none")
        self.assertEqual(template["permitted_run_names"], dict(off="server09-g2-normal-off-03", shadow="server09-g2-normal-shadow-03"))
        sdk = template["qualification_context"]["sdk_toolchain"]
        self.assertIs(sdk["enabled_in_both_modes"], True)
        self.assertIs(sdk["GPU_model_or_JIT_runtime_qualified"], False)
        self.assertEqual(template["qualification_context"]["build_tool"]["source_ref"],
                         dict(path=M.NINJA_SOURCE_REF[0], bytes=M.NINJA_SOURCE_REF[1], sha256=M.NINJA_SOURCE_REF[2]))

    def test_missing_scope_execute_and_launch_exit_before_any_runtime_source(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(M, "verify_source_lock", side_effect=AssertionError("source read")), \
                 patch.object(M, "load_ref", side_effect=AssertionError("module import")):
                for action in ("--launch", "--execute"):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(M.main(["--project", folder, action]), 78)

    def test_cpu_template_and_g1_scope_cannot_reuse_old_gpu_permission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); scope = root / "scope.json"
            for value in (M.scope_template(), {"status": "USER_AUTHORIZED_P4_GPU_NATIVE_QUALIFICATION"}):
                scope.write_text(json.dumps(value))
                with patch.object(M, "verify_source_lock", side_effect=AssertionError("source read")), \
                     patch.object(M, "load_ref", side_effect=AssertionError("module import")), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(M.main(["--project", folder, "--scope-record", "scope.json", "--execute"]), 78)

    def test_strict_context_rejects_bool_float_and_extra_fields(self):
        for actual in (True, 1.0):
            with self.assertRaises(ValueError): M.same(actual, 1, "exact type")
        changed = M.context(); changed["normal_outputs_per_request"] = 128.0
        with self.assertRaises(ValueError): M.same(changed, M.context(), "typed context")
        changed = M.context(); changed["native_io"] = "warm_h2d"
        with self.assertRaises(ValueError): M.same(changed, M.context(), "no scope expansion")
        serialized = json.loads(json.dumps(M.context()))
        self.assertIs(type(serialized["sampling"]["temperature"]), float)
        self.assertEqual(serialized["sampling"]["temperature"], 0.0)
        M.same(serialized, M.context(), "exact real JSON context")
        serialized["sampling"]["temperature"] = 0
        with self.assertRaises(ValueError): M.same(serialized, M.context(), "temperature float retained")

    def test_exact_child_command_preserves_guard_budget_bound_and_bytecode_off(self):
        a = NS(project=Path("/cpu-fixture"), mode="shadow", name="server09-g2-normal-shadow-03",
               scope_record="scope.json", source_lock="new-lock.json")
        command = M.child_command(a)
        self.assertEqual(command[:3], [".venv/bin/python", "-B", M.SCRIPT])
        self.assertEqual(command[-1], "--execute")
        self.assertIn(a.source_lock, command); self.assertIn(a.scope_record, command)

    def test_direct_execution_requires_actual_guard_session_exact_command_and_permission(self):
        a = NS(project=Path.cwd(), mode="off", name="server09-g2-normal-off-03",
               scope_record="scope.json", source_lock="lock.json")
        permit = dict(path=M.PERMISSIONS, bytes=671, sha256="a" * 64)
        active = dict(label=a.name, gpu_uuid=M.GPU_UUID, seconds_limit=300, reserved_seconds=320,
                      command=M.child_command(a), session_id=17, permissions=permit)
        gates = dict(ledger=dict(active_reservation=active), scope=dict(base_permissions=permit))
        env = dict(CUDA_VISIBLE_DEVICES=M.GPU_UUID, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        with patch.object(M, "scope_source_gates", return_value=gates), \
             patch.object(M.os, "getsid", return_value=17, create=True), patch.dict(M.os.environ, env):
            self.assertIs(M.execution_gates(Path.cwd(), a), gates)
            for key, bad in (("session_id", 18), ("reserved_seconds", 300), ("command", ["python"]),
                             ("permissions", None), ("gpu_uuid", "GPU-other")):
                wrong = copy.deepcopy(gates); wrong["ledger"]["active_reservation"][key] = bad
                with patch.object(M, "scope_source_gates", return_value=wrong):
                    with self.assertRaises(ValueError): M.execution_gates(Path.cwd(), a)
            with patch.object(M, "scope_source_gates", return_value=dict(ledger=dict(active_reservation=None))):
                with self.assertRaises(ValueError): M.execution_gates(Path.cwd(), a)

    def test_locked_paths_reject_traversal_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for path in ("../escape", "/absolute", "a//b", "a\\b"):
                with self.assertRaises(ValueError): M.safe(root, path)
            (root / "bad.json").write_text('{"x": 1, "x": 2}')
            with self.assertRaises(ValueError): M.read_json(root / "bad.json")

    def test_source_stat_or_sha_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / "small.py").write_bytes(b"x=1\n")
            row = dict(path="small.py", bytes=4, sha256=hashlib.sha256(b"x=1\n").hexdigest())
            self.assertEqual(M.checked_ref(root, row), root / "small.py")
            with self.assertRaises(ValueError): M.checked_ref(root, dict(row, sha256="0" * 64))
            with self.assertRaises(ValueError): M.checked_ref(root, dict(row, bytes=3))

    def test_old_baseline_lock_cannot_be_used_as_new_g2_lock(self):
        with self.assertRaises(ValueError): M.verify_source_lock(Path.cwd(), M.BASELINE_LOCK)

    def test_off_worker_rpc_does_not_read_worker_or_refs(self):
        class Poison:
            def __getattribute__(self, name): raise AssertionError("off read " + name)
        result = M.install_worker_observation(Poison(), {"mode": "off"})
        self.assertEqual(result["status"], "OFF_ORIGINAL_PATH")
        self.assertEqual(M.export_worker_observation(Poison(), "off")["frames"], [])

    def test_source_only_preflight_without_human_remains_zero_gpu_and_no_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(M, "verify_source_lock", return_value={"fake": 1}) as source_check, \
                 patch.object(M, "scope_source_gates", side_effect=AssertionError("scope launch")), \
                 patch.object(M, "execute_guarded", side_effect=AssertionError("GPU")):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(M.main(["--project", folder, "--preflight", "--source-lock", "new.json"]), 0)
                self.assertEqual(source_check.call_count, 1)
                result = json.loads(output.getvalue())
                self.assertEqual(result["status"], "SOURCE_LOCK_VERIFIED_BUT_NO_NEW_G2_HUMAN_SCOPE")
                self.assertIs(result["gpu_initialized"], False)
                self.assertEqual(result["actual_gpu_runs"], 0)
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_shadow_cannot_run_after_failed_off_or_before_off(self):
        a = NS(mode="shadow")
        scope = dict(permitted_run_names=M.scope_template()["permitted_run_names"], source_lock_sha256="a" * 64)
        for events in ([], [dict(label="server09-g2-normal-off-03", child_exit=1, exit=1, session_drained=True)]):
            with self.assertRaises(ValueError): M.shadow_prerequisite(Path.cwd(), a, dict(scope=scope, ledger=dict(events=events)))
        successful = dict(label="server09-g2-normal-off-03", child_exit=0, exit=0, session_drained=True)
        result = dict(status="PASSED_NORMAL_MODEL_FULL_OUTPUT_ONLY", purpose=M.PURPOSE,
                      original_engine_shutdown_returned=True, source_lock_sha256="a" * 64)
        with patch.object(M, "read_json", return_value=result):
            M.shadow_prerequisite(Path.cwd(), a, dict(scope=scope, ledger=dict(events=[successful])))
        with patch.object(M, "read_json", return_value=dict(result, original_engine_shutdown_returned=False)):
            with self.assertRaises(ValueError): M.shadow_prerequisite(Path.cwd(), a, dict(scope=scope, ledger=dict(events=[successful])))

    def test_locked_sdk_preflight_reads_exact_refs_but_never_creates_overlay(self):
        refs = {p: dict(path=p, bytes=n, sha256=s) for p, n, s in M.SDK_SOURCE_REFS}
        calls = []
        pin = object()
        def read_assets(inventory, proof): calls.append((inventory, proof)); return pin
        module = NS(load_audited_assets=read_assets, prepare_overlay=lambda **kw: self.fail("preflight mkdir"))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); before = dict(os.environ)
            with patch.object(M, "checked_ref", side_effect=lambda r, row: r / row["path"]) as checked, \
                 patch.object(M, "load_ref", return_value=module) as loaded:
                actual_module, actual_pin = M.load_sdk_assets(root, refs)
                self.assertIs(actual_module, module); self.assertIs(actual_pin, pin)
                self.assertEqual(checked.call_count, len(M.SDK_SOURCE_REFS)); self.assertEqual(loaded.call_count, 1)
                self.assertEqual(calls[0][0], dict(refs[M.SDK_INVENTORY], path=str(root / M.SDK_INVENTORY)))
                self.assertEqual(calls[0][1], dict(refs[M.SDK_PROOF], path=str(root / M.SDK_PROOF)))
                for wrong in (dict(refs, **{M.SDK_MODULE: dict(refs[M.SDK_MODULE], sha256="0" * 64)}),
                              {p: row for p, row in refs.items() if p != M.SDK_INVENTORY}):
                    with self.assertRaises(ValueError): M.load_sdk_assets(root, wrong)
                error = RuntimeError("source target drift")
                module.load_audited_assets = lambda *a: (_ for _ in ()).throw(error)
                with self.assertRaises(RuntimeError) as caught: M.load_sdk_assets(root, refs)
                self.assertIs(caught.exception, error)
            self.assertEqual(list(root.iterdir()), []); self.assertEqual(dict(os.environ), before)
        self.assertNotIn("torch", sys.modules); self.assertNotIn("vllm", sys.modules)

    def test_sdk_hook_applies_child_copy_and_exports_only_plain_selected_environment(self):
        refs = {p: dict(path=p, bytes=n, sha256=s) for p, n, s in M.SDK_SOURCE_REFS}
        pin = object(); calls = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            for mode in ("off", "shadow"):
                out = root / (mode + "-02") / "details"
                overlay = out / "runtime-cache/cuda13-sdk"
                env = {key: str(overlay / key.lower()) for key in M.SDK_PROCESS_KEYS}
                def create(**kw):
                    calls.append(kw)
                    return NS(overlay=str(overlay), links=(("new", "existing"),), environment=tuple(env.items()))
                module = NS(prepare_overlay=create)
                with patch.object(M, "load_sdk_assets", return_value=(module, pin)), \
                     patch.dict(os.environ, {"SDK_CPU_SECRET_SENTINEL": "never_export_this_value"}):
                    result = M.prepare_process_sdk(root, refs, out)
                    self.assertEqual({k: os.environ[k] for k in M.SDK_PROCESS_KEYS}, env)
                    self.assertEqual(result["selected_environment"], env)
                    self.assertNotIn("never_export_this_value", json.dumps(result))
                    self.assertIs(result["GPU_model_or_JIT_runtime_qualified"], False)
                    self.assertIs(result["production_qualified"], False)
                self.assertEqual(calls[-1]["approved_run_root"], out.parent)
                self.assertEqual(calls[-1]["runtime_cache"], out / "runtime-cache")
                self.assertIs(calls[-1]["pin"], pin)
            self.assertEqual(len(calls), 2)
            self.assertEqual(list(root.iterdir()), [])

    def test_original_driver_sdk_fault_precedes_framework_attempt_and_follows_cache_setup(self):
        tree = ast.parse(Path(M.__file__).read_bytes())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute_guarded")
        block = next(n for n in fn.body if isinstance(n, ast.Try))
        body = copy.deepcopy(block.body[:2])
        self.assertEqual(body[0].value.func.id, "configure_process_caches")
        self.assertEqual(body[1].value.func.id, "prepare_process_sdk")
        imports = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Import) and
                   any(a.name in ("torch", "vllm") for a in n.names)]
        self.assertLess(body[1].lineno, min(imports))
        order = []; error = RuntimeError("SDK gate")
        def cache(*a): order.append("cache"); return {}
        def sdk(*a): order.append("sdk"); raise error
        result = dict(framework_import_attempted=False, gpu_initialization_attempted=False)
        namespace = dict(result=result, root=Path.cwd(), refs={}, out=Path.cwd(), base=object(),
                         configure_process_caches=cache, prepare_process_sdk=sdk)
        replay = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        with self.assertRaises(RuntimeError) as caught:
            exec(compile(replay, str(M.__file__), "exec"), namespace)
        self.assertIs(caught.exception, error); self.assertEqual(order, ["cache", "sdk"])
        self.assertFalse(result["framework_import_attempted"]); self.assertFalse(result["gpu_initialization_attempted"])
        self.assertNotIn("torch", sys.modules); self.assertNotIn("vllm", sys.modules)

    def test_ninja_readonly_gate_rejects_byte_ELF_execute_or_ref_drift_without_paths_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); target = root / M.NINJA
            target.parent.mkdir(parents=True); content = b"\x7fELF CPU fixture only, never executed"
            target.write_bytes(content)
            pin = (M.NINJA, len(content), hashlib.sha256(content).hexdigest())
            refs = {M.NINJA: dict(path=pin[0], bytes=pin[1], sha256=pin[2])}
            before = dict(os.environ); paths = set(root.rglob("*"))
            with patch.object(M, "NINJA_SOURCE_REF", pin), patch.object(M.os, "access", return_value=True):
                self.assertEqual(M.verify_ninja(root, refs), target)
                with self.assertRaises(ValueError): M.verify_ninja(root, {M.NINJA: dict(refs[M.NINJA], sha256="0" * 64)})
                target.write_bytes(content + b"!")
                with self.assertRaises(ValueError): M.verify_ninja(root, refs)
                target.write_bytes(content)
                with patch.object(M.os, "access", return_value=False):
                    with self.assertRaises(ValueError): M.verify_ninja(root, refs)
            fake = b"not ELF or executable code"; target.write_bytes(fake)
            fake_pin = (M.NINJA, len(fake), hashlib.sha256(fake).hexdigest())
            with patch.object(M, "NINJA_SOURCE_REF", fake_pin), patch.object(M.os, "access", return_value=True):
                with self.assertRaises(ValueError): M.verify_ninja(root, {M.NINJA: dict(path=fake_pin[0], bytes=fake_pin[1], sha256=fake_pin[2])})
            self.assertEqual(set(root.rglob("*")), paths); self.assertEqual(dict(os.environ), before)
        self.assertNotIn("torch", sys.modules); self.assertNotIn("vllm", sys.modules)

    @unittest.skipUnless(os.name == "posix", "Actual Linux private symlink/PATH contract runs on server; Windows lacks link privilege")
    def test_actual_private_single_ninja_link_keeps_sdk_first_and_OS_tail_rejects_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); target = root / M.NINJA
            target.parent.mkdir(parents=True); target.write_bytes(b"\x7fELF CPU fixture, no binary execution"); target.chmod(0o755)
            pin = (M.NINJA, target.stat().st_size, hashlib.sha256(target.read_bytes()).hexdigest())
            refs = {M.NINJA: dict(path=pin[0], bytes=pin[1], sha256=pin[2])}
            out = root / "run03/details"; sdk = out / "runtime-cache/cuda13-sdk"; (sdk / "bin").mkdir(parents=True)
            original = os.pathsep.join([str(sdk / "bin"), "/usr/bin", "/bin"])
            evidence = dict(overlay=str(sdk), selected_environment=dict(PATH=original))
            with patch.object(M, "NINJA_SOURCE_REF", pin), patch.dict(os.environ, {"PATH": original}):
                result = M.prepare_process_ninja(root, refs, out, evidence)
                private = out / "runtime-cache/build-tools/bin"
                self.assertEqual(os.environ["PATH"].split(os.pathsep), [str(sdk / "bin"), str(private), "/usr/bin", "/bin"])
                self.assertEqual([p.name for p in private.iterdir()], ["ninja"])
                self.assertEqual((private / "ninja").resolve(), target)
                self.assertNotIn(str(root / ".venv/bin"), os.environ["PATH"].split(os.pathsep))
                self.assertIs(result["GPU_model_or_JIT_runtime_qualified"], False)
                self.assertIs(result["whole_virtualenv_bin_PATH"], False)
                with self.assertRaises(ValueError): M.prepare_process_ninja(root, refs, out, evidence)
                os.environ["PATH"] = original
                with self.assertRaises(FileExistsError): M.prepare_process_ninja(root, refs, out, evidence)
                (private / "ninja").unlink()
                with self.assertRaises(FileExistsError): M.prepare_process_ninja(root, refs, out, evidence)

    def test_exact_original_driver_ninja_fault_after_sdk_before_any_framework_import(self):
        tree = ast.parse(Path(M.__file__).read_bytes())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute_guarded")
        block = next(n for n in fn.body if isinstance(n, ast.Try))
        body = copy.deepcopy(block.body[:3])
        self.assertEqual([n.value.func.id for n in body], ["configure_process_caches", "prepare_process_sdk", "prepare_process_ninja"])
        imports = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Import) and any(a.name in ("torch", "vllm") for a in n.names)]
        self.assertLess(body[2].lineno, min(imports))
        calls = []; error = RuntimeError("Ninja bytes drift")
        def cache(*a): calls.append("cache"); return {}
        def sdk(*a): calls.append("sdk"); return {}
        def ninja(*a): calls.append("ninja"); raise error
        result = dict(framework_import_attempted=False, gpu_initialization_attempted=False)
        ns = dict(result=result, root=Path.cwd(), out=Path.cwd(), refs={}, base=object(),
                  configure_process_caches=cache, prepare_process_sdk=sdk, prepare_process_ninja=ninja)
        replay = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        with self.assertRaises(RuntimeError) as caught: exec(compile(replay, str(M.__file__), "exec"), ns)
        self.assertIs(caught.exception, error); self.assertEqual(calls, ["cache", "sdk", "ninja"])
        self.assertFalse(result["framework_import_attempted"]); self.assertFalse(result["gpu_initialization_attempted"])
        self.assertNotIn("torch", sys.modules); self.assertNotIn("vllm", sys.modules)


class FakeFrontend:
    def __init__(self, *, count=128, error=None, internal="native-id", duplicate=False):
        self.count, self.error, self.internal, self.duplicate = count, error, internal, duplicate
        self.add_calls = self.step_calls = 0
        self.open = False
        self.tokens = []
    def add_request(self, rid, prompt, sampling):
        self.add_calls += 1; self.rid = rid; self.prompt = prompt; self.sampling = sampling
        self.open = True; return self.internal
    def has_unfinished_requests(self): return self.open
    def step(self):
        self.step_calls += 1
        if self.error is not None: raise self.error
        self.tokens.append(self.step_calls - 1)
        final = self.step_calls >= self.count
        if final: self.open = False
        row = NS(request_id=self.rid, finished=final, prompt_token_ids=self.prompt["prompt_token_ids"],
                 outputs=[NS(token_ids=list(self.tokens), finish_reason="length")], num_cached_tokens=0)
        return [row, row] if self.duplicate and final else [row]


class LifecycleContracts(unittest.TestCase):
    def test_original_frontend_add_once_step_128_full_output_and_returned_native_id(self):
        engine = FakeFrontend(); sampling = object()
        result = M.run_original_request(engine, sampling, "front")
        self.assertEqual((engine.add_calls, engine.step_calls), (1, 128))
        self.assertIs(engine.sampling, sampling)
        self.assertEqual(result["native_request_id"], "native-id")
        self.assertEqual(result["output_token_ids"], list(range(128)))

    def test_partial127_or_extra129_outputs_never_truncated_into_success(self):
        for count in (127, 129):
            engine = FakeFrontend(count=count)
            with self.assertRaises(ValueError): M.run_original_request(engine, object(), "front")
            self.assertEqual(engine.add_calls, 1); self.assertEqual(engine.step_calls, count)

    def test_original_step_error_identity_no_retry(self):
        error = RuntimeError("engine")
        engine = FakeFrontend(error=error)
        with self.assertRaises(RuntimeError) as caught: M.run_original_request(engine, object(), "front")
        self.assertIs(caught.exception, error); self.assertEqual(engine.step_calls, 1)

    def test_existing_unfinished_request_rejected_before_new_add(self):
        engine = FakeFrontend(); engine.open = True
        with self.assertRaises(ValueError): M.run_original_request(engine, object(), "front")
        self.assertEqual(engine.add_calls, 0)

    def test_native_returned_identity_absent_or_duplicate_cohort_rejected(self):
        for engine in (FakeFrontend(internal=None), FakeFrontend(duplicate=True)):
            with self.assertRaises(ValueError): M.run_original_request(engine, object(), "front")

    def test_frame_reconciliation_requires_returned_native_identity_and_every_token(self):
        frontend = dict(native_request_id="native-id", output_token_ids=list(range(128)))
        rows = [dict(gpu_elapsed_ns=None, outputs=(("native-id", (i,)),)) for i in range(128)]
        observed = dict(valid=True, open_event_pair=False, frames=rows)
        result = M.verify_frame_outputs(observed, frontend)
        self.assertEqual(result["complete_output_count"], 128); self.assertFalse(result["cost_qualified"])
        for wrong in (dict(observed, frames=rows[1:]), dict(observed, open_event_pair=True),
                      dict(observed, valid=False)):
            with self.assertRaises(ValueError): M.verify_frame_outputs(wrong, frontend)
        with self.assertRaises(ValueError): M.verify_frame_outputs(observed, dict(frontend, native_request_id="other"))

    def test_actual_author_frontend_and_shutdown_signatures_are_source_bound(self):
        # The server uses its project root; a separate mirror manifest is not
        # required. These exact pins were independently checked on server09.
        refs = [
            dict(path=M.AUTHOR + "/vllm/entrypoints/llm.py", bytes=41208,
                 sha256="2ca6ad3ac3b6b7fea1518eb795473e81f71beeba617393a1cb2282b6c09a92a0"),
            dict(path=M.AUTHOR + "/vllm/v1/engine/llm_engine.py", bytes=16586,
                 sha256="2900f6732a1e507f6ced6f75f05f1fcaf2cc2929df0f5066574e9077ad3fbde1"),
            dict(path=M.AUTHOR + "/vllm/v1/engine/core_client.py", bytes=69599,
                 sha256="d4ba36d9b086e297a12b289887ba04552ccfc81fdedc10972bebffe14ddff8a3"),
            dict(path=M.AUTHOR + "/vllm/v1/engine/core.py", bytes=92611,
                 sha256="e9e522b559d1c79fd3e2943bbb42203180d8341823834fa3e202912b7e48945d"),
            dict(path=M.AUTHOR + "/vllm/engine/llm_engine.py", bytes=296,
                 sha256="3ab9abd1e1689301897760c1589cc91ab688e3bb1cdd33db041a6b584bf25342"),
        ]
        trees = {}
        for row in refs:
            raw = (args.source_root / row["path"]).read_bytes()
            self.assertEqual(len(raw), row["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])
            trees[row["path"]] = ast.parse(raw)
        engine_path = M.AUTHOR + "/vllm/v1/engine/llm_engine.py"
        author = next(n for n in trees[engine_path].body if isinstance(n, ast.ClassDef) and n.name == "LLMEngine")
        methods = {n.name: n for n in author.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual([a.arg for a in methods["add_request"].args.args[:4]], ["self", "request_id", "prompt", "params"])
        self.assertEqual(len(methods["step"].args.args), 1)
        self.assertEqual(len(methods["has_unfinished_requests"].args.args), 1)
        client = next(n for n in trees[M.AUTHOR + "/vllm/v1/engine/core_client.py"].body
                      if isinstance(n, ast.ClassDef) and n.name == "InprocClient")
        shutdown = next(n for n in client.body if isinstance(n, ast.FunctionDef) and n.name == "shutdown")
        self.assertEqual([a.arg for a in shutdown.args.args], ["self", "timeout"])
        self.assertIn("self.engine_core.shutdown()", ast.unparse(shutdown))


if __name__ == "__main__": unittest.main(verbosity=2)
