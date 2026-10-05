"""Execute the original patched parent until a CPU-only subprocess intercept.

No subprocess is launched, no framework is imported, and no actual guard or
positive GPU receipt is fabricated. Temporary fixture files are discarded.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent / "runner"
spec = importlib.util.spec_from_file_location("_cpu_relative_raw_v3", RUNNER / "strong_native_cost_runner_v3.py")
W = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = W
spec.loader.exec_module(W)


class CPUChildInterceptStop(BaseException):
    """Stops the real original parent before any process/model can start."""


def original_source():
    local = HERE.parents[1] / "i_pilot_cpu_preparation_20261004/calibration_v2/run_native_cost_experiment.py"
    if local.exists():
        return local
    return HERE.parents[1] / "server12-i-pilot-cpu-preparation-20261004/calibration_v2/run_native_cost_experiment.py"


def fixture_parent(root, config_path, intercepted, metadata):
    raw = original_source().read_bytes()
    assert hashlib.sha256(raw).hexdigest() == W.OLD_SHA
    tree, _ = W.source_patch(raw)
    R = W.driver_module()
    def refuse_subprocess(command, **kwargs):
        intercepted.append(dict(argv=list(command), kwargs=kwargs))
        raise CPUChildInterceptStop("CPU-only interception; zero subprocess/GPU runs")
    config = dict(out="experiments/cpu-fixture/run/details", gpu_uuid="CPU_FIXTURE_NOT_GPU_AUTHORITY",
                  gpu_entry_binding_ref=dict(origin="CPU_fixture_only"))
    (root / config["out"]).parent.mkdir(parents=True)
    namespace = dict(__name__="_cpu_original_parent_AST_only", CONFIG_PATH=config_path,
        SCRIPT="runner/strong_native_cost_runner_v3.py", DELIVERY="unused_cpu_fixture",
        PURPOSE="CPU_ONLY_ARGUMENT_REGRESSION_NOT_GPU_QUALIFICATION", LABEL="cpu-only-parent-argument-check",
        verify_execution_inputs=lambda r,c,refs,guard: guard,
        safe=R.safe, require=R.require, input_groups=lambda r,c: None,
        write_json=lambda path,value: metadata.append((str(path), value)),
        os=NS(environ={}, getsid=lambda _: 101, getpgid=lambda _: 101),
        sys=sys, time=time, subprocess=NS(run=refuse_subprocess),
        traceback=__import__("traceback"))
    exec(compile(tree, "CPU_fixture_original_patched_parent", "exec", dont_inherit=True), namespace)
    return namespace["execute_parent"], config


class RelativeConfigCPU(unittest.TestCase):
    def test_actual_patched_parent_first_child_uses_guard_config_relative_path_and_aborts(self):
        with tempfile.TemporaryDirectory(prefix="cpu-parent-argv-") as directory:
            root = Path(directory).resolve()
            relative = "artifacts/strong-cal02/CONFIG.json"
            calls, metadata = [], []
            parent, config = fixture_parent(root, root / relative, calls, metadata)
            with self.assertRaisesRegex(CPUChildInterceptStop, "zero subprocess/GPU"):
                parent(root, config, {}, dict(origin="CPU_fixture_only_not_guard_authority"))
            self.assertEqual(len(calls), 1)
            child = calls[0]["argv"]
            self.assertEqual(child.count("--config"), 1)
            child_config = child[child.index("--config") + 1]
            self.assertEqual(child_config, relative)
            self.assertFalse(Path(child_config).is_absolute())
            self.assertNotIn("\\", child_config)
            self.assertEqual(W.driver_module().safe(root, child_config), root / relative)
            parent_config = dict(runner_ref=dict(path="runner/strong_native_cost_runner_v3.py"))
            guarded_parent = W.command(root, relative, parent_config)
            self.assertEqual(child_config, guarded_parent[guarded_parent.index("--config") + 1])
            self.assertEqual(child[child.index("--project") + 1], str(root))
            self.assertEqual(child[-2:], ["--window-index", "0"])
            self.assertFalse(calls[0]["kwargs"].get("start_new_session", False))
            self.assertEqual(calls[0]["kwargs"]["cwd"], root)
            command_metadata = [value for path,value in metadata if path.endswith("CHILD_COMMAND.json")]
            self.assertEqual(len(command_metadata), 1)
            self.assertFalse(command_metadata[0]["starts_new_session"])
            self.assertFalse(any(path.endswith("CHILD_RESULT.json") or path.endswith("native-cost-runtime-result.json")
                                 for path,value in metadata))

    def test_outside_project_config_never_reaches_subprocess(self):
        with tempfile.TemporaryDirectory(prefix="cpu-parent-reject-") as directory:
            root = Path(directory).resolve()
            calls, metadata = [], []
            parent, config = fixture_parent(root, root.parent / "outside-project.json", calls, metadata)
            result = parent(root, config, {}, dict(origin="CPU_fixture_only_not_guard_authority"))
            self.assertEqual(calls, [])
            self.assertEqual(result["status"], "FAILED_NATIVE_SIX_PROCESS_DIAGNOSTIC")
            self.assertEqual(result["error_type"], "ValueError")
            self.assertEqual(result["original_model_subprocesses_started"], 0)
            self.assertFalse(result["production_qualified"])

    def test_v3_changes_only_source_patch_and_all_sealed_previous_bytes_unchanged(self):
        v2 = (RUNNER / "strong_native_cost_runner_v2.py").read_bytes()
        v3 = (RUNNER / "strong_native_cost_runner_v3.py").read_bytes()
        old, new = ast.parse(v2), ast.parse(v3)
        self.assertEqual(len(old.body), len(new.body))
        changed = [getattr(a,"name",type(a).__name__) for a,b in zip(old.body,new.body)
                   if ast.dump(a,include_attributes=False) != ast.dump(b,include_attributes=False)]
        self.assertEqual(changed, ["source_patch"])
        self.assertEqual(hashlib.sha256(v2).hexdigest(), "36b1f20a106a823cef372795b9e65323cce6f7944ddc756d792878d74cbe0d7b")
        self.assertEqual(hashlib.sha256((HERE.parent / "raw_device_binding/uuid_minor_device.py").read_bytes()).hexdigest(),
                         "f43769e132a1fe28c5d1c244f93785e63cc895c8a991b89ae6e4205c4308d427")
        manifest = json.loads((RUNNER / "RUNNER_SOURCE_MANIFEST.json").read_bytes())
        self.assertEqual(len(manifest["files"]), 16)
        for row in manifest["files"]:
            raw = (HERE.parent / row["path"]).read_bytes()
            self.assertEqual(len(raw), row["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
