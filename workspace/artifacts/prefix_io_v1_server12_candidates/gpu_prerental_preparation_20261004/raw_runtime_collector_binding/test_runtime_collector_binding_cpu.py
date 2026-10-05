"""CPU source binding only; no deadline/table/GPU authority is synthesized."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

HERE=Path(__file__).resolve().parent
PREP=HERE.parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value
    spec.loader.exec_module(value)
    return value


N=load("_CPU_only_native_runtime_v2",PREP/"runner/native_runtime_v2.py")
R=load("_CPU_only_strong_trace_runner_v2",PREP/"runner/strong_trace_runner_v2.py")
OLD=PREP/"runner/bounded_native_full_step_collector.py"
NEW=PREP/"runner/bounded_native_full_step_collector_v2.py"
OLD_SHA="9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d"
NEW_SHA="a710f3f55ecbc6d8fbce8f3cf85ce4587daf0e1a863783df25a4ba8bc37e2aa0"
assert hashlib.sha256(OLD.read_bytes()).hexdigest()==OLD_SHA
assert hashlib.sha256(NEW.read_bytes()).hexdigest()==NEW_SHA


def fixture(root,*,phase="development",selected_old=False,missing=None,drift=None,bad_ancestry=False,
            mismatched_plan=False,runtime_name="native_runtime_v2.py"):
    """Actual source bytes, explicit CPU-only JSON/source gate, no real activation."""
    root=root.resolve()
    runtime="CPU_fixture_preparation/runner/"+runtime_name
    old="CPU_fixture_preparation/runner/bounded_native_full_step_collector.py"
    new="CPU_fixture_preparation/runner/bounded_native_full_step_collector_v2.py"
    raw_source=PREP/"runner"/runtime_name
    for path,raw in ((runtime,raw_source.read_bytes()),(old,OLD.read_bytes()),(new,NEW.read_bytes())):
        full=root/path
        full.parent.mkdir(parents=True,exist_ok=True)
        full.write_bytes(raw)
    refs={path:R.ref(root,path) for path in (runtime,old,new)}
    selected=refs[old if selected_old else new]
    calibration=dict(CPU_fixture_only=True,GPU_qualification_issued=False,files=[deepcopy(refs[old]),deepcopy(refs[new])])
    if bad_ancestry:
        calibration["files"][1]["sha256"]="b"*64
    lock_path="CPU_fixture_preparation/CPU_CALIBRATION_SOURCE_ROWS.json"
    R.new_json(root/lock_path,calibration)
    lock_ref=R.ref(root,lock_path)
    plan=dict(CPU_fixture_only=True,GPU_qualification_issued=False,collector_source_ref=deepcopy(selected),
        original_collector_source_ref=refs[old],source_lock_ref=lock_ref,
        gpu_uuid="CPU_FIXTURE_NOT_GPU_AUTHORITY",common_runtime_domain_sha256="a"*64)
    if mismatched_plan:
        plan["collector_source_ref"]=refs[old]
    plan_path="CPU_fixture_preparation/CPU_PLAN_NOT_ISSUABLE.json"
    R.new_json(root/plan_path,plan)
    refs[plan_path]=R.ref(root,plan_path)
    refs[lock_path]=lock_ref
    descriptor=dict(CPU_fixture_only=True,collector_ref=selected,plan_ref=refs[plan_path],calibration_source_lock_ref=lock_ref)
    if missing:
        refs.pop(new if missing=="new" else old)
    if drift:
        (root/(new if drift=="new" else old)).write_bytes(b"CPU deliberate drift; not a runtime source")
    gates=dict(config=dict(phase=phase,runtime_ref=refs[runtime],run_id="CPU-only-source-binding",
                           gpu_uuid="CPU_FIXTURE_NOT_GPU_AUTHORITY"),refs=refs,
        common_runtime_domain_sha256="a"*64,finite_activation=dict(descriptor=descriptor,
            calibration_plan=plan,runtime_refs=refs,actual_table_issued=False,formal_effect_qualified=False))
    return root,gates,selected


def actual_entry_tail(root,gates):
    # Execute the new CPU selection branch from the actual verify_configuration
    # AST. Surrounding original complete-phase/source/permission gates are kept
    # in the source and are not claimed passed by this explicit CPU fixture.
    tree=ast.parse((PREP/"runner/strong_trace_runner_v2.py").read_bytes())
    verify=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="verify_configuration")
    added=verify.body[-2]
    assert isinstance(added,ast.If)
    function=ast.parse("def CPU_tail(root,gates): pass").body[0]
    function.body=[ast.parse('config=gates["config"]').body[0],deepcopy(added),deepcopy(verify.body[-1])]
    namespace=dict(vars(R))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                 "CPU_actual_preflight_runtime_selection_AST","exec",dont_inherit=True),namespace)
    return namespace["CPU_tail"](root,gates)


class RuntimeCollectorSourceCPU(unittest.TestCase):
    def test_actual_new_leaf_selected_with_exact_inherited_rows_and_appended_runtime(self):
        for phase in ("development","effect"):
            with tempfile.TemporaryDirectory(prefix="CPU-only-collector-source-") as temp:
                root,gates,expected=fixture(Path(temp),phase=phase)
                self.assertEqual(N.preflight_collector_binding(root,gates,driver=R),expected)
                self.assertEqual(expected["sha256"],NEW_SHA)
                self.assertFalse(gates["finite_activation"]["actual_table_issued"])
                self.assertFalse(gates["finite_activation"]["formal_effect_qualified"])
                current=gates["refs"]
                inherited=R.read(R.check_ref(root,gates["finite_activation"]["descriptor"]["calibration_source_lock_ref"]))
                self.assertGreater(len(current),len(inherited["files"]))
    def test_qualification_and_no_table_shadow_preserve_old_collector_without_activation(self):
        for phase in ("qualification","shadow"):
            with tempfile.TemporaryDirectory(prefix="CPU-only-native-off-source-") as temp:
                root,gates,expected=fixture(Path(temp),phase=phase)
                gates["finite_activation"]=None
                self.assertEqual(N.preflight_collector_binding(root,gates,driver=R)["sha256"],OLD_SHA)
                self.assertIsNone(gates["finite_activation"])
    def test_old_collector_rejects_from_actual_execute_before_model_or_output_access(self):
        for phase in ("development","effect"):
            with tempfile.TemporaryDirectory(prefix="CPU-only-old-collector-reject-") as temp:
                root,gates,expected=fixture(Path(temp),phase=phase,selected_old=True)
                # No out/storage/model keys provided: actual execute must reject
                # at the early source gate before reading or creating any of them.
                with self.assertRaisesRegex(ValueError,"exact actual V2 collector"):
                    N.execute(root,gates,dict(label=gates["config"]["run_id"]),driver=R)
    def test_missing_drift_or_wrong_ancestry_rejects_before_runtime(self):
        cases=[dict(missing="new"),dict(missing="old"),dict(drift="new"),dict(drift="old"),
               dict(bad_ancestry=True),dict(mismatched_plan=True)]
        for case in cases:
            with tempfile.TemporaryDirectory(prefix="CPU-only-invalid-source-gate-") as temp:
                root,gates,expected=fixture(Path(temp),**case)
                with self.assertRaises(ValueError):
                    N.preflight_collector_binding(root,gates,driver=R)
    def test_actual_cpu_entry_checks_same_runtime_leaf_and_rejects_old_before_gpu(self):
        with tempfile.TemporaryDirectory(prefix="CPU-only-entry-good-source-") as temp:
            root,gates,expected=fixture(Path(temp))
            actual=actual_entry_tail(root,gates)
            evidence=actual["collector_binding_preflight"]
            self.assertEqual(evidence["source_ref"],expected)
            self.assertTrue(evidence["source_only"])
            self.assertFalse(evidence["actual_table_issued"])
            self.assertFalse(evidence["GPU_qualification_issued"])
        for case in (dict(selected_old=True),dict(runtime_name="native_runtime.py")):
            with tempfile.TemporaryDirectory(prefix="CPU-only-entry-old-source-") as temp:
                root,gates,expected=fixture(Path(temp),**case)
                with self.assertRaises(ValueError):
                    actual_entry_tail(root,gates)
    def test_original_phase_workload_pipeline_guard_and_qualification_unchanged(self):
        native_old=ast.parse((PREP/"runner/native_runtime.py").read_bytes())
        native_new=ast.parse((PREP/"runner/native_runtime_v2.py").read_bytes())
        old={node.name:ast.dump(node,include_attributes=False) for node in native_old.body if isinstance(node,ast.FunctionDef)}
        new={node.name:ast.dump(node,include_attributes=False) for node in native_new.body if isinstance(node,ast.FunctionDef)}
        self.assertEqual([name for name in old if old[name]!=new[name]],["execute"])
        self.assertEqual(set(new)-set(old),{"preflight_collector_binding"})
        entry_old=ast.parse((PREP/"runner/strong_trace_runner.py").read_bytes())
        entry_new=ast.parse((PREP/"runner/strong_trace_runner_v2.py").read_bytes())
        old={node.name:ast.dump(node,include_attributes=False) for node in entry_old.body if isinstance(node,ast.FunctionDef)}
        new={node.name:ast.dump(node,include_attributes=False) for node in entry_new.body if isinstance(node,ast.FunctionDef)}
        self.assertEqual([name for name in old if old[name]!=new[name]],["verify_configuration"])
        self.assertEqual(set(old),set(new))
        self.assertEqual(hashlib.sha256((PREP/"runner/native_runtime.py").read_bytes()).hexdigest(),
                         "a293f98841e34bb9357e8dacb62cda43be7dfc7da66167f3df5577c4b5b11709")
        self.assertEqual(hashlib.sha256((PREP/"runner/strong_trace_runner.py").read_bytes()).hexdigest(),
                         "239771d85459b795d82624e23aa96510954ce301b2fc6554b2e575057edb9ce9")
        self.assertEqual(hashlib.sha256(NEW.read_bytes()).hexdigest(),NEW_SHA)
        self.assertEqual(hashlib.sha256(OLD.read_bytes()).hexdigest(),OLD_SHA)
        self.assertFalse(any(name=="torch" or name.startswith(("torch.","vllm","py_kvcache")) for name in sys.modules))


if __name__=="__main__":
    unittest.main(verbosity=2)
