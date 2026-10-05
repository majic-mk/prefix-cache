"""Actual V6 prepare_plan source/leaf replay; fixture plans cannot use GPUs."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest

HERE=Path(__file__).resolve().parent
PREP=HERE.parent
STRUCTURE_ONLY="--structure-only" in sys.argv
if STRUCTURE_ONLY:
    sys.argv.remove("--structure-only")
OLD_SHA="9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d"


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value
    spec.loader.exec_module(value)
    return value


W=load("_CPU_only_raw_v6_binding",PREP/"runner/strong_native_cost_runner_v6.py")
R=W.driver_module()
OLD_RAW=(PREP/"runner/bounded_native_full_step_collector.py").read_bytes()
NEW_PATH=PREP/"runner/bounded_native_full_step_collector_v2.py"
assert hashlib.sha256(OLD_RAW).hexdigest()==OLD_SHA
FUNCTION=next(node for node in ast.parse((PREP/"runner/strong_native_cost_runner_v6.py").read_bytes()).body
              if isinstance(node,ast.FunctionDef) and node.name=="prepare_plan")


def fixture(root, *, absent=None, drift=None, tamper_old_hash=False):
    root=root.resolve()
    relative="CPU_fixture_preparation/runner/strong_native_cost_runner_v6.py"
    source=root/relative
    source.parent.mkdir(parents=True)
    source.write_bytes((PREP/"runner/strong_native_cost_runner_v6.py").read_bytes())
    pair_path="CPU_fixture_preparation/PRIVATE_U_I_CONFIG.json"
    pair_ref=dict(path=pair_path)
    geometry_path="CPU_fixture_preparation/runner/geometry.json"
    scope=dict(Path=Path,R=R,relative=relative,new_root=Path(relative).parent.parent,
               pair_ref=pair_ref,geometry_ref=dict(path=geometry_path))
    assignment=next(node for node in FUNCTION.body if isinstance(node,ast.Assign)
                    and any(isinstance(target,ast.Name) and target.id=="required" for target in node.targets))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(assignment)],type_ignores=[])),
                 "CPU_actual_required_leaf_AST","exec",dont_inherit=True),scope)
    required=scope["required"]
    for key,path in required.items():
        target=root/path
        target.parent.mkdir(parents=True,exist_ok=True)
        if key=="wrapper_source_ref":
            continue
        if key=="original_collector_source_ref":
            raw=OLD_RAW if not tamper_old_hash else b"CPU-fixture-invalid-original-delegate"
        elif key=="collector_source_ref":
            if NEW_PATH.is_file():
                raw=NEW_PATH.read_bytes()
            elif STRUCTURE_ONLY:
                raw=b"CPU-fixture-structure-only-pending-new-collector-not-GPU-usable"
            else:
                raise ValueError("Actual collector V2 source missing; not ready for final byte test")
        elif key=="runtime_pair_ref":
            raw=json.dumps(dict(schema="strong_native_u_i_cpu_configuration_v1",
                                configurations=dict(CPU_fixture_only=True))).encode()
        else:
            raw=b"CPU-fixture-only-other-leaf; not a runtime/GPU proof"
        target.write_bytes(raw)
    input_path="CPU_fixture_preparation/input_manifest.json"
    (root/input_path).write_text('{"CPU_fixture_only":true}',encoding="utf-8")
    refs={path:R.ref(root,path) for path in required.values()}
    refs[input_path]=R.ref(root,input_path)
    if absent is not None:
        refs.pop(required[absent])
    if drift is not None:
        (root/required[drift]).write_bytes(b"CPU-labelled byte drift after source rows frozen")
    validated=[]
    full_calls=[]
    def source_rows(project,lock,*,full):
        assert project==root
        full_calls.append(full)
        return refs
    def validate_pair(value):
        assert value==dict(CPU_fixture_only=True)
        validated.append(value)
    # R.check_ref/ref/read/new_json are actual original byte checks. The surrounding
    # complete lock and pair validator here are explicit CPU fixtures, not receipts.
    bridge=NS(source_rows=source_rows,check_ref=R.check_ref,require=R.require,read=R.read,
        load=lambda *args:NS(validate_runtime_pair=validate_pair),common_domain_sha=R.canonical_sha,
        new_json=R.new_json,STRONG=R.STRONG,NATIVE=R.NATIVE,GUARD=R.GUARD,MODEL_PLAN=R.MODEL_PLAN,MODEL=R.MODEL)
    namespace=dict(Path=Path,time=time,__file__=str(source),driver_module=lambda:bridge,
                   cell_descriptor=W.cell_descriptor)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(FUNCTION)],type_ignores=[])),
                 "CPU_actual_V6_prepare_plan_AST","exec",dont_inherit=True),namespace)
    def invoke():
        return namespace["prepare_plan"](root,source_lock_ref=dict(CPU_fixture_only=True),
            pair_ref=refs[pair_path],input_manifest_ref=refs[input_path],geometry_ref=refs[geometry_path],
            output=root/"CPU_ONLY_NON_ISSUABLE_PLAN.json",job_id="CPU-fixture-job")
    return invoke,required,refs,full_calls


class CollectorBindingCPU(unittest.TestCase):
    def test_only_prepare_plan_changed_and_diagnostics_preserved(self):
        old=ast.parse((PREP/"runner/strong_native_cost_runner_v5.py").read_bytes())
        new=ast.parse((PREP/"runner/strong_native_cost_runner_v6.py").read_bytes())
        previous={node.name:ast.dump(node,include_attributes=False) for node in old.body if isinstance(node,ast.FunctionDef)}
        current={node.name:ast.dump(node,include_attributes=False) for node in new.body if isinstance(node,ast.FunctionDef)}
        self.assertEqual([name for name in previous if previous[name]!=current[name]],["prepare_plan"])
        self.assertEqual(hashlib.sha256((PREP/"runner/strong_native_cost_runner_v5.py").read_bytes()).hexdigest(),
                         "e05921ea213a9b6c6d8da15e9fea2c22ccef62020a934a1ddb3c36859c47978f")
    def test_actual_prepare_plan_selects_both_byte_bound_leaves(self):
        with tempfile.TemporaryDirectory(prefix="CPU-only-leaf-plan-") as temp:
            invoke,required,refs,full_calls=fixture(Path(temp))
            plan=invoke()
            self.assertEqual(full_calls,[True])
            self.assertTrue(required["collector_source_ref"].endswith("/runner/bounded_native_full_step_collector_v2.py"))
            self.assertTrue(required["original_collector_source_ref"].endswith("/runner/bounded_native_full_step_collector.py"))
            self.assertEqual(plan["collector_source_ref"],refs[required["collector_source_ref"]])
            self.assertEqual(plan["original_collector_source_ref"]["sha256"],OLD_SHA)
            self.assertIsNone(plan["gpu_uuid"])
            self.assertTrue(plan["cpu_preparation_only"])
            self.assertFalse(plan["GPU_qualification_issued"])
            self.assertEqual(plan["actual_gpu_runs"],0)
    def test_absent_new_or_original_leaf_rejects_before_plan(self):
        for key in ("collector_source_ref","original_collector_source_ref"):
            with tempfile.TemporaryDirectory(prefix="CPU-only-missing-leaf-") as temp:
                root=Path(temp)
                invoke,required,refs,calls=fixture(root,absent=key)
                with self.assertRaisesRegex(ValueError,"full cost leaf frozen before template: "+key):
                    invoke()
                self.assertFalse((root/"CPU_ONLY_NON_ISSUABLE_PLAN.json").exists())
    def test_drift_or_relocked_changed_original_delegate_rejects(self):
        cases=[dict(drift="collector_source_ref"),dict(drift="original_collector_source_ref"),dict(tamper_old_hash=True)]
        for case in cases:
            with tempfile.TemporaryDirectory(prefix="CPU-only-invalid-leaf-") as temp:
                root=Path(temp)
                invoke,required,refs,calls=fixture(root,**case)
                with self.assertRaises(ValueError):
                    invoke()
                self.assertFalse((root/"CPU_ONLY_NON_ISSUABLE_PLAN.json").exists())
    def test_actual_two_collector_files_and_delegate_pin(self):
        self.assertEqual(hashlib.sha256(OLD_RAW).hexdigest(),OLD_SHA)
        if STRUCTURE_ONLY:
            self.assertTrue(STRUCTURE_ONLY) # explicitly pending; never FINAL_READY
            return
        self.assertTrue(NEW_PATH.is_file(),"V2 source must exist for final byte verification")
        new=NEW_PATH.read_bytes()
        self.assertIn(OLD_SHA.encode(),new)
        self.assertIn(b"bounded_native_full_step_collector.py",new)
        self.assertGreater(len(new),0)
        self.assertNotEqual(hashlib.sha256(new).hexdigest(),OLD_SHA)


if __name__=="__main__":
    unittest.main(verbosity=2)
