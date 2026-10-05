"""Four raw-only CPU checks; no actual model/library/GPU/RPC imports."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    exec(compile(path.read_bytes(),str(path),"exec",dont_inherit=True),vars(module))
    return module


def actual(canonical,local):
    for root in (Path.cwd(),*HERE.parents):
        for name in (canonical,local):
            path=root/name
            if path.is_file():return path
    raise RuntimeError("UNBOUND actual source "+canonical)


M=load(HERE/"natural_unit_raw_runner.py","_fixture_raw_unit")
NATURAL=actual(M.NATURAL["path"],"artifacts/prefix_io_v1_server12_candidates/natural_source_audit_20261005/ACTUAL_PUBLIC_ORIGINAL_MANIFEST_01.json")
MANIFEST=json.loads(NATURAL.read_bytes())
STRONG=actual(M.STRONG_RAW,"artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/runner/strong_native_cost_runner_v7.py")
CORE=actual("artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/calibration_v2/run_native_cost_experiment.py",
    "artifacts/prefix_io_v1_server12_candidates/i_pilot_cpu_preparation_20261004/calibration_v2/run_native_cost_experiment.py")


class RawChecks(unittest.TestCase):
    def test_exact_actual_calibration185_geometry(self):
        record=M.selected_natural_record(MANIFEST)
        cell=M.cell_descriptor(record)
        self.assertEqual((cell["prompt_tokens"],cell["cached_prompt_tokens"],cell["frontend_cached_prompt_tokens"],
            cell["measured_offset"],cell["transfer_quantum_bytes"]),(185,176,176,16,917504))
        self.assertEqual(cell["prompt_tokens"]+cell["measured_offset"]-1,200)
        self.assertEqual(cell["entries"][0]["prompt_token_ids"],MANIFEST["records"][0]["prompt_token_ids"])
        self.assertEqual(cell["entries"][0]["seed"],0)

    def test_reject_injected_dev_or_padded_record(self):
        for edit in (lambda x:x["records"].__setitem__(0,x["records"][30]),
            lambda x:x["records"][0]["prompt_token_ids"].append(1),
            lambda x:x["records"][0].update(seed=4030),lambda x:x["records"][0].update(split="development")):
            bad=deepcopy(MANIFEST);edit(bad)
            with self.assertRaises(ValueError):M.selected_natural_record(bad)

    def test_original_window_finally_and_sdk_interface_preserved(self):
        S=load(STRONG,"_fixture_strong_original")
        S.driver_module=lambda:types.SimpleNamespace(require=M.require)
        baseline,_=S.source_patch(CORE.read_bytes())
        original_window=next(n for n in baseline.body if n.name=="execute_window")
        original_finally=next(n.finalbody for n in original_window.body if isinstance(n,ast.Try))
        captured={}
        def configured(root,gates,guard,relative):
            tree,proof=S.source_patch(CORE.read_bytes());captured["tree"]=tree
            return types.SimpleNamespace(),{},proof
        R=types.SimpleNamespace(load=lambda *args:S,require=M.require)
        config={"job_id":"synthetic-A","gpu_uuid":"GPU-synthetic", "plan_ref":{"path":"synthetic-plan.json"}}
        record=M.selected_natural_record(MANIFEST)
        gates=dict(config=config,refs={M.STRONG_RAW:{}},plan=dict(natural_source_record=record,
            raw_sdk_binding_ref=dict(path=M.SDK_PATH)),current_model_leaf_stats={})
        with mock.patch.object(S,"configure_original",side_effect=configured):
            _,old,_,_=M.configure_window(Path.cwd(),gates,{},"synthetic.json",R=R)
        window=next(n for n in captured["tree"].body if n.name=="execute_window")
        actual_finally=next(n.finalbody for n in window.body if isinstance(n,ast.Try))
        self.assertEqual(ast.dump(ast.Module(body=original_finally,type_ignores=[]),include_attributes=False),
            ast.dump(ast.Module(body=actual_finally,type_ignores=[]),include_attributes=False))
        self.assertEqual((old.PROMPT_TOKENS,old.CACHED_TOKENS,old.SEEDS[0],old.SITE_SDK),(185,176,0,M.SDK_PATH))
        self.assertEqual(S.CACHED_TOKENS,176)
        calls=[n for n in ast.walk(window) if isinstance(n,ast.Call)]
        sdk=next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=="prepare_site_sdk")
        self.assertEqual(len(sdk.args),3)
        self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=="_inherited_actual_model" for n in calls),1)

    def test_single_window_never_executes_parent_estimator_or_issuer(self):
        tree=ast.parse((HERE/"natural_unit_raw_runner.py").read_bytes())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
        calls=[n for n in ast.walk(main) if isinstance(n,ast.Call)]
        names=[n.func.attr for n in calls if isinstance(n.func,ast.Attribute)]
        self.assertEqual(names.count("execute_window"),1)
        self.assertNotIn("execute_parent",names)
        self.assertNotIn("issue_verified_gpu_table",names)
        self.assertNotIn("normalize_actual",names)


if __name__=="__main__":unittest.main()
