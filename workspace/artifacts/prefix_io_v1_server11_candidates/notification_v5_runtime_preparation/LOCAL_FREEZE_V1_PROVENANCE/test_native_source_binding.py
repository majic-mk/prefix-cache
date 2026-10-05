"""CPU actual-source counterexamples; synthetic owners never qualify a GPU run."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load('_p4_c4_actual_source_cpu', HERE/'run_p4_single_file_experiment.py')
V = load('_p4_c4_actual_source_verifier_cpu', HERE/'verify_p4_single_file.py')
PROJECT = os.environ.get('SERVER11_AUTHOR_SOURCE_ROOT')
RUNTIME_PATH = (Path(PROJECT)/M.RUNTIME if PROJECT else
    HERE.parents[2]/'artifacts/prefix_io_v1_server09_candidates/g3_calibration_launcher_cpu_v1/g3_calibration_runtime_metrics_v2.py')
R = load('_p4_c4_original_method_cpu',RUNTIME_PATH)


class ActualNativeSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.code = b'class IoReactor:\n    def _run(self):\n        return None\n'
        self.refs = {}
        for relative in (M.REACTOR,M.LEGACY_COLLECTOR_REACTOR_KEY):
            path = self.root/relative; path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(self.code); self.refs[relative] = M.file_ref(self.root,relative)
        self.module = types.ModuleType('py_kvcache.reactor')
        self.module.__file__ = str(self.root/M.REACTOR)
        exec(compile(self.code,self.module.__file__,'exec',dont_inherit=True),self.module.__dict__)
        self.owner = self.module.IoReactor()
        self.owner._max_accepted_parents = 8; self.owner._prefix_p4_bridge = None
        self.patch = patch.dict(sys.modules,{self.module.__name__:self.module})
        self.patch.start(); self.addCleanup(self.patch.stop)

    def binding(self, mode='off', bridge=None):
        return M.native_source_binding(self.root,self.refs,self.owner,R,bridge=bridge,mode=mode)

    def test_actual_compiled_class_and_original_run_are_bound_for_all_modes(self):
        for mode in M.MODES:
            bridge = None if mode == 'off' else types.SimpleNamespace(mode='interference',single_file_shadow=True)
            self.owner._prefix_p4_bridge = bridge
            value = self.binding(mode,bridge)
            self.assertEqual(value['actual_source_ref'],self.refs[M.REACTOR])
            self.assertEqual(value['original_run_source_ref'],self.refs[M.REACTOR])
            self.assertTrue(value['original_run_code_verified'])
            self.assertIs(value['bridge_is_none'],mode == 'off')

    def test_old_module_rejected_even_when_both_source_files_are_locked(self):
        self.module.__file__ = str(self.root/M.LEGACY_COLLECTOR_REACTOR_KEY)
        with self.assertRaisesRegex(ValueError,'actual common reactor source path'): self.binding()

    def test_old_run_code_filename_rejected_even_when_both_files_are_locked(self):
        exec(compile(self.code,str(self.root/M.LEGACY_COLLECTOR_REACTOR_KEY),'exec',dont_inherit=True),self.module.__dict__)
        self.owner = self.module.IoReactor()
        self.owner._max_accepted_parents = 8; self.owner._prefix_p4_bridge = None
        with self.assertRaisesRegex(ValueError,'actual original source code'): self.binding()

    def test_instance_run_override_rejected(self):
        self.owner._run = lambda: None
        with self.assertRaisesRegex(ValueError,'instance override'): self.binding()

    def test_wrong_parent_cap_or_bridge_identity_rejected(self):
        for value in (64,True):
            self.owner._max_accepted_parents = value
            with self.assertRaisesRegex(ValueError,'bounded common owner'): self.binding()
        self.owner._max_accepted_parents = 8
        bridge = types.SimpleNamespace(mode='interference')
        self.owner._prefix_p4_bridge = bridge
        with self.assertRaisesRegex(ValueError,'bounded common owner'): self.binding('off')
        with self.assertRaisesRegex(ValueError,'bounded common owner'):
            self.binding('on',types.SimpleNamespace(mode='interference'))

    def test_legacy_package_module_rejected_even_if_its_hash_is_locked(self):
        extra = types.ModuleType('prefix_io_control.legacy')
        extra.__file__ = str(self.root/M.LEGACY_COLLECTOR_REACTOR_KEY)
        with patch.dict(sys.modules,{extra.__name__:extra}):
            with self.assertRaisesRegex(ValueError,'escaped common overlay'): self.binding()

    def test_actual_native_source_drift_rejected(self):
        (self.root/M.REACTOR).write_bytes(self.code+b'# changed\n')
        with self.assertRaises(ValueError): self.binding()

    def verification_fixture(self):
        native = self.refs[M.REACTOR]
        raw = dict(native_source_binding=self.binding(),verified_runtime_identity=dict(source_refs=[native]))
        receipt = types.SimpleNamespace(calibration_native_source_sha256=native['sha256'],
            runtime_common_refs=(types.SimpleNamespace(mapping=lambda:native),))
        return raw, dict(mode='off',overlay_relative=M.OVERLAY), receipt

    def test_independent_validator_accepts_actual_common_binding(self):
        raw, config, receipt = self.verification_fixture()
        V.validate_native_source_binding(self.root,self.refs,raw,config,receipt)

    def test_independent_validator_rejects_old_actual_refs_even_if_locked(self):
        for field in ('actual_source_ref','original_run_source_ref'):
            raw, config, receipt = self.verification_fixture()
            raw['native_source_binding'][field] = self.refs[M.LEGACY_COLLECTOR_REACTOR_KEY]
            with self.assertRaisesRegex(ValueError,'common C4 owner'):
                V.validate_native_source_binding(self.root,self.refs,raw,config,receipt)

    def test_independent_validator_checks_bridge_owner_and_actual_identity(self):
        for key,value in (('original_run_code_verified',False),('max_accepted_parents',64),
                ('max_accepted_parents',True),('bridge_is_none',False),('same_requested_bridge',False),
                ('bridge_policy_mode','interference'),('requested_mode','on')):
            raw, config, receipt = self.verification_fixture()
            raw['native_source_binding'][key] = value
            with self.assertRaises(ValueError):
                V.validate_native_source_binding(self.root,self.refs,raw,config,receipt)
        raw, config, receipt = self.verification_fixture()
        raw['verified_runtime_identity']['source_refs'] = [self.refs[M.LEGACY_COLLECTOR_REACTOR_KEY]]
        with self.assertRaisesRegex(ValueError,'identity helper'):
            V.validate_native_source_binding(self.root,self.refs,raw,config,receipt)

    def test_independent_validator_rejects_legacy_duplicate_and_omitted_module_rows(self):
        for variant in ('legacy','duplicate','missing'):
            raw, config, receipt = self.verification_fixture()
            rows = raw['native_source_binding']['loaded_native_modules']
            if variant == 'legacy': rows[0]['source_ref'] = self.refs[M.LEGACY_COLLECTOR_REACTOR_KEY]
            if variant == 'duplicate': rows.append(deepcopy(rows[0]))
            if variant == 'missing': rows.clear()
            with self.assertRaises(ValueError):
                V.validate_native_source_binding(self.root,self.refs,raw,config,receipt)


if __name__ == '__main__': unittest.main(verbosity=2)
