"""Run unchanged original pure policy/bridge tests against this source overlay."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
DEFAULT = HERE.parents[2] / 'artifacts/prefix_io_v1_server08_p4_cpu_extended_20261001/verified-v3/source'
parser = argparse.ArgumentParser()
parser.add_argument('--project-source', type=Path, default=DEFAULT)
parser.add_argument('--test', action='append', default=[])
args = parser.parse_args()
base = args.project_source.resolve()
control = 'third_party/work/prefix-io-p4-02-cpu/src'
sys.path.insert(0, str(base / control))
import prefix_io_control
prefix_io_control.__path__ = [str(HERE / 'source' / control / 'prefix_io_control'),
                             *prefix_io_control.__path__]

class NoBackend:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'vllm', 'py_kvcache', 'cupy', 'cuda', 'numpy'):
            raise RuntimeError('backend import forbidden in pure CPU regression: ' + fullname)
sys.meta_path.insert(0, NoBackend())

tests = args.test or ['tests/prefix_io_v1_p4_policy/test_p4_policy.py',
                     'tests/prefix_io_v1_p4_bridge/test_value_abi_independent.py']
suite = unittest.TestSuite()
refs = []
for index, relative in enumerate(tests):
    path = (base / relative).resolve()
    if not path.is_relative_to(base): raise ValueError('original test outside source tree')
    data = path.read_bytes()
    refs.append(dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    spec = importlib.util.spec_from_file_location('_unchanged_original_cpu_' + str(index), path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
from prefix_io_control import p4_policy, p4_bridge
for module in (p4_policy, p4_bridge):
    if not Path(module.__file__).resolve().is_relative_to(HERE):
        raise ValueError('regression did not load candidate overlay')
result = unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps(dict(scope='unchanged_original_cpu_policy_bridge_regression',
    tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
    skipped=len(result.skipped), passed=result.wasSuccessful(), original_test_refs=refs,
    actual_overlay_modules=[str(Path(m.__file__).resolve()) for m in (p4_policy,p4_bridge)],
    gpu_operations=0), sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() else 1)
