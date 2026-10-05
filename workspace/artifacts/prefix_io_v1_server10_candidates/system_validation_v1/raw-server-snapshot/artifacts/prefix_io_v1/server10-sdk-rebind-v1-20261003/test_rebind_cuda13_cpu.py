"""Critical CPU receipt boundaries; no CUDA compilation or library loading."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('rebind', Path(__file__).with_name('rebind_cuda13_cpu.py'))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class CPURebindBoundaries(unittest.TestCase):
    def test_same_size_drift_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder).resolve() / 'asset'
            path.write_bytes(b'original')
            pin = M.file_ref(path)
            M.verify_ref(pin)
            path.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError, 'drift'):
                M.verify_ref(pin)

    def test_boolean_size_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder).resolve() / 'asset'
            path.write_bytes(b'x')
            pin = M.file_ref(path)
            pin['bytes'] = True
            with self.assertRaises(ValueError):
                M.verify_ref(pin)

    def test_receipt_creation_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'receipt.json'
            M.write_json(path, {'real': True})
            with self.assertRaises(FileExistsError):
                M.write_json(path, {'real': False})
            self.assertEqual(json.loads(path.read_text()), {'real': True})

    def test_cpu_command_environment_excludes_inherited_injection(self):
        with patch.dict(M.os.environ, {'LD_PRELOAD': '/bad.so', 'LD_LIBRARY_PATH': '/stubs',
                                      'NVCC_PREPEND_FLAGS': 'bad', 'CUDA_VISIBLE_DEVICES': '0'}):
            env = M.cpu_environment()
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
            for name in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'NVCC_PREPEND_FLAGS'):
                self.assertNotIn(name, env)


if __name__ == '__main__':
    unittest.main(verbosity=2)
