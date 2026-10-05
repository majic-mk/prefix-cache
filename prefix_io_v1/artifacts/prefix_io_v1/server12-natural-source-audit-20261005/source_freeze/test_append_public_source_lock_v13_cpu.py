"""Small CPU fixtures only: no server/model assets or GPU receipts are used."""
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'append_public_source_lock_v13.py'
spec = importlib.util.spec_from_file_location('_v13_cpu_fixture_module', SOURCE)
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


class Fixture:
    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix='v13_CPU_fixture_')
        self.root = Path(self.temp.name).resolve()
        self.stack = ExitStack()
        self.stack.enter_context(patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        self.put('source/original.py', b'# CPU fixture only\n')
        self.put('source/layout.json', b'{"CPU_fixture":true}\n')
        self.put('inputs/raw.jsonl', b'{"prompt":"CPU synthetic fixture"}\n')
        self.put('inputs/request.json', b'{"CPU_fixture":true,"GPU_eligible":false}\n')
        self.put(F.SELF, SOURCE.read_bytes())
        self.put(F.LEDGER, b'{"active_reservation":null,"CPU_fixture":true}\n')
        self.ledger = (self.root / F.LEDGER).read_bytes()
        self.ancestor = dict(schema='strong_trace_source_lock_v1',
                             files=[F.file_ref(self.root, name) for name in ['source/original.py', 'source/layout.json']],
                             CPU_fixture=True)
        self.write_ancestor()
        self.manifest = F.AUDIT + '/addition_manifest_CPU_fixture.json'
        self.additions = ['source/original.py', 'inputs/raw.jsonl', 'inputs/request.json']
        self.write_manifest()
        self.lock = F.AUDIT + '/LOCK_CPU_fixture.json'
        self.proof = F.AUDIT + '/PROOF_CPU_fixture.json'
        self.stack.enter_context(patch.object(F, 'ANCESTOR_COUNT', 2))
        self.stack.enter_context(patch.object(F, 'LEDGER_SHA', hashlib.sha256(self.ledger).hexdigest()))
        self.stack.enter_context(patch.object(F, '__file__', str(self.root / F.SELF)))
        return self

    def put(self, name, raw):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def write_ancestor(self):
        raw = F.serialized(self.ancestor)
        self.put(F.ANCESTOR, raw)
        if hasattr(self, 'ancestor_patch'):
            self.ancestor_patch.stop()
        self.ancestor_patch = patch.object(F, 'ANCESTOR_SHA', hashlib.sha256(raw).hexdigest())
        self.ancestor_patch.start()

    def write_manifest(self):
        self.put(self.manifest, F.serialized(dict(schema=F.MANIFEST_SCHEMA, files=self.additions,
                                                note='CPU fixtures are never actual source or GPU evidence')))

    def run(self, **overrides):
        values = dict(addition_manifest=self.manifest, source_lock_output=self.lock, proof_output=self.proof)
        values.update(overrides)
        return F.freeze(self.root, **values)

    def __exit__(self, *args):
        self.ancestor_patch.stop()
        self.stack.close()
        # This recursive fixture cleanup targets only the explicit temp directory
        # created above, never the workspace or inherited real project files.
        assert self.root == Path(self.temp.name).resolve() and self.root.name.startswith('v13_CPU_fixture_')
        self.temp.cleanup()


class AppendTests(unittest.TestCase):
    def assert_no_output(self, fixture):
        self.assertFalse((fixture.root / fixture.lock).exists())
        self.assertFalse((fixture.root / fixture.proof).exists())

    def test_actual_CPU_fixture_inheritance_overlap_and_CLI(self):
        with Fixture() as x:
            with patch('builtins.print'):
                F.main(['--project-root', str(x.root), '--addition-manifest', x.manifest,
                        '--source-lock-output', x.lock, '--proof-output', x.proof])
            lock = json.loads((x.root / x.lock).read_bytes())
            proof = json.loads((x.root / x.proof).read_bytes())
            rows = {row['path']: row for row in lock['files']}
            self.assertEqual(len(rows), len(lock['files']))
            for row in x.ancestor['files']:
                self.assertEqual(rows[row['path']], row)
            for name in [F.ANCESTOR, F.SELF, x.manifest, *x.additions]:
                self.assertEqual(rows[name], F.file_ref(x.root, name))
            self.assertEqual(proof['source_lock_ref'], F.file_ref(x.root, x.lock))
            self.assertEqual(proof['inherited_source_count'], 2)
            self.assertTrue(proof['strict_all_ancestor_rows_preserved'])
            self.assertFalse(lock['GPU_launch_allowed'])
            self.assertFalse(proof['GPU_qualification'])
            self.assertEqual(proof['actual_gpu_runs'], 0)
            self.assertEqual((x.root / F.LEDGER).read_bytes(), x.ledger)

    def test_inherited_bytes_and_duplicate_ancestor_rejected(self):
        with Fixture() as x:
            x.put('source/original.py', b'# changed CPU fixture\n')
            with self.assertRaisesRegex(ValueError, 'inherited actual byte drift'):
                x.run()
            self.assert_no_output(x)
        with Fixture() as x:
            x.ancestor['files'][1] = x.ancestor['files'][0]
            x.write_ancestor()
            with self.assertRaisesRegex(ValueError, 'duplicate inherited'):
                x.run()
            self.assert_no_output(x)

    def test_explicit_addition_list_rejects_duplicates_traversal_partial_and_directory(self):
        cases = [('source/original.py', 'source/original.py'), ('../escape.json',),
                 ('inputs/unclosed.partial',), ('inputs',)]
        for names in cases:
            with self.subTest(names=names), Fixture() as x:
                x.additions = list(names)
                x.write_manifest()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    x.run()
                self.assert_no_output(x)

    def test_existing_missing_parent_outside_audit_and_same_outputs_rejected(self):
        for option in ['existing', 'missing-parent', 'outside-audit', 'same']:
            with self.subTest(option=option), Fixture() as x:
                if option == 'existing':
                    x.put(x.lock, b'preserve existing bytes')
                    kwargs = {}
                elif option == 'missing-parent':
                    kwargs = {'source_lock_output': F.AUDIT + '/not-created/lock.json'}
                elif option == 'outside-audit':
                    kwargs = {'source_lock_output': 'inputs/lock.json'}
                else:
                    kwargs = {'proof_output': x.lock}
                with self.assertRaises(ValueError):
                    x.run(**kwargs)
                self.assertFalse((x.root / x.proof).exists())
                if option == 'existing':
                    self.assertEqual((x.root / x.lock).read_bytes(), b'preserve existing bytes')

    def test_late_source_and_ledger_drift_rejected_before_output(self):
        for kind in ['source', 'ledger']:
            with self.subTest(kind=kind), Fixture() as x:
                actual = F.capture_file_ref

                def drift(root, name):
                    result = actual(root, name)
                    if kind == 'source' and name == 'source/layout.json':
                        x.put(name, b'{"changed_CPU_fixture":true}\n')
                    if kind == 'ledger' and name == F.SELF:
                        x.put(F.LEDGER, b'{"active_reservation":null,"changed_CPU_fixture":true}\n')
                    return result

                with patch.object(F, 'capture_file_ref', drift):
                    with self.assertRaisesRegex(ValueError, 'drift after hashing|ledger bytes unchanged'):
                        x.run()
                self.assert_no_output(x)

    def test_symlink_addition_rejected(self):
        with Fixture() as x:
            link = x.root / 'inputs/link.json'
            try:
                link.symlink_to(x.root / 'inputs/request.json')
            except OSError as error:
                self.skipTest('CPU fixture host cannot create a symlink: ' + str(error))
            x.additions.append('inputs/link.json')
            x.write_manifest()
            with self.assertRaisesRegex(ValueError, 'symlink'):
                x.run()
            self.assert_no_output(x)

    def test_active_ledger_rejected(self):
        with Fixture() as x:
            raw = b'{"active_reservation":"CPU_fixture_not_real_GPU"}\n'
            x.put(F.LEDGER, raw)
            with patch.object(F, 'LEDGER_SHA', hashlib.sha256(raw).hexdigest()):
                with self.assertRaisesRegex(ValueError, 'idle original GPU ledger'):
                    x.run()
            self.assert_no_output(x)


if __name__ == '__main__':
    unittest.main(verbosity=2)
