"""CPU mock call contracts only; none of these objects are GPU receipts."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_CPU_mock_only_formal_parent_producer', HERE / 'close_formal_peer_cpu.py')
P = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = P
spec.loader.exec_module(P)


class MockContext:
    """In-memory source/guard/checker fixtures. No GPU qualification or frames."""
    def __init__(self, root, *, role=('development', 'off', 'U')):
        self.root = root.resolve()
        self.output = 'artifacts/prefix_io_v1/CPU_MOCK_ONLY/CPU_MOCK_PARENT_OUTPUT.json'
        (self.root / self.output).parent.mkdir(parents=True)
        self.ledger = self.root / 'CPU_MOCK_ONLY_BUDGET.json'
        self.calls = []
        phase, mode, arm = role
        self.refs = {}
        self.docs = {}
        def leaf(name, document):
            row = dict(path='CPU_MOCK_ONLY/' + name + '.json', bytes=0, sha256='0' * 64)
            self.refs[row['path']] = row
            self.docs[row['path']] = document
            return row
        self.config = dict(phase=phase, mode=mode, arm=arm)
        self.config_ref = leaf('CPU_MOCK_CONFIG', self.config)
        self.child = dict(phase=phase, mode=mode, arm=arm, os_session_drained=False,
                          actual_run_config_ref=self.config_ref, guard_reservation_id='CPU_MOCK_NOT_RESERVATION',
                          nested={'unchanged': [1, {'CPU_TEST_ONLY': True}]},
                          CPU_TEST_MOCK_ONLY=True, NOT_ACTUAL_GPU_EVIDENCE=True)
        self.child_ref = leaf('CPU_MOCK_CHILD_IN_MEMORY_ONLY', self.child)
        self.guard = dict(exit=0, child_exit=0, gpu_job_attempted=True, timed_out=False,
                          interrupted_signal=None, error=None, session_drained=True,
                          session_members_before_cleanup=[], session_members_after_cleanup=[],
                          reservation_id='CPU_MOCK_NOT_RESERVATION', session_id=987654321,
                          CPU_TEST_MOCK_ONLY=True, NOT_ACTUAL_GPU_EVIDENCE=True)
        self.guard_ref = leaf('CPU_MOCK_GUARD_IN_MEMORY_ONLY', self.guard)
        pair = {'U': {'engine': {'kv_transfer_config': {'kv_connector_extra_config': {}}}},
                'I': {'engine': {'kv_transfer_config': {'kv_connector_extra_config': {}}}}}
        for name in ('U', 'I'):
            path = self.root / ('experiments/prefix_io_v1/runs/CPU_MOCK_ONLY_NAMESPACE_' + name)
            path.mkdir(parents=True)
            pair[name]['engine']['kv_transfer_config']['kv_connector_extra_config']['shared_storage_path'] = str(path)
        self.config['pair_config_ref'] = leaf('CPU_MOCK_PAIR_IN_MEMORY_ONLY',
            dict(schema='strong_native_u_i_cpu_configuration_v1', configurations=pair))
        self.config['workload_ref'] = leaf('CPU_MOCK_MANIFEST_IN_MEMORY_ONLY',
            dict(CPU_TEST_MOCK_ONLY=True, NOT_ACTUAL_GPU_EVIDENCE=True))
        self.lock_ref = leaf('CPU_MOCK_SOURCE_LOCK', {})
        self.budget = dict(gpu_wall_seconds=0, active_reservation=None, events=[deepcopy(self.guard)],
                           CPU_TEST_MOCK_ONLY=True, NOT_ACTUAL_GPU_EVIDENCE=True)
        self.save_budget()
        @contextmanager
        def lock(root, **kwargs):
            self.calls.append('READ_ONLY_BUDGET_LOCK_MOCK')
            yield self.ledger
        self.lock = lock
        def source_rows(*args, **kwargs):
            self.calls.append(('SOURCE_ROWS_MOCK', kwargs))
            return self.refs
        def read(path):
            self.calls.append(('READ_SOURCE_MOCK', path))
            return deepcopy(self.docs[path])
        def check(root, row):
            self.calls.append(('CHECK_SOURCE_MOCK', row))
            return row['path']
        def write(path, value):
            self.calls.append('ONE_CPU_MOCK_PARENT_APPEND')
            with path.open('x', encoding='utf-8') as stream:
                json.dump(value, stream)
        self.driver = NS(MAX_GPU_SECONDS=28800, source_rows=source_rows, read=read, check_ref=check,
                         safe=P.safe, new_json=write, ref=P.actual_ref)
        self.probe = lambda sid: []
        def checker(parent, root, config, refs, pair, **kwargs):
            self.calls.append(('PURE_PEER_CHECKER_MOCK_NO_QUALIFICATION', deepcopy(parent), kwargs))
            return parent
        self.checker = checker

    def save_budget(self):
        self.ledger.write_text(json.dumps(self.budget), encoding='utf-8')

    def run(self):
        return P.produce_parent(self.root, source_lock_ref=self.lock_ref, child_ref=self.child_ref,
            guard_ref=self.guard_ref, config_ref=self.config_ref, output_relative=self.output,
            driver=self.driver, peer_checker=self.checker, session_probe=self.probe, budget_lock=self.lock)

    def no_parent(self):
        return not (self.root / self.output).exists()


class CloseFormalPeerCPU(unittest.TestCase):
    def test_parent_changes_only_OS_bit_and_three_closure_fields_then_single_append(self):
        with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_parent_contract_') as temp:
            c = MockContext(Path(temp))
            before_child, before_budget = deepcopy(c.child), c.ledger.read_bytes()
            result = c.run()
            parent = P.parse((c.root / c.output).read_bytes())
            self.assertEqual(c.child, before_child)
            self.assertEqual(set(parent) - set(c.child),
                             {'completed_guard_ref', 'child_result_ref', 'formal_parent_closure_schema'})
            self.assertEqual([key for key in c.child if c.child[key] != parent[key]], ['os_session_drained'])
            self.assertTrue(parent['CPU_TEST_MOCK_ONLY'])
            self.assertTrue(parent['NOT_ACTUAL_GPU_EVIDENCE'])
            self.assertEqual(c.calls.count('ONE_CPU_MOCK_PARENT_APPEND'), 1)
            self.assertEqual(c.ledger.read_bytes(), before_budget)
            self.assertEqual(result['actual_GPU_operations'], 0)
            self.assertFalse(result['gpu_eligible'])
            self.assertFalse(result['formal_effect_qualified'])
            self.assertTrue(result['output_must_enter_later_source_lock_before_consumption'])

    def test_both_effect_roles_delegate_same_pure_complete_peer_verifier(self):
        for role in [('effect', 'off', 'U'), ('effect', 'on', 'I')]:
            with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_effect_peer_') as temp:
                c = MockContext(Path(temp), role=role)
                c.run()
                call = next(item for item in c.calls if isinstance(item, tuple) and
                            item[0] == 'PURE_PEER_CHECKER_MOCK_NO_QUALIFICATION')
                self.assertEqual(call[2]['partition'], 'evaluation')
                self.assertEqual(call[2]['peer_arm'], role[2])

    def test_missing_raw_source_is_rejected_before_checker_or_parent_write(self):
        with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_missing_raw_') as temp:
            c = MockContext(Path(temp))
            c.refs.pop(c.child_ref['path'])
            with self.assertRaisesRegex(ValueError, 'current frozen source closure'):
                c.run()
            self.assertTrue(c.no_parent())
            self.assertFalse(any(isinstance(item, tuple) and item[0].startswith('PURE_PEER') for item in c.calls))

    def test_original_checker_raw_or_CUDA_rejection_never_writes_draft_parent(self):
        with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_verifier_reject_') as temp:
            c = MockContext(Path(temp))
            def reject(*args, **kwargs):
                raise ValueError('CPU mock: original full capture/raw evidence missing')
            c.checker = reject
            with self.assertRaisesRegex(ValueError, 'full capture/raw evidence missing'):
                c.run()
            self.assertTrue(c.no_parent())
            self.assertNotIn('ONE_CPU_MOCK_PARENT_APPEND', c.calls)

    def test_guard_not_naturally_ended_never_checks_peer_or_writes_parent(self):
        cases = [('exit', 1), ('child_exit', -15), ('timed_out', True), ('interrupted_signal', 15),
                 ('error', 'CPU_MOCK_ERROR'), ('session_drained', False),
                 ('session_members_after_cleanup', [{'CPU_TEST_MOCK_ONLY': True}])]
        for key, value in cases:
            with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_bad_guard_') as temp:
                c = MockContext(Path(temp))
                c.guard[key] = value
                with self.assertRaisesRegex(ValueError, 'guard ended naturally and drained'):
                    c.run()
                self.assertTrue(c.no_parent())

    def test_busy_budget_rejects_before_source_replay(self):
        with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_busy_budget_') as temp:
            c = MockContext(Path(temp))
            c.budget['active_reservation'] = {'CPU_TEST_MOCK_ONLY': True}
            c.save_budget()
            with self.assertRaisesRegex(ValueError, 'budget must be idle'):
                c.run()
            self.assertTrue(c.no_parent())
            self.assertFalse(any(isinstance(item, tuple) and item[0] == 'SOURCE_ROWS_MOCK' for item in c.calls))

    def test_missing_or_duplicate_budget_event_rejects_without_modifying_budget(self):
        for events in ([], [1], ['CPU_MOCK_NON_EVENT']):
            with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_bad_budget_event_') as temp:
                c = MockContext(Path(temp))
                c.budget['events'] = events
                c.save_budget()
                before = c.ledger.read_bytes()
                with self.assertRaisesRegex(ValueError, 'unique actual completed guard event'):
                    c.run()
                self.assertTrue(c.no_parent())
                self.assertEqual(c.ledger.read_bytes(), before)
        with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_duplicate_event_') as temp:
            c = MockContext(Path(temp))
            c.budget['events'].append(deepcopy(c.guard))
            c.save_budget()
            with self.assertRaisesRegex(ValueError, 'unique actual completed guard event'):
                c.run()
            self.assertTrue(c.no_parent())

    def test_OS_session_busy_before_and_after_verification_never_writes(self):
        for late in (False, True):
            with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_session_busy_') as temp:
                c = MockContext(Path(temp))
                values = iter([[], [{'CPU_TEST_MOCK_ONLY': True}]] if late else [[{'CPU_TEST_MOCK_ONLY': True}]])
                c.probe = lambda sid: next(values)
                with self.assertRaises(ValueError):
                    c.run()
                self.assertTrue(c.no_parent())

    def test_budget_bytes_changed_during_verifier_refuses_append(self):
        with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_stale_ledger_') as temp:
            c = MockContext(Path(temp))
            def mutate_for_test(parent, *args, **kwargs):
                c.budget['gpu_wall_seconds'] = 1
                c.save_budget()
                return parent
            c.checker = mutate_for_test
            with self.assertRaisesRegex(ValueError, 'budget bytes changed'):
                c.run()
            self.assertTrue(c.no_parent())

    def test_parent_checker_cannot_rewrite_child_or_return_different_document(self):
        for inplace in (False, True):
            with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_rewritten_parent_') as temp:
                c = MockContext(Path(temp))
                def rewrite(parent, *args, **kwargs):
                    if inplace:
                        parent['nested']['unchanged'][0] = 123
                        return parent
                    return dict(parent, invented_source='CPU_MOCK_REJECT')
                c.checker = rewrite
                with self.assertRaisesRegex(ValueError, 'unchanged complete parent'):
                    c.run()
                self.assertTrue(c.no_parent())

    def test_already_closed_child_or_mismatched_config_ref_rejects(self):
        for change in ('closed', 'closure_field', 'config'):
            with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_bad_child_') as temp:
                c = MockContext(Path(temp))
                if change == 'closed':
                    c.child['os_session_drained'] = True
                elif change == 'closure_field':
                    c.child['child_result_ref'] = c.child_ref
                else:
                    c.child['actual_run_config_ref'] = {'CPU_MOCK_WRONG': True}
                with self.assertRaises(ValueError):
                    c.run()
                self.assertTrue(c.no_parent())

    def test_no_output_overwrite_or_directory_creation(self):
        for exists in (False, True):
            with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_output_scope_') as temp:
                c = MockContext(Path(temp))
                if exists:
                    target = c.root / c.output
                    target.write_text('CPU_MOCK_EXISTING_DO_NOT_CHANGE', encoding='utf-8')
                else:
                    c.output = 'artifacts/prefix_io_v1/CPU_MOCK_NEVER_CREATE/parent.json'
                with self.assertRaisesRegex(ValueError, 'fresh bounded parent output'):
                    c.run()
                if exists:
                    self.assertEqual(target.read_text(), 'CPU_MOCK_EXISTING_DO_NOT_CHANGE')
                else:
                    self.assertFalse((c.root / c.output).parent.exists())

    def test_producer_output_cannot_be_part_of_input_lock(self):
        with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_ref_cycle_') as temp:
            c = MockContext(Path(temp))
            c.refs[c.output] = {'CPU_MOCK_NOT_ACTUAL_OUTPUT_REF': True}
            with self.assertRaisesRegex(ValueError, 'no draft/ref cycle'):
                c.run()
            self.assertTrue(c.no_parent())

    def test_unknown_roles_and_absent_backend_imports(self):
        for role in [('qualification', 'off', 'U'), ('development', 'shadow', 'I')]:
            with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_wrong_role_') as temp:
                c = MockContext(Path(temp), role=role)
                with self.assertRaisesRegex(ValueError, 'bounded development Uoff or effect'):
                    c.run()
                self.assertTrue(c.no_parent())
        self.assertFalse(any(name == 'torch' or name.startswith(('torch.', 'vllm', 'py_kvcache')) for name in sys.modules))

    def test_actual_proc_stat_parser_does_not_confuse_parentheses_with_session_fields(self):
        self.assertEqual(P.proc_session_fields('123 (CPU mock ) command(with)name) R 12 15 99 0 0'), ('R', 99))
        for raw in ('CPU_MOCK_NO_STAT', '123 (CPU) R 1 2', '123 (CPU) R 1 2 not_a_session'):
            with self.assertRaisesRegex(ValueError, 'proc session stat shape'):
                P.proc_session_fields(raw)

    def test_original_budget_lock_is_existing_read_only_shared_nonblocking(self):
        with tempfile.TemporaryDirectory(prefix='CPU_MOCK_ONLY_budget_lock_') as temp:
            root = Path(temp)
            ledger = root / 'CPU_MOCK_ONLY_BUDGET.json'
            ledger.write_text('CPU_MOCK_ONLY_NOT_GPU_LEDGER', encoding='utf-8')
            lock_path = ledger.with_suffix('.lock')
            lock_path.write_bytes(b'CPU_MOCK_EXISTING_LOCK_DONT_TOUCH')
            flags = []
            fcntl_mock = NS(LOCK_SH=1, LOCK_NB=4, LOCK_UN=8,
                            flock=lambda stream, value: flags.append((stream.mode, value)))
            driver = NS(LEDGER='CPU_MOCK_ONLY_BUDGET.json', safe=P.safe)
            with patch.object(P, 'os', NS(name='posix')), patch.dict(sys.modules, {'fcntl': fcntl_mock}):
                with P.original_budget_read_lock(root, driver=driver) as path:
                    self.assertEqual(path, ledger.resolve())
            self.assertEqual(flags, [('rb', 5), ('rb', 8)])
            self.assertEqual(lock_path.read_bytes(), b'CPU_MOCK_EXISTING_LOCK_DONT_TOUCH')


if __name__ == '__main__':
    unittest.main(verbosity=2)
