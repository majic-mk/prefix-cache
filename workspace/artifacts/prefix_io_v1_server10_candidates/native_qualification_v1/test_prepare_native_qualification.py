import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('native_cpu_config', Path(__file__).with_name('prepare_native_qualification.py'))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class NativeOuterLimits(unittest.TestCase):
    def ledger(self):
        return dict(gpu_wall_seconds=18000.0, active_reservation=None, events=[])

    def success(self):
        return dict(label=M.NAMES['off'], exit=0, child_exit=0, timed_out=False,
                    error=None, session_drained=True, session_members_before_cleanup=[],
                    session_members_after_cleanup=[])

    def test_exact_two_original_entry_commands(self):
        for mode in ('off', 'shadow'):
            cmd = M.arguments(mode)
            self.assertEqual(cmd[2], M.SCRIPT)
            self.assertIn(M.NAMES[mode], cmd)
            self.assertEqual(cmd[-1], '--launch')
        with self.assertRaises(ValueError):
            M.arguments('full')

    def test_no_shadow_without_successful_off(self):
        ledger = self.ledger()
        M.check_outer_ledger(ledger, 'off')
        with self.assertRaises(ValueError):
            M.check_outer_ledger(ledger, 'shadow')
        ledger['events'] = [self.success()]
        M.check_outer_ledger(ledger, 'shadow')
        for key, bad in (('exit', 1), ('session_drained', False), ('timed_out', True),
                         ('session_members_before_cleanup', [42]), ('child_exit', False)):
            broken = copy.deepcopy(ledger)
            broken['events'][0][key] = bad
            with self.assertRaises(ValueError):
                M.check_outer_ledger(broken, 'shadow')

    def test_no_retries_or_active_reservation(self):
        ledger = self.ledger()
        ledger['events'] = [self.success()]
        with self.assertRaises(ValueError):
            M.check_outer_ledger(ledger, 'off')
        ledger['active_reservation'] = {'id': 'active'}
        with self.assertRaises(ValueError):
            M.check_outer_ledger(ledger, 'shadow')

    def test_budget_not_reset(self):
        ledger = self.ledger()
        ledger['gpu_wall_seconds'] = 28401.0
        with self.assertRaises(ValueError):
            M.check_outer_ledger(ledger, 'off')
        ledger['events'] = [self.success()]
        M.check_outer_ledger(ledger, 'shadow')
        ledger['gpu_wall_seconds'] = 28601.0
        with self.assertRaises(ValueError):
            M.check_outer_ledger(ledger, 'shadow')


if __name__ == '__main__':
    unittest.main(verbosity=2)
