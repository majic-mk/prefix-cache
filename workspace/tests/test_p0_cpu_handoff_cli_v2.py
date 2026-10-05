"""Argument/error boundaries only; archive semantics have independent real tests."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_decoupled_v2_cpu_handoff as cli


class CPUHandoffCliTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.refs = self.root / 'refs.json'; self.refs.write_text('{}')
        self.archive = self.root / 'new.zip'
        self.common = ['--archive', str(self.archive), '--max-archive-bytes', '100000',
                       '--max-uncompressed-bytes', '200000', '--max-members', '100']
        self.metadata = dict(base_commit='CPU-TEST-ONLY',worktree_digest='CPU-TEST-ONLY',
                             members={},excluded_worktree_members={})

    def test_build_forwards_explicit_limits_never_deploys(self):
        out = io.StringIO()
        with patch.object(cli, 'build_cpu_handoff', return_value=dict(
                archive_sha256='a'*64, metadata=self.metadata)) as build, contextlib.redirect_stdout(out):
            result = cli.main(['build', '--workspace', str(self.root), '--evidence', str(self.refs), *self.common])
        self.assertEqual(result, 0)
        self.assertEqual(build.call_args.kwargs['max_members'], 100)
        self.assertEqual(build.call_args.kwargs['evidence_refs'], {})
        value = json.loads(out.getvalue())
        self.assertFalse(value['remote_deployment_performed'])
        self.assertFalse(value['gpu_execution_allowed'])

    def test_verify_never_rebuilds_or_rents(self):
        with patch.object(cli, 'verify_cpu_handoff_archive', return_value=self.metadata) as verify, \
                patch.object(cli, 'build_cpu_handoff') as build, contextlib.redirect_stdout(io.StringIO()):
            cli.main(['verify', '--sha256', 'a'*64, *self.common])
        verify.assert_called_once(); build.assert_not_called()

    def test_missing_evidence_and_duplicate_keys_not_replaced(self):
        self.refs.unlink()
        args = ['build', '--workspace', str(self.root), '--evidence', str(self.refs), *self.common]
        with self.assertRaises(ValueError):
            cli.main(args)
        self.refs.write_text('{"a":1,"a":2}')
        with self.assertRaises(ValueError):
            cli.main(args)
        self.assertFalse(self.archive.exists())

    def test_no_defaults_for_limits_and_no_execute_option(self):
        for args in (['verify', '--archive', str(self.archive), '--sha256', 'a'*64],
                     ['verify', '--sha256', 'a'*64, *self.common, '--execute']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(args)


if __name__ == '__main__':
    unittest.main()
