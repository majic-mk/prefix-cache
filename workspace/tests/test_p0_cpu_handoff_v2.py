"""CPU-only archive transport tests; fixture successes are not model evidence."""
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from probekv.p0_cpu_handoff_v2 import (
    build_cpu_handoff, verify_cpu_handoff_archive, PACKET, PACKET_MEMBERS,
    EVIDENCE_KEYS, _safe_name, _code_allowed, _excluded_head_member,
)
from probekv.p0_stage_readiness_v2 import _BASE_REQUIRED, _STAGE_REQUIRED


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CPUHandoffFixture:
    """Reusable tiny real-file evidence fixture; not a qualification record."""
    def __init__(self, root):
        self.root = root
        self.workspace = root / 'workspace'; self.workspace.mkdir()
        self.evidence = root / 'evidence'; self.evidence.mkdir()
        self.output = root / 'handoff.zip'
        self.head = 'a' * 40
        self.files = {}
        names = set(_BASE_REQUIRED + sum(_STAGE_REQUIRED.values(), ()))
        names.update(('docs/中文.md', 'setup.py', 'src/probekv/extra.py'))
        for name in names:
            path = self.workspace / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'# CPU fixture, never native qualification\n')
            self.files[name] = sha(path)
        self.packet_checks = []
        for name in PACKET_MEMBERS:
            path = self.workspace / PACKET / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('fixture ' + name).encode('utf-8'))
            self.packet_checks.append(dict(file=name, expected=sha(path), actual=sha(path), passed=True))
        (self.workspace / PACKET / 'checksums.sha256').write_text('\n'.join(
            r['expected'] + '  ' + r['file'] for r in self.packet_checks), encoding='utf-8')
        self.log = self.evidence / 'cpu.log'; self.log.write_text('fixture test passed\n')
        self.checks = []
        for name in ('compileall.log', 'contract_validator.log', 'diff_check.log', 'packet_reference_only.log'):
            path = self.evidence / name; path.write_text('fixture command passed\n')
            self.checks.append(dict(log=name, log_sha256=sha(path), returncode=0, passed=True))
        self.refs = {}
        self.flush()

    def put(self, key, value):
        path = self.evidence / (key + '.json')
        path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        self.refs[key] = dict(path=str(path), sha256=sha(path))

    def flush(self):
        digest = hashlib.sha256(json.dumps(self.files, ensure_ascii=False, sort_keys=True,
                                          separators=(',', ':')).encode()).hexdigest()
        self.worktree = dict(base_commit=self.head, branch='codex/fixture', digest=digest, files=self.files)
        self.report = dict(status='CPU_CHECKS_PASS_NATIVE_PENDING', base_commit=self.head,
            worktree_digest=digest, tests_run=1, passed=1, skipped=0, failures=0, errors=0,
            locked_test_accessed=False, gpu_actions_executed=0, cuda_visible_devices='')
        self.tests = [dict(test_id='fixture.test_cpu', observed='PASS', passed=True,
            runtime_commit=self.head, runtime_worktree_digest=digest,
            evidence_path=str(self.log), evidence_hash=sha(self.log))]
        self.commands = dict(repository_checks=self.checks[:3], packet_reference=self.checks[3])
        for name, value in (('cpu_report', self.report), ('worktree_files', self.worktree),
                ('test_results', self.tests), ('commands', self.commands), ('package_checksums', self.packet_checks)):
            self.put(name, value)

    def inventory(self, unused):
        return self.head, set(self.files), set(self.files)


class CPUHandoffTests(unittest.TestCase):
    limits = dict(max_archive_bytes=2 * 1024 * 1024, max_uncompressed_bytes=4 * 1024 * 1024, max_members=200)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.fixture = CPUHandoffFixture(Path(temporary.name))
        self.mock_inventory = patch('probekv.p0_cpu_handoff_v2._git_inventory', side_effect=self.fixture.inventory)
        self.mock_inventory.start(); self.addCleanup(self.mock_inventory.stop)

    def build(self, **changes):
        args = dict(self.limits); args.update(changes)
        return build_cpu_handoff(self.fixture.workspace, evidence_refs=self.fixture.refs,
                                 output_archive=self.fixture.output, **args)

    def verify(self, path=None, **changes):
        path = path or self.fixture.output
        args = dict(self.limits); args.update(changes)
        return verify_cpu_handoff_archive(path, expected_sha256=sha(path), **args)

    def rewrite(self, mutate=None, extra=None):
        path = self.fixture.root / 'rewritten.zip'
        with zipfile.ZipFile(self.fixture.output) as original, zipfile.ZipFile(path, 'w') as target:
            for info in original.infolist():
                data = original.read(info.filename)
                if mutate:
                    info, data = mutate(info, data)
                target.writestr(info, data)
            if extra:
                info = zipfile.ZipInfo(extra); info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                target.writestr(info, b'extra')
        return path

    def test_real_files_roundtrip_preserves_dirty_identity_and_denies_gpu(self):
        result = self.build(); metadata = self.verify()
        self.assertEqual(metadata, result['metadata'])
        self.assertEqual(metadata['base_commit'], self.fixture.head)
        self.assertEqual(metadata['worktree_digest'], self.fixture.worktree['digest'])
        self.assertTrue(metadata['commit_is_not_snapshot'])
        self.assertFalse(metadata['gpu_execution_allowed'])
        self.assertFalse(metadata['P1_M_execution_allowed'])
        self.assertNotIn('handoff.json', metadata['members'])
        self.assertEqual(len(metadata['evidence_index']), 5)
        with zipfile.ZipFile(self.fixture.output) as archive:
            self.assertEqual(archive.read('workspace/docs/中文.md'),
                             (self.fixture.workspace / 'docs/中文.md').read_bytes())

    def test_output_overwrite_is_rejected_without_changing_bytes(self):
        self.build(); old = sha(self.fixture.output)
        with self.assertRaises(FileExistsError): self.build()
        self.assertEqual(sha(self.fixture.output), old)

    def test_changed_file_and_bad_evidence_hash_reject(self):
        (self.fixture.workspace / 'src/probekv/extra.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'stale validation'): self.build()
        self.assertFalse(self.fixture.output.exists())
        self.fixture.refs['cpu_report']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'changed evidence'): self.build()

    def test_new_implementation_after_validation_rejected(self):
        path = self.fixture.workspace / 'src/probekv/new.py'; path.write_text('# new')
        with patch('probekv.p0_cpu_handoff_v2._git_inventory', return_value=(
                self.fixture.head, set(self.fixture.files), set(self.fixture.files) | {'src/probekv/new.py'})):
            with self.assertRaisesRegex(ValueError, 'inventory differs'): self.build()

    def test_head_change_rejected(self):
        with patch('probekv.p0_cpu_handoff_v2._git_inventory', return_value=(
                'b' * 40, set(self.fixture.files), set(self.fixture.files))):
            with self.assertRaisesRegex(ValueError, 'Git HEAD'): self.build()

    def test_deleted_tracked_file_cannot_silently_reappear_on_deployment(self):
        with patch('probekv.p0_cpu_handoff_v2._git_inventory', return_value=(
                self.fixture.head, set(self.fixture.files) | {'src/removed.py'},
                set(self.fixture.files) | {'src/removed.py'})):
            with self.assertRaisesRegex(ValueError, 'removal contract'): self.build()

    def test_actual_failed_test_cannot_pass_by_manual_summary(self):
        self.fixture.tests[0]['observed'] = 'FAIL'
        self.fixture.put('test_results', self.fixture.tests)
        with self.assertRaisesRegex(ValueError, 'failed or unknown test'): self.build()

    def test_raw_cpu_log_and_taskbook_tampering_rejected(self):
        self.fixture.log.write_text('changed raw log')
        with self.assertRaisesRegex(ValueError, 'raw log digest'): self.build()
        self.fixture.flush()
        (self.fixture.workspace / PACKET / PACKET_MEMBERS[0]).write_text('changed taskbook')
        with self.assertRaisesRegex(ValueError, 'member changed'): self.build()

    def test_packet_list_cannot_smuggle_extra_input(self):
        checksum = self.fixture.workspace / PACKET / 'checksums.sha256'
        checksum.write_text(checksum.read_text(encoding='utf-8') + '\n' + '0' * 64 + '  ../secret.txt', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'not explicitly allowed'): self.build()

    def test_changed_packet_reference_status_rejects(self):
        self.fixture.commands['packet_reference']['passed'] = False
        self.fixture.put('commands', self.fixture.commands)
        with self.assertRaisesRegex(ValueError, 'packet reference'): self.build()

    def test_packet_validation_record_must_match_actual_checksum_index(self):
        self.fixture.packet_checks[0]['actual'] = '0' * 64
        self.fixture.put('package_checksums', self.fixture.packet_checks)
        with self.assertRaisesRegex(ValueError, 'taskbook validation report'): self.build()

    def test_untracked_bundles_credentials_weights_and_old_artifacts_are_not_scanned(self):
        for name in ('credential.pem', 'source.bundle', 'model.safetensors', 'artifacts/old.json'):
            path = self.fixture.workspace / name; path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'private fixture not allowed to leave workspace')
        result = self.build()
        names = result['metadata']['members']
        self.assertFalse(any('credential' in p or '.bundle' in p or 'old.json' in p for p in names))

    def test_allowlist_forbids_hidden_secrets_and_binary_data(self):
        for name in ('src/credentials/token.py', 'tests/.ssh/key.py', 'src/model.pt',
                     'scripts/a.bundle', 'artifacts/old.json', 'datasets/train.json'):
            self.assertFalse(_code_allowed(name))

    def test_symlink_member_rejected_before_reading(self):
        path = self.fixture.workspace / 'src/probekv/extra.py'
        real = self.fixture.root / 'real.py'; real.write_bytes(path.read_bytes()); path.unlink()
        try:
            path.symlink_to(real)
        except OSError:
            # Windows privilege-independent simulation exercises the same lstat policy.
            original = Path.lstat
            def fake(item):
                if item.name == 'extra.py':
                    value = original(real)
                    return type('Stat', (), dict(st_mode=stat.S_IFLNK, st_file_attributes=0, st_size=value.st_size))()
                return original(item)
            path.write_bytes(real.read_bytes())
            with patch.object(Path, 'lstat', fake):
                with self.assertRaisesRegex(ValueError, 'symlink'): self.build()
        else:
            with self.assertRaisesRegex(ValueError, 'symlink'): self.build()

    def test_positive_explicit_budgets_and_size_limits(self):
        for values in (dict(max_members=True), dict(max_archive_bytes=0), dict(max_uncompressed_bytes=-1)):
            with self.assertRaisesRegex(ValueError, 'positive'): self.build(**values)
        with self.assertRaisesRegex(ValueError, 'budget'): self.build(max_members=1)
        with self.assertRaisesRegex(ValueError, 'budget'): self.build(max_uncompressed_bytes=1)
        with self.assertRaisesRegex(ValueError, 'budget'): self.build(max_archive_bytes=1)
        self.assertFalse(self.fixture.output.exists())
        self.assertEqual(list(self.fixture.root.glob('.cpu-handoff-*')), [])

    def test_zip_traversal_and_unknown_extra_members_rejected(self):
        self.build()
        for extra in ('../escape.py', 'workspace/src/unknown.py'):
            path = self.rewrite(extra=extra)
            with self.assertRaises(ValueError): self.verify(path)

    def test_zip_duplicate_and_symlink_rejected(self):
        self.build()
        path = self.rewrite(extra='workspace/setup.py')
        with self.assertRaisesRegex(ValueError, 'duplicate'): self.verify(path)
        def mutate(info, data):
            if info.filename == 'workspace/setup.py':
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            return info, data
        with self.assertRaisesRegex(ValueError, 'regular files'): self.verify(self.rewrite(mutate))

    def test_member_bytes_digest_and_false_gpu_authority_rejected(self):
        self.build()
        def corrupt(info, data):
            return info, b'bad' if info.filename == 'workspace/setup.py' else data
        with self.assertRaisesRegex(ValueError, 'digest/size'): self.verify(self.rewrite(corrupt))
        def authorize(info, data):
            if info.filename == 'handoff.json':
                metadata = json.loads(data); metadata['gpu_execution_allowed'] = True
                data = json.dumps(metadata).encode()
            return info, data
        with self.assertRaisesRegex(ValueError, 'CPU-only'): self.verify(self.rewrite(authorize))

    def test_zip_budget_checked_before_json_decode(self):
        self.build()
        with patch('probekv.p0_cpu_handoff_v2.json.loads', side_effect=AssertionError('must not decode')):
            with self.assertRaisesRegex(ValueError, 'budget'): self.verify(max_uncompressed_bytes=1)
        with self.assertRaisesRegex(ValueError, 'archive digest mismatch'):
            verify_cpu_handoff_archive(self.fixture.output, expected_sha256='0' * 64, **self.limits)

    def test_portable_paths_reject_windows_and_posix_escape(self):
        for name in ('/a', '../a', 'x/../a', 'x\\a', 'C:/a', 'x//a', 'CON.txt', 'a./x', 'a/ '):
            with self.assertRaises(ValueError): _safe_name(name)

    def test_excluded_head_eol_reconstruction_never_ships_payload(self):
        raw = b'a\nb\n'; expected = hashlib.sha256(b'a\r\nb\r\n').hexdigest()
        with patch('probekv.p0_cpu_handoff_v2.subprocess.check_output', return_value=raw):
            row = _excluded_head_member(self.fixture.workspace, self.fixture.head, '.log', expected, {'.log'})
            self.assertEqual(row['reconstruction'], 'LF_TO_CRLF')
            with self.assertRaisesRegex(ValueError, 'beyond explicit checkout'):
                _excluded_head_member(self.fixture.workspace, self.fixture.head, '.log', '0' * 64, {'.log'})
            with self.assertRaisesRegex(ValueError, 'untracked excluded'):
                _excluded_head_member(self.fixture.workspace, self.fixture.head, '.log', expected, set())

    def test_inventory_race_fails_without_publishing(self):
        initial = self.fixture.inventory(None)
        with patch('probekv.p0_cpu_handoff_v2._git_inventory', side_effect=[initial, ('b' * 40, initial[1], initial[2])]):
            with self.assertRaisesRegex(ValueError, 'changed during packaging'): self.build()
        self.assertFalse(self.fixture.output.exists())

    def test_content_race_after_copy_fails_before_publication(self):
        from probekv.p0_cpu_handoff_v2 import verify_cpu_handoff_archive as verify
        def mutate_after_verify(*args, **kwargs):
            result = verify(*args, **kwargs)
            (self.fixture.workspace / 'src/probekv/extra.py').write_text('changed after copy')
            return result
        with patch('probekv.p0_cpu_handoff_v2.verify_cpu_handoff_archive', side_effect=mutate_after_verify):
            with self.assertRaisesRegex(ValueError, 'worktree changed during packaging'): self.build()
        self.assertFalse(self.fixture.output.exists())


if __name__ == '__main__':
    unittest.main()
