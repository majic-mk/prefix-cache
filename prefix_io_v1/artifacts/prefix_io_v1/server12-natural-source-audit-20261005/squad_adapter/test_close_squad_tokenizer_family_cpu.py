"""CPU_FIXTURE_ONLY: pure source/block checks and failure paths, no receipts.

No actual tokenizers library/network/GPU is executed. Synthetic positive checks
never invoke the full receipt writer and cannot become actual family evidence.
"""
from copy import deepcopy
import hashlib
import importlib.machinery
import importlib.util
import json
import marshal
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest


HERE = Path(__file__).resolve().parent
MODULE = HERE / 'close_squad_tokenizer_family_cpu.py'


def load(path):
    module = ModuleType('_CPU_FIXTURE_FAMILY_CLOSE')
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


M = load(MODULE)
A = load(HERE / 'squad_source_adapter_cpu_v2.py')


class FamilyCloseCPUFixtureTests(unittest.TestCase):
    def setUp(self):
        self.source = dict(version='1.1', data=[dict(title='CPU_FIXTURE_ARTICLE_' + str(a), paragraphs=[
            dict(context='Context CPU_FIXTURE only ' + str(a), qas=[dict(id='CPU_FIXTURE_QID_' + str(a) + '_' + str(q),
                question='CPU_FIXTURE question ' + str(q) + '?', answers=[dict(text='Context', answer_start=0)])
                for q in range(2)])]) for a in range(3)])
        self.contract = deepcopy(A.FIXED_CONTRACT)
        self.contract['synthetic_fixture'] = True
        self.bundle = A.validate_adapter(json.dumps(self.source).encode('utf-8'), self.contract)
        self.mapping = self.bundle['mapping']
        self.rows = []
        self.contexts = {}
        for source in self.mapping['records']:
            a = source['article_index']
            prefix = [a * 100 + i for i in range(16)]
            self.contexts[(a, 0)] = prefix + [999]
            self.rows.append(dict(request_id=source['request_id'], prompt_sha256=source['prompt_sha256'],
                                  prompt_token_ids=prefix + [1000 + source['request_id']]))

    def check(self, **kwargs):
        values = dict(rows=deepcopy(self.rows), mapping=deepcopy(self.mapping), context_ids=deepcopy(self.contexts))
        values.update(kwargs)
        return M.derive_family_records(**values)

    def test_source_families_and_first_blocks_close_only_synthetic_pure_rows(self):
        records, proof = self.check()
        self.assertEqual(len(records), 6)
        self.assertEqual(proof['family_count'], 3)
        self.assertEqual(proof['block_size'], 16)
        self.assertTrue(proof['synthetic_fixture'])
        self.assertEqual(proof['actual_GPU_operations'], 0)
        for original, record, source in zip(self.rows, records, self.mapping['records']):
            self.assertEqual({key: record[key] for key in original}, original)
            self.assertEqual(record['prefix_family'], source['source_prefix_family'])
        self.assertNotEqual(M.digest(records), M.digest(self.rows))
        self.assertNotIn('family_proof', proof)
        self.assertNotIn('tokenization_result_ref', proof)

    def test_same_first_block_across_distinct_article_splits_rejects(self):
        rows, contexts = deepcopy(self.rows), deepcopy(self.contexts)
        for row in rows[2:4]:
            row['prompt_token_ids'][:16] = rows[0]['prompt_token_ids'][:16]
        contexts[(1, 0)][:16] = rows[0]['prompt_token_ids'][:16]
        with self.assertRaisesRegex(ValueError, 'first 16-token block leaks'):
            self.check(rows=rows, context_ids=contexts)

    def test_same_article_different_paragraph_across_partitions_rejects(self):
        mapping, contexts = deepcopy(self.mapping), deepcopy(self.contexts)
        paragraph = mapping['selected_paragraphs'][1]
        paragraph['article_index'], paragraph['paragraph_index'] = 0, 1
        paragraph['source_identity']['article_index'], paragraph['source_identity']['paragraph_index'] = 0, 1
        paragraph['source_prefix_family'] = 'squad-source-context:' + M.digest(paragraph['source_identity'])
        for row in mapping['records'][2:4]:
            row['article_index'], row['paragraph_index'] = 0, 1
            row['source_prefix_family'] = paragraph['source_prefix_family']
        contexts[(0, 1)] = contexts.pop((1, 0))
        with self.assertRaisesRegex(ValueError, 'article ancestry leaks'):
            self.check(mapping=mapping, context_ids=contexts)

    def test_source_family_label_cannot_replace_real_identity(self):
        for target in ('paragraph', 'record'):
            mapping = deepcopy(self.mapping)
            if target == 'paragraph':
                mapping['selected_paragraphs'][0]['source_prefix_family'] = 'CPU_FIXTURE_SELF_LABEL'
            else:
                mapping['records'][0]['source_prefix_family'] = 'CPU_FIXTURE_SELF_LABEL'
            with self.assertRaisesRegex(ValueError, 'source family identity|source group/partition'):
                self.check(mapping=mapping)

    def test_source_context_hash_and_source_sha_drift_reject(self):
        for target in ('context_sha256', 'source_sha256'):
            mapping = deepcopy(self.mapping)
            mapping['selected_paragraphs'][0]['source_identity'][target] = '0' * 64
            mapping['selected_paragraphs'][0]['source_prefix_family'] = 'squad-source-context:' + M.digest(
                mapping['selected_paragraphs'][0]['source_identity'])
            with self.assertRaisesRegex(ValueError, 'source family identity'):
                self.check(mapping=mapping)

    def test_wrong_split_cannot_be_fixed_by_different_family_label(self):
        mapping = deepcopy(self.mapping)
        mapping['records'][0]['partition'] = 'evaluation'
        with self.assertRaisesRegex(ValueError, 'source group/partition'):
            self.check(mapping=mapping)

    def test_drop_reorder_change_prompt_or_integer_identity_rejects(self):
        rows = deepcopy(self.rows)
        rows.pop()
        with self.assertRaisesRegex(ValueError, 'denominator'):
            self.check(rows=rows)
        rows = deepcopy(self.rows)
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(ValueError, 'row order/identity'):
            self.check(rows=rows)
        for key, value in (('prompt_sha256', '0' * 64), ('request_id', True), ('request_id', 2)):
            rows = deepcopy(self.rows)
            rows[0][key] = value
            with self.assertRaisesRegex(ValueError, 'row order/identity'):
                self.check(rows=rows)
        mapping = deepcopy(self.mapping)
        mapping['records'][0]['request_id'] = False
        with self.assertRaisesRegex(ValueError, 'integer source positions'):
            self.check(mapping=mapping)

    def test_invalid_short_bool_and_negative_token_blocks_reject(self):
        for ids in ([1] * 15, [True] * 16, [-1] * 16, [152064] * 16, [1] * 4097):
            rows = deepcopy(self.rows)
            rows[0]['prompt_token_ids'] = ids
            with self.assertRaisesRegex(ValueError, 'actual first 16-token block'):
                self.check(rows=rows)

    def test_prompt_block_requires_independent_original_context_encoding(self):
        for replacement in (None, [1] * 15, [999] * 16):
            contexts = deepcopy(self.contexts)
            contexts[(0, 0)] = replacement
            with self.assertRaisesRegex(ValueError, 'encoded full context block|not covered by actual original context'):
                self.check(context_ids=contexts)

    def test_context_shared_across_partitions_is_rejected_even_if_blocks_differ(self):
        mapping = deepcopy(self.mapping)
        paragraph = mapping['selected_paragraphs'][1]
        paragraph['context_sha256'] = mapping['selected_paragraphs'][0]['context_sha256']
        paragraph['source_identity']['context_sha256'] = paragraph['context_sha256']
        paragraph['source_prefix_family'] = 'squad-source-context:' + M.digest(paragraph['source_identity'])
        for row in mapping['records'][2:4]:
            row['context_sha256'], row['source_prefix_family'] = paragraph['context_sha256'], paragraph['source_prefix_family']
        with self.assertRaisesRegex(ValueError, 'same context leaks'):
            self.check(mapping=mapping)

    def test_duplicate_actual_source_paragraph_rejects(self):
        mapping = deepcopy(self.mapping)
        mapping['selected_paragraphs'].append(deepcopy(mapping['selected_paragraphs'][0]))
        with self.assertRaisesRegex(ValueError, 'unique actual source paragraph'):
            self.check(mapping=mapping)

    def test_source_only_compile_ignores_valid_timestamp_unfrozen_pyc(self):
        with tempfile.TemporaryDirectory(prefix='CPU_FIXTURE_SOURCE_COMPILE_') as directory:
            root = Path(directory)
            source = root / 'CPU_FIXTURE_SOURCE.py'
            source.write_text('CPU_FIXTURE_PINNED_SOURCE = True\n', encoding='utf-8')
            cached = Path(importlib.util.cache_from_source(str(source)))
            cached.parent.mkdir()
            stat = source.stat()
            cached.write_bytes(importlib.util.MAGIC_NUMBER + struct.pack('<III', 0, int(stat.st_mtime) & 0xffffffff, stat.st_size) +
                               marshal.dumps(compile('CPU_FIXTURE_UNFROZEN_CACHE = True\n', str(source), 'exec')))
            check = {}
            exec(importlib.machinery.SourceFileLoader('CPU_FIXTURE_DEFAULT_LOADER', str(source)).get_code(
                'CPU_FIXTURE_DEFAULT_LOADER'), check)
            self.assertTrue(check['CPU_FIXTURE_UNFROZEN_CACHE'])
            row = M.file_ref(root, source.name)
            before = cached.read_bytes()
            module = M.load_source_only(root, row, {row['path']: row}, row['sha256'])
            self.assertTrue(module.CPU_FIXTURE_PINNED_SOURCE)
            self.assertFalse(hasattr(module, 'CPU_FIXTURE_UNFROZEN_CACHE'))
            self.assertEqual(before, cached.read_bytes())

    def test_source_loader_sha_drift_missing_ref_and_duplicate_json_reject(self):
        with tempfile.TemporaryDirectory(prefix='CPU_FIXTURE_BAD_SOURCE_') as directory:
            root = Path(directory)
            source = root / 'CPU_FIXTURE.py'
            source.write_text('CPU_FIXTURE=True\n', encoding='utf-8')
            row = M.file_ref(root, source.name)
            with self.assertRaisesRegex(ValueError, 'inside current V13'):
                M.load_source_only(root, row, {}, row['sha256'])
            source.write_text('CPU_FIXTURE=False\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SHA/size mismatch'):
                M.load_source_only(root, row, {row['path']: row}, row['sha256'])
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            M.parse('{"a":1,"a":2}')

    def test_cli_unbound_no_library_import_no_output_or_directories(self):
        with tempfile.TemporaryDirectory(prefix='CPU_FIXTURE_UNBOUND_FAMILY_') as directory:
            root = Path(directory)
            proc = subprocess.run([sys.executable, '-B', '-I', '-S', str(MODULE), '--project-root', str(root)],
                                  capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(proc.returncode, 78, proc.stderr)
            report = json.loads(proc.stdout)
            self.assertEqual(report['actual_GPU_operations'], 0)
            self.assertFalse(report['gpu_eligible'])
            self.assertEqual(list(root.iterdir()), [])

    def test_raw_fixture_mock_or_inherited_qualification_header_rejects(self):
        header = dict(schema='actual_cpu_raw_tokenization_v1', status='ACTUAL_CPU_RAW_IDS_ONLY',
                      synthetic_fixture=False, exit_code=0, tokenizer_imported=True, actual_GPU_operations=0,
                      family_proof_available=False, formal_receipt_ready=False, gpu_eligible=False,
                      formal_effect_qualified=False, tokenizer_backend=dict(
                          schema='actual_local_CPU_tokenizer_backend_v1', version='0.22.2', backend_mocked=False,
                          from_file_only=True, add_special_tokens=False, chat_template_applied=False),
                      CPU_FIXTURE_HEADER_ONLY_NOT_A_RESULT=True)
        M.validate_raw_header(header)  # Predicate only: no records, source or receipt.
        for field, value in (('synthetic_fixture', True), ('exit_code', False), ('family_proof_available', True),
                             ('formal_receipt_ready', True), ('gpu_eligible', True), ('actual_GPU_operations', False)):
            changed = deepcopy(header)
            changed[field] = value
            with self.assertRaises(ValueError):
                M.validate_raw_header(changed)
        changed = deepcopy(header)
        changed['tokenizer_backend']['backend_mocked'] = True
        with self.assertRaisesRegex(ValueError, 'nonmocked local backend header'):
            M.validate_raw_header(changed)

    def actual_metadata_only_fixture(self, root):
        candidates = HERE.parents[1]
        ancestor_source = candidates / 'gpu_prerental_preparation_20261004/PRERENT_SOURCE_LOCK_V12.json'
        freezer_source = HERE.parent / 'source_freeze/append_public_source_lock_v13.py'
        if not ancestor_source.is_file() or not freezer_source.is_file():
            for project in (Path.cwd(), *HERE.parents):
                if (project / M.V12_PATH).is_file() and (project / M.FREEZER_PATH).is_file():
                    ancestor_source, freezer_source = project / M.V12_PATH, project / M.FREEZER_PATH
                    break
        for original, relative in ((ancestor_source, M.V12_PATH), (freezer_source, M.FREEZER_PATH)):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        inherited = M.read_json(root / M.V12_PATH)['files']
        ancestor_ref, freezer_ref = M.file_ref(root, M.V12_PATH), M.file_ref(root, M.FREEZER_PATH)
        self.assertEqual(ancestor_ref['sha256'], M.V12_SHA)
        self.assertEqual(freezer_ref['sha256'], M.FREEZER_SHA)
        audit = M.FREEZER_PATH.split('/source_freeze/')[0]
        addition = audit + '/CPU_FIXTURE_ONLY_ADDITION.txt'
        (root / addition).write_text('CPU_FIXTURE_METADATA_ONLY_NOT_A_RECEIPT\n', encoding='utf-8')
        manifest_relative = audit + '/CPU_FIXTURE_ONLY_MANIFEST.json'
        (root / manifest_relative).write_text(json.dumps(dict(schema='bounded_public_source_append_manifest_v1',
                                                             files=[addition])), encoding='utf-8')
        manifest_ref = M.file_ref(root, manifest_relative)
        lock = dict(schema='strong_trace_source_lock_v1', files=[*inherited, ancestor_ref, freezer_ref,
                         manifest_ref, M.file_ref(root, addition)], ancestry_ref=ancestor_ref,
                    append_freezer_ref=freezer_ref, addition_manifest_ref=manifest_ref, GPU_launch_allowed=False,
                    production_qualified=False, strategy_effect_verified=False, model_and_SDK_loaded=False,
                    observation_scope='this_CPU_public_source_append_freeze_only')
        lock_relative = audit + '/PRERENT_SOURCE_LOCK_V13.json'
        return lock, lock_relative

    def test_actual_4953_metadata_ancestry_replay_and_forged_locks_reject(self):
        with tempfile.TemporaryDirectory(prefix='CPU_FIXTURE_METADATA_NO_MODEL_BYTES_') as directory:
            root = Path(directory)
            lock, relative = self.actual_metadata_only_fixture(root)
            def write(value):
                (root / relative).write_text(json.dumps(value), encoding='utf-8')
                return M.file_ref(root, relative)
            refs = M.verify_v13_closure(root, write(lock))
            self.assertEqual(len(refs), 4957)
            # This is metadata replay only; no inherited model/SDK is copied,
            # loaded or qualified and no receipt writer is called.
            mutations = []
            changed = deepcopy(lock); changed['schema'] = 'CPU_FIXTURE_FORGED'; mutations.append(changed)
            changed = deepcopy(lock); changed.pop('ancestry_ref'); mutations.append(changed)
            changed = deepcopy(lock); changed['ancestry_ref']['sha256'] = '0' * 64; mutations.append(changed)
            changed = deepcopy(lock); changed['files'].pop(0); mutations.append(changed)
            changed = deepcopy(lock); changed['files'][0]['bytes'] += 1; mutations.append(changed)
            changed = deepcopy(lock); changed['files'][0]['bytes'] = True; mutations.append(changed)
            changed = deepcopy(lock); changed['files'].append(deepcopy(changed['files'][0])); mutations.append(changed)
            changed = deepcopy(lock); changed['files'][0]['unfrozen_field'] = 1; mutations.append(changed)
            changed = deepcopy(lock); changed['GPU_launch_allowed'] = 0; mutations.append(changed)
            changed = deepcopy(lock); changed['append_freezer_ref']['sha256'] = '0' * 64; mutations.append(changed)
            changed = deepcopy(lock); changed['files'].append(dict(path='CPU_FIXTURE_UNDECLARED', bytes=0, sha256='0' * 64)); mutations.append(changed)
            for changed in mutations:
                with self.assertRaises(ValueError):
                    M.verify_v13_closure(root, write(changed))


if __name__ == '__main__':
    unittest.main()
