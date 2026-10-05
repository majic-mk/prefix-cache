"""CPU_TEST_FIXTURES_ONLY: no actual tokenizer library, receipt or GPU run.

The positive producer fixture copies exact original parser/protocol bytes but
uses explicitly mocked encoder IDs, mock model JSON and a never-loaded mock
binary. Every positive output remains synthetic_fixture=True. Pins are patched
only inside these tests; production pins remain the actual audited server ones.
"""
from contextlib import contextmanager
import importlib.util
import json
import marshal
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


HERE = Path(__file__).resolve().parent
CANDIDATES = HERE.parents[1]
PROTOCOL = CANDIDATES / 'gpu_prerental_preparation_20261004/protocol/prerental_protocol.py'
AUTHOR = CANDIDATES / 'i_pilot_cpu_preparation_20261004/source_inputs/shared_storage_trace_replay.py'
COMMON = CANDIDATES / 'i_pilot_cpu_preparation_20261004/source_inputs/prefix_cache_common.py'
if not all(path.is_file() for path in (PROTOCOL, AUTHOR, COMMON)):
    # Deployment uses the actual server's existing frozen sources, not copies
    # synthesized by this suite. Each copied source remains subject to its SHA.
    for project in (Path.cwd(), *HERE.parents):
        candidates = (
            project / 'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/protocol/prerental_protocol.py',
            project / 'third_party/upstream/kvcache-experiments/scripts/shared_storage_trace_replay.py',
            project / 'third_party/upstream/kvcache-experiments/common/prefix_cache_common.py')
        if all(path.is_file() for path in candidates):
            PROTOCOL, AUTHOR, COMMON = candidates
            break


def load(path):
    spec = importlib.util.spec_from_file_location('_cpu_fixture_raw_producer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RawCPUFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='CPU_MOCK_TOKENIZATION_ONLY_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'artifacts').mkdir()
        self.producer = self.root / 'artifacts/raw_cpu_fixture.py'
        shutil.copyfile(HERE / 'raw_tokenizer_cpu.py', self.producer)
        self.m = load(self.producer)
        for original, name in ((PROTOCOL, 'protocol.py'), (AUTHOR, 'author.py'), (COMMON, 'common.py')):
            shutil.copyfile(original, self.root / name)
        self.package = 'site-packages/tokenizers'
        self.dist = 'site-packages/tokenizers-0.22.2.dist-info'
        (self.root / self.package).mkdir(parents=True)
        (self.root / self.dist).mkdir()
        self.write(self.package + '/__init__.py', '# CPU_MOCK_ONLY_NOT_EXECUTED\n')
        self.write(self.package + '/tokenizers.abi3.so', 'CPU_MOCK_ONLY_NEVER_LOADED')
        self.write(self.package + '/wrapper.py', '# CPU_MOCK_WRAPPER_NEVER_IMPORTED\n')
        self.write(self.dist + '/METADATA', 'Name: tokenizers\nVersion: 0.22.2\n')
        self.write(self.dist + '/RECORD', 'CPU_MOCK_RECORD_NOT_ACTUAL_PACKAGE\n')
        self.write(self.dist + '/WHEEL', 'CPU_MOCK_WHEEL\n')
        self.write(self.dist + '/INSTALLER', 'CPU_MOCK_INSTALLER\n')
        self.write('mock_model_plan.json', '{}')
        self.tokenizer = dict(model=dict(type='BPE'), truncation=None, padding=None,
                              post_processor=dict(type='ByteLevel', add_prefix_space=False,
                                                  trim_offsets=False, use_regex=False))
        self.config = dict(tokenizer_class='Qwen2Tokenizer', add_bos_token=False,
                           split_special_tokens=False, add_prefix_space=False)
        self.write_json('mock_tokenizer.json', self.tokenizer)
        self.write_json('mock_tokenizer_config.json', self.config)
        self.write_json('CPU_FIXTURE_DATASET.json', [
            '  first fixture  ', {'prompt': 'second fixture'},
            {'conversations': [{'from': 'human', 'value': 'third fixture'},
                               {'from': 'gpt', 'value': 'assistant excluded by original parser'},
                               {'from': 'user', 'value': 'fourth fixture'}]},
            {'prompt': ''}, 'first fixture'])
        self.m.MODEL_PLAN_SHA = self.ref('mock_model_plan.json')['sha256']
        self.m.TOKENIZER_JSON_SHA = self.ref('mock_tokenizer.json')['sha256']
        self.m.TOKENIZER_CONFIG_SHA = self.ref('mock_tokenizer_config.json')['sha256']
        self.m.BACKEND_PINS = {name: self.ref(self.package + '/' + name)['sha256']
                               for name in self.m.BACKEND_PINS}
        self.m.DIST_PINS = {name: self.ref(self.dist + '/' + name)['sha256'] for name in self.m.DIST_PINS}
        self.request = dict(schema=self.m.REQUEST_SCHEMA, dataset_ref=self.ref('CPU_FIXTURE_DATASET.json'),
                            protocol_ref=self.ref('protocol.py'), author_trace_ref=self.ref('author.py'),
                            author_common_ref=self.ref('common.py'), model_manifest_ref=self.ref('mock_model_plan.json'),
                            tokenizer_json_ref=self.ref('mock_tokenizer.json'),
                            tokenizer_config_ref=self.ref('mock_tokenizer_config.json'),
                            tokenizers_package_root=self.package, tokenizers_dist_info_root=self.dist,
                            synthetic_fixture=True)
        self.freeze()
        self.calls = []

    def write(self, relative, text):
        (self.root / relative).write_text(text, encoding='utf-8')

    def write_json(self, relative, value):
        (self.root / relative).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')

    def ref(self, name):
        return self.m.file_ref(self.root, name)

    def freeze(self):
        self.write_json('CPU_FIXTURE_REQUEST.json', self.request)
        rows = [self.ref(p.relative_to(self.root).as_posix()) for p in sorted(self.root.rglob('*'))
                if p.is_file() and p.name != 'CPU_FIXTURE_LOCK.json' and not p.suffix == '.pyc']
        self.write_json('CPU_FIXTURE_LOCK.json', dict(schema='CPU_TEST_SOURCE_LOCK_ONLY', files=rows))
        self.lock_ref = self.ref('CPU_FIXTURE_LOCK.json')
        self.request_ref = self.ref('CPU_FIXTURE_REQUEST.json')

    @contextmanager
    def mock_backend(self, root, request, refs, sources):
        self.calls.append('CPU_TEST_MOCK_BACKEND_ONLY')
        self.assertTrue(request['synthetic_fixture'])
        def encode(prompt):
            self.calls.append(prompt)
            return [len(prompt), 42]
        yield encode, dict(schema='CPU_TEST_MOCK_BACKEND_ONLY', backend_mocked=True)

    def run_fixture(self, **kwargs):
        arguments = dict(request=self.request, request_ref=self.request_ref, source_lock_ref=self.lock_ref,
                         output_relative='artifacts/CPU_FIXTURE_RAW_OUTPUT.json', backend_factory=self.mock_backend)
        arguments.update(kwargs)
        return self.m.produce(self.root, **arguments)

    def rejected(self, pattern, **kwargs):
        with self.assertRaisesRegex((ValueError, OSError), pattern):
            self.run_fixture(**kwargs)
        self.assertFalse((self.root / 'artifacts/CPU_FIXTURE_RAW_OUTPUT.json').exists())

    def test_missing_dataset_precedes_every_source_protocol_and_backend_read(self):
        before = sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob('*'))
        with patch.object(self.m, 'current_refs', side_effect=AssertionError('source must not load')), \
                patch.object(self.m, 'load_protocol', side_effect=AssertionError('protocol must not load')), \
                patch.object(self.m, 'actual_local_backend', side_effect=AssertionError('backend must not import')):
            result = self.m.produce(self.root, request={'dataset_ref': None}, output_relative='new/never.json')
        self.assertEqual(result['status'], 'UNBOUND_NO_ACTUAL_DATASET')
        self.assertEqual(result['actual_GPU_operations'], 0)
        self.assertFalse(result['tokenizer_imported'])
        self.assertEqual(before, sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob('*')))

    def test_actual_author_ast_order_and_all_accepted_rows_remain_fixture(self):
        result = self.run_fixture()
        self.assertEqual(self.calls, ['CPU_TEST_MOCK_BACKEND_ONLY', 'first fixture', 'second fixture',
                                      'third fixture\n\nfourth fixture', 'first fixture'])
        self.assertEqual(result['accepted_prompt_count'], 4)
        self.assertEqual([r['request_id'] for r in result['records']], [0, 1, 2, 3])
        self.assertEqual(result['inspection']['duplicate_exact_prompts'], 1)
        self.assertEqual(result['raw_records_sha256'], self.m.digest(result['records']))
        self.assertTrue(result['synthetic_fixture'])
        self.assertEqual(result['status'], 'CPU_FIXTURE_RAW_IDS_ONLY')
        self.assertFalse(result['formal_receipt_ready'])
        self.assertFalse(result['family_proof_available'])
        self.assertFalse(result['gpu_eligible'])
        for row in result['records']:
            self.assertEqual(set(row), {'request_id', 'prompt_sha256', 'prompt_token_ids'})
        self.assertNotIn('tokenized_records_sha256', result)
        self.assertNotIn('partition_counts', result)
        self.assertNotIn('receipt', result)

    def test_injected_mock_cannot_issue_actual_result(self):
        self.request['synthetic_fixture'] = False
        self.freeze()
        self.rejected('mock/injected backend prohibited')
        self.assertEqual(self.calls, [])

    def test_dataset_byte_drift_rejected_before_backend(self):
        self.write('CPU_FIXTURE_DATASET.json', '["changed"]')
        self.rejected('byte SHA/size mismatch')
        self.assertEqual(self.calls, [])

    def test_producer_itself_must_be_current_source_closed(self):
        lock = self.m.bounded_json(self.root / 'CPU_FIXTURE_LOCK.json')
        lock['files'] = [row for row in lock['files'] if row['path'] != 'artifacts/raw_cpu_fixture.py']
        self.write_json('CPU_FIXTURE_LOCK.json', lock)
        self.lock_ref = self.ref('CPU_FIXTURE_LOCK.json')
        self.rejected('leaf absent')
        self.assertEqual(self.calls, [])

    def test_new_package_code_leaf_cannot_escape_lock(self):
        self.write(self.package + '/unexpected.py', '# CPU_MOCK_NEW_UNFROZEN_CODE\n')
        self.rejected('source leaf not in current lock')
        self.assertEqual(self.calls, [])

    def test_package_binary_drift_rejected_before_import(self):
        self.write(self.package + '/tokenizers.abi3.so', 'CHANGED_MOCK_BINARY_NEVER_LOAD')
        self.rejected('byte SHA/size mismatch')
        self.assertEqual(self.calls, [])

    def test_frozen_author_parser_cannot_be_replaced_by_new_parser(self):
        self.write('author.py', '# CPU_MOCK_REPLACEMENT\n')
        self.request['author_trace_ref'] = self.ref('author.py')
        self.freeze()
        self.rejected('pinned original source drift')
        self.assertEqual(self.calls, [])

    def test_97_accepted_rows_rejects_whole_corpus_before_backend(self):
        self.write_json('CPU_FIXTURE_DATASET.json', ['fixture ' + str(i) for i in range(97)])
        self.request['dataset_ref'] = self.ref('CPU_FIXTURE_DATASET.json')
        self.freeze()
        self.rejected('whole author-accepted corpus')
        self.assertEqual(self.calls, [])

    def test_oversized_dataset_rejects_without_author_parse_or_import(self):
        self.m.MAX_BYTES = (self.root / 'CPU_FIXTURE_DATASET.json').stat().st_size - 1
        # Keep bounded source lock parsing available, but test the actual dataset bound.
        with patch.object(self.m, 'bounded_json', side_effect=lambda p: json.loads(p.read_bytes())), \
                patch.object(self.m, 'load_protocol', side_effect=AssertionError('author parser not reached')):
            self.rejected('dataset exceeds 32 MiB')
        self.assertEqual(self.calls, [])

    def test_empty_original_accepted_corpus_rejects_before_backend(self):
        self.write_json('CPU_FIXTURE_DATASET.json', [{'prompt': ''}])
        self.request['dataset_ref'] = self.ref('CPU_FIXTURE_DATASET.json')
        self.freeze()
        self.rejected('No usable prompts|whole author-accepted corpus')
        self.assertEqual(self.calls, [])

    def test_enabled_truncation_rejects_before_import(self):
        self.tokenizer['truncation'] = dict(max_length=4)
        self.write_json('mock_tokenizer.json', self.tokenizer)
        self.request['tokenizer_json_ref'] = self.ref('mock_tokenizer.json')
        self.m.TOKENIZER_JSON_SHA = self.request['tokenizer_json_ref']['sha256']
        self.freeze()
        self.rejected('no truncation/padding')
        self.assertEqual(self.calls, [])

    def test_request_bytes_and_in_memory_claim_must_match(self):
        self.request['synthetic_fixture'] = False
        self.rejected('mock/injected backend prohibited')
        self.request['synthetic_fixture'] = True
        self.request['tokenizers_package_root'] = 'different/tokenizers'
        self.rejected('actual frozen request JSON bytes')
        self.assertEqual(self.calls, [])

    def test_bad_token_ids_do_not_write_partial_output(self):
        for bad in ([True], [-1], [152064], [], [1] * 4097, [1.0], (1,)):
            @contextmanager
            def bad_backend(*args):
                yield lambda prompt: bad, dict(backend_mocked=True)
            self.rejected('raw integer token IDs', backend_factory=bad_backend)

    def test_dataset_mutated_by_mock_encoder_cannot_write_output(self):
        @contextmanager
        def mutation_backend(*args):
            def encode(prompt):
                self.write('CPU_FIXTURE_DATASET.json', '["MUTATED_CPU_FIXTURE"]')
                return [1]
            yield encode, dict(backend_mocked=True)
        self.rejected('byte SHA/size mismatch', backend_factory=mutation_backend)

    def test_new_package_leaf_during_mock_encoding_is_rejected(self):
        @contextmanager
        def mutation_backend(*args):
            def encode(prompt):
                self.write(self.package + '/during.py', '# CPU_MOCK_NOT_FROZEN\n')
                return [1]
            yield encode, dict(backend_mocked=True)
        self.rejected('source leaf not in current lock', backend_factory=mutation_backend)

    def test_existing_output_never_overwritten_and_missing_parent_never_created(self):
        self.write('artifacts/CPU_FIXTURE_RAW_OUTPUT.json', 'KEEP_CPU_FIXTURE')
        with self.assertRaisesRegex(ValueError, 'new append-only output'):
            self.run_fixture()
        self.assertEqual((self.root / 'artifacts/CPU_FIXTURE_RAW_OUTPUT.json').read_text(), 'KEEP_CPU_FIXTURE')
        with self.assertRaisesRegex(ValueError, 'new append-only output'):
            self.run_fixture(output_relative='artifacts/absent/output.json')
        self.assertFalse((self.root / 'artifacts/absent').exists())

    def test_outside_paths_and_duplicate_json_keys_are_rejected(self):
        for relative in ('../outside', '/outside', 'C:/outside', 'a\\b', 'a//b'):
            with self.assertRaises(ValueError):
                self.m.safe(self.root, relative)
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            self.m.parse('{"dataset_ref":null,"dataset_ref":null}')

    def test_actual_backend_module_origins_and_cleanup_via_spy_only(self):
        fake_package = SimpleNamespace(__file__=str(self.root / self.package / '__init__.py'), __version__='0.22.2')
        fake_binary = SimpleNamespace(__file__=str(self.root / self.package / 'tokenizers.abi3.so'))
        backend = SimpleNamespace(truncation=None, padding=None, encode_special_tokens=False,
                                  encode=Mock(return_value=SimpleNamespace(ids=[7])))
        factory = Mock(return_value=backend)
        fake_package.Tokenizer = SimpleNamespace(from_file=factory)
        original_path = list(sys.path)
        original_meta = list(sys.meta_path)
        def import_spy(name):
            self.assertEqual(name, 'tokenizers')
            sys.modules['tokenizers'] = fake_package
            sys.modules['tokenizers.tokenizers'] = fake_binary
            return fake_package
        refs = self.m.current_refs(self.root, self.lock_ref)
        sources = self.m.backend_sources(self.root, self.request, refs)
        with patch.object(self.m.importlib, 'import_module', side_effect=import_spy):
            with self.m.actual_local_backend(self.root, self.request, refs, sources) as (encode, audit):
                self.assertEqual(encode('CPU_SPY_ONLY'), [7])
                backend.encode.assert_called_once_with('CPU_SPY_ONLY', add_special_tokens=False)
                self.assertEqual(len(audit['loaded_module_sources']), 2)
                self.assertFalse(audit['chat_template_applied'])
        factory.assert_called_once_with(str(self.m.safe(self.root, 'mock_tokenizer.json')))
        self.assertEqual(sys.path, original_path)
        self.assertEqual(sys.meta_path, original_meta)
        self.assertFalse(any(n == 'tokenizers' or n.startswith('tokenizers.') for n in sys.modules))

    def test_wrong_backend_origin_and_forbidden_import_cleanup_via_spy(self):
        fake = SimpleNamespace(__file__=str(self.root / 'author.py'), __version__='0.22.2')
        refs = self.m.current_refs(self.root, self.lock_ref)
        sources = self.m.backend_sources(self.root, self.request, refs)
        original_path = list(sys.path)
        original_meta = list(sys.meta_path)
        def import_spy(name):
            sys.modules['tokenizers'] = fake
            return fake
        with patch.object(self.m.importlib, 'import_module', side_effect=import_spy):
            with self.assertRaisesRegex(ValueError, 'inside pinned package'):
                with self.m.actual_local_backend(self.root, self.request, refs, sources):
                    self.fail('foreign backend cannot yield')
        self.assertEqual(sys.path, original_path)
        self.assertEqual(sys.meta_path, original_meta)
        self.assertNotIn('tokenizers', sys.modules)
        with self.assertRaisesRegex(ValueError, 'import forbidden'):
            self.m._CPUImportGate().find_spec('torch.cuda')

    def timestamp_valid_mock_cache(self, source, code):
        """Actual .pyc fixture: valid source mtime/size, deliberately other code."""
        cached = Path(importlib.util.cache_from_source(str(source)))
        cached.parent.mkdir(parents=True, exist_ok=True)
        stat = source.stat()
        raw = importlib.util.MAGIC_NUMBER + struct.pack('<III', 0, int(stat.st_mtime) & 0xffffffff, stat.st_size)
        cached.write_bytes(raw + marshal.dumps(compile(code, str(source), 'exec')))
        return cached

    def test_source_only_package_ignores_existing_timestamp_valid_mock_pyc(self):
        source = self.root / self.package / '__init__.py'
        self.write(self.package + '/__init__.py', 'CPU_MOCK_PINNED_SOURCE = True\n')
        cached = self.timestamp_valid_mock_cache(source, 'CPU_MOCK_UNFROZEN_CACHE = True\n')
        self.freeze()
        # The fixture demonstrably reproduces default SourceFileLoader's hole.
        ordinary = importlib.machinery.SourceFileLoader('CPU_MOCK_DEFAULT_CACHE_REPRODUCTION', str(source))
        check = {}
        exec(ordinary.get_code('CPU_MOCK_DEFAULT_CACHE_REPRODUCTION'), check)
        self.assertTrue(check['CPU_MOCK_UNFROZEN_CACHE'])
        refs = self.m.current_refs(self.root, self.lock_ref)
        finder = self.m._TokenizersSourceOnlyFinder(self.root, self.request, refs)
        spec = finder.find_spec('tokenizers')
        module = importlib.util.module_from_spec(spec)
        before = cached.read_bytes()
        spec.loader.exec_module(module)
        self.assertTrue(module.CPU_MOCK_PINNED_SOURCE)
        self.assertFalse(hasattr(module, 'CPU_MOCK_UNFROZEN_CACHE'))
        self.assertEqual(cached.read_bytes(), before)
        self.assertEqual(Path(module.__file__).resolve(), source.resolve())

    def test_original_protocol_ignores_existing_timestamp_valid_mock_pyc(self):
        source = self.root / 'protocol.py'
        cached = self.timestamp_valid_mock_cache(source, 'raise RuntimeError("CPU_MOCK_UNFROZEN_PROTOCOL_CACHE")\n')
        before = cached.read_bytes()
        module = self.m.load_protocol(source)
        self.assertTrue(callable(module.inspect_trace))
        self.assertEqual(module.AUTHOR_TRACE_SHA256, self.m.AUTHOR_TRACE_SHA)
        self.assertEqual(cached.read_bytes(), before)

    def test_cached_only_package_child_is_rejected_without_executing_cache(self):
        source = self.root / self.package / 'cached_only.py'
        self.write(self.package + '/cached_only.py', '# CPU_MOCK_TO_REMOVE\n')
        cached = self.timestamp_valid_mock_cache(source, 'CPU_MOCK_UNFROZEN_CACHE = True\n')
        source.unlink()  # Only this explicit temporary fixture, never an installed file.
        refs = self.m.current_refs(self.root, self.lock_ref)
        finder = self.m._TokenizersSourceOnlyFinder(self.root, self.request, refs)
        with self.assertRaisesRegex(ValueError, 'cached-only import forbidden'):
            finder.find_spec('tokenizers.cached_only')
        self.assertTrue(cached.exists())

    def test_cli_missing_request_returns_unbound_before_import_and_writes_nothing(self):
        before = sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob('*'))
        proc = subprocess.run([sys.executable, '-B', '-I', '-S', str(HERE / 'raw_tokenizer_cpu.py'),
                               '--project-root', str(self.root), '--output-relative', 'missing/never.json'],
                              check=False, capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 78, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result['status'], 'UNBOUND_NO_ACTUAL_DATASET')
        self.assertFalse(result['tokenizer_imported'])
        self.assertEqual(before, sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob('*')))


if __name__ == '__main__':
    unittest.main()
