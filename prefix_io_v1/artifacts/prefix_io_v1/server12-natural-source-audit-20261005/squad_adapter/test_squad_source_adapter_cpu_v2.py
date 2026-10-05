"""CPU_FIXTURE_SQUAD_ONLY: source mapping/replay tests, never public acquisition.

Fixtures below are explicitly synthetic. No network/GPU/model/tokenizer call.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest


HERE = Path(__file__).resolve().parent
ADAPTER = HERE / 'squad_source_adapter_cpu_v2.py'
M = ModuleType('_CPU_FIXTURE_SQUAD_ADAPTER')
M.__file__ = str(ADAPTER)
exec(compile(ADAPTER.read_bytes(), str(ADAPTER), 'exec', dont_inherit=True), M.__dict__)


def fixture_source():
    articles = []
    for article_index in range(4):
        context = 'Context for CPU_FIXTURE article ' + str(article_index) + ', retained in full.'
        questions = [dict(id='CPU_FIXTURE_QID_' + str(article_index) + '_' + str(qa),
                          question='CPU_FIXTURE question ' + str(qa) + '?',
                          answers=[dict(text='Context', answer_start=0)]) for qa in range(2)]
        articles.append(dict(title='CPU_FIXTURE_TITLE_' + str(article_index), paragraphs=[
            dict(context=context, qas=questions),
            dict(context='CPU_FIXTURE unselected paragraph must not replace paragraph zero.', qas=[])]))
    return dict(version='1.1', data=articles)


def raw(source):
    return json.dumps(source, ensure_ascii=False, indent=2).encode('utf-8')


def keys_recursive(value):
    if type(value) is dict:
        return set(value) | set().union(*(keys_recursive(v) for v in value.values()))
    if type(value) is list:
        return set().union(*(keys_recursive(v) for v in value)) if value else set()
    return set()


class SquadCPUFixtureTests(unittest.TestCase):
    def setUp(self):
        self.source = fixture_source()
        self.contract = deepcopy(M.FIXED_CONTRACT)
        self.contract['synthetic_fixture'] = True
        self.input = raw(self.source)
        self.result = M.validate_adapter(self.input, self.contract)

    def replay(self, **changes):
        data = dict(source_bytes=self.input, selection_contract=self.contract,
                    trace=deepcopy(self.result['trace']), mapping=deepcopy(self.result['mapping']),
                    report=deepcopy(self.result['report']))
        data.update(changes)
        return M.replay_adapter(**data)

    def test_full_fixed_selection_order_context_and_all_title_metadata(self):
        trace, mapping = self.result['trace'], self.result['mapping']
        self.assertEqual(len(trace), 6)
        self.assertEqual(mapping['partition_counts'], dict(calibration=2, development=2, evaluation=2))
        self.assertEqual([row['qid'] for row in mapping['records']],
                         ['CPU_FIXTURE_QID_' + str(a) + '_' + str(q) for a in range(3) for q in range(2)])
        self.assertEqual([row['raw_request_pos'] for row in mapping['records']], list(range(6)))
        self.assertEqual([row['title'] for row in mapping['all_article_metadata']],
                         ['CPU_FIXTURE_TITLE_' + str(a) for a in range(4)])
        for a, paragraph in enumerate(mapping['selected_paragraphs']):
            self.assertEqual(paragraph['context'], self.source['data'][a]['paragraphs'][0]['context'])
            for q in range(2):
                self.assertEqual(trace[a * 2 + q]['prompt'], paragraph['context'] + '\n\n' +
                                 self.source['data'][a]['paragraphs'][0]['qas'][q]['question'])
        self.assertEqual(self.replay()['status'], 'PASS_CPU_SOURCE_REPLAY_ONLY')

    def test_family_is_bound_to_actual_article_context_partition_not_label(self):
        rows = self.result['mapping']['records']
        self.assertEqual(rows[0]['source_prefix_family'], rows[1]['source_prefix_family'])
        self.assertEqual(len({row['source_prefix_family'] for row in rows}), 3)
        for paragraph in self.result['mapping']['selected_paragraphs']:
            identity = paragraph['source_identity']
            self.assertEqual(identity['source_sha256'], self.result['mapping']['source_sha256'])
            self.assertEqual(identity['context_sha256'], M.text_sha(paragraph['context']))
            self.assertEqual(paragraph['source_prefix_family'], 'squad-source-context:' + M.digest(identity))
        mapping = deepcopy(self.result['mapping'])
        mapping['records'][2]['source_prefix_family'] = rows[0]['source_prefix_family']
        with self.assertRaisesRegex(ValueError, 'mapping/group/partition drift'):
            self.replay(mapping=mapping)

    def test_missing_question_cannot_silently_reduce_denominator(self):
        trace = deepcopy(self.result['trace'])
        trace.pop(1)
        with self.assertRaisesRegex(ValueError, 'complete selected trace'):
            self.replay(trace=trace)

    def test_changed_question_and_reordered_questions_reject(self):
        trace = deepcopy(self.result['trace'])
        trace[0]['prompt'] += ' CHANGED_QUESTION'
        with self.assertRaises(ValueError):
            self.replay(trace=trace)
        trace = deepcopy(self.result['trace'])
        trace[0], trace[1] = trace[1], trace[0]
        with self.assertRaises(ValueError):
            self.replay(trace=trace)

    def test_injected_shared_context_is_not_source_evidence(self):
        trace = deepcopy(self.result['trace'])
        trace[0]['prompt'] = 'ARTIFICIAL_SHARED_PREFIX\n' + trace[0]['prompt']
        with self.assertRaisesRegex(ValueError, 'complete selected trace'):
            self.replay(trace=trace)
        mapping = deepcopy(self.result['mapping'])
        mapping['selected_paragraphs'][0]['context'] += ' INJECTION'
        with self.assertRaises(ValueError):
            self.replay(mapping=mapping)

    def test_source_group_drift_changes_identity_even_with_old_labels(self):
        source = deepcopy(self.source)
        source['data'][0]['title'] = 'CPU_FIXTURE_CHANGED_GROUP'
        with self.assertRaisesRegex(ValueError, 'mapping/group/partition drift'):
            self.replay(source_bytes=raw(source))
        source = deepcopy(self.source)
        source['data'][0], source['data'][1] = source['data'][1], source['data'][0]
        with self.assertRaises(ValueError):
            self.replay(source_bytes=raw(source))

    def test_duplicate_qid_within_or_across_selected_articles_reject(self):
        for a, q in ((0, 1), (2, 1)):
            source = deepcopy(self.source)
            source['data'][a]['paragraphs'][0]['qas'][q]['id'] = 'CPU_FIXTURE_QID_0_0'
            with self.assertRaisesRegex(ValueError, 'duplicate qid'):
                M.validate_adapter(raw(source), self.contract)

    def test_empty_context_question_qid_title_or_annotation_reject(self):
        for field in ('context', 'question', 'id', 'title', 'answer'):
            source = deepcopy(self.source)
            article = source['data'][0]
            paragraph = article['paragraphs'][0]
            if field == 'title':
                article['title'] = ' '
            elif field == 'context':
                paragraph['context'] = ' '
            elif field == 'answer':
                paragraph['qas'][0]['answers'] = []
            else:
                paragraph['qas'][0][field] = ' '
            with self.assertRaises(ValueError):
                M.validate_adapter(raw(source), self.contract)

    def test_wrong_split_or_qa_raw_position_reject(self):
        for field, value in (('partition', 'evaluation'), ('qa_index', 1), ('raw_request_pos', 4)):
            mapping = deepcopy(self.result['mapping'])
            mapping['records'][0][field] = value
            with self.assertRaisesRegex(ValueError, 'mapping/group/partition drift'):
                self.replay(mapping=mapping)

    def test_33_qas_rejects_entire_partition_without_selecting_32(self):
        source = deepcopy(self.source)
        first = source['data'][0]['paragraphs'][0]
        first['qas'] = [dict(id='CPU_FIXTURE_OVERFLOW_' + str(i), question='Overflow fixture question?',
                             answers=[dict(text='Context', answer_start=0)]) for i in range(33)]
        with self.assertRaisesRegex(ValueError, '1..32 qas'):
            M.validate_adapter(raw(source), self.contract)

    def test_all_three_maximum_32_preserve_96_without_cap_or_reorder(self):
        source = deepcopy(self.source)
        for a in range(3):
            source['data'][a]['paragraphs'][0]['qas'] = [dict(id='CPU_FIXTURE_LIMIT_' + str(a) + '_' + str(q),
                question='CPU_FIXTURE_LIMIT question ' + str(q) + '?',
                answers=[dict(text='Context', answer_start=0)]) for q in range(32)]
        result = M.validate_adapter(raw(source), self.contract)
        self.assertEqual(len(result['trace']), 96)
        self.assertEqual(result['mapping']['records'][-1]['qid'], 'CPU_FIXTURE_LIMIT_2_31')

    def test_selection_cannot_swap_article_paragraph_or_limits(self):
        for key, replacement in (('article_indices', [3, 1, 2]), ('paragraph_indices', [1, 0, 0]),
                                 ('max_partition_requests', 33), ('max_total_requests', 97),
                                 ('partitions', ['evaluation', 'development', 'calibration']),
                                 ('selection_by_length_or_outcome', True), ('inject_answers', True)):
            contract = deepcopy(self.contract)
            contract[key] = replacement
            with self.assertRaisesRegex(ValueError, 'fixed prospective rule'):
                M.validate_adapter(self.input, contract)

    def test_no_selected_paragraph_zero_or_qas_means_no_replacement(self):
        for replacement in ([], [dict(context='Context fixture', qas=[])]):
            source = deepcopy(self.source)
            source['data'][0]['paragraphs'] = replacement
            with self.assertRaises(ValueError):
                M.validate_adapter(raw(source), self.contract)

    def test_answers_not_injected_or_copied_and_raw_trace_author_acceptable(self):
        self.assertEqual(set(self.result['trace'][0]), {'prompt'})
        self.assertFalse({'answers', 'answer_start', 'answer'} & keys_recursive(self.result))
        for row in self.result['trace']:
            self.assertEqual(row['prompt'], row['prompt'].strip())
        report = self.result['report']
        self.assertTrue(report['synthetic_fixture'])
        self.assertFalse(report['formal_receipt_ready'])
        self.assertFalse(report['tokenizer_prefix_coverage_proven'])
        self.assertFalse(report['natural_production_traffic_claim'])
        self.assertFalse(report['external_acquisition_authenticity_verified_by_this_API'])
        self.assertEqual(report['actual_GPU_operations'], 0)

    def test_exact_frozen_author_cpu_parser_matches_every_accepted_prompt(self):
        candidates = HERE.parents[1]
        sources = (candidates / 'gpu_prerental_preparation_20261004/protocol/prerental_protocol.py',
                   candidates / 'i_pilot_cpu_preparation_20261004/source_inputs/shared_storage_trace_replay.py',
                   candidates / 'i_pilot_cpu_preparation_20261004/source_inputs/prefix_cache_common.py')
        if not all(path.is_file() for path in sources):
            for project in (Path.cwd(), *HERE.parents):
                actual = (project / 'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/protocol/prerental_protocol.py',
                          project / 'third_party/upstream/kvcache-experiments/scripts/shared_storage_trace_replay.py',
                          project / 'third_party/upstream/kvcache-experiments/common/prefix_cache_common.py')
                if all(path.is_file() for path in actual):
                    sources = actual
                    break
        for path, expected in zip(sources, (
                '7fbc544597c9abe334122d354e7494b590e810ab0682f1d2d155408cc4f618eb',
                '83713db1b011a954616896ec0ccfa05dc766805e80ed8defdef0da65100ff99e',
                'a331d1595b815c3c8fc63d3e2f236264c2f181ece7a9de51f50beb4703338df5')):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        protocol = ModuleType('_CPU_FIXTURE_ORIGINAL_PROTOCOL')
        protocol.__file__ = str(sources[0])
        exec(compile(sources[0].read_bytes(), str(sources[0]), 'exec', dont_inherit=True), protocol.__dict__)
        with tempfile.TemporaryDirectory(prefix='CPU_FIXTURE_SQUAD_AUTHOR_AST_') as directory:
            trace = Path(directory) / 'CPU_FIXTURE_TRACE.json'
            trace.write_bytes(raw(self.result['trace']))
            inspection = protocol.inspect_trace(trace, sources[1], sources[2])
            self.assertEqual(inspection['accepted_prompt_count'], len(self.result['trace']))
            self.assertEqual([row['prompt_sha256'] for row in inspection['records']],
                             [row['prompt_sha256'] for row in self.result['mapping']['records']])

    def test_malformed_answer_span_rejects_and_cannot_be_normalized(self):
        for answer in (dict(text='Context', answer_start=True), dict(text='not-in-context', answer_start=0),
                       dict(text='Context', answer_start=100000)):
            source = deepcopy(self.source)
            source['data'][0]['paragraphs'][0]['qas'][0]['answers'] = [answer]
            with self.assertRaisesRegex(ValueError, 'answer span integer'):
                M.validate_adapter(raw(source), self.contract)

    def test_original_author_boundary_strip_preserves_raw_source_and_exact_accepted_mapping(self):
        source = deepcopy(self.source)
        source['data'][0]['paragraphs'][0]['context'] = ' ' + source['data'][0]['paragraphs'][0]['context']
        source['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['answer_start'] = 1
        source['data'][0]['paragraphs'][0]['qas'][1]['answers'][0]['answer_start'] = 1
        source['data'][0]['paragraphs'][0]['qas'][0]['question'] += '  '
        result = M.validate_adapter(raw(source), self.contract)
        prompt = source['data'][0]['paragraphs'][0]['context'] + '\n\n' + source['data'][0]['paragraphs'][0]['qas'][0]['question']
        row = result['mapping']['records'][0]
        self.assertEqual(result['trace'][0]['prompt'], prompt)
        self.assertEqual(row['raw_prompt_sha256'], M.text_sha(prompt))
        self.assertEqual(row['prompt_sha256'], M.text_sha(prompt.strip()))
        self.assertEqual(row['question'], source['data'][0]['paragraphs'][0]['qas'][0]['question'])
        self.assertTrue(row['original_author_boundary_strip_applied'])
        self.assertTrue(result['report']['original_raw_prompt_unchanged'])
        self.assertTrue(result['report']['original_author_boundary_strip_applied'])
        self.assertEqual(result['report']['original_author_boundary_strip_changed_request_count'], 2)
        M.replay_adapter(raw(source), self.contract, result['trace'], result['mapping'], result['report'])
        self.result = result
        self.test_exact_frozen_author_cpu_parser_matches_every_accepted_prompt()

    def test_full_source_bound_version_and_duplicate_keys_reject(self):
        with self.assertRaisesRegex(ValueError, 'whole actual source byte bound'):
            M.validate_adapter(b' ' * (M.MAX_SOURCE_BYTES + 1), self.contract)
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            M.validate_adapter(b'{"version":"1.1","version":"1.1","data":[]}', self.contract)
        source = deepcopy(self.source)
        source['version'] = '2.0'
        with self.assertRaises(ValueError):
            M.validate_adapter(raw(source), self.contract)

    def test_cli_fixture_adapt_replay_append_only_and_no_gpu_claim(self):
        with tempfile.TemporaryDirectory(prefix='CPU_FIXTURE_SQUAD_ONLY_') as directory:
            root = Path(directory)
            source = root / 'CPU_FIXTURE_SOURCE.json'
            contract = root / 'CPU_FIXTURE_CONTRACT.json'
            source.write_bytes(self.input)
            contract.write_bytes(raw(self.contract))
            paths = [root / ('CPU_FIXTURE_' + name + '.json') for name in ('TRACE', 'MAPPING', 'REPORT')]
            command = [sys.executable, '-B', '-I', '-S', str(ADAPTER), '--source', str(source),
                       '--selection-contract', str(contract), '--trace-output', str(paths[0]),
                       '--mapping-output', str(paths[1]), '--report-output', str(paths[2])]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            report = json.loads(proc.stdout)
            self.assertTrue(report['synthetic_fixture'])
            self.assertFalse(report['GPU_launch_allowed'])
            before = [path.read_bytes() for path in paths]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(proc.returncode, 78)
            self.assertEqual(before, [path.read_bytes() for path in paths])
            replay = command[:command.index('--trace-output')] + ['--replay-trace', str(paths[0]),
                '--replay-mapping', str(paths[1]), '--replay-report', str(paths[2])]
            proc = subprocess.run(replay, capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual(json.loads(proc.stdout)['status'], 'PASS_CPU_SOURCE_REPLAY_ONLY')

    def test_failed_validation_writes_nothing_and_missing_output_parent_not_created(self):
        with tempfile.TemporaryDirectory(prefix='CPU_FIXTURE_SQUAD_REJECT_') as directory:
            root = Path(directory)
            bad = deepcopy(self.source)
            bad['data'][0]['paragraphs'][0]['qas'] = []
            source, contract = root / 'CPU_FIXTURE_BAD_SOURCE.json', root / 'CPU_FIXTURE_CONTRACT.json'
            source.write_bytes(raw(bad))
            contract.write_bytes(raw(self.contract))
            outputs = [root / 'absent' / (name + '.json') for name in ('trace', 'mapping', 'report')]
            proc = subprocess.run([sys.executable, '-B', '-I', '-S', str(ADAPTER), '--source', str(source),
                '--selection-contract', str(contract), '--trace-output', str(outputs[0]),
                '--mapping-output', str(outputs[1]), '--report-output', str(outputs[2])],
                capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(proc.returncode, 78)
            self.assertFalse((root / 'absent').exists())
            with self.assertRaisesRegex(ValueError, 'existing output parent'):
                M.append_outputs(trace_path=outputs[0], mapping_path=outputs[1], report_path=outputs[2], result=self.result)
            self.assertFalse((root / 'absent').exists())


if __name__ == '__main__':
    unittest.main()
