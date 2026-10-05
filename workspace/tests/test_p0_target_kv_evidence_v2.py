import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from probekv.p0_evidence_v2 import P0EvidenceWriter, read_p0_target_kv
from probekv.v8_schema10_storage import file_digest


class P0TargetKVEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)/'new'
        self.writer = P0EvidenceWriter(self.root, binding={'fixture': True}, manifest={'fixture': True})
        self.layers = tuple((torch.arange(12, dtype=torch.bfloat16).reshape(3, 2, 2)+i,
                             torch.full((3, 2, 2), .125+i, dtype=torch.bfloat16))
                            for i in range(3))

    def write(self, layers=None, action='mixed'):
        return self.writer.write_action(action, audit={'status': 'COMPLETED', 'answer': None},
            logits=[], origin='cpu_fixture', target_layers=self.layers if layers is None else layers)

    def read(self, action='mixed', max_bytes=144, sha=None):
        directory = self.root/action
        return read_p0_target_kv(directory,
            expected_record_sha256=sha or file_digest(directory/'record.json'), max_bytes=max_bytes)

    def mutate_record(self, mutation):
        path = self.root/'mixed/record.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        mutation(data)
        path.write_text(json.dumps(data), encoding='utf-8')

    def replace_tensor(self, array, layer=1, field='key'):
        path = self.root/'mixed/target-kv'/'layer-{:04d}-{}.npy'.format(layer, 'K' if field == 'key' else 'V')
        with path.open('wb') as stream:
            np.save(stream, array, allow_pickle=False)
        self.mutate_record(lambda record: record['target_kv']['layers'][layer-1][field].update(sha256=file_digest(path)))

    def test_roundtrip_preserves_raw_bits_and_never_qualifies(self):
        # Include signed zero and a BF16 subnormal: equality of values alone
        # would not establish lossless storage of these original bit patterns.
        self.layers[0][0].view(torch.int16).flatten()[0] = -32768
        self.layers[0][0].view(torch.int16).flatten()[1] = 1
        record = self.write()
        self.assertEqual(record['target_kv']['encoding'], 'bfloat16_bits_int16')
        self.assertEqual(record['target_kv']['total_tensor_bytes'], 144)
        self.assertIsNone(record['logits'])
        self.assertEqual(record['numerical_verdict'], 'NOT_EVALUATED')
        self.assertFalse(record['gpu_runtime_qualified'])
        actual = self.read()
        for expected_pair, actual_pair in zip(self.layers, actual):
            for expected, tensor in zip(expected_pair, actual_pair):
                self.assertEqual(tensor.device.type, 'cpu')
                self.assertEqual(tensor.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(expected.view(torch.int16), tensor.view(torch.int16)))
        self.assertEqual(len(list((self.root/'mixed/target-kv').iterdir())), 6)

    def test_regular_action_has_explicit_none_and_cannot_be_read_as_target(self):
        record = self.writer.write_action('ordinary', audit={'status': 'FAILED'}, logits=[], origin='cpu_fixture')
        self.assertIsNone(record['target_kv'])
        with self.assertRaises(ValueError): self.read(action='ordinary')

    def test_writer_does_not_stack_or_run_cuda(self):
        with patch('numpy.stack', side_effect=AssertionError('no all-layer stack')), \
                patch('torch.cuda.synchronize', side_effect=AssertionError('no CUDA work')):
            self.write(); result = self.read()
        self.assertEqual(len(result), 3)

    def test_noncontiguous_cpu_input_preserves_values(self):
        layers = tuple(tuple(t.transpose(1, 2) for t in pair) for pair in self.layers)
        self.write(layers)
        result = self.read()
        for expected_pair, actual_pair in zip(layers, result):
            for expected, actual in zip(expected_pair, actual_pair):
                self.assertTrue(torch.equal(expected, actual))

    def test_writer_rejects_dtype_shape_partial_empty_and_nonfinite(self):
        invalid = [(), ((self.layers[0][0],),),
                   ((torch.ones(3, 2, 2), self.layers[0][1]),),
                   ((torch.ones(3, 2, 1, dtype=torch.bfloat16), self.layers[0][1]),),
                   ((torch.empty(0, 2, 2, dtype=torch.bfloat16), self.layers[0][1]),),
                   ((torch.full((3, 2, 2), float('inf'), dtype=torch.bfloat16), self.layers[0][1]),),
                   ((torch.full((3, 2, 2), float('nan'), dtype=torch.bfloat16), self.layers[0][1]),)]
        for index, layers in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.write(layers, action='bad{}'.format(index))
            self.assertTrue((self.root/'bad{}'.format(index)/'request.json').exists())
            self.assertFalse((self.root/'bad{}'.format(index)/'record.json').exists())

    def test_writer_rejects_non_host_before_value_access(self):
        # Meta tensors let this CPU test prove device rejection before any
        # finite check or conversion, without allocating CUDA memory.
        pair = tuple(torch.empty(3, 2, 2, dtype=torch.bfloat16, device='meta') for _ in range(2))
        with self.assertRaises(ValueError): self.write((pair,))

    def test_action_is_immutable(self):
        self.write()
        digest = file_digest(self.root/'mixed/record.json')
        with self.assertRaises(FileExistsError): self.write()
        self.assertEqual(file_digest(self.root/'mixed/record.json'), digest)

    def test_wrong_record_and_request_checksums_rejected(self):
        self.write()
        with self.assertRaises(ValueError): self.read(sha='0'*64)
        with (self.root/'mixed/request.json').open('a', encoding='utf-8') as stream: stream.write(' ')
        with self.assertRaises(ValueError): self.read()

    def test_wrong_tensor_checksum_rejected(self):
        self.write()
        path = self.root/'mixed/target-kv/layer-0001-K.npy'
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        with self.assertRaises(ValueError): self.read()

    def test_budget_is_required_and_checked_before_tensor_loading(self):
        self.write()
        with self.assertRaises(TypeError):
            read_p0_target_kv(self.root/'mixed', expected_record_sha256=file_digest(self.root/'mixed/record.json'))
        with patch('numpy.fromfile', side_effect=AssertionError('must reject before allocation')):
            with self.assertRaises(MemoryError): self.read(max_bytes=143)
        for budget in (0, -1, True, 144.):
            with self.subTest(budget=budget), self.assertRaises(ValueError): self.read(max_bytes=budget)

    def test_partial_missing_and_unexpected_files_rejected(self):
        self.write()
        path = self.root/'mixed/target-kv/layer-0002-V.npy'
        original = path.read_bytes(); path.unlink()
        with self.assertRaises(ValueError): self.read()
        path.write_bytes(original)
        (path.parent/'unexpected.npy').write_bytes(b'not part of descriptor')
        with self.assertRaises(ValueError): self.read()

    def test_descriptor_cannot_redirect_request_or_tensor(self):
        self.write()
        self.mutate_record(lambda r: r['target_kv']['layers'][0]['key'].update(file='../outside.npy'))
        with self.assertRaises(ValueError): self.read()
        self.mutate_record(lambda r: r['target_kv']['layers'][0]['key'].update(file='target-kv/layer-0001-K.npy'))
        self.mutate_record(lambda r: r['request'].update(file='../request.json'))
        with self.assertRaises(ValueError): self.read()

    def test_descriptor_requires_encoding_layer_order_geometry_and_bytes(self):
        self.write()
        path = self.root/'mixed/record.json'; original = path.read_text(encoding='utf-8')
        changes = [lambda r: r['target_kv'].update(encoding='float32'),
                   lambda r: r['target_kv'].update(logical_dtype='float32'),
                   lambda r: r['target_kv'].update(layer_count=4),
                   lambda r: r['target_kv'].update(shape=[3, 2, True]),
                   lambda r: r['target_kv'].update(total_tensor_bytes=143),
                   lambda r: r['target_kv']['layers'][1].update(layer_1based=3)]
        for index, change in enumerate(changes):
            path.write_text(original, encoding='utf-8'); self.mutate_record(change)
            with self.subTest(index=index), self.assertRaises(ValueError): self.read()

    def test_forged_self_consistent_descriptor_does_not_override_raw_geometry(self):
        self.write()
        self.replace_tensor(np.zeros((3, 2, 1), dtype=np.int16))
        with self.assertRaises(ValueError): self.read()

    def test_wrong_raw_dtype_even_with_rehashed_file_rejected(self):
        self.write()
        self.replace_tensor(np.ones((3, 2, 2), dtype=np.float16))
        with self.assertRaises(ValueError): self.read()

    def test_nonfinite_bf16_bits_even_with_rehashed_file_rejected(self):
        self.write()
        bits = np.zeros((3, 2, 2), dtype=np.int16); bits.flat[0] = 0x7f80
        self.replace_tensor(bits)
        with self.assertRaises(ValueError): self.read()

    def test_truncated_or_appended_payload_is_rejected_after_rehash(self):
        self.write()
        path = self.root/'mixed/target-kv/layer-0001-K.npy'; original = path.read_bytes()
        for payload in (original[:-1], original+b'garbage'):
            path.write_bytes(payload)
            self.mutate_record(lambda r: r['target_kv']['layers'][0]['key'].update(sha256=file_digest(path)))
            with self.assertRaises(ValueError): self.read()


if __name__ == '__main__': unittest.main()
