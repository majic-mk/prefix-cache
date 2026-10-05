"""Immutable P0 raw evidence. A completed request is not numerical PASS.

No pickle loading, fabricated timing, tolerance defaults, or automatic resume.
CPU fixture evidence can be inspected, but cannot qualify native GPU execution.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import time

from .v8_schema10_event_log import atomic_json
from .v8_schema10_execution import digest_json
from .v8_schema10_storage import file_digest


def write_new_json(path, value):
    path = Path(path)
    if path.exists() or path.with_name(path.name+'.tmp').exists():
        raise FileExistsError('P0 evidence is immutable')
    atomic_json(path, value)
    return dict(file=path.name, sha256=file_digest(path))


class P0EvidenceWriter:
    def __init__(self, root, *, binding, manifest):
        self.root = Path(root)
        # One owner only; a torn/failed directory is never resumed implicitly.
        self.root.mkdir(parents=True, exist_ok=False)
        self.binding = json.loads(json.dumps(binding, allow_nan=False))
        self.previous = '0'*64
        self.sequence = 0
        self.finished = False
        self.manifest = write_new_json(self.root/'manifest.json', manifest)
        self.log_path = self.root/'actions.jsonl'
        self.log_path.touch(exist_ok=False)

    def append(self, kind, action_id, payload):
        if self.finished:
            raise RuntimeError('P0 evidence stream finalized')
        row = dict(sequence=self.sequence+1, previous_sha256=self.previous,
                   binding=self.binding, kind=kind, action_id=action_id, payload=payload)
        row['event_sha256'] = digest_json(row)
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n'
        with self.log_path.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        self.sequence += 1
        self.previous = row['event_sha256']

    def write_action(self, action_id, *, audit, logits, origin, target_layers=None):
        """Serialize already-host logits and optional raw BF16 target slices.

        Transfers/fences belong to the caller. Target K/V is saved layer by
        layer without stacking, floating-point conversion, or pickle.
        """
        import numpy as np
        if not re.fullmatch('[A-Za-z0-9_-]{1,80}', action_id):
            raise ValueError('safe explicit action ID required')
        if origin not in ('real_cuda_execution', 'cpu_fixture'):
            raise ValueError('explicit evidence origin required')
        directory = self.root/action_id
        directory.mkdir(exist_ok=False)
        started = time.perf_counter_ns()
        # Save the answer/partial failure before potentially failing logit I/O.
        raw = write_new_json(directory/'request.json', audit)
        logit_descriptor = None
        if logits:
            rows = []
            for tensor in logits:
                if str(tensor.device) != 'cpu' or str(tensor.dtype) != 'torch.float32':
                    raise ValueError('raw logits must already be CPU FP32; no hidden GPU/D2H work')
                array = tensor.detach().numpy()
                if array.ndim == 2 and array.shape[0] == 1:
                    array = array[0]
                if array.ndim != 1 or not np.isfinite(array).all():
                    raise ValueError('finite one-row logits required')
                rows.append(array)
            array = np.stack(rows)
            path = directory/'logits.npy'
            with path.open('xb') as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush(); os.fsync(stream.fileno())
            logit_descriptor = dict(file=path.name, sha256=file_digest(path), shape=list(array.shape), dtype='float32')
        target_descriptor = (_write_target_kv(directory, target_layers)
                             if target_layers is not None else None)
        record = dict(action_id=action_id, evidence_origin=origin, request=raw,
            logits=logit_descriptor, execution_completed=audit.get('status')=='COMPLETED',
            target_kv=target_descriptor,
            numerical_verdict='NOT_EVALUATED', gpu_runtime_qualified=False,
            evidence_write_host_ms=(time.perf_counter_ns()-started)/1e6)
        descriptor = write_new_json(directory/'record.json', record)
        self.append('action_recorded', action_id, dict(record=descriptor, numerical_verdict='NOT_EVALUATED'))
        return record

    def finalize(self, report):
        self.append('batch_stopped', None, report)
        result = {**report, 'raw_event_sha256': file_digest(self.log_path),
                  'final_event_sha256': self.previous, 'event_count': self.sequence,
                  'gpu_runtime_qualified': False, 'P1_execution_allowed': False,
                  'paper_evidence': False}
        write_new_json(self.root/'result.json', result)
        self.finished = True
        return result


def _write_target_kv(directory, layers):
    import numpy as np
    import torch
    if not isinstance(layers, (tuple, list)) or not layers:
        raise ValueError('target K/V requires an explicit nonempty layer sequence')
    shape = None
    # Validate before writing any target file. Never inspect a CUDA tensor's
    # values (or silently move it) while serializing purported host evidence.
    for pair in layers:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError('every target layer requires both K and V')
        for tensor in pair:
            if (not torch.is_tensor(tensor) or tensor.device.type != 'cpu'
                    or tensor.dtype != torch.bfloat16 or tensor.ndim != 3
                    or min(tensor.shape) < 1):
                raise ValueError('target K/V must already be nonempty CPU BF16 [N,H,D]')
            if shape is None:
                shape = tuple(tensor.shape)
            if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError('target K/V requires finite identical geometry across all layers')
    target_root = directory/'target-kv'
    target_root.mkdir(exist_ok=False)
    descriptors = []
    for index, pair in enumerate(layers, 1):
        row = {'layer_1based': index}
        for field, suffix, tensor in zip(('key', 'value'), ('K', 'V'), pair):
            relative = 'target-kv/layer-{:04d}-{}.npy'.format(index, suffix)
            path = directory/relative
            # The int16 view retains signed zero and every BF16 mantissa bit.
            bits = tensor.detach().contiguous().view(torch.int16).numpy()
            with path.open('xb') as stream:
                np.save(stream, bits, allow_pickle=False)
                stream.flush(); os.fsync(stream.fileno())
            row[field] = {'file': relative, 'sha256': file_digest(path)}
        descriptors.append(row)
    return dict(encoding='bfloat16_bits_int16', logical_dtype='bfloat16',
        shape=list(shape), layer_count=len(layers),
        total_tensor_bytes=len(layers)*2*math.prod(shape)*2, layers=descriptors)


def _fixed_evidence_file(directory, relative):
    path = directory/relative
    if (directory.is_symlink() or path.is_symlink() or not path.is_file()
            or path.resolve().parent != (directory/Path(relative).parent).resolve()):
        raise ValueError('missing or redirected fixed P0 evidence file')
    return path


def _bounded_evidence_json(path):
    # This is a metadata ceiling, independent of the caller's tensor budget.
    if path.stat().st_size > 4*1024*1024:
        raise ValueError('P0 evidence metadata exceeds bounded reader limit')
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def read_p0_target_kv(directory, *, expected_record_sha256, max_bytes):
    """Read hash-bound CPU BF16 target evidence, never a GPU qualification.

    ``max_bytes`` bounds the aggregate returned tensor payload. Files are
    checked sequentially and no all-layer stack or floating copy is created.
    The caller must independently bind layer count to the model/recipe and
    establish whether this is the correct exact or explicit mixed reference.
    """
    import numpy as np
    import torch
    if (type(max_bytes) is not int or max_bytes <= 0
            or not isinstance(expected_record_sha256, str)
            or re.fullmatch('[0-9a-f]{64}', expected_record_sha256) is None):
        raise ValueError('explicit positive tensor budget and record SHA256 required')
    directory = Path(directory)
    record_path = _fixed_evidence_file(directory, 'record.json')
    if file_digest(record_path) != expected_record_sha256:
        raise ValueError('P0 target record digest mismatch')
    record = _bounded_evidence_json(record_path)
    if (record.get('action_id') != directory.name
            or record.get('evidence_origin') not in ('cpu_fixture', 'real_cuda_execution')
            or not isinstance(record.get('request'), dict)
            or record['request'].get('file') != 'request.json'):
        raise ValueError('fixed matching P0 action/request identity required')
    request_path = _fixed_evidence_file(directory, 'request.json')
    if file_digest(request_path) != record['request'].get('sha256'):
        raise ValueError('P0 target request audit digest mismatch')
    _bounded_evidence_json(request_path)
    descriptor = record.get('target_kv')
    if (not isinstance(descriptor, dict)
            or set(descriptor) != {'encoding','logical_dtype','shape','layer_count','total_tensor_bytes','layers'}
            or descriptor['encoding'] != 'bfloat16_bits_int16'
            or descriptor['logical_dtype'] != 'bfloat16'):
        raise ValueError('explicit raw BF16 target descriptor required')
    shape, count = descriptor['shape'], descriptor['layer_count']
    if (type(shape) is not list or len(shape) != 3
            or any(type(n) is not int or n < 1 for n in shape)
            or type(count) is not int or count < 1
            or type(descriptor['layers']) is not list or len(descriptor['layers']) != count):
        raise ValueError('target descriptor must have complete positive layer geometry')
    tensor_bytes = math.prod(shape)*2
    required = count*2*tensor_bytes
    if type(descriptor['total_tensor_bytes']) is not int or descriptor['total_tensor_bytes'] != required:
        raise ValueError('target descriptor byte count differs from geometry')
    if required > max_bytes:
        raise MemoryError('target evidence exceeds explicit aggregate tensor budget')
    target_root = directory/'target-kv'
    if target_root.is_symlink() or not target_root.is_dir():
        raise ValueError('missing or redirected target evidence directory')
    expected_names = {'layer-{:04d}-{}.npy'.format(i, suffix)
                      for i in range(1, count+1) for suffix in ('K', 'V')}
    if {p.name for p in target_root.iterdir()} != expected_names:
        raise ValueError('partial or unexpected target layer files')
    result = []
    for index, row in enumerate(descriptor['layers'], 1):
        if (not isinstance(row, dict) or set(row) != {'layer_1based','key','value'}
                or type(row['layer_1based']) is not int or row['layer_1based'] != index):
            raise ValueError('target layers must be contiguous and ordered from one')
        pair = []
        for field, suffix in (('key','K'), ('value','V')):
            entry = row[field]
            relative = 'target-kv/layer-{:04d}-{}.npy'.format(index, suffix)
            if (not isinstance(entry, dict) or set(entry) != {'file','sha256'}
                    or entry['file'] != relative):
                raise ValueError('target tensor cannot redirect its fixed layer path')
            path = _fixed_evidence_file(directory, relative)
            if path.stat().st_size > tensor_bytes+10000:
                raise ValueError('target tensor file exceeds declared payload/header')
            if file_digest(path) != entry['sha256']:
                raise ValueError('target tensor file digest mismatch')
            with path.open('rb') as stream:
                # Our immutable writer emits NPY v1. Parsing the bounded
                # header before allocation prevents forged shapes/pickle.
                if np.lib.format.read_magic(stream) != (1, 0):
                    raise ValueError('target tensor requires the writer NPY v1 encoding')
                actual_shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                    stream, max_header_size=10000)
                if (tuple(actual_shape) != tuple(shape) or fortran or dtype != np.dtype('int16')
                        or path.stat().st_size != stream.tell()+tensor_bytes):
                    raise ValueError('raw target tensor shape/dtype/length differs')
                bits = np.fromfile(stream, dtype=np.int16, count=math.prod(shape)).reshape(shape)
            tensor = torch.from_numpy(bits).view(torch.bfloat16)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError('raw target BF16 bits contain nonfinite values')
            pair.append(tensor)
        result.append(tuple(pair))
    return tuple(result)


def read_p0_events(path, *, binding):
    previous, rows = '0'*64, []
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith('\n'):
                raise ValueError('torn P0 event; preserve and start a new output directory')
            row = json.loads(line)
            sha = row.pop('event_sha256')
            if (row['sequence'] != number or row['previous_sha256'] != previous
                    or row['binding'] != binding or digest_json(row) != sha):
                raise ValueError('P0 event chain or binding mismatch')
            previous = sha; rows.append({**row, 'event_sha256': sha})
    return rows


def compare_logit_files(reference, candidate, *, expected_hashes, relative_l2_limit,
                        minimum_positions):
    """Raw numeric comparison only. Caller must validate same execution recipe.

    There is deliberately no default BF16 tolerance or GPU qualification flag.
    Zero reference norm uses exact equality, never an invented epsilon.
    """
    import math
    import numpy as np
    if (type(relative_l2_limit) not in (int, float) or not math.isfinite(relative_l2_limit)
            or relative_l2_limit < 0 or type(minimum_positions) is not int or minimum_positions < 1):
        raise ValueError('preregistered finite tolerance and positive position count required')
    paths = (Path(reference), Path(candidate))
    if len(expected_hashes) != 2 or any(file_digest(p) != h for p,h in zip(paths, expected_hashes)):
        raise ValueError('raw logit digest mismatch')
    a,b = (np.load(p, allow_pickle=False) for p in paths)
    if (a.dtype != np.float32 or b.dtype != np.float32 or a.ndim != 2
            or a.shape != b.shape or a.shape[0] < minimum_positions or a.shape[1] < 1
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('finite matched FP32 teacher-position/vocabulary arrays required')
    a,b = a.astype(np.float64), b.astype(np.float64)
    norm = np.linalg.norm(a, axis=1)
    diff = np.linalg.norm(a-b, axis=1)
    values = [float(d/n) if n else (0. if d == 0 else None) for d,n in zip(diff,norm)]
    passed = all(v is not None and v <= relative_l2_limit for v in values)
    return dict(numeric_passed=passed, positions=len(values), relative_l2_by_position=values,
                maximum_relative_l2=max(values) if all(v is not None for v in values) else None,
                relative_l2_limit=relative_l2_limit, native_runtime_qualified=False,
                recipe_alignment_verified=False, evidence_scope='raw_numeric_only')


def compare_p0_logit_actions(reference_directory, candidate_directory, *,
                            record_hashes, manifest_hashes, relative_l2_limit,
                            minimum_positions):
    """Verify raw action files AND actual teacher feeds before comparing logits.

    Deliberately not an exact/mixed qualification gate: matching decode inputs
    does not prove matching upstream Source injections, masks, or r1 recipes.
    A future recipe verifier must establish those separately. This API cannot
    authorize P1, even when both rows came from CUDA and numeric_passed is true.
    """
    from .p0_decode_evidence_v2 import validate_decode_trace, assert_same_logit_conditioning
    if len(record_hashes) != 2 or len(manifest_hashes) != 2:
        raise ValueError('two frozen raw record and manifest hashes required')
    loaded = []
    for directory, record_sha, manifest_sha in zip(
            (Path(reference_directory), Path(candidate_directory)), record_hashes, manifest_hashes):
        manifest_path, record_path = directory.parent/'manifest.json', directory/'record.json'
        if file_digest(manifest_path) != manifest_sha or file_digest(record_path) != record_sha:
            raise ValueError('P0 manifest or action record digest mismatch')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        record = json.loads(record_path.read_text(encoding='utf-8'))
        jobs = [j for j in manifest['jobs'] if j['action_id'] == directory.name]
        if len(jobs) != 1 or record['action_id'] != directory.name:
            raise ValueError('action is not uniquely bound in frozen manifest')
        job = jobs[0]
        if digest_json(job['request']) != job['request_sha256']:
            raise ValueError('bound request digest mismatch')
        if (record['request']['file'] != 'request.json' or record.get('logits') is None
                or record['logits']['file'] != 'logits.npy'):
            raise ValueError('fixed raw request/logit files required; no redirected paths')
        if file_digest(directory/'request.json') != record['request']['sha256']:
            raise ValueError('raw request audit digest mismatch')
        audit = json.loads((directory/'request.json').read_text(encoding='utf-8'))
        if (audit.get('status') != 'COMPLETED' or audit.get('cleanup', {}).get('passed') is not True
                or record.get('execution_completed') is not True
                or audit.get('request_id') != job['request']['request_id']):
            raise ValueError('completed matching action and successful cleanup required')
        answer = audit['answer']
        if (record['evidence_origin'] not in ('cpu_fixture', 'real_cuda_execution')
                or record['logits'].get('dtype') != 'float32'
                or len(record['logits']['shape']) != 2):
            raise ValueError('raw evidence origin and FP32 logit geometry required')
        trace = validate_decode_trace(answer.get('decode_input_trace_v2'),
            request=job['request'], answer=answer, logit_rows=record['logits']['shape'][0])
        if (trace['mode'] != 'teacher_forced'
                or answer.get('generation_mode') != 'teacher_forced_logit_diagnostic'):
            raise ValueError('numerical teacher comparison cannot consume ordinary QA logits')
        loaded.append((manifest, record, trace))
    binding_fields = ('code_commit', 'runtime_digest', 'patch_sha256', 'model_signature',
                      'tokenizer_hash', 'gpu_uuid', 'instance_id')
    for key in binding_fields:
        left, right = loaded[0][0]['binding'].get(key), loaded[1][0]['binding'].get(key)
        if not isinstance(left, str) or not left or left != right:
            raise ValueError('numerical pair runtime/model/device binding mismatch: '+key)
    assert_same_logit_conditioning(loaded[0][2], loaded[1][2])
    result = compare_logit_files(Path(reference_directory)/'logits.npy', Path(candidate_directory)/'logits.npy',
        expected_hashes=tuple(item[1]['logits']['sha256'] for item in loaded),
        relative_l2_limit=relative_l2_limit, minimum_positions=minimum_positions)
    # The descriptor shape is also checked against the actual array, not just
    # against a potentially self-consistent but incorrect decode trace.
    import numpy as np
    for directory, item in zip((reference_directory, candidate_directory), loaded):
        actual = np.load(Path(directory)/'logits.npy', allow_pickle=False)
        if list(actual.shape) != item[1]['logits']['shape']:
            raise ValueError('logit descriptor does not match actual raw array')
    return {**result, 'conditioning_alignment_verified': True,
            'predicted_token_ids_identical': loaded[0][2]['predicted_token_ids'] == loaded[1][2]['predicted_token_ids'],
            'evidence_origins': [item[1]['evidence_origin'] for item in loaded],
            'recipe_alignment_verified': False, 'P1_execution_allowed': False,
            'pending_recipe_checks': ['upstream_execution', 'source_artifacts', 'repair_masks',
                                      'target_full_layer_ledger', 'exact_or_explicit_mixed_reference'],
            'evidence_scope': 'conditioning_aligned_logits_only_not_native_qualification'}
