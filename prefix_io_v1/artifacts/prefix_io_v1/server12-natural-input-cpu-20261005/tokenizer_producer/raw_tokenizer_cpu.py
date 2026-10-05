"""Bounded, local CPU raw token IDs from the unchanged author's prompt parser.

This is deliberately not a formal tokenizer/family receipt. No family labels,
partitions, deadlines, model execution, cache directories or GPU authority are
produced. With no actual dataset, UNBOUND precedes every tokenizer import.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import re
import sys


REQUEST_SCHEMA = 'bounded_author_cpu_raw_tokenization_request_v1'
RESULT_SCHEMA = 'actual_cpu_raw_tokenization_v1'
PROTOCOL_SHA = '7fbc544597c9abe334122d354e7494b590e810ab0682f1d2d155408cc4f618eb'
AUTHOR_TRACE_SHA = '83713db1b011a954616896ec0ccfa05dc766805e80ed8defdef0da65100ff99e'
AUTHOR_COMMON_SHA = 'a331d1595b815c3c8fc63d3e2f236264c2f181ece7a9de51f50beb4703338df5'
MODEL_PLAN_SHA = '9b1c79986aa1fd62eb9064ecb8b4e2a03fc235a34ac2d69c33a4fcb531a91017'
TOKENIZER_JSON_SHA = 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'
TOKENIZER_CONFIG_SHA = '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583'
BACKEND_VERSION = '0.22.2'
BACKEND_PINS = {
    '__init__.py': '148ecb122f3fee03be9abb1fe213dd85fbc691bf2ad2b437ff531533781a6a3a',
    'tokenizers.abi3.so': 'c116fcf1e80d461ce0a35c332974f25949e8359416f50b3d53371810d2ce1ccc',
}
DIST_PINS = {
    'METADATA': '15a5ddaf489f592b77e0a934c0eeb46b51130a94064ca129232afdb0a3868efc',
    'RECORD': 'f6cf016a34d1eb2dbf38bb6b0896b18f1d81651c97ddf514674a42ec3e958bc7',
}
REQUEST_FIELDS = {'schema', 'dataset_ref', 'protocol_ref', 'author_trace_ref', 'author_common_ref',
                  'model_manifest_ref', 'tokenizer_json_ref', 'tokenizer_config_ref',
                  'tokenizers_package_root', 'tokenizers_dist_info_root', 'synthetic_fixture'}
MAX_BYTES = 32 * 1024**2
MAX_REQUESTS = 96
MAX_PROMPT_TOKENS = 4096
MAX_TOKEN_ID = 152064
FORBIDDEN_IMPORTS = ('torch', 'transformers', 'vllm', 'py_kvcache', 'kvcache',
                     'huggingface_hub', 'requests', 'httpx', 'urllib.request')


def require(condition, reason):
    if not condition:
        raise ValueError('RAW_CPU_TOKENIZATION_REJECTED: ' + reason)


def parse(raw):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, 'nonfinite JSON'))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def safe(root, relative):
    root = Path(root).resolve(strict=True)
    require(type(relative) is str and relative and not relative.startswith('/') and
            ':' not in relative and '\\' not in relative and '\0' not in relative and
            all(part not in ('', '.', '..') for part in relative.split('/')), 'project-relative POSIX path')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'symlink input/output')
    require(path.resolve().is_relative_to(root), 'actual project containment')
    return path


def file_ref(root, relative):
    path = safe(root, relative)
    require(path.is_file(), 'actual regular file required: ' + relative)
    size = path.stat().st_size
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            hasher.update(block)
    require(path.stat().st_size == size, 'file changed while hashing')
    return dict(path=relative, bytes=size, sha256=hasher.hexdigest())


def check_ref(root, row):
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'} and
            type(row['bytes']) is int and row['bytes'] >= 0 and type(row['sha256']) is str and
            re.fullmatch('[0-9a-f]{64}', row['sha256']), 'strict actual file ref')
    require(file_ref(root, row['path']) == row, 'actual byte SHA/size mismatch: ' + row['path'])
    return safe(root, row['path'])


def bounded_json(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_BYTES,
            'bounded actual JSON source')
    return parse(path.read_bytes())


def current_refs(root, lock_ref):
    lock = bounded_json(check_ref(root, lock_ref))
    require(type(lock) is dict and type(lock.get('files')) is list and lock['files'], 'current source lock leaves')
    refs = {}
    for row in lock['files']:
        require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'} and
                type(row['path']) is str and row['path'] not in refs, 'unique exact current source refs')
        refs[row['path']] = row
    return refs


def closed(root, row, refs, *, expected_sha=None):
    require(type(row) is dict and refs.get(row.get('path')) == row, 'leaf absent from current source lock')
    path = check_ref(root, row)
    require(expected_sha is None or row['sha256'] == expected_sha, 'pinned original source drift')
    return path


def cpu_modules_only():
    require(not any(name == blocked or name.startswith(blocked + '.')
                    for name in sys.modules for blocked in FORBIDDEN_IMPORTS), 'CPU-only import namespace')


def load_protocol(path):
    """Load the exact frozen standard-library CPU protocol, never the author client."""
    spec = importlib.util.spec_from_file_location('_frozen_raw_cpu_protocol', path)
    require(spec is not None and spec.loader is not None, 'actual original protocol loader')
    module = importlib.util.module_from_spec(spec)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PROTOCOL_SHA, 'exact original protocol execution bytes')
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        # SourceFileLoader can read an old timestamp-valid .pyc even when
        # dont_write_bytecode=True. Execute only the actual pinned source bytes.
        exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _tree_leaves(root, relative, predicate):
    base = safe(root, relative)
    require(base.is_dir(), 'actual tokenizer package/dist-info directory')
    paths = []
    for path in base.rglob('*'):
        require(not path.is_symlink(), 'symlink tokenizer tree leaf')
        if path.is_file() and predicate(path):
            paths.append(path.relative_to(Path(root).resolve()).as_posix())
    return sorted(paths)


def backend_sources(root, request, refs):
    package = request['tokenizers_package_root']
    dist = request['tokenizers_dist_info_root']
    require(safe(root, package).name == 'tokenizers' and
            safe(root, dist).name == 'tokenizers-' + BACKEND_VERSION + '.dist-info' and
            safe(root, package).parent == safe(root, dist).parent, 'fixed local tokenizer package/version')
    pkgpaths = _tree_leaves(root, package, lambda p: p.suffix in ('.py', '.so', '.pyd'))
    distpaths = _tree_leaves(root, dist, lambda p: p.suffix not in ('.pyc', '.pyo'))
    require(pkgpaths and distpaths, 'full actual package and metadata leaves')
    selected = []
    for name in pkgpaths + distpaths:
        require(name in refs, 'tokenizer source leaf not in current lock: ' + name)
        closed(root, refs[name], refs)
        selected.append(refs[name])
    for name, pin in BACKEND_PINS.items():
        closed(root, refs.get(package + '/' + name), refs, expected_sha=pin)
    for name, pin in DIST_PINS.items():
        closed(root, refs.get(dist + '/' + name), refs, expected_sha=pin)
    require(dist + '/WHEEL' in refs and dist + '/WHEEL' in distpaths, 'installed wheel metadata closure')
    metadata = safe(root, dist + '/METADATA').read_text(encoding='utf-8')
    require('\nName: tokenizers\n' in '\n' + metadata and
            '\nVersion: ' + BACKEND_VERSION + '\n' in '\n' + metadata, 'installed tokenizers version metadata')
    return selected


def validate_assets(tokenizer, config):
    require(type(tokenizer) is dict and type(config) is dict and
            tokenizer.get('model', {}).get('type') == 'BPE' and
            tokenizer.get('truncation') is None and tokenizer.get('padding') is None,
            'local BPE tokenizer with no truncation/padding')
    require(config.get('tokenizer_class') == 'Qwen2Tokenizer' and config.get('add_bos_token') is False and
            config.get('add_eos_token') in (None, False) and config.get('split_special_tokens') is False and
            config.get('add_prefix_space') is False, 'frozen plain Qwen text tokenizer flags')
    post = tokenizer.get('post_processor')
    require(type(post) is dict and post == dict(type='ByteLevel', add_prefix_space=False,
                                               trim_offsets=False, use_regex=False),
            'unchanged ByteLevel postprocessor')


class _CPUImportGate:
    def find_spec(self, fullname, path=None, target=None):
        require(not any(fullname == name or fullname.startswith(name + '.')
                        for name in FORBIDDEN_IMPORTS), 'framework/model/network client import forbidden: ' + fullname)
        return None


class _ClosedSourceLoader:
    """Read no .pyc: compile the exact current frozen source leaf directly."""
    def __init__(self, root, row, refs):
        self.root, self.row, self.refs = root, row, refs

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = closed(self.root, self.row, self.refs)
        raw = path.read_bytes()
        require(len(raw) == self.row['bytes'] and hashlib.sha256(raw).hexdigest() == self.row['sha256'],
                'exact frozen tokenizer execution source bytes')
        exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)


class _TokenizersSourceOnlyFinder:
    """Resolve only closed package .py or original pinned extension leaves.

    Existing __pycache__ files may remain on disk. They are never candidates.
    The original CPython extension loader still loads the actual frozen .so.
    """
    def __init__(self, root, request, refs):
        self.root, self.request, self.refs = root, request, refs

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'tokenizers' and not fullname.startswith('tokenizers.'):
            return None
        parts = fullname.split('.')[1:]
        require(all(part.isidentifier() for part in parts), 'actual tokenizer module name')
        relative = self.request['tokenizers_package_root']
        if parts:
            relative += '/' + '/'.join(parts)
        candidates = ((relative + '/__init__.py', True), (relative + '.py', False))
        for relative_file, package in candidates:
            actual = safe(self.root, relative_file)
            if actual.is_file():
                row = self.refs.get(relative_file)
                closed(self.root, row, self.refs)
                return importlib.util.spec_from_file_location(
                    fullname, actual, loader=_ClosedSourceLoader(self.root, row, self.refs),
                    submodule_search_locations=[str(actual.parent)] if package else None)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            relative_file = relative + suffix
            actual = safe(self.root, relative_file)
            if actual.is_file():
                closed(self.root, self.refs.get(relative_file), self.refs)
                return importlib.util.spec_from_file_location(
                    fullname, actual, loader=importlib.machinery.ExtensionFileLoader(fullname, str(actual)))
        require(False, 'tokenizer module has no frozen source or extension; cached-only import forbidden')


@contextmanager
def actual_local_backend(root, request, refs, sources):
    """Only local Tokenizer.from_file + encode, all package module origins closed."""
    cpu_modules_only()
    require(backend_sources(root, request, refs) == sources, 'same complete frozen backend source leaves')
    closed(root, request['tokenizer_json_ref'], refs, expected_sha=TOKENIZER_JSON_SHA)
    require(not any(name == 'tokenizers' or name.startswith('tokenizers.') for name in sys.modules),
            'fresh actual tokenizers module namespace required')
    previous_path = list(sys.path)
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    gate = _CPUImportGate()
    source_finder = _TokenizersSourceOnlyFinder(root, request, refs)
    sys.meta_path.insert(0, gate)
    sys.meta_path.insert(1, source_finder)
    sys.path.insert(0, str(safe(root, request['tokenizers_package_root']).parent))
    try:
        module = importlib.import_module('tokenizers')
        require(module.__version__ == BACKEND_VERSION, 'loaded pinned tokenizer backend version')
        loaded = []
        for name, mod in list(sys.modules.items()):
            if name == 'tokenizers' or name.startswith('tokenizers.'):
                location = Path(getattr(mod, '__file__', '')).resolve(strict=True)
                require(location.is_relative_to(safe(root, request['tokenizers_package_root'])),
                        'loaded tokenizer origin inside pinned package')
                relative = location.relative_to(Path(root).resolve()).as_posix()
                closed(root, refs.get(relative), refs)
                loaded.append(dict(module=name, source_ref=refs[relative]))
        backend = module.Tokenizer.from_file(str(safe(root, request['tokenizer_json_ref']['path'])))
        require(backend.truncation is None and backend.padding is None and backend.encode_special_tokens is False,
                'loaded backend never truncates/pads/splits original special tokens')
        audit = dict(schema='actual_local_CPU_tokenizer_backend_v1', version=module.__version__,
                     loaded_module_sources=sorted(loaded, key=lambda row: row['module']),
                     from_file_only=True, add_special_tokens=False, chat_template_applied=False,
                     backend_mocked=False)
        yield (lambda prompt: backend.encode(prompt, add_special_tokens=False).ids), audit
        cpu_modules_only()
        closed(root, request['tokenizer_json_ref'], refs, expected_sha=TOKENIZER_JSON_SHA)
        require(backend_sources(root, request, refs) == sources, 'backend sources unchanged after local CPU encoding')
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode
        if gate in sys.meta_path:
            sys.meta_path.remove(gate)
        if source_finder in sys.meta_path:
            sys.meta_path.remove(source_finder)
        for name in list(sys.modules):
            if name == 'tokenizers' or name.startswith('tokenizers.'):
                del sys.modules[name]


def unbound():
    return dict(schema=RESULT_SCHEMA, status='UNBOUND_NO_ACTUAL_DATASET', exit_code=78,
                tokenizer_imported=False, records=[], family_proof_available=False,
                formal_receipt_ready=False, gpu_eligible=False, formal_effect_qualified=False,
                actual_GPU_operations=0)


def validate_author_rows(prompts, inspection):
    """Recheck original parser denominator and exact prompt order before import."""
    require(type(prompts) is list and 1 <= len(prompts) <= MAX_REQUESTS,
            'whole author-accepted corpus must contain 1..96 requests; no subset selection')
    require(len(prompts) == inspection.get('accepted_prompt_count') and
            type(inspection.get('records')) is list and len(inspection['records']) == len(prompts),
            'actual author inspection denominator')
    observed = [dict(request_id=index, prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                     utf8_bytes=len(prompt.encode('utf-8'))) for index, prompt in enumerate(prompts)]
    require(observed == inspection['records'] and digest(observed) == inspection['ordered_prompt_digest'],
            'original author parser order/prompt bytes')
    return observed


def tokenize_author_rows(prompts, inspection, encode):
    """Keep the complete accepted corpus and original order; reject whole overflow."""
    observed = validate_author_rows(prompts, inspection)
    rows = []
    for prompt, source in zip(prompts, observed):
        ids = encode(prompt)
        require(type(ids) is list and 1 <= len(ids) <= MAX_PROMPT_TOKENS and
                all(type(value) is int and 0 <= value < MAX_TOKEN_ID for value in ids),
                'complete raw integer token IDs 1..4096 in original model vocabulary')
        rows.append(dict(request_id=source['request_id'], prompt_sha256=source['prompt_sha256'],
                         prompt_token_ids=ids))
    return rows


def produce(root, *, request, request_ref=None, source_lock_ref=None,
            output_relative=None, backend_factory=None):
    """Append raw IDs once. Injected backends are fixture-only, never receipts."""
    if type(request) is dict and request.get('dataset_ref') is None:
        return unbound()
    cpu_modules_only()
    require(type(request) is dict and set(request) == REQUEST_FIELDS and request['schema'] == REQUEST_SCHEMA and
            type(request['synthetic_fixture']) is bool, 'strict bounded raw tokenization request')
    require(backend_factory is None or request['synthetic_fixture'] is True,
            'mock/injected backend prohibited for an actual result')
    refs = current_refs(root, source_lock_ref)
    source_relative = Path(__file__).resolve(strict=True).relative_to(Path(root).resolve(strict=True)).as_posix()
    producer_ref = refs.get(source_relative)
    closed(root, producer_ref, refs)
    request_path = closed(root, request_ref, refs)
    require(bounded_json(request_path) == request, 'actual frozen request JSON bytes')
    pins = {'protocol_ref': PROTOCOL_SHA, 'author_trace_ref': AUTHOR_TRACE_SHA,
            'author_common_ref': AUTHOR_COMMON_SHA, 'model_manifest_ref': MODEL_PLAN_SHA,
            'tokenizer_json_ref': TOKENIZER_JSON_SHA, 'tokenizer_config_ref': TOKENIZER_CONFIG_SHA}
    paths = {name: closed(root, request[name], refs, expected_sha=pin) for name, pin in pins.items()}
    dataset = closed(root, request['dataset_ref'], refs)
    require(dataset.stat().st_size <= MAX_BYTES, 'complete actual dataset exceeds 32 MiB; no truncation')
    tokenizer = bounded_json(paths['tokenizer_json_ref'])
    config = bounded_json(paths['tokenizer_config_ref'])
    validate_assets(tokenizer, config)
    sources = backend_sources(root, request, refs)
    output = safe(root, output_relative)
    require(output_relative.startswith('artifacts/') and output.parent.is_dir() and not output.exists(),
            'new append-only output in an existing artifact directory')
    protocol = load_protocol(paths['protocol_ref'])
    inspection = protocol.inspect_trace(dataset, paths['author_trace_ref'], paths['author_common_ref'],
                                        request['dataset_ref']['sha256'])
    prompts = protocol.author_cpu_namespace(paths['author_trace_ref'], paths['author_common_ref'])[
        'load_trace_prompts'](str(dataset), None)
    validate_author_rows(prompts, inspection)
    factory = actual_local_backend if backend_factory is None else backend_factory
    with factory(root, request, refs, sources) as (encode, backend_audit):
        require(type(backend_audit) is dict and
                backend_audit.get('backend_mocked') is (backend_factory is not None), 'actual versus fixture backend identity')
        rows = tokenize_author_rows(prompts, inspection, encode)
    cpu_modules_only()
    selected = [producer_ref, request_ref, request['dataset_ref'], *[request[name] for name in pins], *sources]
    # Actual bytes are closed again after parsing/encoding, before the first output write.
    for row in selected:
        closed(root, row, refs)
    require(backend_sources(root, request, refs) == sources, 'unchanged whole tokenizer source tree after encoding')
    check_ref(root, source_lock_ref)
    result = dict(schema=RESULT_SCHEMA, status='CPU_FIXTURE_RAW_IDS_ONLY' if request['synthetic_fixture'] else
                  'ACTUAL_CPU_RAW_IDS_ONLY', exit_code=0, synthetic_fixture=request['synthetic_fixture'],
                  dataset_ref=request['dataset_ref'], request_ref=request_ref, source_lock_ref=source_lock_ref,
                  model_manifest_ref=request['model_manifest_ref'], tokenizer_backend=backend_audit,
                  tokenizer_imported=backend_factory is None,
                  sources=selected, inspection=inspection, records=rows,
                  raw_records_sha256=digest(rows), accepted_prompt_count=len(rows),
                  author_max_prompts=None, all_author_accepted_prompts_preserved=True,
                  prompt_mode='unchanged_author_plain_text_no_chat_template',
                  add_special_tokens=False, truncation=None, padding=None,
                  family_proof_available=False, formal_receipt_ready=False,
                  gpu_eligible=False, formal_effect_qualified=False, actual_GPU_operations=0)
    require(safe(root, output_relative) == output and output.parent.is_dir() and not output.exists(),
            'output still fresh after all read-only CPU validation')
    with output.open('xb') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode('utf-8') + b'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--source-lock', help='actual project-relative frozen CPU source lock')
    parser.add_argument('--request', help='actual project-relative raw request JSON; absent dataset returns UNBOUND')
    parser.add_argument('--output-relative', help='new JSON inside an already existing artifacts directory')
    args = parser.parse_args(argv)
    try:
        request = bounded_json(safe(args.project_root, args.request)) if args.request else {'dataset_ref': None}
        if type(request) is dict and request.get('dataset_ref') is None:
            result = unbound()
        else:
            require(args.source_lock and args.output_relative, 'bound input requires source lock and output path')
            result = produce(args.project_root, request=request,
                             request_ref=file_ref(args.project_root, args.request),
                             source_lock_ref=file_ref(args.project_root, args.source_lock),
                             output_relative=args.output_relative)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return result['exit_code']
    except (ValueError, OSError, ImportError, AttributeError) as error:
        print(json.dumps(dict(schema=RESULT_SCHEMA, status='REJECTED_CPU_INPUT_OR_SOURCE',
                              exit_code=78, error=str(error), formal_receipt_ready=False,
                              gpu_eligible=False, actual_GPU_operations=0), ensure_ascii=False, sort_keys=True))
        return 78


if __name__ == '__main__':
    raise SystemExit(main())
