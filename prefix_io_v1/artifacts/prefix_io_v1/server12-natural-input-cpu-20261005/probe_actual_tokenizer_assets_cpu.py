"""Diagnostic CPU check of existing tokenizer assets, never a natural trace receipt."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

MODEL = 'models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444'
TOKENIZER_SHA = 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'
CONFIG_SHA = '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583'
LEDGER_SHA = '774847ccdb7ec4c1785c19a87e718b2a91e15fba44248eb949dab559e08fdd6b'
SITE = '.venv/lib/python3.12/site-packages'


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def safe(root, name):
    need(type(name) is str and name and not name.startswith('/') and ':' not in name and '\\' not in name and
         all(x not in ('', '.', '..') for x in name.split('/')), 'contained relative path')
    p = root / name
    need(not any(x.is_symlink() for x in (p,) + tuple(p.parents)), 'no symlink')
    need(p.resolve().is_relative_to(root), 'project containment')
    return p


def ref(root, name):
    p = safe(root, name)
    before = p.stat()
    with p.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = p.stat()
    need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'source changed while hashing')
    return dict(path=name, bytes=after.st_size, sha256=sha)


def no_GPU_modules():
    need(not any(n == 'torch' or n.startswith(('torch.', 'vllm', 'py_kvcache')) for n in sys.modules),
         'no model or GPU framework imports')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--source-closure', required=True)
    parser.add_argument('--output-relative', required=True)
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only invocation')
    root = args.project_root.resolve(strict=True)
    output = safe(root, args.output_relative)
    need(args.output_relative.startswith('artifacts/prefix_io_v1/server12-natural-input-cpu-20261005/') and
         output.parent.is_dir() and not output.exists(), 'fresh diagnostic output only')
    no_GPU_modules()
    need(not any(n == 'tokenizers' or n.startswith('tokenizers.') for n in sys.modules), 'fresh CPU tokenizer import')
    ledger_path = root / 'experiments/prefix_io_v1/gpu-budget-ledger.json'
    budget = ledger_path.read_bytes()
    need(hashlib.sha256(budget).hexdigest() == LEDGER_SHA and json.loads(budget)['active_reservation'] is None,
         'unchanged idle original GPU budget')
    closure_path = safe(root, args.source_closure)
    closure_raw = closure_path.read_bytes()
    closure = json.loads(closure_raw)
    need(closure['schema'] == 'actual_existing_CPU_tokenizers_import_source_closure_v1', 'actual package source inventory')
    rows = {row['path']: row for row in closure['files']}
    need(len(rows) == len(closure['files']), 'unique package source refs')
    actual_package_files = {p.relative_to(root).as_posix() for p in (root / SITE / 'tokenizers').rglob('*')
                            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.so')}
    need(actual_package_files == {name for name in rows if name.startswith(SITE + '/tokenizers/')},
         'complete existing package module source closure')
    for name, row in rows.items():
        need(name.startswith(SITE + '/tokenizers') and ref(root, name) == row, 'actual package source bytes')
    tokenizer_ref = ref(root, MODEL + '/tokenizer.json')
    config_ref = ref(root, MODEL + '/tokenizer_config.json')
    need(tokenizer_ref['sha256'] == TOKENIZER_SHA and config_ref['sha256'] == CONFIG_SHA, 'existing frozen Qwen tokenizer assets')
    cfg = json.loads((root / config_ref['path']).read_bytes())
    spec = json.loads((root / tokenizer_ref['path']).read_bytes())
    need(cfg['tokenizer_class'] == 'Qwen2Tokenizer' and cfg['add_bos_token'] is False and
         cfg.get('add_eos_token') in (None, False) and cfg['split_special_tokens'] is False,
         'supported existing raw prompt semantics')
    need(spec['truncation'] is None and spec['padding'] is None and spec['post_processor']['type'] == 'ByteLevel',
         'no padding, truncation or template additions')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(root / SITE))
        module = importlib.import_module('tokenizers')
        need(module.__version__ == '0.22.2', 'existing tokenizer version')
        tokenizer = module.Tokenizer.from_file(str(root / tokenizer_ref['path']))
        need(tokenizer.truncation is None and tokenizer.padding is None and tokenizer.num_special_tokens_to_add(False) == 0,
             'actual tokenizer has no hidden additions')
        # These are declared unit probes, never user requests or natural data.
        fixtures = ['hello world', '\u4e2d\u6587\u5206\u8bcd\u68c0\u67e5', '<|im_start|>user\nhello<|im_end|>\n']
        probes = []
        for text in fixtures:
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            need(ids and all(type(i) is int and 0 <= i < 152064 for i in ids), 'actual finite model token IDs')
            need(ids == tokenizer.encode(text, add_special_tokens=True).ids, 'diagnostic special token policy equality')
            need(tokenizer.decode(ids, skip_special_tokens=False) == text, 'diagnostic exact round trip')
            probes.append(dict(text_sha256=hashlib.sha256(text.encode()).hexdigest(), actual_token_ids=ids,
                               exact_round_trip=True, special_token_policy_equal=True, synthetic_fixture=True))
        imported = []
        for name, item in list(sys.modules.items()):
            if name == 'tokenizers' or name.startswith('tokenizers.'):
                origin = getattr(item, '__file__', None)
                if origin:
                    relative = Path(origin).resolve(strict=True).relative_to(root).as_posix()
                    need(rows.get(relative) == ref(root, relative), 'every actual imported module source bound')
                    imported.append(dict(module=name, source_ref=rows[relative]))
        no_GPU_modules()
        need(ledger_path.read_bytes() == budget and closure_path.read_bytes() == closure_raw, 'original budget and source inventory unchanged')
        need(ref(root, tokenizer_ref['path']) == tokenizer_ref and ref(root, config_ref['path']) == config_ref, 'asset bytes unchanged')
        for name, row in rows.items():
            need(ref(root, name) == row, 'package source unchanged after diagnostic')
        result = dict(schema='existing_cpu_tokenizer_asset_diagnostic_v1', status='PASS_ACTUAL_CPU_ASSETS_DIAGNOSTIC_ONLY',
                      synthetic_fixture=True, actual_GPU_operations=0, actual_natural_dataset_processed=False,
                      tokenizer_library_version=module.__version__, tokenizer_ref=tokenizer_ref, config_ref=config_ref,
                      package_source_count=len(rows), actual_imported_modules=imported, probes=probes,
                      no_model_or_GPU_framework_imports=True, original_GPU_budget_sha256=LEDGER_SHA,
                      family_proof_created=False, formal_receipt_created=False, gpu_eligible=False, strategy_effect_verified=False)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        print(json.dumps(dict(status=result['status'], probe_count=len(probes), source_count=len(rows),
                              actual_GPU_operations=0, synthetic_fixture=True, natural_receipt=False), sort_keys=True))
    finally:
        sys.path[:] = original_path


if __name__ == '__main__':
    main()
