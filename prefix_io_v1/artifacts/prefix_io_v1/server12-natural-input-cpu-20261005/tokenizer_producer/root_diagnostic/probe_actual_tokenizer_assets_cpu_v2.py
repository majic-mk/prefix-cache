"""Source-only CPU asset diagnostic; synthetic probes never qualify natural input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types

STAGE = 'artifacts/prefix_io_v1/server12-natural-input-cpu-20261005'
PRODUCER = STAGE + '/tokenizer_producer/raw_tokenizer_cpu.py'
PRODUCER_SHA = '2d252d0647f6a1617a4222fd86009da27344d29391c109fa3ccecf7cd4f68c63'
LEDGER_SHA = '774847ccdb7ec4c1785c19a87e718b2a91e15fba44248eb949dab559e08fdd6b'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--source-lock', required=True)
    parser.add_argument('--output-relative', required=True)
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only actual invocation')
    root = args.project_root.resolve(strict=True)
    need(not any(name == 'torch' or name.startswith(('torch.', 'transformers', 'vllm', 'py_kvcache', 'tokenizers'))
                 for name in sys.modules), 'fresh CPU-only module namespace')
    source_path = root / PRODUCER
    raw = source_path.read_bytes()
    need(not source_path.is_symlink() and hashlib.sha256(raw).hexdigest() == PRODUCER_SHA, 'actual stable producer bytes')
    # Do not use a default SourceFileLoader for this bootstrap either.
    module_name = '_actual_source_only_CPU_tokenizer_diagnostic'
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    sys.modules[module_name] = module
    exec(compile(raw, str(source_path), 'exec'), module.__dict__)
    refs = module.current_refs(root, module.file_ref(root, args.source_lock))
    module.closed(root, refs.get(PRODUCER), refs, expected_sha=PRODUCER_SHA)
    self_relative = Path(__file__).resolve().relative_to(root).as_posix()
    module.closed(root, refs.get(self_relative), refs)
    request_path = root / STAGE / 'RAW_TOKENIZATION_REQUEST_UNBOUND_TEMPLATE.json'
    request_raw = request_path.read_bytes()
    request = module.parse(request_raw)
    need(request['dataset_ref'] is None, 'diagnostic cannot process natural data')
    module.closed(root, request['tokenizer_json_ref'], refs, expected_sha=module.TOKENIZER_JSON_SHA)
    module.closed(root, request['tokenizer_config_ref'], refs, expected_sha=module.TOKENIZER_CONFIG_SHA)
    module.validate_assets(module.bounded_json(root / request['tokenizer_json_ref']['path']),
                           module.bounded_json(root / request['tokenizer_config_ref']['path']))
    sources = module.backend_sources(root, request, refs)
    ledger_path = root / 'experiments/prefix_io_v1/gpu-budget-ledger.json'
    before = ledger_path.read_bytes()
    need(hashlib.sha256(before).hexdigest() == LEDGER_SHA and json.loads(before)['active_reservation'] is None,
         'unchanged idle original GPU budget')
    target = module.safe(root, args.output_relative)
    need(args.output_relative.startswith(STAGE + '/') and target.parent.is_dir() and not target.exists(),
         'fresh existing diagnostic directory')
    fixtures = ['hello world', '\u4e2d\u6587\u5206\u8bcd\u68c0\u67e5', '<|im_start|>user\nhello<|im_end|>\n']
    probes = []
    with module.actual_local_backend(root, request, refs, sources) as (encode, audit):
        backend = sys.modules['tokenizers'].Tokenizer.from_file(str(root / request['tokenizer_json_ref']['path']))
        need(backend.num_special_tokens_to_add(False) == 0 and backend.encode_special_tokens is False,
             'actual current Qwen raw token semantics')
        for text in fixtures:
            ids = encode(text)
            need(ids and all(type(token) is int and 0 <= token < 152064 for token in ids), 'finite actual CPU token IDs')
            need(ids == backend.encode(text, add_special_tokens=True).ids, 'diagnostic special token policy equality')
            need(backend.decode(ids, skip_special_tokens=False) == text, 'diagnostic exact round trip')
            probes.append(dict(text_sha256=hashlib.sha256(text.encode()).hexdigest(), actual_token_ids=ids,
                               exact_round_trip=True, special_token_policy_equal=True, synthetic_fixture=True))
    module.cpu_modules_only()
    need(not any(name == 'tokenizers' or name.startswith('tokenizers.') for name in sys.modules), 'backend namespace restored')
    need(ledger_path.read_bytes() == before and request_path.read_bytes() == request_raw,
         'original budget and input template unchanged')
    result = dict(schema='existing_CPU_source_only_tokenizer_asset_diagnostic_v2',
                  status='PASS_ACTUAL_SOURCE_ONLY_CPU_ASSETS_DIAGNOSTIC_ONLY',
                  synthetic_fixture=True, probes=probes, producer_ref=refs[PRODUCER],
                  source_lock_ref=module.file_ref(root, args.source_lock), tokenizer_backend=audit,
                  package_source_count=len(sources), actual_GPU_operations=0,
                  actual_natural_dataset_processed=False, family_proof_created=False,
                  formal_receipt_created=False, gpu_eligible=False, strategy_effect_verified=False,
                  supersedes_default_import_diagnostics_for_executed_source_provenance=True,
                  original_GPU_budget_sha256=LEDGER_SHA)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], probe_count=len(probes), source_count=len(sources),
                          actual_GPU_operations=0, synthetic_fixture=True, natural_receipt=False), sort_keys=True))


if __name__ == '__main__':
    main()
