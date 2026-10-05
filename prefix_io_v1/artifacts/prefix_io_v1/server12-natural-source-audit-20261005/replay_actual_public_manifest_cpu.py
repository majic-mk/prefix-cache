"""CPU original-protocol replay of all actual source-closed public QA token/family records."""
import pathlib,types,sys,hashlib,json,os
ROOT=pathlib.Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D=ROOT/'artifacts/prefix_io_v1/server12-natural-source-audit-20261005'
PROTOCOL_SHA='7fbc544597c9abe334122d354e7494b590e810ab0682f1d2d155408cc4f618eb'
LOCK_SHA='1d1aef9591a8c319668260d56b1cbe55b8775f0f18b39ea89a36ca83952070b1'
LEDGER_SHA='774847ccdb7ec4c1785c19a87e718b2a91e15fba44248eb949dab559e08fdd6b'
def ref(p):
 b=p.read_bytes();return dict(path=p.relative_to(ROOT).as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 ledger=ROOT/'experiments/prefix_io_v1/gpu-budget-ledger.json'
 assert ref(ledger)['sha256']==LEDGER_SHA
 raw=json.loads((D/'ACTUAL_CPU_RAW_TOKEN_IDS_01.json').read_bytes())
 receipt=json.loads((D/'ACTUAL_CPU_TOKENIZER_FAMILY_RECEIPT_01.json').read_bytes())
 decl=json.loads((D/'ACTUAL_PUBLIC_REPLAY_DECLARATION_01.json').read_bytes())
 source=ROOT/'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/protocol/prerental_protocol.py'
 b=source.read_bytes();assert hashlib.sha256(b).hexdigest()==PROTOCOL_SHA
 lock=D/'PRERENT_SOURCE_LOCK_V13.json';assert ref(lock)['sha256']==LOCK_SHA
 refs={r['path']:r for r in json.loads(lock.read_bytes())['files']}
 assert refs[source.relative_to(ROOT).as_posix()]==ref(source)
 m=types.ModuleType('_actual_public_original_protocol_replay');m.__file__=str(source);sys.modules[m.__name__]=m
 exec(compile(b,str(source),'exec'),m.__dict__)
 trace=ROOT/raw['dataset_ref']['path']
 author=ROOT/raw['inspection']['records'][0].get('unused','third_party/upstream/kvcache-experiments/scripts/shared_storage_trace_replay.py')
 common=ROOT/'third_party/upstream/kvcache-experiments/common/prefix_cache_common.py'
 manifest=m.freeze_trace(trace,author,common,decl,receipt)
 assert manifest['partition_counts']==dict(calibration=30,development=5,evaluation=5) and len(manifest['records'])==40
 assert manifest['gpu_effect_qualified'] is False and manifest['gpu_operations']==0
 for r in manifest['records']:assert r['prompt_token_ids']==raw['records'][r['request_id']]['prompt_token_ids']
 output=D/'ACTUAL_PUBLIC_ORIGINAL_MANIFEST_01.json'
 with output.open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write('\n')
 assert ref(ledger)['sha256']==LEDGER_SHA and source.read_bytes()==b
 proof=dict(schema='actual_original_protocol_public_manifest_replay_CPU_v1',status='PASS_ALL_40_ORIGINAL_PROTOCOL_CPU_REPLAY',manifest_ref=ref(output),source_lock_ref=ref(lock),protocol_source_ref=ref(source),declaration_ref=ref(D/'ACTUAL_PUBLIC_REPLAY_DECLARATION_01.json'),raw_result_ref=ref(D/'ACTUAL_CPU_RAW_TOKEN_IDS_01.json'),family_receipt_ref=ref(D/'ACTUAL_CPU_TOKENIZER_FAMILY_RECEIPT_01.json'),family_result_ref=ref(D/'ACTUAL_CPU_TOKENIZER_FAMILY_RESULT_01.json'),partition_counts=manifest['partition_counts'],actual_all_prompt_token_count=sum(len(r['prompt_token_ids']) for r in manifest['records']),prompt_token_length_min=min(len(r['prompt_token_ids'])for r in manifest['records']),prompt_token_length_max=max(len(r['prompt_token_ids'])for r in manifest['records']),model_context_limit=decl['max_model_len'],controlled_output_tokens=decl['output_tokens'],all_declared_context_checks_passed=True,source_refs_in_future_consumption_lock_required=True,formal_GPU_configuration_ready=False,GPU_operations=0,ordinary_strategy_authorized=False,service_SLO=None,control_window_deadline=None,budget_ledger_sha256=LEDGER_SHA)
 with (D/'ACTUAL_PUBLIC_ORIGINAL_MANIFEST_CPU_PROOF_01.json').open('x') as f:json.dump(proof,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(proof,sort_keys=True))
if __name__=='__main__':main()
