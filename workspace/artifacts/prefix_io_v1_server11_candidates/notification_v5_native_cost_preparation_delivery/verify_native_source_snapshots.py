"""Verify source map against already validated extracted archive, no execution."""
import argparse
import hashlib
import json
from pathlib import Path

LOCK_SHA='47a445ba5fc1fae5f9fba5f32871f5af771e66083753654122d7e96bc6f64e2a'
RECEIPT_SHA='acad37b8fa48abcdf9c96fd0207b5a09093d38f87cc4181268d0fd19ce5b0171'
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extracted',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True);parser.add_argument('--result',type=Path,required=True)
    a=parser.parse_args();root=a.extracted.resolve(strict=True)
    receiptbytes=a.receipt.read_bytes();require(sha(receiptbytes)==RECEIPT_SHA,'authenticated receipt hash')
    receipt=json.loads(receiptbytes)
    manifest=json.loads((root/'ARCHIVE_CONTENTS_MANIFEST.json').read_bytes())
    declared={r['path']:r for r in manifest['files']}
    require(len(declared)==len(manifest['files'])==receipt['data_files']==145,'complete validated member list')
    for row in declared.values():
        path=root/row['path'];require(path.resolve().is_relative_to(root) and not path.is_symlink(),'member containment')
        b=path.read_bytes();require(len(b)==row['bytes'] and sha(b)==row['sha256'],'manifest file drift')
    delivery=root/'server11-c5-native-cost-preparation-delivery-cpu-20261004'
    lockbytes=(delivery/'SOURCE_LOCK_NATIVE_PREPARATION_CPU.json').read_bytes()
    require(sha(lockbytes)==LOCK_SHA==receipt['source_lock_sha256'],'locked source identity')
    lock=json.loads(lockbytes);mapping=json.loads((delivery/'SOURCE_SNAPSHOT_MAP.json').read_bytes())
    expected={r['path']:r for r in lock['files']}
    require(len(expected)==len(lock['files'])==len(mapping['files'])==mapping['source_rows']==159,'complete source mapping')
    seen=set()
    for row in mapping['files']:
        original={k:row[k] for k in ('path','bytes','sha256')}
        require(original==expected[row['path']] and row['path'] not in seen,'unique exact mapped source')
        member=declared[row['snapshot_path']]
        require(member['bytes']==row['bytes'] and member['sha256']==row['sha256'],'source snapshot value')
        seen.add(row['path'])
    require(seen==set(expected),'missing source map')
    audit=json.loads((delivery/'SERVER_FINAL_AUDIT.json').read_bytes())
    require(audit['prior_references_verified']==797 and audit['GPU_runs']==audit['formal_benchmark_runs']==0,'server audit boundary')
    require(all(audit[k] is False for k in ('gpu_launch_allowed','native_execution_verified','native_cost_qualified',
                'full_runtime_cost_qualified','on_observation_cost_measured')),'no qualification')
    report=dict(status='PASS_ALL_159_SOURCE_REFERENCES_FROM_VERIFIED_NATIVE_CPU_ARCHIVE',
        extracted=str(root),source_rows=159,archive_files=145,unique_external_snapshots=mapping['unique_external_snapshots'],
        source_lock_sha256=LOCK_SHA,receipt_sha256=RECEIPT_SHA,actual_gpu_runs=0,formal_benchmark_runs=0,
        source_byte_verification_only=True,source_code_executed=False,model_weights_included=False,
        server_factory_tests=audit['factory_tests'],server_independent_tests=audit['independent_tests'],
        server_protocol_tests=audit['protocol_tests'],prior_references_verified_by_server=797,
        native_cost_qualified=False,full_runtime_cost_qualified=False,on_observation_cost_measured=False)
    with a.result.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
