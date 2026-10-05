"""Verify source map against already validated extracted archive, no execution."""
import argparse
import hashlib
import json
from pathlib import Path

LOCK_SHA='bb7522102bddee35e5878551e98d23ae475897e24663d4c3aa9ae235329b176d'
RECEIPT_SHA='250b043988732c590663c3d10686012c78287a264aaa79850267f0aa1bc6427d'
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
    require(len(declared)==len(manifest['files'])==receipt['data_files']==187,'complete validated member list')
    for row in declared.values():
        path=root/row['path'];require(path.resolve().is_relative_to(root) and not path.is_symlink(),'member containment')
        b=path.read_bytes();require(len(b)==row['bytes'] and sha(b)==row['sha256'],'manifest file drift')
    delivery=root/'server11-c5-combined-runtime-delivery-cpu-20261004'
    lockbytes=(delivery/'SOURCE_LOCK_COMBINED_RUNTIME_CPU.json').read_bytes()
    require(sha(lockbytes)==LOCK_SHA==receipt['source_lock_sha256'],'locked source identity')
    lock=json.loads(lockbytes);mapping=json.loads((delivery/'SOURCE_SNAPSHOT_MAP.json').read_bytes())
    expected={r['path']:r for r in lock['files']}
    require(len(expected)==len(lock['files'])==len(mapping['files'])==mapping['source_rows']==202,'complete source mapping')
    seen=set()
    for row in mapping['files']:
        original={k:row[k] for k in ('path','bytes','sha256')}
        require(original==expected[row['path']] and row['path'] not in seen,'unique exact mapped source')
        member=declared[row['snapshot_path']]
        require(member['bytes']==row['bytes'] and member['sha256']==row['sha256'],'source snapshot value')
        seen.add(row['path'])
    require(seen==set(expected),'missing source map')
    audit=json.loads((delivery/'SERVER_FINAL_AUDIT.json').read_bytes())
    require(audit['prior_references_verified']==952 and audit['GPU_runs']==audit['formal_benchmark_runs']==0,'server audit boundary')
    require(all(audit[k] is False for k in ('gpu_launch_allowed','native_execution_verified','native_cost_qualified',
                'full_runtime_cost_qualified','on_observation_cost_measured')),'no qualification')
    report=dict(status='PASS_ALL_202_SOURCE_REFERENCES_FROM_VERIFIED_COMBINED_CPU_ARCHIVE',
        extracted=str(root),source_rows=202,archive_files=187,unique_external_snapshots=mapping['unique_external_snapshots'],
        source_lock_sha256=LOCK_SHA,receipt_sha256=RECEIPT_SHA,actual_gpu_runs=0,formal_benchmark_runs=0,
        source_byte_verification_only=True,source_code_executed=False,model_weights_included=False,
        server_factory_tests=audit['factory_tests'],server_independent_tests=audit['independent_tests'],
        server_resource_tests=audit['resource_tests'],prior_references_verified_by_server=952,
        native_cost_qualified=False,full_runtime_cost_qualified=False,on_observation_cost_measured=False)
    with a.result.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
