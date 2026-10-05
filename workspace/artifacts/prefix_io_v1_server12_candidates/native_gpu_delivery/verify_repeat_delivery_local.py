"""Verify the downloaded fixed-three diagnostic archive and save flat bytes.

Adapted from verify_normal_delivery.py. This imports no engine, downloads
nothing and grants no scientific qualification. The trusted server result SHA
is supplied by the caller; original server paths remain in the mapping report.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import tarfile

A='artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004'
D='artifacts/prefix_io_v1/server12-c5-normal-repeatability-v1-20261004'
STEM=A+'/REPEAT_REPEAT_DELIVERY_fixed3-final'
SCHEMA='server12_fixed_repeat_delivery_manifest_v1'
LOCK_SHA='4aab66888592a8fdf6544513003e9b7382f2194f3da2407b8991073a8e86ffef'
LEDGER='experiments/prefix_io_v1/gpu-budget-ledger.json'
MAX_FILE=32*1024**2
MAX_TOTAL=100*1024**2
MAX_TAR=104*1024**2
MAX_MEMBERS=2048
EXCLUDED_PARTS={'__pycache__','runtime-cache','private-storage','fixture-tmp','recorded_files',
    'server_evidence_root','normal_off01_verified','.private-sdk','private-sdk','.private-ninja',
    'private-ninja','ninja','torch_extensions','compiled','compiledso','.git'}
SUFFIXES={'.py','.md','.json','.log','.txt','.yaml'}
RUN_PREFIXES=tuple('experiments/prefix_io_v1/runs/server12-c5-native-repeat-off%02d/'%(i+1) for i in range(3))
EXTRA_PAYLOADS={'experiments/prefix_io_v1/configs/permissions.yaml',
    'experiments/prefix_io_v1/scripts/run_gpu_stage.py'}


def require(value,reason):
    if not value:raise ValueError('REPEAT_LOCAL_DELIVERY_REJECTED: '+reason)


def relative_name(value):
    require(type(value) is str and 0<len(value)<=240 and all(32<=ord(c)<127 for c in value)
        and ':' not in value and '\\' not in value and not value.startswith('/')
        and all(part not in ('','.','..') for part in value.split('/')),'plain bounded project-relative name')
    return value


def payload_name(value):
    relative_name(value)
    require(not set(value.split('/')).intersection(EXCLUDED_PARTS)
        and not any(part.startswith('fixture-') for part in value.split('/')),'excluded cache/SDK/fixture payload')
    require((value.startswith((D+'/',A+'/',*RUN_PREFIXES)) or value in EXTRA_PAYLOADS)
        and Path(value).suffix in SUFFIXES,'only actual small diagnostic evidence payloads')
    return value


def no_links(path,regular=False):
    path=Path(path).absolute()
    for candidate in (path,*path.parents):
        require(not candidate.is_symlink() and not getattr(candidate,'is_junction',lambda:False)(),
            'local symlink/junction refused')
    if regular:require(path.is_file() and stat.S_ISREG(path.stat().st_mode),'regular actual local input')
    return path


def pairs(items):
    value={}
    for key,item in items:
        require(key not in value,'duplicate JSON field');value[key]=item
    return value


def parse(raw):
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda value:require(False,'nonfinite JSON'))


def reference(row,max_bytes=MAX_FILE):
    require(type(row) is dict and set(row)=={'path','bytes','sha256'},'exact typed file reference')
    relative_name(row['path'])
    require(type(row['bytes']) is int and 0<=row['bytes']<=max_bytes
        and type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}',row['sha256']) is not None,
        'bounded bytes and SHA reference')
    return row


def actual_ref(path,server_path,max_bytes=MAX_FILE):
    path=no_links(path,regular=True);before=path.stat()
    require(0<=before.st_size<=max_bytes,'bounded local file')
    digest=hashlib.sha256();size=0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):
            size+=len(block);require(size<=max_bytes,'local file grew above bound');digest.update(block)
    after=path.stat()
    require(size==before.st_size==after.st_size and (before.st_dev,before.st_ino,before.st_mtime_ns)==
        (after.st_dev,after.st_ino,after.st_mtime_ns),'local file drift while hashing')
    return dict(path=server_path,bytes=size,sha256=digest.hexdigest())


def plain_tar(archive_path):
    """Stream literal USTAR headers; tarfile iteration cannot hide extensions."""
    members=[];seen=set();payload=0;expanded=0
    with gzip.open(archive_path,'rb') as stream:
        def bounded_read(size):
            nonlocal expanded
            raw=stream.read(size);expanded+=len(raw)
            require(expanded<=MAX_TAR,'tar expansion above104MiB');return raw
        def exact(size):
            parts=[];remaining=size
            while remaining:
                block=bounded_read(min(remaining,1024**2));require(block,'truncated tar data')
                parts.append(block);remaining-=len(block)
            return b''.join(parts)
        while True:
            header=bounded_read(512)
            require(len(header)==512,'complete header and terminal zero blocks required')
            if header==b'\0'*512:
                require(exact(512)==b'\0'*512,'two terminal zero blocks required')
                trailing=0
                while True:
                    block=bounded_read(1024**2)
                    if not block:break
                    require(block==b'\0'*len(block),'hidden trailing tar bytes')
                    trailing+=len(block)
                require(trailing%512==0,'terminal tar block alignment')
                break
            require(header[156:157]==b'0' and header[257:263]==b'ustar\0'
                and header[263:265]==b'00','literal regular USTAR only; extensions refused')
            info=tarfile.TarInfo.frombuf(header,encoding='ascii',errors='strict')
            payload_name(info.name)
            require(info.type==tarfile.REGTYPE and not info.linkname and not info.pax_headers
                and info.name not in seen and type(info.size) is int and 0<=info.size<=MAX_FILE,
                'duplicate/link/extension/oversized member')
            require(info.uid==info.gid==0 and info.mode==0o644 and info.mtime==0,'regular evidence header metadata')
            require(info.tobuf(format=tarfile.USTAR_FORMAT,encoding='ascii',errors='strict')==header,
                'canonical literal USTAR header; hidden header bytes refused')
            seen.add(info.name);payload+=info.size
            require(len(seen)<=MAX_MEMBERS and payload<=MAX_TOTAL,'bounded member bytes/count')
            data=exact(info.size)
            padding=(-info.size)%512
            require(exact(padding)==b'\0'*padding,'nonzero tar data padding')
            members.append((info.name,data))
    return members


def local_name(server_path):
    prefix=hashlib.sha256(server_path.encode()).hexdigest()[:12]
    base=re.sub('[^A-Za-z0-9._-]','_',server_path.rsplit('/',1)[-1])[:64].rstrip('.')
    require(base,'nonempty flat basename');return prefix+'_'+base


def verify(archive_path,result_path,manifest_path,output_dir,result_sha256,
           prior_calibration_archive=None,prior_normal_archive=None):
    require(type(result_sha256) is str and re.fullmatch('[0-9a-f]{64}',result_sha256) is not None,
        'trusted actual server result SHA required')
    archive_path=no_links(archive_path,regular=True);result_path=no_links(result_path,regular=True)
    manifest_path=no_links(manifest_path,regular=True)
    result_ref=actual_ref(result_path,STEM+'_RESULT.json')
    require(result_ref['sha256']==result_sha256,'downloaded RESULT SHA differs from actual server receipt')
    result_bytes=result_path.read_bytes()
    require(len(result_bytes)==result_ref['bytes'] and hashlib.sha256(result_bytes).hexdigest()==result_ref['sha256'],
        'RESULT bytes drift before parsing')
    result=parse(result_bytes)
    require(type(result) is dict and result.get('status')=='PASS_FIXED_REPEAT_DELIVERY_BYTE_ARCHIVE'
        and result.get('gpu_ledger_byte_unchanged') is True
        and type(result.get('complete_runtime_source_files_verified_before_and_after')) is int
        and result['complete_runtime_source_files_verified_before_and_after']==4816
        and type(result.get('GPU_operations_this_action')) is int and result['GPU_operations_this_action']==0
        and type(result.get('data_deletions_this_action')) is int and result['data_deletions_this_action']==0
        and result.get('experiment_qualification_issued_this_action') is False,'actual finite byte-archive result')
    archive_ref=reference(result['archive'],MAX_TOTAL);manifest_ref=reference(result['manifest'])
    require(archive_ref['path']==STEM+'.tar.gz' and manifest_ref['path']==STEM+'_MANIFEST.json',
        'fixed3-final exact archive identity')
    require(actual_ref(archive_path,archive_ref['path'],MAX_TOTAL)==archive_ref,'downloaded archive SHA/size')
    require(actual_ref(manifest_path,manifest_ref['path'])==manifest_ref,'downloaded manifest SHA/size')
    manifest_bytes=manifest_path.read_bytes()
    require(len(manifest_bytes)==manifest_ref['bytes'] and hashlib.sha256(manifest_bytes).hexdigest()==manifest_ref['sha256'],
        'manifest bytes drift before parsing')
    manifest=parse(manifest_bytes)
    require(type(manifest) is dict and manifest.get('schema')==SCHEMA
        and manifest.get('manifest_self_excluded') is True
        and manifest.get('experiment_qualification_issued_this_action') is False
        and type(manifest.get('source_files_fully_verified')) is int and manifest['source_files_fully_verified']==4816
        and type(manifest.get('files')) is list and 1<=len(manifest['files'])<MAX_MEMBERS,
        'fixed actual manifest and source-count metadata')
    refs={}
    for row in manifest['files']:
        reference(row);payload_name(row['path'])
        require(row['path'] not in refs and row['path']!=manifest_ref['path'],'unique nonself manifest rows')
        refs[row['path']]=row
    payload=sum(row['bytes'] for row in refs.values())
    for key,expected in (('count',len(refs)),('payload_bytes',payload)):
        require(type(manifest.get(key)) is int and manifest[key]==expected,'manifest exact total: '+key)
    for key,expected in (('source_files',len(refs)),('payload_bytes',payload),('archive_members',len(refs)+1)):
        require(type(result.get(key)) is int and result[key]==expected,'result exact total: '+key)
    members=plain_tar(archive_path)
    require(len(members)==len(refs)+1 and members[-1]==(manifest_ref['path'],manifest_bytes)
        and {name for name,_ in members[:-1]}==set(refs),'exact terminal manifest and payload membership')
    names=set();mappings=[];data_by_path={}
    for name,raw in members:
        expected=manifest_ref if name==manifest_ref['path'] else refs[name]
        require(len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'],
            'actual member SHA/bytes: '+name)
        mapped=local_name(name);require(mapped.lower() not in names,'flat filename collision')
        names.add(mapped.lower());data_by_path[name]=raw
        if name!=manifest_ref['path']:mappings.append(dict(expected,local_name=mapped))
    snapshot=reference(result['ledger_snapshot'])
    require(refs.get(snapshot['path'])==snapshot and manifest.get('original_ledger_snapshot_ref')==snapshot,
        'actual immutable ledger snapshot member')
    ledger_bytes=data_by_path[snapshot['path']];ledger=parse(ledger_bytes)
    ledger_ref=reference(result['original_gpu_ledger_before'])
    require(type(ledger) is dict and type(ledger.get('gpu_wall_seconds')) in (int,float)
        and math.isfinite(ledger['gpu_wall_seconds']) and 0<=ledger['gpu_wall_seconds']<=28800,
        'finite original eight-hour ledger scalar')
    require(ledger_ref==result.get('original_gpu_ledger_after')==manifest.get('original_ledger_ref_at_pack')
        and ledger_ref['path']==LEDGER and ledger.get('active_reservation') is None
        and type(ledger.get('events')) is list
        and len(ledger_bytes)==ledger_ref['bytes'] and hashlib.sha256(ledger_bytes).hexdigest()==ledger_ref['sha256'],
        'actual snapshot equals original idle ledger bytes')
    lock_ref=reference(manifest['source_lock_ref'])
    require(lock_ref['path']==D+'/COMMON_SOURCE_LOCK.json' and lock_ref['sha256']==LOCK_SHA
        and refs.get(lock_ref['path'])==lock_ref,'actual frozen4816 source map member')
    lock=parse(data_by_path[lock_ref['path']])
    require(lock.get('schema')=='c5_repeatability_common_source_lock_v1'
        and lock.get('scope')=='server12_c5_normal_repeatability_v1'
        and type(lock.get('files')) is list and len(lock['files'])==4816,'actual complete source-map metadata')
    source_map={}
    for row in lock['files']:
        reference(row,64*1024**3)
        require(row['path'] not in source_map,'unique actual source-map paths');source_map[row['path']]=row
    for name,row in refs.items():
        if name.startswith(D+'/') and name.endswith('.py'):
            require(source_map.get(name)==row,'archived frozen diagnostic source agrees with source map')
    deps=manifest.get('immutable_dependency_archives')
    require(type(deps) is list and len(deps)==2 and deps==result.get('immutable_dependency_archives'),
        'the same two separately pinned dependency archives')
    supplied=(prior_calibration_archive,prior_normal_archive)
    expected_paths=(A+'/GPU_EVIDENCE.tar.gz',A+'/NORMAL_NORMAL_DELIVERY_off01-final.tar.gz')
    dependency_status=[]
    for index,dep in enumerate(deps):
        require(type(dep) is dict and dep.get('archive_embedded') is False,'prior dependency remains separate')
        dependency_ref=reference(dep['archive_ref'],MAX_TOTAL)
        reference(dep['manifest_ref']);reference(dep['result_ref'])
        require(dependency_ref['path']==expected_paths[index] and dependency_ref['path'] not in refs,
            'known prior archive excluded from new payload')
        status='SEPARATELY_PINNED_NOT_REHASHED_BY_THIS_CALL'
        if supplied[index] is not None:
            require(actual_ref(supplied[index],dependency_ref['path'],MAX_TOTAL)==dependency_ref,
                'actual local prior archive byte pin');status='ACTUAL_LOCAL_PRIOR_ARCHIVE_BYTES_REVERIFIED'
        dependency_status.append(dict(archive_ref=dependency_ref,status=status))
    require(actual_ref(archive_path,archive_ref['path'],MAX_TOTAL)==archive_ref
        and actual_ref(result_path,result_ref['path'])==result_ref
        and actual_ref(manifest_path,manifest_ref['path'])==manifest_ref,'downloaded input drift after verification')
    output_dir=no_links(output_dir)
    require(not output_dir.exists() and output_dir.parent.is_dir(),'new append-only local output directory')
    output_dir.mkdir(exist_ok=False);recorded=output_dir/'recorded_files';recorded.mkdir(exist_ok=False)
    for name,raw in members:
        target=recorded/local_name(name)
        with target.open('xb') as stream:stream.write(raw)
        require(actual_ref(target,name)==(manifest_ref if name==manifest_ref['path'] else refs[name]),
            'actual flat local output SHA verification')
    for mapping in mappings:mapping['local_path']=str(recorded/mapping.pop('local_name'))
    report=dict(status='PASS_SERVER12_REPEAT_DELIVERY_LOCAL_BYTE_VERIFICATION',
        origin='actual_downloaded_archive_result_manifest_and_flat_output_SHA_verification',
        result=result_ref,archive=archive_ref,manifest=manifest_ref,ledger_snapshot=snapshot,
        source_files_verified=len(refs),archive_members=len(members),payload_bytes=payload,
        archived_source_map_rows=4816,actual_archived_diagnostic_source_bytes_match_source_map=True,
        server_full_source_verification_before_and_after=4816,
        local_full4816_source_entities_rehashed=False,local_model_weight_or_SDK_binary_payload_read=False,
        prior_dependency_archive_verification=dependency_status,
        experiment_qualification_issued=False,normal_qualification_passed=False,permits_next_mode=None,
        performance_benefit_proved=False,GPU_operations=0,data_deletions=0,
        local_filename_mapping='original projectrelative paths retained; recorded_files/pathhash12_basename64',
        files=mappings,manifest_local_path=str(recorded/local_name(manifest_ref['path'])))
    with (output_dir/'LOCAL_REPEAT_DELIVERY_BYTE_VERIFICATION.json').open('xb') as stream:
        stream.write((json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
    return {key:value for key,value in report.items() if key!='files'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--result-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--prior-calibration-archive',type=Path)
    parser.add_argument('--prior-normal-archive',type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(verify(args.archive,args.result,args.manifest,args.output_dir,args.result_sha256,
        args.prior_calibration_archive,args.prior_normal_archive),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
