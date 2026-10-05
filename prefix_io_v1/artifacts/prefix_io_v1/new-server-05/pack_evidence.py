from pathlib import Path
import hashlib,json,zipfile,datetime
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/prefix_io_v1/new-server-05'
ZIP=ROOT.parent/'prefix-io-v1-new-server-05-feasibility-20260927.zip'
sha=lambda b:hashlib.sha256(b).hexdigest()
assert ROOT.name=='project' and not ZIP.exists()
binary=OUT/'io_uring_syscall_probe'
provenance=dict(source_sha256=sha((OUT/'io_uring_syscall_probe.c').read_bytes()),compiled_binary_sha256=sha(binary.read_bytes()),compiled_binary_bytes=binary.stat().st_size,compiler='/usr/bin/gcc',build_evidence='compile-probe.json',binary_excluded_from_portable_delivery=True)
with (OUT/'probe-build-provenance.json').open('x') as f:json.dump(provenance,f,indent=2)
selected=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['io_uring_syscall_probe','delivery-manifest.json','package-check.json'] and '__pycache__' not in p.parts]
selected += [ROOT/p for p in [
 'docs/prefix_io_v1/NEW_SERVER_05_FEASIBILITY_REPORT.md',
 'docs/prefix_io_v1/PLATFORM_IO_URING_FEASIBILITY_REQUEST.md',
 'docs/prefix_io_v1/REPRODUCE_NEW_SERVER_05.md',
 'docs/prefix_io_v1/CAPABILITY_MATRIX.md',
 'docs/prefix_io_v1/GPU_STAGE_REPORT.md',
 'experiments/prefix_io_v1/configs/permissions.yaml',
 'experiments/prefix_io_v1/configs/authorizations/io_uring_20260927.json',
 'experiments/prefix_io_v1/execution_state.json',
 'experiments/prefix_io_v1/runs/native-prefix-04/details/smoke-result.json',
 'artifacts/prefix_io_v1/new-server-03/final-budget-summary.json',
 'artifacts/prefix_io_v1/new-server-03/platform-probe/io_uring_setup_probe.py']]
records=[]
with zipfile.ZipFile(ZIP,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(set(selected)):
  assert p.is_file() and not p.is_symlink()
  p.resolve().relative_to(ROOT.resolve())
  data=p.read_bytes();rel=p.relative_to(ROOT).as_posix();name='project/'+rel
  records.append(dict(path=rel,archive_path=name,bytes=len(data),sha256=sha(data)))
  z.writestr(name,data)
 manifest=dict(schema_version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  delivery_type='standalone feasibility evidence; not complete engine release',verdict='NO_GO_CURRENT_CONTAINER',
  architecture_feasibility='UNDETERMINED',gpu_execution_this_delivery=False,model_download_this_delivery=False,
  historical_gpu_evidence_explicitly_not_rerun=True,files=records,file_count=len(records))
 payload=(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode()
 z.writestr('delivery-manifest.json',payload)
with zipfile.ZipFile(ZIP) as z:
 assert z.testzip() is None and len(z.namelist())==len(records)+1
 for row in records:
  data=z.read(row['archive_path'])
  assert len(data)==row['bytes'] and sha(data)==row['sha256']
(OUT/'delivery-manifest.json').write_bytes(payload)
result=dict(archive=str(ZIP),archive_bytes=ZIP.stat().st_size,archive_sha256=sha(ZIP.read_bytes()),
 manifest_records=len(records),zip_entries=len(records)+1,manifest_sha256=sha(payload),
 all_file_sizes_and_sha256_verified=True,gpu_execution=False)
(OUT/'package-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
