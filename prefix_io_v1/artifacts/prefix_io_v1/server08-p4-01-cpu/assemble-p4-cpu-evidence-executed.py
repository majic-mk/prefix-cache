from pathlib import Path
import hashlib,json,zipfile,datetime
root=Path('.');out=root/'artifacts/prefix_io_v1/server08-p4-01-cpu'
def digest(raw):return hashlib.sha256(raw).hexdigest()
refs=json.loads((out/'test-input-lock-v2.json').read_text())['files']
paths={r['path'] for r in refs}
for r in refs:
    raw=Path(r['path']).read_bytes()
    assert len(raw)==r['bytes'] and digest(raw)==r['sha256']
allowed={'.json','.jsonl','.md','.py','.xml','.log','.patch','.txt','.yaml'}
for p in out.rglob('*'):
    if not p.is_file() or p.suffix not in allowed:continue
    parts=p.relative_to(out).parts
    if any('pytest-tmp' in v or v in {'scratch','common-off-import','__pycache__','.pytest_cache'} for v in parts):continue
    if p.name.startswith('server08-p4-cpu-evidence-v1') or p.name=='package-server-verification.json':continue
    assert not p.is_symlink()
    paths.add(str(p))
for p in (root/'docs/prefix_io_v1').glob('0[0-5]_*.md'):paths.add(str(p))
for p in (root/'docs/prefix_io_v1/templates').iterdir():
    if p.is_file() and p.suffix in allowed:paths.add(str(p))
paths.update(['docs/prefix_io_v1/SERVER08_P4_CPU_REPORT.md','experiments/prefix_io_v1/configs/permissions.yaml',
              'artifacts/prefix_io_v1/server08-p3-16/execution-lock-12-final-p3.json',
              'artifacts/prefix_io_v1/server08-p3-16/p3-final-root-review.json'])
assert all(Path(s).suffix in allowed for s in paths)
manifest={'schema_version':1,'status':'P4_CPU_PACKAGE_SELECTED_BYTES',
    'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'file_count':len(paths),'files':[],
    'model_or_private_KV_payload_included':False,'ELF_or_venv_included':False,'GPU_runs_added':0}
for s in sorted(paths):
    p=Path(s);raw=p.read_bytes()
    manifest['files'].append({'path':s,'bytes':len(raw),'sha256':digest(raw)})
mp=out/'server08-p4-cpu-evidence-v1-manifest.json';assert not mp.exists();mp.write_text(json.dumps(manifest,indent=2)+'\n')
zp=out/'server08-p4-cpu-evidence-v1.zip';assert not zp.exists()
with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for r in manifest['files']:z.writestr(r['path'],Path(r['path']).read_bytes())
    z.writestr(str(mp),mp.read_bytes())
with zipfile.ZipFile(zp) as z:
    assert z.testzip() is None
    assert set(z.namelist())=={r['path'] for r in manifest['files']}|{str(mp)}
    for r in manifest['files']:
        raw=z.read(r['path']);assert len(raw)==r['bytes'] and digest(raw)==r['sha256']
    assert z.read(str(mp))==mp.read_bytes()
ledger=root/'experiments/prefix_io_v1/gpu-budget-ledger.json'
assert digest(ledger.read_bytes())=='31199998369e35fcd40daddd7af340353b0feaf134bbe37d2dd8f1f35391efc1'
result={'status':'PASS_PACKAGE_CRC_AND_ALL_FILE_SHA256','archive':{'path':str(zp),'bytes':zp.stat().st_size,'sha256':digest(zp.read_bytes())},
    'manifest':{'path':str(mp),'bytes':mp.stat().st_size,'sha256':digest(mp.read_bytes())},
    'files_verified':len(manifest['files']),'tested_source_refs_verified':len(refs),'GPU_ledger_unchanged':True,'GPU_runs_added':0}
vp=out/'package-server-verification.json';assert not vp.exists();vp.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

