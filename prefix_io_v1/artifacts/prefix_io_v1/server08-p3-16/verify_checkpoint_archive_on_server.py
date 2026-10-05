from pathlib import Path,PurePosixPath
import hashlib,io,json,zipfile,stat,datetime,time
root=Path(__file__).resolve().parent
archive=root/"server08-p3-16-blocked-checkpoint-evidence-v3.zip"
manifest=root/"server08-p3-16-blocked-checkpoint-evidence-v3-manifest.json"
out=root/"evidence-archive-server-verification.json"
assert not out.exists()
def sha_file(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def check_names(z,rows,index):
    infos=z.infolist();names=[i.filename for i in infos]
    assert len(names)==len(set(names))
    for i in infos:
        p=PurePosixPath(i.filename)
        assert not p.is_absolute() and ".." not in p.parts and "\\" not in i.filename
        assert not stat.S_ISLNK(i.external_attr>>16)
    expected={r["entry"] for r in rows}|{index}
    assert len(expected)==len(rows)+1 and set(names)==expected
def check_rows(z,rows):
    for r in rows:
        assert z.getinfo(r["entry"]).file_size==r["bytes"]
        h=hashlib.sha256();size=0
        with z.open(r["entry"]) as f:
            for b in iter(lambda:f.read(1024*1024),b""):h.update(b);size+=len(b)
        assert size==r["bytes"] and h.hexdigest()==r["sha256"],r["entry"]
import subprocess,shutil
ledger=json.loads((root.parents[2]/"experiments/prefix_io_v1/gpu-budget-ledger.json").read_text())
assert ledger["active_reservation"] is None
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"],text=True).strip()
started=time.monotonic()
assert sha_file(manifest)=="bb527e02547d07e013c53c5d41c31a7b807557f7a6d7923a5e48b1aacc56fd90"
m=json.loads(manifest.read_text(encoding="utf-8"))
assert archive.stat().st_size==m["archive"]["bytes"]==52574727
assert sha_file(archive)==m["archive"]["sha256"]=="958ebfbcfa36c0bc51007b9ddde2fdbd226eeac9d5a7c902ee8a5cd933998d0f"
assert m["p3_phase_closed"] is False and m["p4_enabled"] is False
assert (m["CPU_unique_passed"],m["CPU_skipped"],m["GPU_P316_runs"],m["remaining_GPU_runs"])==(1613,16,29,8)
with zipfile.ZipFile(archive) as z:
    inner=json.loads(z.read("evidence-file-manifest.json"))
    assert all(inner[k]==m[k] for k in inner)
    check_names(z,m["files"],"evidence-file-manifest.json")
    check_rows(z,m["files"])
    source_meta=json.loads(z.read("project/artifacts/prefix_io_v1/server08-p3-16/frozen-input-snapshot-at-storage-block-v2.json"))
    snapshot=z.read(m["source_snapshot"])
    assert len(snapshot)==source_meta["archive"]["bytes"]==11861986
    assert hashlib.sha256(snapshot).hexdigest()==source_meta["archive"]["sha256"]=="dcefc31e51208f8ff4e91de34389bd032642d20f43946595617506c07c42fa59"
    with zipfile.ZipFile(io.BytesIO(snapshot)) as sz:
        si=json.loads(sz.read("snapshot-entries.json"))
        assert all(si[k]==source_meta[k] for k in si)
        assert si["input_keys"]==si["unique_source_paths"]==len(si["files"])==2599
        assert si["p3_phase_closed"] is False
        check_names(sz,si["files"],"snapshot-entries.json")
        check_rows(sz,si["files"])
    lookup={r["entry"]:r for r in m["files"]}
sidecars={}
for name,expected in {
    "PILOT_PROBLEM_REPORT-at-storage-block-final-v2.md":"213d15716f2a899f16af3970472aa868221404e544754e4ec16bafd045d6d03f",
    "final-report-count-clarification.json":"e374a75f09acf5a954c06b0b456e47836fd67634c070a6003e01a1c66fd64285",
}.items():
    p=root/name;actual=sha_file(p);assert actual==expected
    sidecars[name]=dict(bytes=p.stat().st_size,sha256=actual,identity_source="post-checkpoint-independent-report-review")
for name in ["REPRODUCTION-at-storage-block.md","private-cache-merge-reviewable-action.json","p3-blocked-checkpoint-final-status.json","frozen-input-snapshot-at-storage-block-v2.json","p3-report-schema-correction-receipt.json","REPORT_CORRECTIONS-at-storage-block.md"]:
    p=root/name;r=lookup["project/artifacts/prefix_io_v1/server08-p3-16/"+name]
    assert p.stat().st_size==r["bytes"] and sha_file(p)==r["sha256"]
    sidecars[name]=dict(bytes=r["bytes"],sha256=r["sha256"],identity_source="verified-archive-file-manifest")
result=dict(status="PASS_ARCHIVE_AND_EVERY_FILE_AND_FROZEN_SOURCE_SNAPSHOT",archive=dict(path=str(archive),bytes=m["archive"]["bytes"],sha256=m["archive"]["sha256"]),metadata_files_verified=len(m["files"]),frozen_input_files_verified=len(si["files"]),sidecars=sidecars,verification_script_sha256=sha_file(Path(__file__)),CPU_unique_passed=1613,CPU_skipped=16,GPU_runs_completed=29,remaining_GPU_runs=8,P3_complete=False,P4_enabled=False,new_GPU_workloads_run=0,verification_location="server08",local_full_archive_download_verified=False,primary_free_bytes=shutil.disk_usage(root.parents[2]/"experiments/prefix_io_v1/runs").free,seconds=time.monotonic()-started,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with out.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2,ensure_ascii=False)
print(json.dumps(dict(status=result["status"],archive_sha256=result["archive"]["sha256"],metadata_files_verified=result["metadata_files_verified"],frozen_input_files_verified=result["frozen_input_files_verified"],sidecars_verified=len(sidecars),seconds=result["seconds"])))
