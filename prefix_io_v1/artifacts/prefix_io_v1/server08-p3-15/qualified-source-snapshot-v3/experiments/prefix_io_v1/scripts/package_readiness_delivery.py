"""Package P311 evidence on approved auxiliary storage, without model/cache payloads."""
import hashlib,json,subprocess,time,zipfile
from pathlib import Path
from experiment_storage import ROOT,preflight
OUT=ROOT/"artifacts/prefix_io_v1/server07-p3-11"
AUX=Path("/root/prefix-io-v1-validation")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ledger=json.loads((ROOT/"experiments/prefix_io_v1/gpu-budget-ledger.json").read_text())
    assert ledger["active_reservation"] is None
    preflight(AUX/"deliveries/server07-p3-dispatch-readiness.zip",256*1024**2)
    for short,repo in [("native-p2","third_party/work/py-kvcache-p2-aio"),("native-p3","third_party/work/py-kvcache-p3-quota-cpu"),("vllm-author","third_party/work/vllm-author-build")]:
        patch=subprocess.check_output(["git","-C",str(ROOT/repo),"diff","--binary"])
        (OUT/(short+"-working-tree.patch")).write_bytes(patch)
    lock=json.loads((OUT/"source-lock.json").read_text())
    package_script="experiments/prefix_io_v1/scripts/package_readiness_delivery.py"
    lock[package_script]=sha(ROOT/package_script)
    (OUT/"source-lock.json").write_text(json.dumps(lock,indent=2))
    version=json.loads((OUT/"version-lock.json").read_text())
    version.update(source_lock_files=len(lock),source_lock_sha256=sha(OUT/"source-lock.json"))
    (OUT/"version-lock.json").write_text(json.dumps(version,indent=2))
    files={}
    def add(p):
        p=Path(p)
        if not p.is_absolute():p=ROOT/p
        assert p.is_file() and not p.is_symlink(),str(p)
        if p.is_relative_to(ROOT):name=p.relative_to(ROOT).as_posix()
        elif p.is_relative_to(AUX):name="auxiliary/"+p.relative_to(AUX).as_posix()
        else:raise ValueError("outside evidence roots")
        files[name]=p
    for p,digest in lock.items():
        p=Path(p);p=p if p.is_absolute() else ROOT/p
        assert sha(p)==digest,str(p)
        add(p)
    for directory,pattern in [
        ("docs/prefix_io_v1","*.md"),("docs/prefix_io_v1/templates","*"),
        ("experiments/prefix_io_v1/locks","*.json"),("patches/prefix_io_v1","*.patch"),
        ("src/prefix_io_control","*.py"),("experiments/prefix_io_v1/scripts","*.py"),
        ("tests/prefix_io_v1_pilot","*.py"),("tests/prefix_io_v1_start_budget","*.py")]:
        for p in (ROOT/directory).rglob(pattern):
            if p.is_file() and "__pycache__" not in p.parts:add(p)
    for p in ["experiments/prefix_io_v1/execution_state.json","experiments/prefix_io_v1/gpu-budget-ledger.json",
        "artifacts/prefix_io_v1/server07-p3-09/private-cache-duplicate-audit.json",
        "artifacts/prefix_io_v1/server07-p3-09/model-analysis.json",
        "artifacts/prefix_io_v1/server07-p3-07/qualified-analysis.json",
        "artifacts/prefix_io_v1/server07-p3-09/version-lock.json",
        "third_party/work/vllm-author-build/vllm/platforms/cuda.py"]:
        add(p)
    for plan_file in ("gpu-plan.json","model-plan.json"):
        entry=json.loads((OUT/plan_file).read_text())
        cmd=entry["command"];folder=Path(cmd[cmd.index("--output")+1])
        for p in folder.iterdir():
            if p.is_file():add(p)
        for name in ("process.log","result.json"):add(ROOT/"experiments/prefix_io_v1/runs"/entry["label"]/name)
        if "--qualification" in cmd:
            permit=json.loads((ROOT/cmd[cmd.index("--qualification")+1]).read_text())
            for key in ("cost_plan","cost_result"):
                if key in permit:add(permit[key])
    for p in (AUX/"cpu-evidence/server07-p3-11-first").iterdir():
        if p.is_file():add(p)
    add(AUX/"runs/server07-p3-c2-fullref-01/details/result.json")
    for p in (AUX/"audits").glob("server07-p3-11-*"):
        if p.is_file():add(p)
    for p in OUT.rglob("*"):
        if p.is_file() and p.name not in {"delivery-manifest.json","delivery-receipt.json","delivery-local-verification.json"}:add(p)
    manifest=dict(schema_version=1,created_unix=time.time(),project_root=str(ROOT),auxiliary_root=str(AUX),
        report="docs/prefix_io_v1/SERVER07_P3_DISPATCH_READINESS_REPORT.md",
        notes="Model weights and cache .bin payloads stay on server; raw native traces, token outputs, locks, patches, manifests and journals are included. Do not rerun completed apply commands.",
        files=[dict(path=name,server_path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for name,p in sorted(files.items())])
    manifest_path=OUT/"delivery-manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2))
    destination=AUX/"deliveries/server07-p3-dispatch-readiness.zip"
    assert not destination.exists()
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,p in sorted(files.items()):z.write(p,name)
        z.write(manifest_path,"delivery-manifest.json")
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)+1
        for item in manifest["files"]:assert hashlib.sha256(z.read(item["path"])).hexdigest()==item["sha256"]
    receipt=dict(path=str(destination),bytes=destination.stat().st_size,sha256=sha(destination),manifest_files=len(files),
        source_lock_files=len(lock),zip_roundtrip_all_hashes_verified=True,storage_after=preflight(AUX/"runs/next",0))
    (OUT/"delivery-receipt.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))
if __name__=="__main__":main()
