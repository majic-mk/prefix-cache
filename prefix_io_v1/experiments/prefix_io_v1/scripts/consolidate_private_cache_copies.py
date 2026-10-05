"""Scoped archival hardlink consolidation, pending explicit approval.
All paths and bytes stay present. Existing shared inodes are never replaced.
"""
import argparse,hashlib,json,os,stat,time
from pathlib import Path
from experiment_storage import ROOT,preflight

ACTION="deduplicate_completed_private_cache_copies_preserving_paths_and_bytes"
AUX=Path("/root/prefix-io-v1-validation")

def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for data in iter(lambda:f.read(1024**2),b""):h.update(data)
    return h.hexdigest()

def require(ok,message):
    if not ok:raise ValueError(message)

def verified_path(record,base,*,private):
    p=Path(record["path"]);require(p.is_absolute() and not p.is_symlink(),"noncanonical file")
    require(p.resolve()==p and p.is_relative_to(base),"path outside authorized root or redirected")
    if private:require(p.is_relative_to(base/"runs"),"target must be in completed runs")
    require(p.suffix==".bin" and p.is_file(),"not a cache file")
    s=p.stat()
    require(stat.S_ISREG(s.st_mode),"not a regular file")
    require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==
        (record["device"],record["inode"],record["bytes"],record["mtime_ns"]),"file changed since audit")
    if private:require(s.st_nlink==1 and record["private_completed_run"],"never replace shared data")
    return p

def consolidate_one(canonical,target,expected,*,before_replace=None):
    """Caller has validated scope/metadata. Back up the private inode until verified."""
    require(digest(canonical)==digest(target)==expected,"content mismatch")
    require(target.stat().st_nlink==1,"target acquired another hardlink")
    token=str(os.getpid())+"-"+str(time.monotonic_ns())
    backup=target.with_name(target.name+".dedup-backup-"+token)
    link=target.with_name(target.name+".dedup-link-"+token)
    require(not backup.exists() and not link.exists(),"temporary name collision")
    os.link(target,backup)  # Retain the original private inode through verification.
    replaced=False
    try:
        os.link(canonical,link)
        if before_replace is not None:before_replace()
        require(digest(backup)==expected and digest(link)==expected,"content changed before replacement")
        os.replace(link,target);replaced=True
        require(os.path.samefile(canonical,target) and digest(target)==expected,"post-replacement verification failed")
        backup.unlink()  # Retire only this verified redundant private inode.
        fd=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    except BaseException:
        if replaced and backup.exists():os.replace(backup,target)
        if link.exists():link.unlink()
        if backup.exists():backup.unlink()
        raise

def evidence_path(path,*,project=ROOT,aux=AUX,fresh=False):
    """Permit receipts in project evidence or the explicitly granted auxiliary audits."""
    path=Path(path).absolute()
    require(path.resolve()==path and not path.is_symlink(),"evidence path redirected")
    allowed=(project/"artifacts/prefix_io_v1",aux/"audits")
    require(any(path!=base and path.is_relative_to(base) for base in allowed),"unexpected evidence path")
    require(not fresh or not path.exists(),"journal must be fresh evidence")
    return path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--audit",type=Path,required=True)
    ap.add_argument("--authorization",type=Path)
    ap.add_argument("--journal",type=Path,required=True)
    ap.add_argument("--apply",action="store_true")
    a=ap.parse_args()
    audit=evidence_path(a.audit);journal=evidence_path(a.journal,fresh=True)
    ledger=json.loads((ROOT/"experiments/prefix_io_v1/gpu-budget-ledger.json").read_text())
    require(ledger.get("active_reservation") is None,"no consolidation during an experiment")
    payload=json.loads(audit.read_text());require(payload["read_only"] and not payload["data_changed"],"wrong audit")
    audit_sha=digest(audit)
    todo=[]
    for group in payload["proposals"]:
        canonical=verified_path(group["canonical"],AUX,private=False)
        for record in group["targets"]:
            target=verified_path(record,AUX,private=True)
            result=next((parent/"result.json" for parent in target.parents
                if parent.name=="details"),None)
            require(result is not None and result.is_file(),"missing completed-run receipt")
            run=json.loads(result.read_text())
            require(run.get("engine_shutdown")=="completed" and run.get("status","").startswith("PASSED_"),"run not completed")
            require(target.stat().st_dev==canonical.stat().st_dev,"cross-filesystem hardlink")
            todo.append((canonical,target,group["sha256"],record["allocated_bytes"]))
    require(len({str(t) for _,t,_,_ in todo})==len(todo)==payload["candidate_files"],"duplicate target")
    storage=preflight(AUX/"runs/next",0)
    plan=dict(action=ACTION,apply=a.apply,audit_sha256=audit_sha,target_files=len(todo),
        possible_reclaim_bytes=sum(n for _,_,_,n in todo),storage=storage,
        preserved="all original cache paths and bytes; existing shared files and model files are not replaced",
        effect="verified private archival copies acquire shared immutable inodes",
        general_permissions_changed=False,approval_required=True)
    if not a.apply:print(json.dumps(plan,indent=2));return 0
    require(a.authorization is not None,"explicit scoped authorization required")
    auth_path=a.authorization.resolve()
    require(auth_path.is_relative_to(ROOT/"experiments/prefix_io_v1/configs/authorizations"),"authorization outside project")
    auth=json.loads(auth_path.read_text())
    require(auth.get("approved") is True and auth.get("action")==ACTION and
        auth.get("audit_sha256")==audit_sha and auth.get("user_instruction"),"authorization mismatch")
    # Reserve the receipt bytes on the filesystem that will actually store them.
    # Scope approval remains bound to this exact manifest hash, regardless of location.
    receipt_reserve=max(16*1024**2,len(todo)*2048)
    receipt_path=journal if journal.is_relative_to(AUX/"audits") else ROOT/"experiments/prefix_io_v1/runs/next"
    preflight(receipt_path,receipt_reserve)
    journal.parent.mkdir(parents=True,exist_ok=True)
    with journal.open("x") as f:
        def record(data):
            f.write(json.dumps(data)+"\n");f.flush();os.fsync(f.fileno())
        record(dict(kind="begin",plan=plan,authorization_sha256=digest(auth_path),unix=time.time()))
        done=0
        try:
            for canonical,target,sha,size in todo:
                require(target.stat().st_nlink==1,"target no longer private")
                record(dict(kind="intent",canonical=str(canonical),target=str(target),sha256=sha))
                consolidate_one(canonical,target,sha)
                done+=1
                record(dict(kind="verified",target=str(target),sha256=sha,reclaimed_allocated_bytes=size,
                    canonical_inode=canonical.stat().st_ino))
        except BaseException as exc:
            record(dict(kind="stopped",completed=done,error=repr(exc)));raise
        record(dict(kind="complete",completed=done,storage_after=preflight(AUX/"runs/next",0),unix=time.time()))
    print(json.dumps(dict(completed=done,journal=str(journal))))
    return 0
if __name__=="__main__":raise SystemExit(main())
