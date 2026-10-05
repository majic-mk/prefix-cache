"""Read-only verification of an approved archival dedup manifest after application."""
import argparse,hashlib,json,os
from pathlib import Path

def verify(path):
    audit=json.loads(Path(path).read_text());count=0
    base=Path("/root/prefix-io-v1-validation")
    seen=set()
    for group in audit["proposals"]:
        canonical=Path(group["canonical"]["path"])
        if canonical.is_symlink() or canonical.resolve()!=canonical or not canonical.is_relative_to(base):
            raise ValueError("canonical scope changed")
        if hashlib.sha256(canonical.read_bytes()).hexdigest()!=group["sha256"]:
            raise ValueError("canonical content mismatch")
        for record in group["targets"]:
            target=Path(record["path"])
            if str(target) in seen:raise ValueError("duplicate target")
            seen.add(str(target))
            if (target.is_symlink() or target.resolve()!=target or not target.is_relative_to(base/"runs")
                    or target.stat().st_size!=group["bytes"] or not os.path.samefile(target,canonical)):
                raise ValueError("target path/content identity mismatch")
            count+=1
    if count!=audit["candidate_files"]:raise ValueError("wrong target count")
    return dict(passed=True,manifest=str(path),audit_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        targets=count,canonical_sha256_checks=len(audit["proposals"]),all_target_paths_retained=True,
        all_target_bytes_equal_hashed_canonical=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--audit",action="append",required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();results=[verify(path) for path in a.audit]
    debris=[str(p) for p in Path("/root/prefix-io-v1-validation/runs").rglob("*") if ".dedup-" in p.name]
    if debris:raise ValueError("dedup temporary files remain")
    report=dict(passed=True,manifests=results,no_temporary_debris=True)
    with a.output.open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
