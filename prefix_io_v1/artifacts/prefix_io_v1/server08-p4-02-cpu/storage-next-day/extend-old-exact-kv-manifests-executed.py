"""Extend registered metadata with earlier exact consolidation manifests only."""
from pathlib import Path
import json,gzip,importlib.util,time,stat
base=Path("artifacts/prefix_io_v1/server08-p4-02-cpu/storage-next-day").resolve()
spec=importlib.util.spec_from_file_location("registered_audit",base/"audit-registered-private-kv-metadata-executed.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with gzip.open(base/"metadata-state.json.gz","rt") as f:state=json.load(f)
inventory,_=m.read_json(m.INVENTORY)
closed_roots={row["cache_root"] for row in inventory}
sources=[m.OLD_MERGE]+sorted((m.AUX/"audits").glob("*audit.json"))
memo={};seen_paths={row["path"] for inode in state["inodes"].values() for row in inode["paths"]}
summary=[];started=time.monotonic()
for source in sources:
    data,sref=m.read_json(source)
    m.require(data.get("read_only") is True and data.get("data_changed") is False,"earlier read-only exact audit required")
    state["metadata_refs"].append(sref)
    examined=0;new=0;skipped=0
    for group in data["proposals"]:
        for old in [group["canonical"]]+group["targets"]:
            p=Path(old["path"])
            m.require(p.is_absolute() and str(p)==old["path"],"canonical absolute old cache path")
            if str(p) in seen_paths:continue
            root=next((Path(root) for root in closed_roots if str(p).startswith(root+"/")),None)
            if root is not None:kind="completed_private_run"
            elif p.is_relative_to(m.AUX/"storage"):root=m.AUX/"storage";kind="protected_readonly_source"
            elif p.is_relative_to(m.PUBLISHED):root=m.PUBLISHED;kind="protected_readonly_source"
            else:
                skipped+=1
                state["errors"].append(dict(source=str(source),path=str(p),error="not a completed registered private cache or protected registered source"))
                continue
            relative=p.relative_to(root).as_posix()
            row={"path":relative,"bytes":group["bytes"],"sha256":group["sha256"]}
            try:
                record=m.member(root,row,memo,kind);m.add(state,record)
                seen_paths.add(str(p));new+=1
            except (OSError,ValueError) as exc:
                state["errors"].append(dict(source=str(source),path=str(p),error=str(exc)))
            examined+=1
            if time.monotonic()-started>95:raise RuntimeError("bounded extra metadata scan exceeded95s")
    summary.append({"manifest_ref":sref,"new_registered_paths":new,"examined_new":examined,"excluded":skipped})
with gzip.open(base/"metadata-state-with-old-audits.json.gz","wt",compresslevel=6) as f:
    json.dump(state,f,separators=(",",":"))
potential={}
for ino,v in state["inodes"].items():
    if len(v["expected_hashes"])!=1:continue
    k=(v["device"],v["bytes"],v["expected_hashes"][0])
    potential.setdefault(k,[]).append(ino)
duplicates=[dict(device=k[0],bytes=k[1],expected_sha256=k[2],inodes=ids) for k,ids in potential.items() if len(ids)>1]
m.write_new(base/"candidate-inode-groups-before-real-hash.json",duplicates)
out={"status":"READ_ONLY_METADATA_SCOPE_COMPLETE","completed_registered_roots":len(inventory),
     "unique_inodes":len(state["inodes"]),"registered_alias_paths":len(seen_paths),"errors":len(state["errors"]),
     "single_link_private_inodes":sum(v["links"]==1 and all(p["kind"]=="completed_private_run" for p in v["paths"])
               for v in state["inodes"].values()),
     "content_conflicting_inodes":sum(len(v["expected_hashes"])!=1 for v in state["inodes"].values()),
     "distinct_inode_expected_duplicate_groups":len(duplicates),"elapsed_seconds":time.monotonic()-started,
     "old_manifest_summaries":summary,"payload_files_read":0,"gpu_operations":0,"data_changed":False}
m.write_new(base/"metadata-scope-complete.json",out)
print(json.dumps({k:v for k,v in out.items() if k!="old_manifest_summaries"}))
