"""Freeze a read-only exact private-cache proposal; no application code."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import os,json,gzip,stat,importlib.util,time
BASE=Path("artifacts/prefix_io_v1/server08-p4-02-cpu/storage-next-day").resolve()
spec=importlib.util.spec_from_file_location("metadata_audit",BASE/"audit-registered-private-kv-metadata-executed.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with gzip.open(BASE/"metadata-state-with-old-audits.json.gz","rt") as f:state=json.load(f)
groups=json.loads((BASE/"candidate-inode-groups-before-real-hash.json").read_text())
def sha_file(path,expected):
    memo={};m.validate_dir(path.parent,memo)
    before=path.lstat()
    m.require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),"regular private KV only")
    m.require((before.st_dev,before.st_ino,before.st_size,before.st_nlink,before.st_mtime_ns,before.st_ctime_ns,before.st_blocks*512)==
      (expected["device"],expected["inode"],expected["bytes"],expected["links"],expected["mtime_ns"],
       expected["ctime_ns"],expected["allocated_bytes"]),"inode metadata changed since read-only inventory")
    digest=sha256()
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
    with os.fdopen(fd,"rb") as stream:
        for chunk in iter(lambda:stream.read(1024**2),b""):digest.update(chunk)
    after=path.lstat()
    m.require(m.stable(before)==m.stable(after),"private KV inode changed while hashing")
    return digest.hexdigest()
started=time.monotonic()
ledger,lref=m.read_json(m.ROOT/"experiments/prefix_io_v1/gpu-budget-ledger.json")
m.require(ledger.get("active_reservation") is None,"no audit during actual GPU run")
# Independently fixed private completion and manifest metadata must stay unchanged.
for r in state["metadata_refs"]:
    m.require(m.ref(Path(r["path"]))==r,"registered source/completion metadata changed")
real_hashes={};hash_log=[]
for group in groups:
    for ino in group["inodes"]:
        if ino in real_hashes:continue
        v=state["inodes"][ino];path=Path(v["paths"][0]["path"])
        digest=sha_file(path,v)
        m.require(digest in v["expected_hashes"] and digest==group["expected_sha256"],
                  "actual KV SHA differs from registered declaration")
        real_hashes[ino]=digest
        hash_log.append(dict(inode_key=ino,path=str(path),bytes=v["bytes"],sha256=digest))
proposals=[];shared_alternatives=[];scope_roots=[]
def public_record(v,path):
    return {k:v[k] for k in ("device","inode","bytes","allocated_bytes","links","mtime_ns","ctime_ns")} | {
       "path":path,"private_completed_run":True}
for group in groups:
    inodes=[state["inodes"][ino] for ino in group["inodes"]]
    inodes.sort(key=lambda v:(not any(p["kind"]=="protected_readonly_source" for p in v["paths"]),-v["links"],v["paths"][0]["path"]))
    canonical=inodes[0]
    canonical_path=next((p["path"] for p in canonical["paths"] if p["kind"]=="protected_readonly_source"),
                        canonical["paths"][0]["path"])
    targets=[]
    for v in inodes[1:]:
        private=all(p["kind"]=="completed_private_run" for p in v["paths"])
        if private and v["links"]==1 and len(v["paths"])==1:
            targets.append(public_record(v,v["paths"][0]["path"]))
        elif private and v["links"]==len(v["paths"]) and not any(p["kind"]=="protected_readonly_source" for p in v["paths"]):
            shared_alternatives.append(dict(canonical=public_record(canonical,canonical_path),
                  closed_private_aliases=[public_record(v,p["path"]) for p in v["paths"]],
                  sha256=group["expected_sha256"],reclaim_upper_bound_bytes=v["allocated_bytes"],
                  never_applied=True,not_covered_by_existing_apply_tool=True))
    if targets:
        proposals.append({"sha256":group["expected_sha256"],"bytes":group["bytes"],
         "canonical":public_record(canonical,canonical_path),"targets":targets,
         "reclaimable_allocated_bytes":sum(t["allocated_bytes"] for t in targets)})
def open_private_fds():
    matches=[];prefixes=tuple(r["cache_root"]+"/" for r in state["roots"])
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():continue
        fds=process/"fd"
        try:entries=list(fds.iterdir())
        except OSError:continue
        for fd in entries:
            try:value=os.readlink(fd)
            except OSError:continue
            if value.startswith(prefixes) and value.endswith(".bin"):
                matches.append(dict(pid=int(process.name),fd=fd.name,path=value))
    return matches
open_fds=open_private_fds()
m.require(not open_fds,"registered private KV currently has open file descriptors")
# Check every proposed target/canonical after hashing; no duplicate target inode.
target_inodes=set()
for g in proposals:
    for r in [g["canonical"]]+g["targets"]:
        p=Path(r["path"]);s=p.lstat()
        m.require((s.st_dev,s.st_ino,s.st_size,s.st_nlink,s.st_mtime_ns,s.st_ctime_ns)==
          (r["device"],r["inode"],r["bytes"],r["links"],r["mtime_ns"],r["ctime_ns"]),
          "proposed cache metadata changed after hashing")
    for r in g["targets"]:
        k=(r["device"],r["inode"]);m.require(k not in target_inodes,"reclaim inode duplicated")
        target_inodes.add(k)
for r in state["metadata_refs"]:
    m.require(m.ref(Path(r["path"]))==r,"source/completion metadata changed after hashing")
m.require(m.ref(m.ROOT/"experiments/prefix_io_v1/gpu-budget-ledger.json")==lref,"GPU ledger changed")
permissions_ref=m.ref(m.ROOT/"experiments/prefix_io_v1/configs/permissions.yaml")
primary_free=os.statvfs(m.PRIMARY).f_bavail*os.statvfs(m.PRIMARY).f_frsize
aux_free=os.statvfs(m.AUX).f_bavail*os.statvfs(m.AUX).f_frsize
floor=8*1024**3;reserve=3*1024**3;aux_max=20*1024**3
# Parent supplied current actual metadata; deliberately do not rewalk unregistered data.
aux_used=20095848448
primary_reclaim=sum(r["allocated_bytes"] for g in proposals for r in g["targets"] if Path(r["path"]).is_relative_to(m.PRIMARY))
aux_reclaim=sum(r["allocated_bytes"] for g in proposals for r in g["targets"] if Path(r["path"]).is_relative_to(m.AUX/"runs"))
observed=datetime.now(timezone.utc).isoformat()
storage={"observation_utc":observed,"reserve_bytes":reserve,"minimum_free_bytes":floor,
 "PRIMARY":{"live_free_bytes":primary_free,"above_floor_bytes":max(0,primary_free-floor),
     "current_deficit_bytes":max(0,reserve+floor-primary_free),"proposal_reclaim_upper_bound_bytes":primary_reclaim,
     "still_missing_after_proposal_bytes":max(0,reserve+floor-primary_free-primary_reclaim),
     "fits_after_proposal_upper_bound":primary_free+primary_reclaim-reserve>=floor},
 "AUX":{"live_free_bytes":aux_free,"max_bytes":aux_max,"carried_forward_used_bytes":aux_used,
     "usage_provenance":"trusted root task message 2026-10-01: current CPU stage actual metadata; not rescanned",
     "usage_is_freshly_measured_by_this_audit":False,"capacity_remaining_bytes":aux_max-aux_used,
     "above_floor_bytes":max(0,aux_free-floor),"current_usable_bytes":min(aux_max-aux_used,max(0,aux_free-floor)),
     "current_deficit_bytes":max(0,reserve-min(aux_max-aux_used,max(0,aux_free-floor))),
     "proposal_reclaim_upper_bound_bytes":aux_reclaim,
     "still_missing_after_proposal_bytes":max(0,reserve-min(aux_max-aux_used+aux_reclaim,max(0,aux_free+aux_reclaim-floor))),
     "fits_after_proposal_upper_bound":aux_used-aux_reclaim+reserve<=aux_max and aux_free+aux_reclaim-reserve>=floor}}
manifest={"schema_version":1,"observation_utc":observed,"read_only":True,"data_changed":False,
 "approval_required":True,"authorization_created":False,"apply_performed":False,"gpu_operations":0,
 "action":"deduplicate_completed_private_cache_copies_preserving_paths_and_bytes",
 "scope":"registered completed private KV payloads only; model weights/libraries/user shared data excluded",
 "completed_cache_roots":len(state["roots"]),
 "completed_cache_roots_allowlist":[r["cache_root"] for r in state["roots"]],
 "protected_registered_sources_not_replaced":True,"unique_inodes_in_metadata":len(state["inodes"]),
 "registered_alias_paths":sum(len(v["paths"]) for v in state["inodes"].values()),
 "metadata_errors":len(state["errors"]),"unique_files_hashed":len(real_hashes),
 "unique_bytes_hashed":sum(r["bytes"] for r in hash_log),"hash_once_per_inode":True,
 "candidate_files":len(target_inodes),"candidate_groups":len(proposals),
 "possible_reclaim_bytes":sum(g["reclaimable_allocated_bytes"] for g in proposals),
 "proposals":proposals,"closed_private_alias_alternatives":shared_alternatives,
 "closed_shared_alias_alternative_count":len(shared_alternatives),
 "open_registered_payload_fd_count":len(open_fds),"current_storage":storage,
 "permissions_ref":permissions_ref,"ledger_ref":lref,
 "input_metadata_refs":state["metadata_refs"],
 "hash_scope":"only candidate distinct-inode expected-equal KV members; all were actually SHA-verified",
 "future_apply_requires":"new exact manifest authorization and full revalidation; post-apply real disk free/cap check",
 "no_space_recovered_now":True,"current_3GiB_model_reserve_status":"BLOCKED_STORAGE",
 "caution":"Old already-shared aliases are never counted twice; this proposal remains insufficient for the original 3GiB reserve."}
m.write_new(BASE/"private-cache-space-proposal-frozen.json",manifest)
m.write_new(BASE/"actual-candidate-inode-hashes.json",hash_log)
receipt={"status":"READ_ONLY_PROPOSAL_COMPLETE_3GIB_RESERVE_STILL_BLOCKED",
  "manifest_ref":m.ref(BASE/"private-cache-space-proposal-frozen.json"),"candidate_files":len(target_inodes),
  "possible_reclaim_bytes":manifest["possible_reclaim_bytes"],"actual_inode_hashes":len(real_hashes),
  "actual_bytes_read":manifest["unique_bytes_hashed"],"elapsed_seconds":time.monotonic()-started,
  "storage":storage,"gpu_operations":0,"hardlinks_created":0,"files_deleted":0,"files_moved":0,
  "cache_files_written":0,"protected_sources_replaced":0}
m.write_new(BASE/"read-only-space-review-result.json",receipt)
print(json.dumps(receipt))
