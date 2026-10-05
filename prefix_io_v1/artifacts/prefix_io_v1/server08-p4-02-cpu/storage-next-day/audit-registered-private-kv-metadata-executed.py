"""Read-only registered private KV metadata audit. Never consolidate or unlink."""
from pathlib import Path, PurePosixPath
from hashlib import sha256
import os,stat,json,gzip,time,argparse

ROOT=Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
OUT=ROOT/"artifacts/prefix_io_v1/server08-p4-02-cpu/storage-next-day"
PRIMARY=ROOT/"experiments/prefix_io_v1/runs"
AUX=Path("/root/prefix-io-v1-validation")
INVENTORY=OUT/"registered-completed-run-inventory.json"
PUBLISHED=PRIMARY/"server08-p3-14-source/heldout-long"
REGISTRATION=PRIMARY/"server08-p3-14-source/registration.json"
OLD_MERGE=ROOT/"artifacts/prefix_io_v1/server08-p3-16/private-cache-merge-manifest.json"

def require(ok,message):
    if not ok:raise ValueError(message)
def ref(path):
    data=path.read_bytes()
    return dict(path=str(path),bytes=len(data),sha256=sha256(data).hexdigest())
def strict_object(pairs):
    d={}
    for k,v in pairs:
        require(k not in d,"duplicate metadata JSON key")
        d[k]=v
    return d
def read_json(path,limit=16*1024**2):
    require(path.is_file() and not path.is_symlink(),"regular metadata path")
    require(path.stat().st_size<=limit,"bounded metadata file")
    data=path.read_bytes()
    return json.loads(data,object_pairs_hook=strict_object),dict(path=str(path),bytes=len(data),sha256=sha256(data).hexdigest())

def key(st):return str(st.st_dev)+":"+str(st.st_ino)
def stable(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_nlink,st.st_blocks)
def record(path,st,expected,kind):
    return {"path":str(path),"device":st.st_dev,"inode":st.st_ino,"bytes":st.st_size,
        "allocated_bytes":st.st_blocks*512,"links":st.st_nlink,"mtime_ns":st.st_mtime_ns,
        "ctime_ns":st.st_ctime_ns,"expected_sha256":expected,"kind":kind}
def validate_dir(path,memo):
    if str(path) in memo:return
    cursor=path
    while cursor!=cursor.parent:
        if str(cursor) in memo:break
        st=cursor.lstat()
        require(stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode),"cache parent symlink/nondirectory")
        memo[str(cursor)]=[st.st_dev,st.st_ino]
        cursor=cursor.parent
def member(root,row,memo,kind):
    require(type(row) is dict and set(row)=={"path","bytes","sha256"},"exact registered member keys")
    relative=row["path"]
    require(type(relative) is str and "\\" not in relative and "\x00" not in relative,"member text")
    parts=relative.split("/")
    require(all(p not in ("",".","..") and ":" not in p for p in parts),"unsafe registered member path")
    pure=PurePosixPath(relative)
    require(not pure.is_absolute() and pure.suffix==".bin" and "block_size_" in relative,
            "only explicit KV layout .bin members")
    require(type(row["bytes"]) is int and 0<row["bytes"]<=32*1024**2,"bounded KV storage unit")
    require(type(row["sha256"]) is str and len(row["sha256"])==64 and
            all(c in "0123456789abcdef" for c in row["sha256"]),"declared SHA")
    path=root/relative
    validate_dir(path.parent,memo)
    st=path.lstat()
    require(stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode),"payload symlink/nonregular")
    require(st.st_size==row["bytes"],"registered KV bytes changed")
    return record(path,st,row["sha256"],kind)
def write_new(path,data):
    with path.open("x",encoding="utf-8") as f:json.dump(data,f,sort_keys=True,indent=2);f.write("\n")
def add(state,r):
    ino=str(r["device"])+":"+str(r["inode"])
    obj=state["inodes"].get(ino)
    if obj is None:
        obj=dict(r,paths=[dict(path=r["path"],kind=r["kind"])],expected_hashes=[r["expected_sha256"]])
        state["inodes"][ino]=obj
    else:
        require((obj["bytes"],obj["allocated_bytes"],obj["links"],obj["mtime_ns"],obj["ctime_ns"])==
                (r["bytes"],r["allocated_bytes"],r["links"],r["mtime_ns"],r["ctime_ns"]),
                "inode metadata changed across registered aliases")
        if r["path"] not in {v["path"] for v in obj["paths"]}:
            obj["paths"].append(dict(path=r["path"],kind=r["kind"]))
        if r["expected_sha256"] not in obj["expected_hashes"]:obj["expected_hashes"].append(r["expected_sha256"])

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--start",type=int,required=True)
    ap.add_argument("--count",type=int,default=16);a=ap.parse_args()
    started=time.monotonic();rows,iref=read_json(INVENTORY)
    state_path=OUT/"metadata-state.json.gz"
    if state_path.exists():
        with gzip.open(state_path,"rt") as f:state=json.load(f)
    else:
        require(a.start==0,"first batch must start0")
        state={"next_index":0,"inodes":{},"metadata_refs":[iref],"roots":[],"errors":[],"read_only":True}
    require(state["next_index"]==a.start,"metadata batch sequence mismatch")
    memo={};end=min(a.start+a.count,len(rows))
    ledger,lref=read_json(ROOT/"experiments/prefix_io_v1/gpu-budget-ledger.json")
    require(ledger.get("active_reservation") is None,"no audit during active GPU run")
    if a.start==0:
        state["gpu_ledger_start"]=lref
        registration,rref=read_json(REGISTRATION)
        state["metadata_refs"].append(rref)
        protected=((Path(registration["source_root"]),Path(registration["source_manifest"])),
                   (Path(registration["origin_root"]),Path(registration["origin_manifest"])))
        for cache,manifest in protected:
            require(cache==PUBLISHED or cache==AUX/"storage/heldout-long","unexpected readonly canonical")
            validate_dir(cache,memo);members,mref=read_json(manifest)
            state["metadata_refs"].append(mref)
            for row in members:add(state,member(cache,row,memo,"protected_readonly_source"))
    for index in range(a.start,end):
        entry=rows[index];cache=Path(entry["cache_root"]);run=Path(entry["run"])
        require(cache==run/"details/storage" and (cache.is_relative_to(PRIMARY) or
                cache.is_relative_to(AUX/"runs")),"registered completed private cache scope")
        validate_dir(cache,memo)
        result,rref=read_json(Path(entry["result_path"]))
        require(result.get("engine_shutdown")=="completed" and
                result.get("status","").startswith("PASSED_"),"completed native run required")
        members,mref=read_json(Path(entry["manifest_path"]))
        require(mref["sha256"]==entry["manifest_sha256"],"registered manifest changed after inventory")
        require(len(members)<=4096,"registered cache manifest member limit")
        before_count=len(state["inodes"]);seen=set()
        for row in members:
            require(row["path"] not in seen,"duplicate registered relative member")
            seen.add(row["path"])
            try:add(state,member(cache,row,memo,"completed_private_run"))
            except (OSError,ValueError) as exc:
                state["errors"].append({"root":str(cache),"member":row["path"],"error":str(exc)})
            if time.monotonic()-started>85:raise RuntimeError("bounded metadata batch exceeded85s")
        require(ref(Path(entry["result_path"]))==rref and ref(Path(entry["manifest_path"]))==mref,
                "completion/manifest changed during metadata batch")
        state["roots"].append({"cache_root":str(cache),"file_count":len(members),
                 "new_unique_inodes":len(state["inodes"])-before_count,"result_ref":rref,"manifest_ref":mref})
        state["metadata_refs"].extend((rref,mref));state["next_index"]=index+1
    # Output is audit metadata only; no data/cache path is written.
    with gzip.open(state_path,"wt",compresslevel=6) as f:json.dump(state,f,separators=(",",":"))
    require(ref(ROOT/"experiments/prefix_io_v1/gpu-budget-ledger.json")==lref,"GPU ledger changed during audit")
    summary={"status":"READ_ONLY_REGISTERED_METADATA_BATCH_COMPLETE","start":a.start,"end":end,
        "registered_roots_total":len(rows),"unique_inodes_seen":len(state["inodes"]),
        "metadata_errors":len(state["errors"]),"elapsed_seconds":time.monotonic()-started,
        "private_single_link_inodes":sum(v["links"]==1 and not any(p["kind"]=="protected_readonly_source"
                 for p in v["paths"]) for v in state["inodes"].values()),"payload_files_read":0,
        "gpu_operations":0,"consolidation_performed":False}
    write_new(OUT/("metadata-batch-"+str(a.start)+".json"),summary)
    print(json.dumps(summary))
if __name__=="__main__":main()
