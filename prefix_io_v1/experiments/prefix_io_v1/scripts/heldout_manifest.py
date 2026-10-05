"""CPU-only manifest and conservative disk checks for held-out acquisition."""
import hashlib,json
from pathlib import Path

FLOOR=8*1024**3
LOG_RESERVE=64*1024**2

def require(value,message):
    if not value: raise ValueError(message)

def validate_manifest(data,sizes,reps):
    require(data.get("schema_version")==1 and data.get("partition")=="validation",
            "validation partition/schema required")
    require(data.get("used_for_fit") is False and data.get("formal_evaluation") is False,
            "manifest cannot be fit/evaluation data")
    require(len(sizes)==len(set(sizes)) and all(type(n) is int and 16<=n<=16384 and n%16==0 for n in sizes)
            and type(reps) is int and 1<=reps<=8,"invalid domain")
    require(data.get("sizes")==sizes and data.get("reps")==reps,"manifest domain mismatch")
    rows=data.get("prompts");require(isinstance(rows,list) and len(rows)==len(sizes)*reps,"manifest coverage")
    result={}; families=set(); prefixes=set()
    calibration_first_blocks={tuple([2000+n+r*1500]+[1000+i%500 for i in range(15)])
                              for n in [16,64,128,256,512,1024,2048,4096,8192,16384] for r in range(18)}
    for row in rows:
        n=row["prefix_tokens"];rep=row["rep"];key=(n,rep);tokens=row["token_ids"]
        require(type(n) is int and n in sizes and type(rep) is int and 0<=rep<reps and key not in result,
                "duplicate/unexpected prompt")
        require(isinstance(tokens,list) and len(tokens)==n+1 and
                all(type(t) is int and 0<=t<151643 for t in tokens),"invalid token sequence")
        family=row.get("family")
        require(isinstance(family,str) and family.startswith("validation-") and family not in families,
                "duplicate/invalid family")
        first=tuple(tokens[:16])
        require(first not in prefixes and first not in calibration_first_blocks,"prefix family leakage")
        families.add(family);prefixes.add(first);result[key]=tokens
    return result

def load_manifest(path,sizes,reps):
    raw=Path(path).read_bytes();data=json.loads(raw)
    return validate_manifest(data,sizes,reps),dict(path=str(Path(path).resolve()),
        sha256=hashlib.sha256(raw).hexdigest(),partition=data["partition"],
        used_for_fit=False,formal_evaluation=False,families=[r["family"] for r in data["prompts"]])

def disk_requirement(sizes,reps,populate):
    # 16-token files; one-token output cannot complete another block.
    # Include one extra block per prompt for conservative publication accounting.
    payload=sum((n//16+1)*917504*reps for n in sizes) if populate else 0
    metadata=((payload+49)//50) if payload else 0
    return dict(floor_bytes=FLOOR,log_reserve_bytes=LOG_RESERVE,
                maximum_new_kv_bytes=payload,metadata_reserve_bytes=metadata,
                required_free_bytes=FLOOR+LOG_RESERVE+payload+metadata)

def check_disk(free,sizes,reps,populate):
    result=disk_requirement(sizes,reps,populate)
    require(free>=result["required_free_bytes"],"heldout disk budget insufficient")
    return dict(result,free_before_bytes=free)
