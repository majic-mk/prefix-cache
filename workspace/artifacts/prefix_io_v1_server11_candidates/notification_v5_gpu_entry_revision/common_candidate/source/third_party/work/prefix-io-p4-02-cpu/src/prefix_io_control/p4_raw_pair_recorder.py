"""Scalar raw-pair preparation, never a GPU timer or an execution authority.

Only complete explicit observations are accepted. P3 host timestamps, missing
per-step IO, changing context, and scheduled-work-v1 cannot be converted into
the active-decode exact9 domain. This journal has no task/resource ownership.
"""
from dataclasses import dataclass,asdict
from hashlib import sha256
from pathlib import Path,PurePosixPath
import json
from .dispatch_budget import STAGES
from .p4_production_table_contract import TableContext,EvidenceRef

ROLES=("baseline_wrapper","action_wrapper","baseline_observations","action_observations")
RUN_KEYS={"pair_id","trace_sha256","prefix_family_sha256","seed","split","arm_order",
 "warmup_windows","measured_windows","input_tokens","output_tokens","request_count",
 "workload_sha256","start_ns","end_ns","exit_code","accepted_io_drained","completed_new_io"}
WINDOW_KEYS={"pair_id","window_id","phase","start_ns","end_ns","load","existing_io","new_io","output_tokens"}
LOAD_KEYS={"active_decode","batch","prefill_tokens","context_length"}
MAX_PAIRS=64;MAX_WINDOWS=4096;MAX_JSON_BYTES=2*1024**2

def require(ok,message):
    if not ok:raise ValueError(message)

def integer(value,name,minimum=0):
    require(type(value) is int and value>=minimum,name+" must be an explicit integer")
    return value

def text(value,name):
    require(type(value) is str and value and len(value)<=128,name+" must be bounded text")

def counters(value):
    require(type(value) is list and len(value)==4,"SSD read/write/H2D/D2H explicit counters")
    out=[]
    for amount in value:
        require(type(amount) is dict and set(amount)=={"ops","bytes"},"strict stage amount")
        for key in ("ops","bytes"):integer(amount[key],key)
        require((amount["ops"]==0)==(amount["bytes"]==0),"stage physical amount mismatch")
        out.append(dict(amount))
    return out

def load_values(value):
    require(type(value) is dict and set(value)==LOAD_KEYS,"actual active-decode exact9 fields required")
    for key,val in value.items():integer(val,key,1 if key in ("active_decode","batch","context_length") else 0)
    require(value["active_decode"]<=value["batch"],"active decode exceeds actual batch")
    return dict(value)

def raw_runs(values):
    require(type(values) is list and 1<=len(values)<=MAX_PAIRS,"bounded complete raw run list")
    result={}
    for run in values:
        require(type(run) is dict and set(run)==RUN_KEYS,"complete explicit wrapper run required")
        text(run["pair_id"],"pair_id");require(run["pair_id"] not in result,"duplicate raw pair")
        for key in ("trace_sha256","prefix_family_sha256","workload_sha256"):
            require(type(run[key]) is str and len(run[key])==64 and all(c in "0123456789abcdef" for c in run[key]),
                    "frozen "+key+" required")
        for key in ("seed","start_ns","end_ns"):integer(run[key],key)
        for key in ("warmup_windows","measured_windows","input_tokens","output_tokens","request_count"):
            integer(run[key],key,1)
        require(run["output_tokens"]>=2,"continuous decode requires at least two measured tokens")
        require(run["split"] in ("calibration","validation") and run["arm_order"] in ("AB","BA"),
                "explicit split/order")
        require(run["start_ns"]<run["end_ns"],"positive actual run interval")
        require(type(run["exit_code"]) is int and run["exit_code"]==0 and run["accepted_io_drained"] is True,
                "normal exit and complete accepted-IO drain required")
        counters(run["completed_new_io"])
        result[run["pair_id"]]=run
    return result

def raw_windows(values,runs,load,expected_existing,expected_new):
    require(type(values) is list and 1<=len(values)<=MAX_WINDOWS,"bounded full per-step raw windows")
    keys=set();by_pair={}
    for row in values:
        require(type(row) is dict and set(row)==WINDOW_KEYS,"complete explicit per-step window required")
        text(row["pair_id"],"pair_id");text(row["window_id"],"window_id")
        require(row["pair_id"] in runs and row["phase"] in ("warmup","measured"),"known pair and phase")
        key=(row["pair_id"],row["window_id"],row["phase"])
        require(key not in keys,"duplicate scalar window");keys.add(key)
        integer(row["start_ns"],"start_ns");integer(row["end_ns"],"end_ns")
        run=runs[row["pair_id"]]
        require(run["start_ns"]<=row["start_ns"]<row["end_ns"]<=run["end_ns"],"actual window inside run")
        require(load_values(row["load"])==load,"context/load mismatch remains unknown; no bucketing")
        require(counters(row["existing_io"])==expected_existing,"actual existing-IO state mismatch")
        require(counters(row["new_io"])==expected_new,"only exact frozen one-stage action allowed")
        integer(row["output_tokens"],"window output tokens",1)
        by_pair.setdefault(row["pair_id"],[]).append(row)
    for pair,run in runs.items():
        rows=sorted(by_pair.get(pair,()),key=lambda r:r["start_ns"])
        require(all(a["end_ns"]<=b["start_ns"] for a,b in zip(rows,rows[1:])),"overlapping windows")
        require([r["phase"] for r in rows]==["warmup"]*run["warmup_windows"]+["measured"]*run["measured_windows"],
                "warmup first and measured counts explicit")
        require(sum(r["output_tokens"] for r in rows if r["phase"]=="measured")==run["output_tokens"],
                "complete measured output token work")
        completed=[dict(ops=a["ops"]*len(rows),bytes=a["bytes"]*len(rows)) for a in expected_new]
        require(counters(run["completed_new_io"])==completed,"all actual added IO must complete at drain")
    return keys

@dataclass(frozen=True)
class PreparedRawPair:
    context: TableContext
    cell_id: str
    origin: str
    role_json: tuple
    paired_runs: int
    measured_windows: int
    @property
    def gpu_verified(self):return False
    @property
    def production_qualified(self):return False
    @property
    def status(self):return "RAW_PAIR_PREPARED_GPU_AUTHENTICITY_UNPROVEN"

def prepare_raw_pair(context,cell_id,*,load,stage,units,action_operations,existing_io,
                     baseline_runs,action_runs,baseline_windows,action_windows,origin):
    require(type(context) is TableContext,"frozen TableContext required")
    text(cell_id,"cell_id");require(origin in ("cpu_fixture","native_gpu_recording"),"explicit recording origin")
    load=load_values(load)
    require(stage in STAGES,"original four physical stages only")
    require(type(units) is int and units in (1,2,4,8),"finite native storage-unit action")
    integer(action_operations,"action_operations",1);require(action_operations<=units,"physical ops bound")
    existing=counters(existing_io);zero=[dict(ops=0,bytes=0) for _ in STAGES]
    new=[dict(x) for x in zero];new[STAGES.index(stage)]=dict(ops=action_operations,bytes=units*context.transfer_quantum_bytes)
    b=raw_runs(baseline_runs);a=raw_runs(action_runs)
    require(set(a)==set(b),"matched baseline/action pair IDs")
    intervals=[]
    for pair,run in b.items():
        other=a[pair]
        require(all(run[k]==other[k] for k in RUN_KEYS-{"start_ns","end_ns","completed_new_io"}),
                "paired trace/work/order changed")
        first,second=(run,other) if run["arm_order"]=="AB" else (other,run)
        require(first["end_ns"]<=second["start_ns"],"actual paired order timestamps")
        intervals.extend((r["start_ns"],r["end_ns"]) for r in (run,other))
    intervals.sort()
    require(all(x[1]<=y[0] for x,y in zip(intervals,intervals[1:])),"independent paired runs overlap")
    base_existing=existing if context.cost_basis=="existing_io_plus_delta" else zero
    keys_b=raw_windows(baseline_windows,b,load,base_existing,zero)
    keys_a=raw_windows(action_windows,a,load,existing,new)
    require(keys_a==keys_b,"same paired window geometry")
    base_rows={(w["pair_id"],w["window_id"],w["phase"]):w for w in baseline_windows}
    require(all(w["output_tokens"]==base_rows[(w["pair_id"],w["window_id"],w["phase"])]["output_tokens"]
                for w in action_windows),"same paired measured token work")
    objects={}
    for role,arm,payload,values in (
        ("baseline_wrapper","baseline","runs",baseline_runs),("action_wrapper","action","runs",action_runs),
        ("baseline_observations","baseline","windows",baseline_windows),
        ("action_observations","action","windows",action_windows)):
        obj=dict(schema_version=1,scope="paired_measurement_wrapper" if payload=="runs" else "paired_window_observations",
                 origin=origin,context=asdict(context),arm=arm,cell_id=cell_id,**{payload:values})
        raw=json.dumps(obj,sort_keys=True,allow_nan=False).encode()
        require(len(raw)<=MAX_JSON_BYTES,"bounded raw role metadata")
        objects[role]=raw
    return PreparedRawPair(context,cell_id,origin,tuple((role,objects[role]) for role in ROLES),
                           len(b),sum(w["phase"]=="measured" for w in baseline_windows))

def write_raw_pair(root,prepared,relative_dir):
    require(type(prepared) is PreparedRawPair,"exact prepared scalar journal")
    require(tuple(role for role,_ in prepared.role_json)==ROLES and
            all(type(data) is bytes and 0<len(data)<=MAX_JSON_BYTES for _,data in prepared.role_json),
            "complete bounded immutable four-role journal")
    root=Path(root).resolve(strict=True)
    require(type(relative_dir) is str and relative_dir and "\\" not in relative_dir,"POSIX output directory")
    p=PurePosixPath(relative_dir)
    require(not p.is_absolute() and all(x not in ("",".","..") for x in relative_dir.split("/")) and
            ":" not in p.parts[0],"bounded project output")
    dest=root
    for part in p.parts:
        dest=dest/part;require(not dest.is_symlink(),"symlink journal output")
    require(not dest.exists(),"raw pair output is append-new")
    dest.mkdir(parents=True,exist_ok=False);refs={}
    for role,raw in prepared.role_json:
        path=dest/(role+".json")
        with path.open("xb") as h:h.write(raw)
        refs[role]=EvidenceRef(path.relative_to(root).as_posix(),len(raw),sha256(raw).hexdigest())
    return refs

def assemble_raw_pair(root,prepared,relative_dir,*,cell_geometry,expected_plan,expected_verifier_ref):
    """Append full metadata and revalidate using the one existing cost estimator.

    The qualification *candidate* reports semantic consistency only. Neither a
    native_gpu_recording label nor this report proves that a GPU ran.
    """
    from .p4_paired_measurement_verifier import build_paired_cell_candidate
    from .p4_verified_cost_loader import load_semantically_verified_table
    require(type(expected_verifier_ref) is EvidenceRef,"byte-bound semantic verifier required")
    require(expected_plan.context==prepared.context and
            {name for name,_ in expected_plan.action_operations}=={prepared.cell_id},
            "one-cell assembly must match exact frozen context and plan")
    expected_verifier_ref.verify(root)
    raw_refs=write_raw_pair(root,prepared,relative_dir+"/raw")
    candidate=build_paired_cell_candidate(root,cell_geometry=cell_geometry,raw_refs=raw_refs,
        expected_plan=expected_plan,analysis_path=relative_dir+"/analysis.json")
    def write(name,data):
        path=Path(root)/relative_dir/name
        data=json.dumps(data,sort_keys=True,allow_nan=False).encode()
        require(len(data)<=MAX_JSON_BYTES,"bounded assembled metadata")
        with path.open("xb") as h:h.write(data)
        return EvidenceRef(path.relative_to(Path(root)).as_posix(),len(data),sha256(data).hexdigest())
    cell=candidate.to_candidate_mapping()
    cref=write("candidate.json",dict(schema_version=1,scope="production_candidate",
        context=asdict(prepared.context),cells=[cell]))
    qref=write("qualification-candidate.json",dict(schema_version=1,
        scope="production_qualification_candidate",status="PASS_REPORTED",candidate_ref=asdict(cref),
        context=asdict(prepared.context),verifier_ref=asdict(expected_verifier_ref),
        measurement_refs=list(cell["measurement_refs"].values())))
    expected_verifier_ref.verify(root)
    loaded=load_semantically_verified_table(root,cref.path,expected_context=prepared.context,
        qualification_ref=qref,expected_verifier_ref=expected_verifier_ref,
        plan_path=expected_plan.plan_ref.path,expected_plan_ref=expected_plan.plan_ref)
    require(not loaded.production_qualified and not loaded.gpu_verified,"CPU assembly cannot qualify GPU")
    return loaded,dict(raw_refs={k:asdict(v) for k,v in raw_refs.items()},
        analysis_ref=asdict(candidate.analysis_ref),candidate_ref=asdict(cref),qualification_ref=asdict(qref),
        status="RAW_PAIR_SEMANTICS_VERIFIED_GPU_AUTHENTICITY_UNPROVEN",
        origin=prepared.origin,production_qualified=False,gpu_verified=False,
        GPU_runs_performed=0,model_loaded=False)
