from pathlib import Path
from hashlib import sha256
import json
import shutil
root=Path.cwd()
out=root/"artifacts/prefix_io_v1/server08-p4-02-cpu/cost-verifier"
base=out/"fixtures-v2-cli-01"
base.mkdir(exist_ok=False)
PERMISSION="experiments/prefix_io_v1/configs/permissions.yaml"
AUTH_CHOICES=("artifacts/prefix_io_v1/server08-p4-02-cpu/CPU_ONLY_AUTHORIZATION.json",
              "artifacts/prefix_io_v1/server08-p4-01-cpu/CPU_ONLY_AUTHORIZATION.json")
auth=next(name for name in AUTH_CHOICES if (root/name).is_file())
def ref(dest,name):
    raw=(dest/name).read_bytes()
    return {"path":name,"bytes":len(raw),"sha256":sha256(raw).hexdigest()}
cases=[]
for name,source,mutation in (
    ("positive-full128-selected1","continuous-context-full128-selected1",None),
    ("positive-full128-selected2","continuous-context-full128-selected2",None),
    ("negative-original-source-drift","continuous-context-full128-selected1","raw"),
    ("negative-complete-trace-drift","continuous-context-full128-selected1","trace"),
    ("negative-v1-role-in-v2-bundle","continuous-context-full128-selected1","version")):
    dest=base/name
    shutil.copytree(out/"fixtures-v2-01"/source,dest)
    for relative in (PERMISSION,auth):
        target=dest/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((root/relative).read_bytes())
    candidate=json.loads((dest/"candidate.json").read_text())
    cell=candidate["cells"][0]
    bundle={"schema_version":2,"scope":"p4_explicit_pair_observations",
        "context":candidate["context"],"cell_id":cell["cell_id"],
        "origin":"cpu_fixture","cell_geometry":{key:cell[key] for key in
            ("cell_id","load","existing_io","stage","physical_bytes")},
        "raw_refs":{key:cell["measurement_refs"][key] for key in
            ("baseline_wrapper","action_wrapper","baseline_observations","action_observations")}}
    if mutation=="version":
        path=dest/"action_wrapper.json"
        value=json.loads(path.read_text());value["schema_version"]=1
        path.write_text(json.dumps(value,sort_keys=True))
        bundle["raw_refs"]["action_wrapper"]=ref(dest,"action_wrapper.json")
    (dest/"bundle.json").write_text(json.dumps(bundle,sort_keys=True))
    for name2,target in (("bundle-ref.json","bundle.json"),("plan-ref.json","plan.json"),
                         ("verifier-ref.json","verifier.py")):
        (dest/name2).write_text(json.dumps(ref(dest,target),sort_keys=True))
    if mutation=="raw":(dest/"action_wrapper.json").write_text("{}")
    if mutation=="trace":(dest/"complete-action-p0.json").write_text("{}")
    cases.append({"name":name,"root":str(dest),"expected_exit":0 if mutation is None else 1,
        "mutation":mutation,"origin":"cpu_fixture","actual_GPU_runs":0})
(base/"case-plan.json").write_text(json.dumps(cases,sort_keys=True,indent=2)+"\n")
print(json.dumps({"cases":cases,"GPU_runs":0,"old_fixtures_modified":False}))
