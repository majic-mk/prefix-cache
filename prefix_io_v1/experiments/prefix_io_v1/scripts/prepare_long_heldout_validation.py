"""CPU preparation for independent 16256-token validation, frozen before runs."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
from transformers import AutoTokenizer
from heldout_manifest import validate_manifest
p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);a=p.parse_args()
root=Path.cwd();out=a.out.resolve();assert out.is_relative_to(root/"artifacts/prefix_io_v1")
tokenizer=AutoTokenizer.from_pretrained(root/"models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444",local_files_only=True)
topics=[
("validation-observatory-long","Observatory operations handbook, edition seven. Weather records and telescope maintenance are indexed by observing night. ",
 "The operator records cloud cover, sky brightness and pointing coordinates before each exposure. Reference images are stored with calibration frames. A maintenance log tracks alignment checks and replaced components. The shift leader reviews the nightly summary and marks incomplete observations for later review. "),
("validation-greenhouse-long","Greenhouse planning dossier, edition nine. Plant propagation records connect seed batches, irrigation schedules and environmental controls. ",
 "Seedlings are grouped by planting date and their trays retain the original identification labels. Water flow is measured before a schedule change. Staff record soil moisture, temperature and growth stage. Each inspection notes ventilation conditions and the location of any damaged leaves. The weekly review compares measurements with the recorded maintenance history. "),
("validation-railway-long","Railway workshop reference, edition twelve. Track inspection schedules and component records are organised by route section. ",
 "The team checks measurement instruments before travelling to the site. Rail joints, fasteners and drainage channels are photographed with location markers. Inspection records include date, weather, observations and the responsible technician. Any uncertain measurement is repeated and both readings are retained. Repairs follow the documented sequence and require a separate completion record. ")]
rows=[];texts=[]
for rep,(family,intro,paragraph) in enumerate(topics):
    text=intro+"".join("Record "+str(i)+": "+paragraph for i in range(1,500))
    ids=tokenizer.encode(text,add_special_tokens=False);assert len(ids)>16257
    rows.append(dict(prefix_tokens=16256,rep=rep,family=family,token_ids=ids[:16257]))
    texts.append(dict(family=family,text=text))
data=dict(schema_version=1,partition="validation",used_for_fit=False,formal_evaluation=False,
    sizes=[16256],reps=3,prompts=rows,construction="Locally authored controlled documents; independent from calibration and previous 2K validation; one warmup and two measured families.")
validate_manifest(data,[16256],3)
old=json.loads((root/"artifacts/prefix_io_v1/server07-p3-04/heldout-prompts.json").read_text())
assert not {tuple(r["token_ids"][:16]) for r in rows}&{tuple(r["token_ids"][:16]) for r in old["prompts"]}
for name,obj in [("long-heldout-prompts.json",data),("long-heldout-source-texts.json",texts)]:
    with (out/name).open("x") as f:json.dump(obj,f,indent=2)
print(json.dumps(dict(sha256=hashlib.sha256((out/"long-heldout-prompts.json").read_bytes()).hexdigest(),tokens=16257,families=[r["family"] for r in rows])))
