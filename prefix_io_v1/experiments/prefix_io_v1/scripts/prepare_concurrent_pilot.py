"""CPU-only freeze of P3 two-sequence controlled text development traces."""
import dataclasses, hashlib, json, os, random, sys, time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"third_party/upstream/kvcache-experiments"))
from common.prefix_cache_common import generate_request_schedule
from transformers import AutoTokenizer
from concurrent_pilot_contract import DELTA, PATTERNS, expected_variant_engine, validate

out=ROOT/"artifacts/prefix_io_v1/server07-p3-06"
old=ROOT/"artifacts/prefix_io_v1/server07-p3-05"
candidate=json.loads((old/"aux-calibration-candidate/curves-v2.json").read_text())
prompts=json.loads((old/"long-heldout-prompts.json").read_text())
families=[dict(name=r["family"],tokens=r["token_ids"],initial_ssd_present=True) for r in prompts["prompts"]]
tokenizer=AutoTokenizer.from_pretrained(ROOT/"models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444",local_files_only=True)
texts=[
("development-library-long","Library preservation manual, edition fourteen. Collections are indexed by accession date and shelf location. ",
 "The conservator records paper condition, binding integrity and previous repairs. Temperature and humidity logs accompany each inspection. Staff compare the current photographs with earlier records and note changes without removing historical annotations. A separate register documents materials used in conservation work. "),
("development-waterworks-long","Waterworks maintenance dossier, edition sixteen. Pump stations and sampling points are catalogued by district. ",
 "The technician verifies the instrument calibration before collecting a measurement. Operational records include flow rate, inspection time and equipment condition. Maintenance steps are logged in order, and replacement parts retain their original identification. The next shift reviews outstanding tasks and records completion independently. ")]
text_rows=[]
for name,intro,paragraph in texts:
    text=intro+"".join("Record "+str(i)+": "+paragraph for i in range(1,500))
    tokens=tokenizer.encode(text,add_special_tokens=False)
    assert len(tokens)>16257
    families.append(dict(name=name,tokens=tokens[:16257],initial_ssd_present=False))
    text_rows.append(dict(name=name,text=text))
schedule=generate_request_schedule(10,16257,16257,0.,.9,.9,1.,random.Random(1704))
arrivals=[x.scheduled_time for x in schedule]
engine=expected_variant_engine({k:v for k,v in candidate["provenance"]["engine"].items() if k!="kv_transfer_config"},DELTA)
index={}
for profile,pattern in PATTERNS.items():
    data=dict(partition="development",profile=profile,scope="Controlled authored documents; Poisson author arrivals; two-sequence problem screen, not production workload or formal performance evaluation",
        engine=engine,staging_bytes=1073741824,output_tokens=128,policy="shadow",load_planner="on",
        artificial_io_delay=False,cache_resets=0,formal_goodput=False,slo=None,
        gpu_uuid=candidate["provenance"]["gpu_uuid"],arrival_seed=1704,arrival_rate=1.,
        scheduler_source="author common.prefix_cache_common.generate_request_schedule",
        families=families,requests=[dict(request_id=i,family_index=f,scheduled_time=arrivals[i]) for i,f in enumerate(pattern)])
    validate(data,candidate,data["gpu_uuid"])
    path=out/(profile+"-manifest.json")
    with path.open("x") as f:json.dump(data,f,indent=2)
    index[profile]=dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
with (out/"development-source-texts.json").open("x") as f:json.dump(text_rows,f,indent=2)
with (out/"manifest-index.json").open("x") as f:json.dump(index,f,indent=2)
print(json.dumps(dict(index=index,arrivals=arrivals,initially_absent_families=[f["name"] for f in families if not f["initial_ssd_present"]])))
