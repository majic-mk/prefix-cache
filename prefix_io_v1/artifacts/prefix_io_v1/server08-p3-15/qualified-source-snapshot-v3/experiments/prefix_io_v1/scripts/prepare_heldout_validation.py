"""Create CPU-tokenized held-out texts before any GPU measurements."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
from transformers import AutoTokenizer
from heldout_manifest import validate_manifest,check_disk
import shutil
p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);args=p.parse_args()
root=Path.cwd();out=args.out.resolve()
if not out.is_relative_to(root/"artifacts/prefix_io_v1"):raise ValueError("output outside project artifacts")
out.mkdir(parents=True,exist_ok=True)
model=root/"models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444"
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
texts=[
("validation-river-warmup","River field notebook: upstream flow measurements and water sampling observations. ",
 "At each sampling station the team recorded water temperature, current velocity, and sediment depth. Samples were labelled before transport. A second technician checked the calibration sheet. Rainfall totals and recent maintenance were kept separate from the instrument readings. "),
("validation-library-measured","Library preservation record: cataloguing, ventilation and book storage procedures. ",
 "The archive contains printed maps, correspondence and reference books. Staff record shelf position before moving a volume. Temperature is measured morning and evening, while humidity is logged continuously. A damaged binding is photographed and entered into the repair register. Visitors consult a copy when the original cannot be handled safely. "),
("validation-factory-measured","Factory inspection log: pump maintenance, spare parts and scheduled production checks. ",
 "Each pump has a serial number, an operating-hour counter and a maintenance history. Inspectors note vibration, lubrication level and seal wear. Replacement parts are checked against the drawing before installation. A trial run follows the documented start sequence. Unexpected pressure readings trigger a second measurement and a written inspection note. ")
]
prompts=[];sources=[]
for rep,(family,intro,paragraph) in enumerate(texts):
    text=intro+"".join("Section "+str(i)+": "+paragraph for i in range(1,90))
    tokens=tokenizer.encode(text,add_special_tokens=False)
    assert len(tokens)>2049
    prompts.append(dict(prefix_tokens=2048,rep=rep,family=family,token_ids=tokens[:2049]))
    sources.append(dict(family=family,text=text,construction="Locally authored controlled document text, truncated to 2049 tokens; no external dataset and no performance result used."))
data=dict(schema_version=1,partition="validation",used_for_fit=False,formal_evaluation=False,
    sizes=[2048],reps=3,prompts=prompts,tokenizer_model=str(model),
    note="One warmup family and two measured held-out families; single length does not validate long-prefix interpolation.")
validate_manifest(data,[2048],3)
budget=check_disk(shutil.disk_usage(root).free,[2048],3,True)
for name,value in [("heldout-prompts.json",data),("heldout-source-texts.json",sources),("heldout-disk-preflight.json",budget)]:
    with (out/name).open("x") as f:json.dump(value,f,indent=2)
print(json.dumps(dict(manifest_sha256=hashlib.sha256((out/"heldout-prompts.json").read_bytes()).hexdigest(),budget=budget)))
