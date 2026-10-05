"""Freeze author-generated arrivals and exact-token controlled development traces."""
import dataclasses,hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"third_party/upstream/kvcache-experiments"))
from common.prefix_cache_common import generate_request_schedule

def main():
    target=ROOT/"experiments/prefix_io_v1/manifests/p3-development-01"
    target.mkdir(parents=True,exist_ok=False)
    profiles=[("low_contention",12,512,.75,.5,0),("mixed_pressure",24,768,.5,4.,1),("low_reuse_pressure",24,768,0.,4.,2)]
    index={}
    for name,n,length,reuse,rate,seed in profiles:
        specs=generate_request_schedule(n,length,length,reuse,.90,.90,rate,random.Random(seed))
        rng=random.Random(seed+100);rows=[]
        for spec in specs:
            tokens=[22000+spec.request_id]+[rng.randrange(1000,20000) for _ in range(spec.doc_tokens-1)]
            if spec.reuse_source_id is not None:
                m=spec.reuse_prefix_len
                tokens[:m]=rows[spec.reuse_source_id]["prompt_token_ids"][:m]
            rows.append(dict(**dataclasses.asdict(spec),prompt_token_ids=tokens))
        data=dict(schema=1,profile=name,seed=seed,scope="controlled synthetic-token development replay; no artificial I/O delay, not a production QA dataset",
            scheduler_source="author common.prefix_cache_common.generate_request_schedule",
            output_tokens=128,stop_rule="fixed 128, ignore_eos=true; all arms identical",
            warmup_prompt_tokens=[30000]+list(range(1000,1128)),warmup_output_tokens=16,
            engine=dict(kv_cache_memory_bytes=268435456,max_model_len=1024,max_num_seqs=4,max_num_batched_tokens=1024),
            storage=dict(staging_mem_mib=128,iodepth=4,io_backend="linux_aio",preload=True,shared_staging=True),
            admission="unchanged original LoadPlanner with P1 paired curves; batch-4 curve accuracy not yet requalified",
            slo=None,formal_goodput_enabled=False,requests=rows)
        p=target/(name+".json");p.write_text(json.dumps(data,indent=2))
        index[name]=dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),requests=n)
    (target/"index.json").write_text(json.dumps(index,indent=2));print(json.dumps(index,indent=2))
if __name__=="__main__":main()
