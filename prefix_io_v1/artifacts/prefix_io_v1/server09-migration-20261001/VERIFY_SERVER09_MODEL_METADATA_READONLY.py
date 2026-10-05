import os,pathlib,json,hashlib
root=pathlib.Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
names=[]
for p in root.iterdir():
 if 'model' in p.name.lower():names.append({'path':str(p),'is_dir':p.is_dir(),'is_symlink':p.is_symlink()})
for base in [root/'models',root/'model']:
 if not base.exists():continue
 for directory,dirs,files in os.walk(base,followlinks=False):
  depth=len(pathlib.Path(directory).relative_to(base).parts)
  if depth>3:dirs[:]=[];continue
  for n in files:
   p=pathlib.Path(directory)/n;s=p.lstat()
   if n in ['config.json','generation_config.json','model.safetensors.index.json','tokenizer_config.json']:
    b=p.read_bytes();d=json.loads(b);names.append({'path':str(p),'bytes':s.st_size,'sha256':hashlib.sha256(b).hexdigest(),'model_type':d.get('model_type'),'architectures':d.get('architectures'),'weight_map_count':len(d.get('weight_map',{})),'total_size':d.get('metadata',{}).get('total_size')})
   elif n.endswith('.safetensors'):names.append({'path':str(p),'bytes':s.st_size,'metadata_only':True,'is_symlink':p.is_symlink()})
print(json.dumps({'model_metadata':names,'GPU_imports':0,'model_weight_payload_hashes':0,'server_writes':0}))

