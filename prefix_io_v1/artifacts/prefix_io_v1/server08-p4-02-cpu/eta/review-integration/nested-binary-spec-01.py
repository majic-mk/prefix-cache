from pathlib import Path
import sys,json,hashlib,importlib.util,importlib.abc,importlib.machinery
root=Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
out=root/'artifacts/prefix_io_v1/server08-p4-02-cpu/eta/review-integration/nested-binary-spec-01'
out.mkdir(parents=True,exist_ok=False);scratch=out/'fixture';scratch.mkdir()
attempts=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','py_kvcache','vllm','cupy','numpy'):
   attempts.append(name);raise RuntimeError('CPU review backend guard '+name)
sys.meta_path.insert(0,Guard())
p=root/'experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py'
expected='370c40dd5b4bc0bcf91a5ac9402d65ee80ea5426f34762632ebda9e11abd5491'
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
spec=importlib.util.spec_from_file_location('independent_qualifier',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
name='vllm.vllm_flash_attn._vllm_fa2_C'
suffix=importlib.machinery.EXTENSION_SUFFIXES[0]
rel=m.BINARY+'/vllm_flash_attn/_vllm_fa2_C'+suffix
binary=scratch/rel;binary.parent.mkdir(parents=True);binary.write_bytes(b'CPU REVIEW FAKE ELF DO NOT EXECUTE')
row={'path':rel,'bytes':binary.stat().st_size,'sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
finder=m.BoundAuthorFinder(scratch,{rel:row});selected=finder.locked_binary_spec(name)
assert type(selected.loader) is importlib.machinery.ExtensionFileLoader and selected.origin==str(binary)
results={'exact_nested_spec':{'accepted':True,'loader_type':type(selected.loader).__name__,'origin':selected.origin,'executed':False}}
# Replace execution entrypoints with faults: this review never calls them.
oldcreate=importlib.machinery.ExtensionFileLoader.create_module;oldexec=importlib.machinery.ExtensionFileLoader.exec_module
def forbidden(*a,**kw):raise AssertionError('binary execution forbidden in CPU review')
importlib.machinery.ExtensionFileLoader.create_module=forbidden
importlib.machinery.ExtensionFileLoader.exec_module=forbidden
try:
 def check(label,fn):
  try:fn();results[label]={'rejected':False}
  except Exception as exc:results[label]={'rejected':True,'error':type(exc).__name__+': '+str(exc)}
 oldpy=binary.parent/'_vllm_fa2_C.py';oldpy.write_text("raise AssertionError('old Python fallback executed')\n")
 check('old_adjacent_python_not_fallback',lambda:m.BoundAuthorFinder(scratch,{}).find_spec(name,[str(binary.parent)]))
 changed=dict(row);changed['sha256']='f'*64
 check('changed_exact_binary',lambda:m.BoundAuthorFinder(scratch,{rel:changed}).locked_binary_spec(name))
 if len(importlib.machinery.EXTENSION_SUFFIXES)>1:
  rel2=m.BINARY+'/vllm_flash_attn/_vllm_fa2_C'+importlib.machinery.EXTENSION_SUFFIXES[1]
  second=scratch/rel2;second.write_bytes(b'CPU REVIEW')
  row2={'path':rel2,'bytes':second.stat().st_size,'sha256':hashlib.sha256(second.read_bytes()).hexdigest()}
  check('ambiguous_extension_suffix',lambda:m.BoundAuthorFinder(scratch,{rel:row,rel2:row2}).locked_binary_spec(name))
 check('invalid_module_name',lambda:finder.locked_binary_spec('vllm.a/../../escape'))
 check('oversized_lock_declared',lambda:m.BoundAuthorFinder(scratch,{rel:{**row,'bytes':512*1024**2+1}}).locked_binary_spec(name))
finally:
 importlib.machinery.ExtensionFileLoader.create_module=oldcreate
 importlib.machinery.ExtensionFileLoader.exec_module=oldexec
assert all(v.get('rejected',True) for k,v in results.items() if k!='exact_nested_spec')
assert not attempts and hashlib.sha256(p.read_bytes()).hexdigest()==expected
result={'cases':results,'gpu_operations':0,'backend_import_attempts':attempts,'binary_execution_calls':0,
 'source_sha256':expected,'source_unchanged':True,'claims_ABI_qualified':False}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
