"""CPU replay of the actual site startup and exact source query; no framework/model execution."""
import argparse,ast,functools,hashlib,importlib,importlib.machinery,json,pathlib,sys,types
p=argparse.ArgumentParser()
p.add_argument("--project",type=pathlib.Path,required=True)
p.add_argument("--helper",required=True);p.add_argument("--source-lock",required=True)
a=p.parse_args();root=a.project.resolve()
assert not any(n=="torch" or n.startswith("torch.") or n=="vllm" or n.startswith("vllm.") for n in sys.modules)
assert not sys.flags.no_site and len(sys.path_hooks)==3
refs={row["path"]:row for row in json.loads((root/a.source_lock).read_bytes())["files"]}
row=refs[a.helper];raw=(root/a.helper).read_bytes()
assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]
H=types.ModuleType("_normal_site_cpu_optional_helper");H.__file__=str(root/a.helper)
sys.modules[H.__name__]=H;exec(compile(raw,H.__file__,"exec"),H.__dict__)
assert H.verify_optional_absence(root,refs)["physical_absence_verified"] is True
original_source=(root/H.QUALIFY).read_bytes()
base=types.ModuleType("_normal_site_cpu_original_finder");base.__file__=str(root/H.QUALIFY)
sys.modules[base.__name__]=base;exec(compile(original_source,base.__file__,"exec",dont_inherit=True),base.__dict__)
finder=base.BoundAuthorFinder(root,refs)
# Only a synthetic top package prevents importing the real framework.
# Its actual empty third_party child is loaded by the unchanged source finder.
top=types.ModuleType("vllm");top.__path__=[str(root/H.AUTHOR/"vllm")]
top.__spec__=importlib.machinery.ModuleSpec("vllm",None,is_package=True)
top.__spec__.submodule_search_locations=top.__path__
sys.modules["vllm"]=top
module=types.ModuleType("vllm.utils.import_utils");module.__file__=str(root/H.IMPORT_UTILS)
module.__dict__.update(cache=functools.cache,importlib=importlib)
nodes=[n for n in ast.parse((root/H.IMPORT_UTILS).read_bytes(),filename=module.__file__).body
 if isinstance(n,ast.FunctionDef) and n.name in ("_has_module","has_deep_gemm")]
exec(compile(ast.Module(body=nodes,type_ignores=[]),module.__file__,"exec",dont_inherit=True),module.__dict__)
sys.modules[module.__name__]=module
hooks_before=list(sys.path_hooks);meta_before=list(sys.meta_path);handle=None;later_calls=[]
class LaterFinder:
 def find_spec(self,fullname,path=None,target=None):
  if fullname==H.OPTIONAL_NAME:later_calls.append(fullname);raise AssertionError("later finder must never run")
sentinel=LaterFinder();sys.meta_path.insert(0,finder);sys.meta_path.append(sentinel)
try:
 parent=importlib.import_module("vllm.third_party")
 assert pathlib.Path(parent.__file__)==root/H.PARENT_RELATIVE/"__init__.py"
 missing=None
 try:module.has_deep_gemm()
 except ModuleNotFoundError as e:missing=e
 assert missing is not None and type(missing) is ModuleNotFoundError and missing.name==H.OPTIONAL_NAME
 cached=sys.path_importer_cache[str(root/H.PARENT_RELATIVE)]
 assert type(cached) is importlib.machinery.FileFinder
 original=module._has_module;handle=H.install_capability_probe(module,finder,root,refs)
 assert module.has_deep_gemm() is False
 assert sys.path_importer_cache[str(root/H.PARENT_RELATIVE)] is cached
 assert module._has_module(H.OPTIONAL_NAME) is False
 try:importlib.import_module(H.OPTIONAL_NAME)
 except ModuleNotFoundError as e:assert type(e) is ModuleNotFoundError and e.name==H.OPTIONAL_NAME
 else:raise AssertionError("required import was allowed")
 assert not later_calls
 assert handle.detach() is True and module._has_module is original;handle=None
 assert all(left is right for left,right in zip(hooks_before,sys.path_hooks)) and len(hooks_before)==len(sys.path_hooks)
 assert not any(n=="torch" or n.startswith("torch.") for n in sys.modules)
 print(json.dumps(dict(status="PASS_ACTUAL_NORMAL_SITE_CPU_EXACT_ORIGINAL_QUERY_REPLAY",
 synthetic_top_package_and_capability_namespace=True,original_capability_functions_AST=True,
 actual_empty_parent_loaded_by_original_finder=True,
 actual_site_path_hooks_count=len(sys.path_hooks),actual_site_hooks_unchanged=True,
 canonical_cached_FileFinder_same_object=True,original_missing_exception_reproduced=True,
 patched_optional_capability=False,required_import_still_rejected=True,later_unknown_meta_finder_calls=0,
 original_cached_function_restored=True,framework_imported=False,new_GPU_jobs=0,
 GPU_collector_verified=False,model_output_qualified=False,performance_claim=False)))
finally:
 if handle is not None:handle.detach()
 sys.meta_path[:]=meta_before
 for name in ("vllm","vllm.third_party","vllm.utils.import_utils",base.__name__,H.__name__):sys.modules.pop(name,None)
