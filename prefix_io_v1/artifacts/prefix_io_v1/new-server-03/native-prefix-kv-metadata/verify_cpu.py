import ast, difflib, hashlib, json, os, subprocess
from pathlib import Path
root=Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
e=root/"artifacts/prefix_io_v1/new-server-03/native-prefix-kv-metadata"
target=root/"experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py"
before=(e/"native_gpu_prefix_smoke.before.py").read_text()
after=target.read_text()
compile(after,str(target),"exec")
tree=ast.parse(after)
prior=ast.parse(before)
def consts(tree):
 return {t.targets[0].id:ast.dump(t.value) for t in tree.body if isinstance(t,ast.Assign) and isinstance(t.targets[0],ast.Name) and t.targets[0].id in {"ENGINE","SAMPLING","PROMPT_TOKEN_IDS","EXPECTED_HOT_TOKENS"}}
assert consts(tree)==consts(prior)
rpc=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="collective_rpc"]
assert len(rpc)==1 and isinstance(rpc[0].args[0],ast.Name) and rpc[0].args[0].id=="worker_kv_storage_metadata"
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="worker_kv_storage_metadata")
attrs={n.func.attr for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
assert not attrs.intersection({"cpu","cuda","to","numpy","item","tolist","synchronize","clone","copy_","empty_cache"})
assert {"untyped_storage","nbytes","data_ptr"}.issubset(attrs)
assert 'os.chdir(output_dir)' in after and after.index('os.chdir(output_dir)')<after.index('        import vllm')
assert after.index("llm = LLM(")<after.index("storage_reports = llm.collective_rpc")<after.index('for name in ("cold", "repeat")')
rel="experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py"
patch=e/"0003-native-prefix-kv-storage-metadata.patch"
patch.write_text("".join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile="a/"+rel,tofile="b/"+rel)))
check=subprocess.run(["git","apply","--reverse","--check",str(patch)],cwd=root,text=True,capture_output=True)
assert check.returncode==0,check.stderr
# Forward check on a temporary, evidence-local copy of only the prior script.
scratch=e/"patch-forward-check"
(scratch/Path(rel).parent).mkdir(parents=True,exist_ok=True)
(scratch/rel).write_text(before)
forward=subprocess.run(["git","apply","--check",str(patch)],cwd=scratch,text=True,capture_output=True)
assert forward.returncode==0,forward.stderr
env=dict(os.environ,CUDA_VISIBLE_DEVICES="")
cli=subprocess.run([str(root/".venv/bin/python"),str(target),"--help"],cwd=root,env=env,text=True,capture_output=True)
(e/"cli-help.log").write_text(cli.stdout+cli.stderr)
assert cli.returncode==0
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out={"status":"PASSED_CPU_ONLY","gpu_executed":False,"model_loaded":False,"torch_or_vllm_imported_by_verification":False,
"compile":"passed","engine_sampling_prompt_unchanged":True,"rpc_count":len(rpc),"metadata_only_source_check":"passed",
"cwd_preserved":True,"forward_patch_check":forward.returncode,"reverse_patch_check":check.returncode,
"cli_help_exit":cli.returncode,"before_sha256":sha(e/"native_gpu_prefix_smoke.before.py"),"after_sha256":sha(target),"patch_sha256":sha(patch),
"tests_sha256":sha(root/"tests/prefix_io_v1_native_prefix/test_kv_storage_metadata.py"),
"qualification":"synthetic CPU metadata tests only; actual tensor bytes require real native GPU run"}
(e/"verification.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
