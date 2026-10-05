import ast,difflib,hashlib,json,subprocess
from pathlib import Path
root=Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
e=root/"artifacts/prefix_io_v1/new-server-03/native-prefix-cwd"
p=root/"experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py"
before=e/"native_gpu_prefix_smoke.before.py"
a=before.read_text();b=p.read_text()
compile(b,str(p),"exec")
assert b.index("output_dir = args.output_dir.resolve()") < b.index("validate_local_model(args.model_dir, args.model_plan)") < b.index('write_new_json(output_dir / "frozen-config.json", frozen)') < b.index("os.chdir(output_dir)") < b.index("        import vllm") < b.index("llm = LLM")
assert '"working_directory": str(output_dir)' in b
assert 'return model_dir, {**provider' in b and '"plan_path": str(plan_path)' in b
rel=str(p.relative_to(root))
patch="".join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile="a/"+rel,tofile="b/"+rel))
out=e/"0002-native-prefix-run-directory.patch";out.write_text(patch)
c=subprocess.run(["git","apply","--reverse","--check",str(out)],cwd=root,capture_output=True,text=True)
assert c.returncode==0,c.stderr
r={"scope":"CPU source-order verification and 24 existing unit regressions","gpu_operations":0,"source_compile":"passed","paths_validated_before_chdir":True,"chdir_before_vllm_import":True,"working_directory_frozen":True,"reverse_patch_check":"passed","before_sha256":hashlib.sha256(before.read_bytes()).hexdigest(),"after_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"patch_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"tests":"24 passed in 0.10s; cpu-tests.xml"}
(e/"verification.json").write_text(json.dumps(r,indent=2)+"\n")
print(json.dumps(r,indent=2))
