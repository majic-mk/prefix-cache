import ast, difflib, hashlib, json, os, subprocess
from pathlib import Path
from types import SimpleNamespace
r=Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
e=r/"artifacts/prefix_io_v1/new-server-03/native-prefix-kv-dtype"
p=r/"experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py"
before=(e/"native_gpu_prefix_smoke.before.py").read_text();after=p.read_text()
compile(after,str(p),"exec")
a,b=ast.parse(before),ast.parse(after)
def constants(tree):
 return {n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {"ENGINE","SAMPLING"}}
ca,cb=constants(a),constants(b)
assert ca["ENGINE"]["kv_cache_dtype"]=="bfloat16"
assert cb["ENGINE"]==dict(ca["ENGINE"],kv_cache_dtype="auto")
assert cb["SAMPLING"]==ca["SAMPLING"]
assert cb["ENGINE"]["dtype"]=="bfloat16" and cb["ENGINE"]["quantization"] is None
for name in ["worker_kv_storage_metadata","summarize_output","configure_runtime_environment","validate_local_model"]:
 get=lambda t:ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name))
 assert get(a)==get(b),name
assert 'PROMPT_TOKEN_IDS = list(range(1000, 1128))' in after
assert 'cold["output_token_ids"] == repeat["output_token_ids"]' in after
assert 'cold["num_cached_tokens"] == 0' in after and 'repeat["num_cached_tokens"] == EXPECTED_HOT_TOKENS' in after
assert 'validate_effective_config(config, torch.bfloat16)' in after
assert 'report["actual_kv_tensor_dtype"] = storage_report["dtype"]' in after
u=r/"third_party/work/vllm-author-build/vllm/utils/torch_utils.py"
source=u.read_text();tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="kv_cache_dtype_str_to_dtype")
module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),fn],type_ignores=[])
ast.fix_missing_locations(module)
bf16=object();half=object()
ns={"torch":SimpleNamespace(half=half),"STR_DTYPE_TO_TORCH_DTYPE":{"bfloat16":bf16}}
exec(compile(module,str(u),"exec"),ns)
assert ns["kv_cache_dtype_str_to_dtype"]("auto",SimpleNamespace(dtype=bf16)) is bf16
rel="experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py"
patch=e/"0005-native-prefix-auto-kv-bf16.patch"
patch.write_text("".join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile="a/"+rel,tofile="b/"+rel)))
rev=subprocess.run(["git","apply","--reverse","--check",str(patch)],cwd=r,text=True,capture_output=True);assert rev.returncode==0,rev.stderr
scratch=e/"patch-forward-check";(scratch/Path(rel).parent).mkdir(parents=True,exist_ok=True);(scratch/rel).write_text(before)
fwd=subprocess.run(["git","apply","--check",str(patch)],cwd=scratch,text=True,capture_output=True);assert fwd.returncode==0,fwd.stderr
cli=subprocess.run([str(r/".venv/bin/python"),str(p),"--help"],cwd=r,text=True,capture_output=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))
(e/"cli-help.log").write_text(cli.stdout+cli.stderr);assert cli.returncode==0
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
log=r/"experiments/prefix_io_v1/runs/native-prefix-03/process.log"
lines=log.read_text().splitlines()
evidence=[next({"line":i,"text":l} for i,l in enumerate(lines,1) if needle in l) for needle in ["GPU KV cache size: 1,168 tokens","unsupported kv_cache_dtype (str), got bfloat16."]]
v={"status":"PASSED_CPU_ONLY","gpu_executed":False,"model_loaded":False,"compile":"passed","cli_help_exit":cli.returncode,
"only_engine_change":{"kv_cache_dtype":{"before":"bfloat16","after":"auto"}},"unchanged":["weights BF16/quantization None","64 MiB KV budget","128 input IDs/16 output tokens","cached 0/112","exact output comparison","worker strict 28-layer CUDA BF16 metadata","runtime environment/cache/IPC/cwd"],
"guards":"effective cache dtype equals ENGINE auto; model dtype equals actual torch.bfloat16; model quantization None",
"actual_kv_dtype_evidence":"existing strict worker Tensor dtype verification plus returned metadata dtype; no CPU substitute",
"original_auto_mapping_cpu_check":"real AST-extracted kv_cache_dtype_str_to_dtype(auto, model_config) returns same model dtype; isolated metadata objects, no torch import",
"mapping_source":str(u.relative_to(r)),"mapping_source_sha256":sha(u),
"failed_run_03_evidence":evidence,"failed_run_03_log_sha256":sha(log),"failed_run_03_preserved":True,
"before_sha256":sha(e/"native_gpu_prefix_smoke.before.py"),"after_sha256":sha(p),"patch_sha256":sha(patch),
"forward_patch_check":fwd.returncode,"reverse_patch_check":rev.returncode}
(e/"verification.json").write_text(json.dumps(v,indent=2)+"\n");print(json.dumps(v,indent=2))
