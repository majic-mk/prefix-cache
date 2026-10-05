import ast, difflib, hashlib, importlib.util, json, os, subprocess
from pathlib import Path
from uuid import uuid4
import zmq

root=Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
ev=root/"artifacts/prefix_io_v1/new-server-03/native-prefix-ipc-path"
target=root/"experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py"
before=(ev/"native_gpu_prefix_smoke.before.py").read_text()
after=target.read_text()
compile(after,str(target),"exec")
def constants(source):
 tree=ast.parse(source)
 return {n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {"ENGINE","SAMPLING"}}
a,b=constants(before),constants(after)
expected=dict(a["ENGINE"], enable_chunked_prefill=True)
assert a["ENGINE"]["enable_chunked_prefill"] is False
assert b["ENGINE"] == expected and b["SAMPLING"]==a["SAMPLING"]
assert 'PROMPT_TOKEN_IDS = list(range(1000, 1128))' in before and 'PROMPT_TOKEN_IDS = list(range(1000, 1128))' in after
assert '((len(PROMPT_TOKEN_IDS) - 1) // ENGINE["block_size"]' in after
assert 'cold["output_token_ids"] == repeat["output_token_ids"]' in after
assert before[before.index("def worker_kv_storage_metadata"):before.index("def main")] == after[after.index("def worker_kv_storage_metadata"):after.index("def main")]
assert 'os.chdir(output_dir)' in after
spec=importlib.util.spec_from_file_location("ipc_verify",target)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.configure_runtime_environment()
directory=Path(os.environ["TMPDIR"])
assert directory==root/".p1tmp" and not directory.is_symlink()
assert os.environ["VLLM_RPC_BASE_PATH"]==str(directory)
assert "VLLM_RPC_BASE_PATH" in m.RUNTIME_PATH_ENV_KEYS
filename=str(uuid4());path=directory/filename
legacy=root/"experiments/prefix_io_v1/runtime-tmp"/filename
limit=getattr(zmq,"IPC_PATH_MAX_LEN",107)
new_bytes=len(os.fsencode(str(path)));old_bytes=len(os.fsencode(str(legacy)))
assert new_bytes<=limit<old_bytes
context=zmq.Context();socket=context.socket(zmq.ROUTER)
try:
 socket.bind("ipc://"+str(path))
 endpoint=socket.getsockopt_string(zmq.LAST_ENDPOINT)
 assert endpoint=="ipc://"+str(path)
finally:
 socket.close(linger=0);context.term()
 if path.exists():path.unlink()
rel="experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py"
patch=ev/"0004-native-prefix-short-ipc-and-prefill.patch"
patch.write_text("".join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile="a/"+rel,tofile="b/"+rel)))
reverse=subprocess.run(["git","apply","--reverse","--check",str(patch)],cwd=root,capture_output=True,text=True)
assert reverse.returncode==0,reverse.stderr
scratch=ev/"patch-forward-check";(scratch/Path(rel).parent).mkdir(parents=True,exist_ok=True);(scratch/rel).write_text(before)
forward=subprocess.run(["git","apply","--check",str(patch)],cwd=scratch,capture_output=True,text=True)
assert forward.returncode==0,forward.stderr
cli=subprocess.run([str(root/".venv/bin/python"),str(target),"--help"],cwd=root,capture_output=True,text=True)
(ev/"cli-help.log").write_text(cli.stdout+cli.stderr);assert cli.returncode==0
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={"status":"PASSED_CPU_ONLY","gpu_executed":False,"model_loaded":False,"native_prefix_01_preserved":True,
"changes":{"TMPDIR":str(directory),"VLLM_RPC_BASE_PATH":str(directory),"enable_chunked_prefill":{"before":False,"after":True}},
"remaining_engine_sampling_prompt_and_kv_rpc_unchanged":True,"ipc":{"real_zmq_bind_close":"passed","uuid_bytes":36,"old_path_bytes":old_bytes,"new_path_bytes":new_bytes,"zmq_ipc_path_max_len":limit,"symlink":False},
"compile":"passed","cli_help_exit":cli.returncode,"forward_patch_check":forward.returncode,"reverse_patch_check":reverse.returncode,
"before_sha256":sha(ev/"native_gpu_prefix_smoke.before.py"),"after_sha256":sha(target),"patch_sha256":sha(patch),
"failed_run_log_sha256":sha(root/"experiments/prefix_io_v1/runs/native-prefix-01/process.log")}
(ev/"verification.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
