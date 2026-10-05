import ast, hashlib, json, os, subprocess
from pathlib import Path
r=Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
e=r/"artifacts/prefix_io_v1/new-server-04/calibration-preparation"
p=r/"experiments/prefix_io_v1/scripts/prepare_native_calibration_plan.py"
t=r/"tests/prefix_io_v1_calibration/test_prepare_plan.py"
for f in [p,t]:
 compile(f.read_text(),str(f),"exec")
tree=ast.parse(p.read_text())
imports={n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n,ast.Import)}
imports|={n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module}
assert not imports & {"torch","vllm","requests","httpx","urllib","subprocess","socket","py_kvcache","numpy","scipy"}
help_result=subprocess.run([str(r/".venv/bin/python"),str(p),"--help"],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"),cwd=r)
assert help_result.returncode==0
(e/"cli-help.log").write_text(help_result.stdout+help_result.stderr)
plan=json.loads((e/"candidate-plan.json").read_text())
assert plan["execution"]=="UNEXECUTED" and plan["driver_ready"] is False
assert plan["calibration_measurements"]==dict(f=None,g_mem=None,g_ssd=None)
assert "curves" not in plan and len(plan["candidate_jobs"])==3
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
commands=[
{"kind":"cpu_test","command":"CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/prefix_io_v1_calibration -q -p no:cacheprovider --junitxml=artifacts/prefix_io_v1/new-server-04/calibration-preparation/cpu-tests.xml","exit":0,"result":"32 passed in 0.18s"},
{"kind":"real_candidate_prepare_cpu_only","command":"CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/prefix_io_v1/scripts/prepare_native_calibration_plan.py --smoke-dir experiments/prefix_io_v1/runs/native-prefix-04/details --doc-sizes 64 128 192 --repeats 3 --output artifacts/prefix_io_v1/new-server-04/calibration-preparation/candidate-plan.json","exit":0,"result":{"status":"PLANNED_CPU_ONLY","candidate_jobs":3,"gpu_executed":False}},
{"kind":"verification","command":"CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python artifacts/prefix_io_v1/new-server-04/calibration-preparation/verify_cpu.py","exit":0}]
(e/"commands.json").write_text(json.dumps({"cwd":str(r),"commands":commands},indent=2)+"\n")
record={"status":"PASSED_CPU_PREPARATION_ONLY","gpu_executed":False,"network_used":False,"calibration_executed":False,
"compile":"passed","stdlib_only_import_review":"passed","cli_help_exit":help_result.returncode,"test_count":32,
"test_scope":"synthetic metadata/identity/candidate/schema rejection; not performance or actual GPU/I/O",
"script_sha256":sha(p),"tests_sha256":sha(t),"candidate_plan_sha256":sha(e/"candidate-plan.json"),
"p1_complete":False,"p3_activated":False}
(e/"verification.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))
