from pathlib import Path
import hashlib, json, subprocess, inspect
import multiprocessing.popen_fork, multiprocessing.popen_spawn_posix, multiprocessing.util
root=Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
ev=root/"artifacts/prefix_io_v1/new-server-03/native-integration-review"
roles={
"third_party/upstream/kvcache-experiments/common/vllm_server.py":"Original server wrapper creates new session",
"third_party/upstream/kvcache-experiments/common/kv_connector_harness.py":"Dummy forward/model outputs; not a model smoke",
"third_party/upstream/kvcache-experiments/common/prefix_cache_common.py":"HTTP/token workload helpers",
"third_party/work/py-kvcache/tests/test_e2e_kvcache.py":"Original model E2E reference",
"third_party/work/py-kvcache/py_kvcache/vllm.py":"Native planner, connector, handler construction",
"third_party/work/py-kvcache/py_kvcache/reactor.py":"Native CQE success, staging lifecycle, cache source",
"third_party/work/py-kvcache/py_kvcache/transfer.py":"Native parent and successful I/O event emission",
"third_party/work/py-kvcache/py_kvcache/liburing_file.py":"io_size, CQE polling, optional fsync",
"third_party/work/py-kvcache/py_kvcache/profiling.py":"Native profiler activity gate",
"third_party/work/vllm-author-build/vllm/v1/engine/core_client.py":"Native engine client/RPC",
"third_party/work/vllm-author-build/vllm/v1/engine/utils.py":"EngineCore multiprocessing.Process launch",
"third_party/work/vllm-author-build/vllm/utils/system_utils.py":"Native fork/spawn contexts",
"third_party/work/vllm-author-build/vllm/v1/executor/uniproc_executor.py":"Native uni worker executes within EngineCore",
"third_party/work/vllm-author-build/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py":"Native pending-job GPU-source fence",
"third_party/work/vllm-author-build/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py":"Native jobs-to-flush ordering",
"third_party/work/vllm-author-build/vllm/v1/core/block_pool.py":"Free queue/refcount is not an overwrite release witness"}
files=[]
for path,role in roles.items():
 p=root/path
 if not p.is_file():
  raise RuntimeError(path+" missing")
 raw=p.read_bytes()
 files.append({"path":path,"role":role,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
repos={}
for repo in ["third_party/upstream/kvcache-experiments","third_party/work/py-kvcache","third_party/work/vllm-author-build"]:
 repos[repo]=subprocess.check_output(["git","-C",str(root/repo),"rev-parse","HEAD"],text=True).strip()
sources="\n\n".join(inspect.getsource(f) for f in [
 multiprocessing.popen_fork.Popen._launch,
 multiprocessing.popen_spawn_posix.Popen._launch,
 multiprocessing.util.spawnv_passfds])
(ev/"python-launch-source.txt").write_text(sources)
snapshot={"audit_kind":"read_only_source_review","gpu_executed":False,"io_uring_executed":False,"model_loaded":False,
"repositories":repos,"files":files,"stdlib_launch_source_sha256":hashlib.sha256(sources.encode()).hexdigest(),
"blocking_evidence":"artifacts/prefix_io_v1/new-server-03/io-uring-current.json",
"state":"BLOCKED_NATIVE_STAGING_AND_SSD","p2_activated":False,
"next_gates":["real io_uring permission/availability","native model Prefix qualification","legal machine/model BF16 v2 cost curves","bounded native driver token/trace/lifecycle observation","frozen budgets and layout"],
"driver_implemented":False,"no_dummy_or_sync_io_substitute":True,
"report_sha256":hashlib.sha256((ev/"P1_NATIVE_INTEGRATION_REVIEW.md").read_bytes()).hexdigest()}
(ev/"source-review.json").write_text(json.dumps(snapshot,indent=2)+"\n")
print(json.dumps({"files":len(files),"status":snapshot["state"],"report_sha256":snapshot["report_sha256"]},indent=2))
