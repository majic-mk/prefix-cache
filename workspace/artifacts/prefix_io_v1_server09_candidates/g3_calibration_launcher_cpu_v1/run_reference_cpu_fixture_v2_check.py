"""Run the explicit CPU-only fixture suite; no production/GPU claim."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "reference-cpu-evidence-07-v2"
OUTPUT.mkdir(exist_ok=False)
NAMES = ("test_g3_reference_result.py", "test_g3_reference_result_v2.py",
         "g3_reference_result.py", "run_g3_reference_pilot.py")

def refs():
    result = {}
    for name in NAMES:
        raw = (HERE / name).read_bytes()
        result[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return result

before = refs()
source = HERE.parents[1] / "prefix_io_v1_server08_primary_qualification/contents"
argv = [sys.executable, "-I", "-S", "-B", str(HERE / NAMES[1]),
        "--source-root", str(source)]
command = dict(argv=argv, cwd=str(HERE), origin="actual_local_CPU_execution",
               GPU_calls=0, RPC_calls=0, production_qualified=False)
(OUTPUT / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
start = time.monotonic()
process = subprocess.run(argv, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         timeout=90, check=False)
elapsed = time.monotonic() - start
(OUTPUT / "process.log").write_bytes(process.stdout)
after = refs()
result = dict(status="PASS_CPU_FIXTURE_ONLY" if process.returncode == 0 and before == after else "FAIL",
              captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              exit_code=process.returncode, elapsed_seconds=elapsed,
              command=command, source_before=before, source_after=after,
              source_unchanged=before == after, GPU_calls=0, RPC_calls=0,
              backend_import_attempts_reported_by_fixture=0 if process.returncode == 0 else "SEE_LOG",
              actual_GPU_run=False, production_qualified=False,
              server_Python312_replay="PENDING_PARENT_SERVER_REPLAY")
(OUTPUT / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
print(process.stdout.decode("utf-8", errors="replace"))
raise SystemExit(process.returncode if before == after else 2)
