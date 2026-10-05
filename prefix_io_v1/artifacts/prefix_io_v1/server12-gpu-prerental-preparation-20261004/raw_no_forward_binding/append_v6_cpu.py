"""Append collector/delegate source-leaf binding only; no GPU import/run."""
import hashlib
from pathlib import Path

PREP=Path(__file__).resolve().parent.parent
source=PREP/"runner/strong_native_cost_runner_v5.py"
target=PREP/"runner/strong_native_cost_runner_v6.py"
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()=="e05921ea213a9b6c6d8da15e9fea2c22ccef62020a934a1ddb3c36859c47978f"
text=raw.decode("utf-8")
before='required = dict(collector_source_ref=(Path(relative).parent / "bounded_native_full_step_collector.py").as_posix(),'
after='required = dict(collector_source_ref=(Path(relative).parent / "bounded_native_full_step_collector_v2.py").as_posix(),\n        original_collector_source_ref=(Path(relative).parent / "bounded_native_full_step_collector.py").as_posix(),'
assert text.count(before)==1
text=text.replace(before,after)
before='        R.check_ref(root, refs[path])\n    plan = dict(schema="strong_gpu_exact_cell_prelaunch_plan_v1"'
after='        R.check_ref(root, refs[path])\n    R.require(refs[required["original_collector_source_ref"]]["sha256"] ==\n              "9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d",\n              "immutable original full-step collector delegate")\n    plan = dict(schema="strong_gpu_exact_cell_prelaunch_plan_v1"'
assert text.count(before)==1
text=text.replace(before,after)
with target.open("xb") as stream:
    stream.write(text.encode("utf-8"))
print(target.name,len(target.read_bytes()),hashlib.sha256(target.read_bytes()).hexdigest())
