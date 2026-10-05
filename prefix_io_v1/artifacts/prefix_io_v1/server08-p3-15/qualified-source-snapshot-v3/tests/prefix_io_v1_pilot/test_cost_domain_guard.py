"""Extended acquisition must not silently become a production planner domain."""
import os,subprocess,sys,uuid
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/"experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py"

@pytest.mark.parametrize("tail,expected",[
    (["--domain","4096","--mode","planned","--curves","missing.json"],"acquisition only"),
    (["--domain","8192","--mode","planned","--curves","missing.json"],"acquisition only"),
    (["--domain","16384","--mode","planned","--curves","missing.json"],"acquisition only"),
    (["--domain","4096","--sizes","8192","--mode","cold"],"unbounded acquisition"),
    (["--domain","4096","--reps","9","--mode","cold"],"unbounded acquisition"),
    (["--domain","32768","--mode","cold"],"invalid choice"),
    (["--mode","paired","--native-hot-diagnostic"],"requires external cache disabled"),
    (["--mode","cold","--diagnostic-logprobs"],"requires native hot diagnostic"),
    (["--mode","cold","--cached-reference-logprobs"],"cached reference requires paired mode only"),
    (["--mode","paired","--cached-reference-logprobs","--native-hot-diagnostic"],"cached reference requires paired mode only"),
])
def test_rejected_before_model_or_output_creation(tail,expected):
    path=ROOT/"experiments/prefix_io_v1/runs"/("cpu-rejected-"+uuid.uuid4().hex)
    command=[sys.executable,str(SCRIPT),"--output-dir",str(path/"out"),"--storage",str(path/"storage"),
        "--model-dir","missing","--model-plan","missing",*tail]
    result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),
        text=True,capture_output=True,timeout=15)
    assert result.returncode!=0 and expected in result.stderr
    assert not path.exists()
