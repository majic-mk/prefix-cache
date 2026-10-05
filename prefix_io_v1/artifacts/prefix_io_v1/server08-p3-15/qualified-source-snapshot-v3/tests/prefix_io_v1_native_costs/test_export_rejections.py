"""CPU validation against retained real GPU acquisition records."""
import json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments/prefix_io_v1/scripts"))
RUNS=["server07-cal-cold-01","server07-cal-populate-02","server07-cal-paired-01"]

@pytest.fixture
def source(tmp_path):
    paths=[]
    for i,label in enumerate(RUNS):
        src=ROOT/"experiments/prefix_io_v1/runs"/label/"details"
        if not (src/"result.json").exists():pytest.skip("retained GPU acquisition required")
        dst=tmp_path/str(i);dst.mkdir()
        for name in ("result.json","frozen-config.json"):
            (dst/name).write_text((src/name).read_text())
        paths.append(dst/"result.json")
    return paths

def mutate(path,fn):
    d=json.loads(path.read_text());fn(d);path.write_text(json.dumps(d))

@pytest.mark.parametrize("case,expected",[
    ("failed","failed acquisition"),
    ("wrong_gpu","identity/measurement mismatch"),
    ("unpaired","paired SSD/staging"),
    ("wrong_token","token mismatch"),
    ("wrong_bytes","wrong path"),
    ("unpinned","layout/pinned budget"),
    ("negative_service","negative paired storage service"),
    ("corrupted","corrupted output")])
def test_rejects_unqualified_costs(source,tmp_path,case,expected):
    from export_native_aio_costs import export
    if case=="failed":mutate(source[2],lambda d:d.update(status="FAILED"))
    elif case=="wrong_gpu":mutate(source[2].parent/"frozen-config.json",lambda d:d.update(gpu_uuid="GPU-other"))
    elif case=="unpaired":mutate(source[2],lambda d:d.update(mode="restore"))
    elif case=="wrong_token":mutate(source[2],lambda d:d["rows"][0]["output_token_ids"].append(123))
    elif case=="wrong_bytes":mutate(source[2],lambda d:d["rows"][0]["trace"].update(preload_actual_read_bytes=0,foreground_logical_read_bytes=0))
    elif case=="unpinned":mutate(source[2],lambda d:d["rows"][0]["drain"]["handlers"][0].update(pinned=False))
    elif case=="negative_service":
        def change(d):
            for r in d["rows"]:
                if r["kind"]=="g_mem":r["metrics"]["first_token_latency"]=10.0
        mutate(source[2],change)
    elif case=="corrupted":mutate(source[2],lambda d:d["rows"][0]["metrics"].update(is_corrupted=True))
    with pytest.raises(ValueError,match=expected):
        export(*source,tmp_path/"rejected")
    assert not (tmp_path/"rejected").exists()
