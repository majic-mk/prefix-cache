import ast, copy, json
from pathlib import Path
import pytest
from prefix_io_control.simple_stage_options import parse_simple_options,build_simple_kwargs,model_file,PARENT_KEY,POLICY_KEY
from tests.prefix_io_v1_simple_stage_model.default_ast import strip_simple
ROOT=Path(__file__).resolve().parents[2]

def parent(cap=64):
    return {PARENT_KEY:dict(schema_version=1,run_id="r",max_accepted_parents=cap)}

def raw(mode="fixed"):
    unit=4096
    value=dict(schema_version=1,run_id="r",mode=mode,epoch_ns=10_000_000,byte_quantum=unit,
        cumulative={s:dict(ops=16,bytes=unit*16) for s in ("ssd_read","ssd_write","h2d","d2h")},
        inflight={s:dict(ops=32,bytes=unit*32) for s in ("ssd_read","ssd_write","h2d","d2h")},
        shared_ssd_cumulative=dict(ops=32,bytes=unit*32),shared_ssd_inflight=dict(ops=64,bytes=unit*64),
        shared_copy_cumulative_bytes=unit*32,shared_copy_inflight_bytes=unit*64,
        reserve_staging_bytes=unit*4 if mode=="pressure" else 0,max_accepted_parents=64,
        sample_max_age_ns=20_000_000,max_wait_ns=200_000_000)
    return dict(parent(),**{POLICY_KEY:value})

def test_absent_and_explicit_off_create_no_controller():
    assert build_simple_kwargs(parse_simple_options({}))=={}
    assert build_simple_kwargs(parse_simple_options({POLICY_KEY:dict(mode="off")}))=={}
    result=build_simple_kwargs(parse_simple_options(dict(parent(),**{POLICY_KEY:dict(mode="off")})))
    assert result["max_accepted_parents"]==64 and "dispatch_controller" not in result
    assert "stage_accounting" in result

@pytest.mark.parametrize("mode",["shadow","fixed","pressure"])
def test_all_arms_share_common_bound_and_distinct_policy(mode):
    result=build_simple_kwargs(parse_simple_options(raw(mode)))
    assert result["max_accepted_parents"]==64 and result["dispatch_controller"].run_id=="r"
    assert result["dispatch_controller"].config.mode==mode

@pytest.mark.parametrize("field,value",[
    ("schema_version",True),("schema_version",2),("run_id",""),
    ("max_accepted_parents",True),("max_accepted_parents",0),("max_accepted_parents",3),("max_accepted_parents",128)])
def test_bad_common_fields(field,value):
    data=parent();data[PARENT_KEY][field]=value
    with pytest.raises(ValueError):parse_simple_options(data)

@pytest.mark.parametrize("kind",["no_parent","unknown_parent","unknown_policy","identity","cap","missing_stage","missing_bytes","bool_bytes","unquantized","p4","old_policy","full_off"])
def test_bad_protocol(kind):
    data=raw()
    if kind=="no_parent":del data[PARENT_KEY]
    elif kind=="unknown_parent":data[PARENT_KEY]["unexpected"]=1
    elif kind=="unknown_policy":data[POLICY_KEY]["unexpected"]=1
    elif kind=="identity":data[POLICY_KEY]["run_id"]="other"
    elif kind=="cap":data[POLICY_KEY]["max_accepted_parents"]=32
    elif kind=="missing_stage":del data[POLICY_KEY]["cumulative"]["h2d"]
    elif kind=="missing_bytes":del data[POLICY_KEY]["cumulative"]["h2d"]["bytes"]
    elif kind=="bool_bytes":data[POLICY_KEY]["cumulative"]["h2d"]["bytes"]=True
    elif kind=="unquantized":data[POLICY_KEY]["shared_copy_cumulative_bytes"]+=1
    elif kind=="full_off":data[POLICY_KEY]["mode"]="off"
    elif kind=="p4":data[POLICY_KEY]["mode"]="joint"
    elif kind=="old_policy":data["prefix_io_start_budget"]=dict(schema_version=1,mode="shadow",run_id="r")
    with pytest.raises(ValueError):parse_simple_options(data)

def test_model_file_injects_identity_and_rejects_mutable_identity(tmp_path):
    data=raw("pressure");parent_cfg=data[PARENT_KEY];parent_cfg.pop("run_id")
    policy=data[POLICY_KEY];policy.pop("run_id")
    config=dict(schema_version=1,parent_admission=parent_cfg,stage_policy=policy)
    path=tmp_path/"config.json";path.write_text(json.dumps(config))
    result=model_file(path,"cohort")
    assert result[PARENT_KEY]["run_id"]==result[POLICY_KEY]["run_id"]=="cohort"
    assert json.loads(path.read_text())==config
    config["stage_policy"]["run_id"]="foreign";path.write_text(json.dumps(config))
    with pytest.raises(ValueError):model_file(path,"cohort")

def test_closed_p314_default_driver_ast_unchanged():
    prior=ast.parse((ROOT/"artifacts/prefix_io_v1/server08-p3-15/driver-before.py").read_text())
    new=ast.parse((ROOT/"experiments/prefix_io_v1/scripts/run_concurrent_pilot.py").read_text())
    assert ast.dump(strip_simple(new),include_attributes=False)==ast.dump(prior,include_attributes=False)

def test_new_control_validation_precedes_source_copy_and_gpu_runtime():
    tree=ast.parse((ROOT/"experiments/prefix_io_v1/scripts/run_concurrent_pilot.py").read_text())
    main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="main")
    texts=[ast.unparse(x) for x in main.body]
    policy=next(i for i,s in enumerate(texts) if s.startswith("if a.simple_stage_config"))
    clone=next(i for i,s in enumerate(texts) if "inventory = clone_private_storage(" in s)
    runtime=next(i for i,s in enumerate(texts) if "base.configure_runtime_environment()" in s)
    assert policy<clone<runtime and "model_file(" in texts[policy]
