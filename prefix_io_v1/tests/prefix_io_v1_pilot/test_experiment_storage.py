from pathlib import Path
import copy,json,os,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"experiments/prefix_io_v1/scripts"))
from experiment_storage import usage,budget_values,authorized_path,preflight
from prefix_io_control.config import read_yaml,validate_permissions,ConfigError
ROOT=Path(__file__).resolve().parents[2]
def grant():
    p=read_yaml(ROOT/"docs/prefix_io_v1/templates/permissions.yaml")
    p["approved_auxiliary_storage"]=dict(root="/root/dedicated",max_bytes=20*1024**3,
        minimum_free_bytes=8*1024**3,authorization_record="a.json")
    return p
def test_legacy_permission_unchanged():
    assert "approved_auxiliary_storage" not in validate_permissions(read_yaml(ROOT/"docs/prefix_io_v1/templates/permissions.yaml"))
@pytest.mark.parametrize("key,value",[
 ("root","/"),("root","relative"),("max_bytes",21*1024**3),("max_bytes",True),
 ("minimum_free_bytes",7*1024**3),("authorization_record","../a.json"),
 ("authorization_record","/a.json")])
def test_invalid_aux_grant(key,value):
    p=grant();p["approved_auxiliary_storage"][key]=value
    with pytest.raises(ConfigError):validate_permissions(p)
def test_hardlinks_count_once_and_alias_is_not_followed(tmp_path):
    private=tmp_path/"private";private.mkdir()
    data=private/"data";data.write_bytes(b"x"*4096)
    os.link(data,private/"hardlink")
    outside=tmp_path/"model";outside.mkdir();(outside/"weights").write_bytes(b"x"*100000)
    (private/"model_alias").symlink_to(outside,target_is_directory=True)
    assert usage(private)==dict(used_bytes=4096,unique_files=1)
@pytest.mark.parametrize("used,free,new,cap,floor",[
 (9,100,2,10,8),(0,9,2,20,8),(False,100,0,20,8),(0,100,-1,20,8)])
def test_budget_rejects_overage_and_bad_types(used,free,new,cap,floor):
    with pytest.raises(ValueError):budget_values(used,free,new,cap,floor)
def test_budget_boundary_is_exact():
    assert budget_values(8,10,2,10,8)["reserved_new_bytes"]==2
def test_unauthorized_path_rejected():
    with pytest.raises(ValueError):authorized_path("/root/not-this-project/scratch")
