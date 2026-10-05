"""Freeze a CPU test input closure; no model imports or GPU authority."""
import argparse,hashlib,json,pathlib
def ref(root,path):
 p=root/path;assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root)
 assert not any(x.is_symlink() for x in p.parents)
 with p.open("rb") as f:h=hashlib.file_digest(f,"sha256").hexdigest()
 return {"path":path,"bytes":p.stat().st_size,"sha256":h}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--project",required=True);root=pathlib.Path(ap.parse_args().project).resolve(strict=True)
 cal="artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004";normal="artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004";out="artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004/NORMAL_CPU_SOURCE_LOCK.json"
 expected="1c0e7ccaa24aef878ea513df6744ff9037c00f02329b95793801b8a05b3ac5c0"
 parent=cal+"/SITE_SOURCE_LOCK.json";pr=ref(root,parent);assert pr["sha256"]==expected
 rows=json.loads((root/parent).read_text())["files"];assert len(rows)==4756;merged={}
 for row in rows:assert ref(root,row["path"])==row and row["path"] not in merged;merged[row["path"]]=row
 merged[parent]=pr
 for p in sorted((root/normal).rglob("*")):
  if p.is_file() and "__pycache__" not in p.parts:
   assert p.suffix in (".py",".md",".json");rel=p.relative_to(root).as_posix()
   assert p.name!="COMMON_SOURCE_LOCK.json" and not p.name.startswith(("CONFIG_","AUTHORITY_","SCOPE_","HUMAN_GPU_GRANT_"))
   merged[rel]=ref(root,rel)
 doc={"schema":"server12_normal_cpu_source_lock_v1","scope":"CPU_TEST_INPUTS_ONLY_NOT_GPU_AUTHORITY","parent_site_ref":pr,"files":[merged[k] for k in sorted(merged)],
  "GPU_operations":0,"gpu_authority_issued":False,"actual_normal_GPU_execution_verified":False,"all_parent_leaf_bytes_verified":True}
 with (root/out).open("x",encoding="utf-8") as f:json.dump(doc,f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps({"status":"PASS_NORMAL_CPU_INPUT_CLOSURE","source_count":len(merged),"lock_ref":ref(root,out),"GPU_operations":0}))
if __name__=="__main__":main()
