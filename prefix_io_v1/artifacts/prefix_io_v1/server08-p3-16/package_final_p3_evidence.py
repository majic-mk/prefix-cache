from pathlib import Path
import json,hashlib,zipfile,datetime,time,subprocess,sys
root=Path(".").resolve();out=Path("artifacts/prefix_io_v1/server08-p3-16")
sys.path[:0]=[str(root/"src"),str(root/"experiments/prefix_io_v1/scripts")]
from experiment_storage import preflight
def rd(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def put(p,v):
 with Path(p).open("x",encoding="utf8") as f:json.dump(v,f,indent=2,ensure_ascii=False)
 return ref(p)
assert rd("experiments/prefix_io_v1/gpu-budget-ledger.json")["active_reservation"] is None
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"],text=True).strip()
assert rd(out/"p3-final-root-review.json")["p3_phase_closed"] is True
storage=preflight(root/"experiments/prefix_io_v1/runs/server08-p3-16-final-archive-budget/details",536870912)
assert out.stat().st_dev==(root/"experiments/prefix_io_v1/runs").stat().st_dev
assert not any(k in sys.modules for k in ["torch","cupy","py_kvcache","vllm"])
put(out/"final-archive-storage-preflight.json",dict(schema_version=1,status="PASS_ORIGINAL_512MiB_RESERVE_AND_8GiB_FLOOR_SAME_FILESYSTEM",storage=storage,actual_archive_directory=str(out.resolve()),same_filesystem_as_authorized_primary=True,new_GPU_workloads_run=0,GPU_libraries_loaded=False))
matrix="# P3 最终能力矩阵\n\n|能力|实际资格|边界|\n|---|---|---|\n"
for row in [("完整模型/精确Prefix","20次完整cohort各10请求/1280输出一致","已见P3开发域，没有SLO/未见收益"),("原成本准入","共同1GiB原permit、两个容量域各六点门槛和loaded-U通过","25%门槛保持；容量不混入候选选择"),("共享staging/复制合并/异步流水线","真实CUDA/LinuxAIO与完整排空，80成员共享资格通过","软件Event可见性门控不等于真实DMA忙延迟"),("预加载","960/l1与1024/l2各独立loaded-U通过","同时改变容量/horizon，不作单参数因果归因"),("有限fixed/pressure控制","五臂各两次完成，U最好，B=U","无新策略提升证据、无全局最优"),("真实释放依赖","原Event/AIO/parent完成后回收；unknown保守","没有立即GPU释放credit/提前复用协议"),("稀疏干扰额度","七conditional cell原25%稳定/预测门槛通过","production lookup未qualified、unsupported None"),("紧凑观测","quiet optional off/on/on/off实际对照与CPU窗口记录","normal-I/O总开销≤2%没有证据；共同计数常开"),("研究关闭回退","off factory回原stage签发；共同安全修复独立保留","补丁往返/CPU与真实U资格不等于整环境历史重演"),("P3验收","15项strict PASS、root审查关闭、0剩余P3GPU","P4-P7关闭，研究收益未证明"),("版本/来源锁","作者HEAD、16ELF byte identities、最终2724输入身份和3048源SHA","模型/库payload不打包；非全安装环境snapshot")]:
 matrix+="|"+"|".join(row)+"|\n"
with (out/"CAPABILITY_MATRIX-final.md").open("x",encoding="utf8") as f:f.write(matrix)
old=rd(out/"server08-p3-16-blocked-checkpoint-evidence-v3-manifest.json")
source=rd(out/"frozen-input-snapshot-at-storage-block-v2.json")
base=Path(old["archive"]["path"])
assert ref(base)["sha256"]==old["archive"]["sha256"]=="958ebfbcfa36c0bc51007b9ddde2fdbd226eeac9d5a7c902ee8a5cd933998d0f"
proof=rd(out/"evidence-archive-server-verification.json")
assert proof["status"]=="PASS_ARCHIVE_AND_EVERY_FILE_AND_FROZEN_SOURCE_SNAPSHOT"
base_map={str(Path(r["source_path"]).resolve()):r["sha256"] for r in [*old["files"],*source["files"]]}
locked=rd(out/"execution-lock-12-final-p3.json")
required={str((Path(p) if Path(p).is_absolute() else root/p).resolve()):h for p,h in locked.items()}
assert all(sha(p)==h for p,h in required.items())
candidates={Path(p) for p in required if str(Path(p)).startswith(str(root)+"/")}
extensions={".json",".jsonl",".log",".xml",".csv",".md",".yaml",".yml",".txt",".patch",".py"}
excluded_tmp=[]
def add_tree(directory):
 for p in Path(directory).rglob("*"):
  rel=p.relative_to(root)
  if any(s.endswith("-tmp") for s in rel.parts):
   excluded_tmp.append(str(rel));continue
  if p.is_file() and p.suffix.lower() in extensions:candidates.add(p)
add_tree(root/out)
budget=rd("experiments/prefix_io_v1/gpu-budget-ledger.json")
for e in budget["events"]:
 if e["label"].startswith("server08-p3-16-"):add_tree(e["evidence"])
for p in ["experiments/prefix_io_v1/execution_state.json","experiments/prefix_io_v1/configs/permissions.yaml","docs/prefix_io_v1/SERVER08_P3_FINAL_ROOT_REVIEW.md"]:candidates.add(root/p)
delta=[]
for p in sorted(candidates,key=str):
 p=Path(p);rel=p.relative_to(root);cursor=root
 for part in rel.parts:
  cursor=cursor/part;assert not cursor.is_symlink(),str(p)
 assert p.is_file()
 h=sha(p);abskey=str(p.resolve())
 if base_map.get(abskey)==h:continue
 delta.append(dict(entry="project/"+str(rel),source_path=str(p),bytes=p.stat().st_size,sha256=h))
delta_map={r["source_path"]:r["sha256"] for r in delta}
assert all(base_map.get(p)==h or delta_map.get(p)==h for p,h in required.items())
archive=out/"server08-p3-16-complete-evidence-v1.zip";assert not archive.exists()
meta=dict(schema_version=1,status="P3_COMPLETE_EVIDENCE_BASE_PLUS_DELTA",P3_complete=True,P4_enabled=False,root_review=ref(out/"p3-final-root-review.json"),required_source_lock=ref(out/"source-lock-final-p3.json"),required_input_keys=len(locked),required_unique_paths=len(required),all_required_inputs_covered=True,historical_base=ref(base),historical_base_entry="historical-checkpoint/"+base.name,historical_base_proof=ref(out/"evidence-archive-server-verification.json"),historical_base_metadata_files=len(old["files"]),historical_frozen_source_files=len(source["files"]),delta_files=delta,temporary_fixture_metadata_excluded=sorted(set(excluded_tmp)),GPU_stage_successes=37,GPU_stage_failed_attempts=1,CPU_unique_passed=1613,CPU_skipped=16,model_and_private_cache_payload_included=False,compiled_library_payloads_included=False,new_GPU_workloads_run=0)
start=time.monotonic()
with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
 z.write(base,meta["historical_base_entry"],compress_type=zipfile.ZIP_STORED)
 for r in delta:z.write(r["source_path"],r["entry"])
 z.writestr("final-evidence-file-manifest.json",json.dumps(meta,indent=2,ensure_ascii=False))
meta["archive"]=ref(archive);meta["seconds"]=time.monotonic()-start
m=put(out/"server08-p3-16-complete-evidence-v1-manifest.json",meta)
print(json.dumps(dict(status=meta["status"],archive=meta["archive"],manifest=m,delta_files=len(delta),required_inputs=len(locked),base_metadata_files=len(old["files"]),base_source_files=len(source["files"]),seconds=meta["seconds"],model_payloads_included=False)))
