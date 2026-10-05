from pathlib import Path
import ast,hashlib,json,subprocess,shutil,sys,datetime
root=Path(".").resolve();out=root/"artifacts/prefix_io_v1/server08-p3-16";primary=root/"experiments/prefix_io_v1/runs"
tool=root/"experiments/prefix_io_v1/scripts/verify_private_cache_archive.py"
assert hashlib.sha256(tool.read_bytes()).hexdigest()=="ae9711a0373a43add89fc38d68e683ae6abaecf3c12881ba248f2ce5272ac699"
audit=out/"private-cache-merge-manifest.json";assert hashlib.sha256(audit.read_bytes()).hexdigest()=="e692ad4495548d9729423d956ceea405bbcac4202d5b9e50bee2db4ac424e062"
assert json.loads((primary.parent/"gpu-budget-ledger.json").read_text())["active_reservation"] is None
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
post=out/"private-cache-merge-postverify.json";summary=out/"private-cache-merge-space-after-apply.json";assert not post.exists() and not summary.exists()
tree=ast.parse(tool.read_text(),filename=str(tool))
mapping={"/root/prefix-io-v1-validation":str(primary.parent),"/root/prefix-io-v1-validation/runs":str(primary)}
counts={k:0 for k in mapping}
for node in ast.walk(tree):
 if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value in mapping:
  old=node.value;node.value=mapping[old];counts[old]+=1
assert counts=={k:1 for k in mapping}
sys.argv=[str(tool),"--audit",str(audit),"--output",str(post)]
exec(compile(ast.fix_missing_locations(tree),str(tool),"exec"),{"__name__":"__main__","__file__":str(tool)})
report=json.loads(post.read_text());assert report["passed"] is True and report["no_temporary_debris"] is True and report["manifests"][0]["targets"]==10269
rows=[json.loads(x) for x in (out/"private-cache-merge-apply-journal.jsonl").read_text().splitlines()]
expected={t["path"] for g in json.loads(audit.read_text())["proposals"] for t in g["targets"]}
intent=[x for x in rows if x["kind"]=="intent"];verified=[x for x in rows if x["kind"]=="verified"]
assert rows[0]["kind"]=="begin" and rows[-1]["kind"]=="complete" and rows[-1]["completed"]==10269
assert not any(x["kind"]=="stopped" for x in rows)
assert len(intent)==len(verified)==len({x["target"] for x in intent})==len({x["target"] for x in verified})==10269
assert {x["target"] for x in intent}=={x["target"] for x in verified}==expected
assert rows[0]["authorization_sha256"]=="8b7d58b0379203e0d7e7e663d1c1808eb81a9d03a3c7885b5be3bc931599ecab"
before=json.loads((out/"private-cache-merge-space-before-apply.json").read_text());free=shutil.disk_usage(primary).free
assert free>before["primary_free_bytes"] and free-3*1024**3>=8*1024**3
d=dict(schema_version=1,status="PASS_EXACT_APPROVED_PRIVATE_CACHE_MERGE_POSTVERIFY_AND_JOURNAL",target_files=10269,canonical_groups=2060,all_paths_and_bytes_preserved=True,intents=len(intent),verified=len(verified),complete=10269,journal_bytes=(out/"private-cache-merge-apply-journal.jsonl").stat().st_size,primary_free_before=before["primary_free_bytes"],primary_free_after=free,actual_primary_net_free_increase_bytes=free-before["primary_free_bytes"],aux_free_before=before["aux_free_bytes"],aux_free_after=shutil.disk_usage("/root/prefix-io-v1-validation").free,df_command=["df","-B1",str(primary),"/root/prefix-io-v1-validation"],df_output=subprocess.check_output(["df","-B1",str(primary),"/root/prefix-io-v1-validation"],text=True),original_verifier_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),runtime_AST_only_two_literal_mappings=mapping,original_source_file_changed=False,next_fixed_3GiB_stage_space_gate=True,no_temporary_debris=True,GPU_workloads_run=0,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with summary.open("x",encoding="utf-8") as f:json.dump(d,f,indent=2)
print(json.dumps(d))
