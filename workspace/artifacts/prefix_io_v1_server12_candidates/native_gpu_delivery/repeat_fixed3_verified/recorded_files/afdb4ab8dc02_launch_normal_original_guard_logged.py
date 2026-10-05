"""Log a frozen controller launch; original guard owns actual GPU usage and cleanup."""
import argparse,json,os,pathlib,subprocess,time
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--project",required=True);ap.add_argument("--mode",choices=("off","shadow","on"),required=True);args=ap.parse_args()
 root=pathlib.Path(args.project).resolve(strict=True);d=root/"artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004";out=root/"artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004"
 tags={"off":"23_NORMAL_OFF_ORIGINAL_GUARD_LAUNCH","shadow":"28_NORMAL_SHADOW_ORIGINAL_GUARD_LAUNCH","on":"33_NORMAL_ON_ORIGINAL_GUARD_LAUNCH"};tag=tags[args.mode]
 argv=[str(root/".venv/bin/python"),"-B","-I",str(d/"control_p4_single_file.py"),"--root",str(root),"launch","--mode",args.mode]
 env=dict(os.environ);env.pop("CUDA_VISIBLE_DEVICES",None);env["PYTHONDONTWRITEBYTECODE"]="1"
 def put(suffix,doc):
  with (out/(tag+suffix)).open("x",encoding="utf-8") as f:json.dump(doc,f,indent=2,sort_keys=True);f.write("\n")
 put("_COMMAND.json",{"argv":argv,"cwd":str(root),"CUDA_VISIBLE_DEVICES_in_parent":env.get("CUDA_VISIBLE_DEVICES"),"authorization":"persistent direct project GPU instruction","per_job_gpu_limit_seconds":300,"reserve_seconds":320,"original_guard_owns_gpu_budget_and_session":True})
 start=time.monotonic();r=subprocess.run(argv,cwd=root,env=env,capture_output=True,timeout=240)
 for suffix,b in (("_STDOUT.log",r.stdout),("_STDERR.log",r.stderr)):
  with (out/(tag+suffix)).open("xb") as f:f.write(b)
 doc={"exit":r.returncode,"controller_elapsed_seconds":time.monotonic()-start,"stdout_bytes":len(r.stdout),"stderr_bytes":len(r.stderr),"controller_invocation_is_not_model_success":True,"actual_guard_result_required":True}
 put("_RESULT.json",doc)
 print(json.dumps(dict(doc,stdout=r.stdout.decode("utf-8","replace")[:6000],stderr=r.stderr.decode("utf-8","replace")[:4000])))
 return 0 if r.returncode==0 else 1
if __name__=="__main__":raise SystemExit(main())
