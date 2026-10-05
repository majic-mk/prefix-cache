"""Exercise the prepared normal CLI while absent authority blocks all model imports."""
import argparse,hashlib,io,json,pathlib,runpy,sys,traceback,contextlib
BLOCKED=[]; FORBIDDEN=("torch","vllm","py_kvcache")
class NoGPU:
 def find_spec(self,fullname,path=None,target=None):
  if any(fullname==x or fullname.startswith(x+".") for x in FORBIDDEN):
   BLOCKED.append(fullname);raise RuntimeError("CPU_ONLY_NEGATIVE_MODEL_IMPORT_REFUSED")
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--project",required=True);r=pathlib.Path(ap.parse_args().project).resolve(strict=True)
 d=r/"artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004";a=r/"artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004"
 script=d/"run_p4_single_file_experiment.py";config=d/"CONFIG_off.json";authority=d/"AUTHORITY_off.json"
 run=r/"experiments/prefix_io_v1/runs/server12-c5-native-normal-off01";ledger=r/"experiments/prefix_io_v1/gpu-budget-ledger.json";before=ledger.read_bytes()
 assert config.is_file() and not authority.exists() and not run.exists() and not any(x in sys.modules for x in FORBIDDEN)
 sys.meta_path.insert(0,NoGPU());old=sys.argv[:];sys.argv=[str(script),"--execute","--config",str(config)]
 output=io.StringIO();caught=None;frames=[]
 try:
  with contextlib.redirect_stdout(output):runpy.run_path(str(script),run_name="__main__")
 except ValueError as e:
  caught=str(e);frames=[{"file":t.filename,"function":t.name,"line":t.lineno} for t in traceback.extract_tb(e.__traceback__)]
 finally:sys.argv=old
 assert caught and caught.startswith("NORMAL_ENTRY_REJECTED:") and any(x["function"]=="load_authority" for x in frames), (caught,frames)
 assert not BLOCKED and not authority.exists() and not run.exists() and before==ledger.read_bytes()
 imported=[n for n in sys.modules if any(n==x or n.startswith(x+".") for x in FORBIDDEN)];assert not imported
 doc={"status":"PASS_REAL_PREPARED_NORMAL_OFF_CLI_AUTHORITY_REJECTION","actual_cli":[str(script),"--execute","--config",str(config)],
  "exception_type":"ValueError","actual_reason":caught,"actual_call_frames":frames,"captured_stdout":output.getvalue(),
  "authority_exists":False,"normal_run_directory_created":False,"model_GPU_import_attempts":BLOCKED,"model_GPU_imports":imported,
  "ledger_unchanged":True,"ledger_sha256":hashlib.sha256(before).hexdigest(),"GPU_operations":0,"normal_runtime_qualified":False}
 with (a/"REAL_NORMAL_UNAUTHORIZED_ENTRY_REJECTION.json").open("x",encoding="utf-8") as f:json.dump(doc,f,indent=2);f.write("\n")
 print(json.dumps({k:v for k,v in doc.items() if k not in ("actual_call_frames","captured_stdout")}))
if __name__=="__main__":main()
