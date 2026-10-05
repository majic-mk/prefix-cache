"""Preserve transient Ninja build evidence without controlling the build."""
from pathlib import Path
import argparse, json, time
p=argparse.ArgumentParser()
p.add_argument("--build-dir",type=Path,required=True)
p.add_argument("--output-dir",type=Path,required=True)
p.add_argument("--seconds",type=int,default=5400)
a=p.parse_args()
a.output_dir.mkdir(exist_ok=True,parents=True)
for name in ("CMakeCache.txt","build.ninja","CMakeFiles/CMakeConfigureLog.yaml"):
 source=a.build_dir/name
 if source.exists():
  (a.output_dir/name.replace("/","-")).write_bytes(source.read_bytes())
deadline=time.monotonic()+a.seconds
last=None
while time.monotonic()<deadline and a.build_dir.exists():
 source=a.build_dir/".ninja_log"
 try: data=source.read_bytes()
 except FileNotFoundError: data=None
 if data is not None and data!=last:
  (a.output_dir/"ninja-execution.log").write_bytes(data)
  last=data
 time.sleep(.5)
(a.output_dir/"preservation.json").write_text(json.dumps({
 "source":str(a.build_dir),"last_observed_actions":0 if last is None else len(last.splitlines())-1,
 "build_directory_still_exists":a.build_dir.exists(),
 "note":"Last observed Ninja log; package install result and installed library hashes are separate evidence.",
 "gpu_execution":False
},indent=2)+"\n")
