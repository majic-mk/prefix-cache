"""Record CPU scalar ETA guard evidence; no backend import is permitted."""
import argparse,importlib.abc,json,sys,time,unittest
from pathlib import Path
import xml.etree.ElementTree as ET
p=argparse.ArgumentParser();p.add_argument("--name",required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2]
out=root/"artifacts/prefix_io_v1/server08-p4-02-cpu/eta"/a.name
out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(root/"third_party/work/prefix-io-p4-02-cpu/src"))
sys.path.insert(0,str(Path(__file__).parent))
blocked=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0] in ("torch","vllm","py_kvcache","cupy","cuda","numpy"):
   blocked.append(fullname);raise RuntimeError("backend import forbidden: "+fullname)
sys.meta_path.insert(0,Guard())
class Result(unittest.TextTestResult):
 def startTest(self,test):
  self.start=time.monotonic_ns();super().startTest(test)
 def stopTest(self,test):
  self.rows.append((test.id(),time.monotonic_ns()-self.start));super().stopTest(test)
class Runner(unittest.TextTestRunner):
 def _makeResult(self):
  r=Result(self.stream,self.descriptions,self.verbosity);r.rows=[];return r
suite=unittest.defaultTestLoader.loadTestsFromName("test_causal_history")
with (out/"process.log").open("x") as f:r=Runner(stream=f,verbosity=2).run(suite)
xml=ET.Element("testsuite",tests=str(r.testsRun),failures=str(len(r.failures)),errors=str(len(r.errors)))
fails={t.id():v for t,v in r.failures+r.errors}
for tid,ns in r.rows:
 c=ET.SubElement(xml,"testcase",name=tid.rsplit(".",1)[-1],classname=tid.rsplit(".",1)[0],time=str(ns/1e9))
 if tid in fails:ET.SubElement(c,"failure").text=fails[tid]
ET.ElementTree(xml).write(out/"cpu.xml",encoding="utf-8",xml_declaration=True)
receipt=dict(status="PASS_CPU_ETA_SCALAR_CONTRACT" if r.wasSuccessful() else "FAIL_CPU_ETA_SCALAR_CONTRACT",
 tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),gpu_workloads_run=0,
 forbidden_import_attempts=blocked,production_eta_qualified=False,
 scope="CPU synthetic causal history arithmetic and guards; no physical GPU measurement")
(out/"result.json").write_text(json.dumps(receipt,indent=2))
(out/"command.json").write_text(json.dumps(dict(executable=sys.executable,argv=sys.argv,cwd=str(Path.cwd()),
 isolated=sys.flags.isolated,no_site=sys.flags.no_site,exit=0 if r.wasSuccessful() else 1),indent=2))
print(json.dumps(receipt));raise SystemExit(0 if r.wasSuccessful() else 1)
