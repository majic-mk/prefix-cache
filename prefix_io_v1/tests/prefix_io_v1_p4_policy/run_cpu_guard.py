"""Run scalar policy tests with backend imports forbidden and save exact evidence."""
import argparse
import importlib.abc
import json
from pathlib import Path
import sys
import time
import unittest
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument("--output",required=True)
parser.add_argument("--name",required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
out = Path(args.output).resolve()
if root not in out.parents:
    raise ValueError("evidence must remain inside the project")
out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root/"third_party/work/prefix-io-p4-01-cpu/src"))
blocked=[]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".")[0] in ("torch","vllm","py_kvcache","cupy","cuda","numpy"):
            blocked.append(fullname)
            raise RuntimeError("backend imports forbidden during CPU policy contract: "+fullname)
sys.meta_path.insert(0,Guard())

class Result(unittest.TextTestResult):
    def startTest(self,test):
        self.started=time.monotonic_ns()
        super().startTest(test)
    def stopTest(self,test):
        self.rows.append((test.id(),time.monotonic_ns()-self.started))
        super().stopTest(test)

paths = {suffix:out/(args.name+suffix) for suffix in (".log",".json",".xml","-command.json")}
if any(p.exists() for p in paths.values()):
    raise FileExistsError("fresh evidence names are required")
suite=unittest.defaultTestLoader.discover(str(root/"tests/prefix_io_v1_p4_policy"))
with paths[".log"].open("x",encoding="utf-8") as f:
    class Runner(unittest.TextTestRunner):
        def _makeResult(self):
            result=Result(self.stream,self.descriptions,self.verbosity)
            result.rows=[]
            return result
    result=Runner(stream=f,verbosity=2).run(suite)
xml=ET.Element("testsuite",name="P4 CPU value-policy contract",tests=str(result.testsRun),
              failures=str(len(result.failures)),errors=str(len(result.errors)),
              skipped=str(len(result.skipped)))
failures={test.id():text for test,text in result.failures}
errors={test.id():text for test,text in result.errors}
skips={test.id():text for test,text in result.skipped}
for name,ns in result.rows:
    case=ET.SubElement(xml,"testcase",name=name.rsplit(".",1)[-1],
                       classname=name.rsplit(".",1)[0],time=str(ns/1e9))
    for tag,items in (("failure",failures),("error",errors),("skipped",skips)):
        if name in items:
            ET.SubElement(case,tag).text=items[name]
ET.ElementTree(xml).write(paths[".xml"],encoding="utf-8",xml_declaration=True)
receipt={"status":"PASS_CPU_POLICY_CONTRACT" if result.wasSuccessful() else "FAIL_CPU_POLICY_CONTRACT",
         "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
         "skipped":len(result.skipped),"gpu_initialized":False,"gpu_workloads":0,
         "forbidden_import_attempts":blocked,
         "scope":"CPU scalar fixtures, not production ownership or interference qualification"}
paths[".json"].write_text(json.dumps(receipt,indent=2),encoding="utf-8")
paths["-command.json"].write_text(json.dumps({
    "executable":sys.executable,"argv":sys.argv,"cwd":str(Path.cwd()),
    "python_isolated":sys.flags.isolated,"python_no_site":sys.flags.no_site,
    "guards":"Torch/vLLM/py_kvcache/CUDA/CuPy/NumPy imports forbidden",
    "environment_required":{"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"},
    "receipt":str(paths[".json"]),"xml":str(paths[".xml"]),"exit":0 if result.wasSuccessful() else 1
},indent=2),encoding="utf-8")
print(json.dumps(receipt))
sys.exit(0 if result.wasSuccessful() else 1)
