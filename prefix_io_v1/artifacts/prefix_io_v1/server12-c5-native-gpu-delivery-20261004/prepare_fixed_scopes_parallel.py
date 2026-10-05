"""Independent CPU source proofs in separate processes; no GPU launch."""
from concurrent.futures import ThreadPoolExecutor
import json,os,subprocess
from pathlib import Path
ROOT=Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D=ROOT/'artifacts/prefix_io_v1/server12-c5-normal-repeatability-v1-20261004'
def stage(index,action):
    argv=[str(ROOT/'.venv/bin/python'),'-B','-I',str(D/'control_p4_single_file.py'),'--root',str(ROOT),action,'--index',str(index)]
    token='CPU_'+action.upper()+'_off'+str(index+1).zfill(2)
    with (D/(token+'_COMMAND.json')).open('x') as f:json.dump(dict(argv=argv,cwd=str(ROOT),CUDA_VISIBLE_DEVICES=''),f,indent=2);f.write('\n')
    value=subprocess.run(argv,cwd=ROOT,capture_output=True,timeout=240)
    for name,data in (('STDOUT',value.stdout),('STDERR',value.stderr)):
        with (D/(token+'_'+name+'.log')).open('xb') as f:f.write(data)
    with (D/(token+'_RESULT.json')).open('x') as f:json.dump(dict(exit=value.returncode,GPU_runs=0),f);f.write('\n')
    if value.returncode:raise RuntimeError(token+': '+value.stderr.decode('utf-8','replace')[-2000:])
    return json.loads(value.stdout)
def prepare(index):
    return dict(diagnostic_index=index,scope_ref=stage(index,'scope'),source_before_ref=stage(index,'before'),GPU_runs=0)
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(prepare,range(3)))
    print(json.dumps(dict(status='PASS_ALL_THREE_INDEPENDENT_CPU_SCOPES_AND_FULL_SOURCE_PROOFS',records=records,GPU_runs=0)),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())

