"""Run only the preregistered frozen supervisor; original guard owns GPUs."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D='artifacts/prefix_io_v1/server12-c5-normal-repeatability-v1-20261004'
PINS={
'COMMON_SOURCE_LOCK.json':'4aab66888592a8fdf6544513003e9b7382f2194f3da2407b8991073a8e86ffef',
'PROTOCOL.json':'ddfcacd283cc9f5fc6d2ec5bd75333c34c8dd956fe02487092b5702c7d420dcd',
'run_fixed_repeatability.py':'702a2f914246c667a7eb363025b602abfb83c1935bb135e4715f48a6b2dfcf33'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true',required=True);args=p.parse_args()
    d=ROOT/D
    assert ROOT.is_dir() and d.is_dir() and not d.is_symlink()
    for name,digest in PINS.items():
        path=d/name
        assert not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==digest
    ledger=ROOT/'experiments/prefix_io_v1/gpu-budget-ledger.json'
    before=json.loads(ledger.read_bytes())
    assert before['active_reservation'] is None and before['gpu_wall_seconds']+960<=28800
    command=[str(ROOT/'.venv/bin/python'),'-B','-I',str(d/'run_fixed_repeatability.py'),'--project-root',str(ROOT),'--execute']
    env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
    env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    record=dict(argv=command,cwd=str(ROOT),source_pins=PINS,persistent_GPU_authority_reused=True,
        original_max_gpu_seconds=28800,ledger_sha256_before=hashlib.sha256(ledger.read_bytes()).hexdigest(),
        GPU_owned_only_by_original_guard=True,CUDA_VISIBLE_DEVICES_outer_removed=True,wall_started_unix=time.time())
    with (d/'GPU_SUPERVISOR_COMMAND.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    started=time.monotonic()
    with (d/'GPU_SUPERVISOR_STDOUT.log').open('xb') as out,(d/'GPU_SUPERVISOR_STDERR.log').open('xb') as err:
        result=subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
    after=json.loads(ledger.read_bytes())
    value=dict(exit=result.returncode,elapsed_including_CPU_verification_seconds=time.monotonic()-started,
        gpu_seconds_before=before['gpu_wall_seconds'],gpu_seconds_after=after['gpu_wall_seconds'],
        active_reservation_after=after['active_reservation'],original_budget_unchanged=True,
        ledger_sha256_after=hashlib.sha256(ledger.read_bytes()).hexdigest(),wall_finished_unix=time.time())
    with (d/'GPU_SUPERVISOR_RESULT.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value),flush=True)
    return result.returncode

if __name__=='__main__':raise SystemExit(main())

