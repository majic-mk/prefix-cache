"""Independently apply the combined patch with Git to a new local C3 copy."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    output=Path(__file__).resolve().parent
    mapping=json.loads((output/'PATCH_MAP.json').read_bytes())
    base=Path(mapping['base_directory']);target=Path(mapping['target_directory'])
    destination=output/'git_reconstructed_candidate_v4'
    if destination.exists():
        raise ValueError('append-only verification destination already exists')
    shutil.copytree(base,destination,ignore=shutil.ignore_patterns('__pycache__'))
    command=['git','-c','core.autocrlf=false','apply','--no-index','--whitespace=nowarn',
             str(output/'C3_TO_C4.patch')]
    checks=[]
    for arguments in (command[:-1]+['--check',command[-1]],command):
        result=subprocess.run(arguments,cwd=destination,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
        checks.append(dict(command=arguments,cwd=str(destination),exit=result.returncode,
            stdout=result.stdout.decode('utf-8',errors='replace'),stderr=result.stderr.decode('utf-8',errors='replace')))
        if result.returncode:
            raise ValueError('independent Git application failed: '+repr(checks[-1]))
    actual={p.relative_to(destination).as_posix():p.read_bytes()
            for p in destination.rglob('*') if p.is_file()}
    expected={p.relative_to(target).as_posix():p.read_bytes()
            for p in target.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual!=expected:
        raise ValueError('Git reconstructed directory differs from actual C4')
    result=dict(scope='INDEPENDENT_GIT_APPLY_BYTES_VERIFICATION',commands=checks,
        files_verified=len(actual),exact_tree_match=True,GPU_operations=0,performance_claim=False,
        patch_map_sha256=hashlib.sha256((output/'PATCH_MAP.json').read_bytes()).hexdigest(),
        combined_patch_sha256=hashlib.sha256((output/'C3_TO_C4.patch').read_bytes()).hexdigest(),
        files=[dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
               for name,raw in sorted(actual.items())])
    with (output/'GIT_APPLY_VERIFICATION.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(files_verified=len(actual),exact_tree_match=True,GPU_operations=0)))


if __name__=='__main__':main()
