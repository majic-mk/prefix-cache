"""Use the authenticated pinned SSH worker; credentials stay in worker memory."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import time
import uuid
import zlib

SPOOL = Path('C:/Users/mamengkui/AppData/Local/Temp/prefix_v1_rpc_server11b')
ROOT = '/root/autodl-tmp/prefix-io-v1-handoff/project'
PYTHON = ROOT + '/.venv/bin/python'
BASE = ROOT + '/artifacts/prefix_io_v1/'
DIRS = {k: BASE + 'server11-c5-native-cost-preparation' + ('' if k == 'candidate' else '-' + k) + '-cpu-20261004'
        for k in ('candidate', 'review', 'protocol', 'delivery')}

def rpc(request):
    key = uuid.uuid4().hex
    source, result = SPOOL / (key + '.request'), SPOOL / (key + '.response')
    with source.open('x', encoding='utf-8') as stream:
        json.dump(request, stream)
    deadline = time.monotonic() + request.get('timeout', 60) + 120
    while not result.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('RPC worker deadline')
        time.sleep(.1)
    value = json.loads(result.read_text())
    source.unlink(); result.unlink()
    if 'error' in value:
        raise RuntimeError(value['error'])
    return value

def remote_python(code, timeout=40):
    return rpc(dict(cmd=shlex.quote(PYTHON) + ' -B -I -S -c ' + shlex.quote(code), timeout=timeout))

def upload(local, scope, names):
    """Bounded append-only upload; identical existing files are only verified."""
    local = Path(local).resolve(strict=True)
    destination = DIRS[scope]
    rows = []
    for name in names:
        if name.startswith('/') or any(p in ('', '.', '..') for p in name.split('/')):
            raise ValueError('relative upload filename')
        path = local / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(local):
            raise ValueError('regular contained upload source')
        data = path.read_bytes()
        if len(data) > 10 * 1024**2:
            raise ValueError('source size cap')
        rows.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                         data=base64.b64encode(data).decode()))
    results = []
    batches=[]
    batch=[]
    def encode(values):return base64.b64encode(zlib.compress(json.dumps(values).encode())).decode()
    for row in rows:
        if batch and len(encode(batch+[row]))>85000:
            batches.append(batch);batch=[]
        batch.append(row)
        if len(encode(batch))>85000:raise ValueError('single encoded source packet too large')
    if batch:batches.append(batch)
    for batch in batches:
        packet = encode(batch)
        code = """import base64,zlib,json,pathlib,hashlib,shutil
d=pathlib.Path(DEST)
assert d.is_dir() and not d.is_symlink() and shutil.disk_usage(d).free>=8*1024**3
rows=json.loads(zlib.decompress(base64.b64decode(PACKET)))
for row in rows:
 p=d/row['path']; assert p.resolve().is_relative_to(d.resolve())
 assert not any(x.is_symlink() for x in (p,)+tuple(p.parents))
 b=base64.b64decode(row['data']); assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): assert p.read_bytes()==b
 else:
  with p.open('xb') as f:f.write(b)
print(json.dumps({'uploaded_or_identical':len(rows),'GPU_runs':0}))
""".replace('DEST', repr(destination)).replace('PACKET', repr(packet))
        response = remote_python(code)
        if response['exit'] != 0:
            raise RuntimeError(response)
        results.append(json.loads(response['stdout']))
    return dict(files=len(rows), batches=results)

def run_logged(scope, tag, argv, timeout=50):
    directory = DIRS[scope]
    code = """import pathlib,subprocess,json,os,time
d=pathlib.Path(DIRECTORY); tag=TAG; argv=ARGV
def put(suffix,value):
 with (d/(tag+suffix)).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\\n')
put('_COMMAND.json',{'argv':argv,'cwd':ROOT,'CUDA_VISIBLE_DEVICES':''})
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
start=time.monotonic(); r=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,timeout=TIMEOUT)
for suffix,b in (('_STDOUT.log',r.stdout),('_STDERR.log',r.stderr)):
 with (d/(tag+suffix)).open('xb') as f:f.write(b)
doc={'exit':r.returncode,'elapsed_seconds':time.monotonic()-start,'stdout_bytes':len(r.stdout),'stderr_bytes':len(r.stderr),'GPU_runs':0}
put('_RESULT.json',doc)
print(json.dumps(dict(doc,stdout=r.stdout.decode('utf-8','replace')[:16000],stderr=r.stderr.decode('utf-8','replace')[:5000])))
""".replace('DIRECTORY', repr(directory)).replace('TAG', repr(tag)).replace('ARGV', repr(argv)).replace('ROOT', repr(ROOT)).replace('TIMEOUT', str(timeout))
    value = remote_python(code, timeout+10)
    if value['exit'] != 0:
        raise RuntimeError(value)
    return json.loads(value['stdout'])

if __name__ == '__main__':
    import sys
    action = sys.argv[1]
    if action == 'upload':
        local, scope = sys.argv[2:4]
        names = [str(p.relative_to(Path(local))).replace('\\','/') for p in Path(local).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md')]
        for filename in ('SOURCE_INHERITANCE.json','LOCAL_DEVELOPMENT_RECORD.json'):
            if (Path(local)/filename).is_file(): names.append(filename)
        result = upload(local,scope,sorted(names))
    elif action == 'run':
        result = run_logged(sys.argv[2],sys.argv[3],sys.argv[4:])
    elif action == 'download':
        result = rpc(dict(op='download',remote=sys.argv[2],local=sys.argv[3],timeout=60))
    else: raise ValueError('action')
    print(json.dumps(result,sort_keys=True))
