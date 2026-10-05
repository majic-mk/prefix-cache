"""Read a bounded allowlist of existing tar members; never extract the archive."""
from pathlib import Path
import argparse,hashlib,json,tarfile,time

HERE=Path(__file__).resolve().parent
PROJECT=Path.cwd().resolve()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=PROJECT/'artifacts/server11_results/p4_v4/SERVER11_RAW_DELTA.tar.gz')
parser.add_argument('--output',type=Path,default=HERE/'SELECTED_ARCHIVE_MEMBERS.json')
args=parser.parse_args()
ARCHIVE=args.archive.resolve(strict=True)
if args.output.exists():raise FileExistsError('append-only review output already exists: '+str(args.output))
A='artifacts/prefix_io_v1/'
C=A+'server11-p4-single-file-candidate-v4-20261003/'
D=A+'server11-p4-single-file-runtime-v4-20261003/'
N=A+'server11-native-cost-v6-20261003/'
P=A+'server11-p4-retry-cpu-profile-v2-20261003/'
R='experiments/prefix_io_v1/runs/'
wanted=set([
C+'CPU_PROFILE_FULL_BORROW.json',C+'FULL_BORROW_CPU_PROFILE_COMMAND.json',
P+'ACTUAL_CPU_full_borrow.json',P+'CPU_PROFILE_full_borrow_COMMAND.json',P+'profile_retry_hotpath.py',
C+'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py',
C+'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_bridge.py',
C+'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_policy.py',
C+'native_full_step_collector.py',C+'test_single_file_retry_observation.py',
D+'run_p4_single_file_experiment.py',D+'verify_p4_single_file.py',
N+'NATIVE_CONDITIONAL_COST_RESULT.json',N+'NATIVE_COST_PLAN.json',N+'NATIVE_MEASUREMENTS.json',
R+'server11-native-cost-six-window-06/details/native-cost-runtime-result.json',
])
for mode in ('off','shadow','on'):
 wanted.add(D+'QUALIFICATION_'+mode+'.json')
 wanted.add(R+'server11-p4-single-file-'+mode+'-04/details/p4-single-file-runtime-result.json')
members={};index=[];started=time.perf_counter()
with ARCHIVE.open('rb') as f:
 digest=hashlib.file_digest(f,'sha256').hexdigest()
with tarfile.open(ARCHIVE,'r|gz',bufsize=1024*1024) as archive:
 for member in archive:
  if not member.isfile():continue
  index.append(dict(name=member.name,bytes=member.size))
  if member.name not in wanted:continue
  assert member.name not in members and member.size<=16*1024**2
  stream=archive.extractfile(member);raw=stream.read()
  assert len(raw)==member.size
  content=json.loads(raw) if member.name.endswith('.json') else raw.decode()
  members[member.name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),content=content)
  print('read',member.name,len(raw),flush=True)
assert set(members)==wanted,sorted(wanted-set(members))
result=dict(scope='bounded_streamed_archive_member_read_no_GPU_RPC',
 archive_ref=dict(path=ARCHIVE.relative_to(PROJECT).as_posix() if ARCHIVE.is_relative_to(PROJECT) else str(ARCHIVE),bytes=ARCHIVE.stat().st_size,sha256=digest),
 archive_file_count=len(index),selected_member_count=len(members),
 selected_member_uncompressed_bytes=sum(m['bytes'] for m in members.values()),
 selected_members=members,
 nonfixture_profile_or_timeline_members=[row for row in index
  if '/cpu-native-tmp/' not in row['name'] and any(k in row['name'].lower() for k in ('profile','trace','timeline','capture.json'))],
 seconds=time.perf_counter()-started)
with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,separators=(',',':'))
print(json.dumps({k:v for k,v in result.items() if k not in ('selected_members','nonfixture_profile_or_timeline_members')},indent=2))

