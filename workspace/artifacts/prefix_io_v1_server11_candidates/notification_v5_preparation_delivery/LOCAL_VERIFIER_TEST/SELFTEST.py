"""Local-only adversarial fixtures for the backup verifier; no network/GPU."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import unittest

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'verify_preparation_backup.py'
spec=importlib.util.spec_from_file_location('_backup_verifier_under_test',SOURCE)
V=importlib.util.module_from_spec(spec);sys.modules[spec.name]=V;spec.loader.exec_module(V)
ROOT=sorted(V.ROOTS)[0]
BASE=[(root+'/data.json',b'{"cpu":true}\n',tarfile.REGTYPE) for root in sorted(V.ROOTS)]

def encoded(doc):return (json.dumps(doc,sort_keys=True)+'\n').encode()
def build(entries,mutate_manifest=None):
 manifest=dict(files=[dict(path=name,bytes=len(data),sha256=V.digest(data)) for name,data,kind in entries],
  data_file_count=len(entries),raw_bytes=sum(len(data) for _,data,_ in entries),GPU_runs=0,formal_benchmark_runs=0)
 if mutate_manifest:mutate_manifest(manifest)
 mb=encoded(manifest);target=io.BytesIO()
 with tarfile.open(fileobj=target,mode='w:gz') as archive:
  for name,data,kind in entries+[(V.MANIFEST,mb,tarfile.REGTYPE)]:
   info=tarfile.TarInfo(name);info.type=kind;info.size=len(data)
   if kind in (tarfile.SYMTYPE,tarfile.LNKTYPE):info.linkname='../../outside'
   archive.addfile(info,io.BytesIO(data))
 ab=target.getvalue()
 receipt=dict(status='PASS_SEALED_CPU_PREPARATION',GPU_runs=0,formal_benchmark_runs=0,
  archive=dict(file='fixture.tar.gz',bytes=len(ab),sha256=V.digest(ab)),
  manifest=dict(file=V.MANIFEST,bytes=len(mb),sha256=V.digest(mb)),
  data_files=len(entries),raw_bytes=sum(len(data) for _,data,_ in entries))
 return ab,receipt

class BackupTests(unittest.TestCase):
 sequence=max((int(p.name.split('-')[1]) for p in HERE.glob('case-*') if p.is_dir()),default=0)
 def fixture(self,ab,receipt):
  type(self).sequence+=1;directory=HERE/('case-%02d'%self.sequence);directory.mkdir()
  (directory/'fixture.tar.gz').write_bytes(ab);(directory/'receipt.json').write_bytes(encoded(receipt))
  return argparse.Namespace(archive=directory/'fixture.tar.gz',receipt=directory/'receipt.json',
   output_dir=directory/'extracted',result=directory/'result.json')
 def rejected(self,entries=None,mutate_manifest=None,mutate_archive=None,mutate_receipt=None):
  ab,receipt=build(BASE if entries is None else entries,mutate_manifest)
  if mutate_archive:ab=mutate_archive(ab)
  if mutate_receipt:mutate_receipt(receipt)
  args=self.fixture(ab,receipt)
  with self.assertRaises((ValueError,KeyError,tarfile.TarError,EOFError,OSError)):V.run(args)
  self.assertFalse(args.output_dir.exists());self.assertFalse(args.result.exists())
 def test_valid_actual_cli_writes_verified_bytes_and_refuses_overwrite(self):
  ab,receipt=build(BASE);args=self.fixture(ab,receipt)
  command=[sys.executable,'-B','-I','-S',str(SOURCE),'--archive',str(args.archive),'--receipt',str(args.receipt),
   '--output-dir',str(args.output_dir),'--result',str(args.result)]
  proc=subprocess.run(command,capture_output=True,text=True)
  self.assertEqual(proc.returncode,0,proc.stderr);doc=json.loads(proc.stdout)
  self.assertEqual(doc['status'],'PASS_SHA_VERIFIED_NEW_CPU_PREPARATION_BACKUP');self.assertEqual(doc['gpu_runs'],0)
  self.assertEqual(doc['data_files'],4);self.assertEqual(doc['verified_members'],5)
  for name,data,_ in BASE:self.assertEqual((args.output_dir/name).read_bytes(),data)
  before={p.relative_to(args.output_dir).as_posix():p.read_bytes() for p in args.output_dir.rglob('*') if p.is_file()}
  with self.assertRaises(ValueError):V.run(args)
  self.assertEqual(before,{p.relative_to(args.output_dir).as_posix():p.read_bytes() for p in args.output_dir.rglob('*') if p.is_file()})
 def test_unsafe_member_paths_rejected_before_creating_output(self):
  names=['../escape','/absolute','C:/drive','other/x',ROOT+'/../escape',ROOT+'//empty',ROOT+'/./dot',
   ROOT+'/a\\b',ROOT+'/file:stream',ROOT+'/bad.',ROOT+'/bad ',ROOT+'/nul.txt',ROOT+'/COM¹.txt',
   ROOT+'/LPT²',ROOT+'/com1 .txt',ROOT+'/CONIN$',ROOT+'/bad?name',ROOT+'/control\x01name']
  for name in names:
   with self.subTest(name=name):self.rejected(BASE+[(name,b'x',tarfile.REGTYPE)])
 def test_duplicates_case_and_file_parent_collisions(self):
  pairs=[(ROOT+'/data.json',ROOT+'/data.json'),(ROOT+'/Same',ROOT+'/same'),
   (ROOT+'/Sub/a',ROOT+'/sub/b'),(ROOT+'/node',ROOT+'/node/child'),(ROOT+'/node/child',ROOT+'/node')]
  for a,b in pairs:
   with self.subTest(a=a,b=b):self.rejected(BASE+[(a,b'a',tarfile.REGTYPE),(b,b'b',tarfile.REGTYPE)])
 def test_nonregular_link_and_directory_members(self):
  for kind in (tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE,tarfile.FIFOTYPE,tarfile.CHRTYPE,tarfile.GNUTYPE_SPARSE):
   with self.subTest(kind=kind):self.rejected(BASE+[(ROOT+'/special',b'',kind)])
 def test_archive_and_manifest_sha_are_independent_trust_checks(self):
  self.rejected(mutate_archive=lambda ab:ab+b'tampered')
  self.rejected(mutate_receipt=lambda r:r['manifest'].update(sha256='0'*64))
  self.rejected(mutate_manifest=lambda m:m['files'][0].update(sha256='0'*64))
  self.rejected(mutate_manifest=lambda m:m['files'][0].update(bytes=123))
 def test_manifest_missing_extra_duplicate_and_count_disagreement(self):
  self.rejected(mutate_manifest=lambda m:m['files'].pop())
  self.rejected(mutate_manifest=lambda m:m['files'][0].update(path=ROOT+'/missing'))
  self.rejected(mutate_manifest=lambda m:m['files'][0].update(path=m['files'][1]['path']))
  self.rejected(mutate_receipt=lambda r:r.update(data_files=3))
  self.rejected(mutate_receipt=lambda r:r.update(raw_bytes=1))
  self.rejected(BASE[:-1])
 def test_resource_caps_and_exact_integer_not_bool(self):
  self.assertEqual((V.MAX_ARCHIVE,V.MAX_RAW,V.MAX_FILE,V.MAX_FILES),(20*V.MIB,50*V.MIB,10*V.MIB,300))
  self.rejected(BASE+[(ROOT+'/oversize',b'x'*(10*V.MIB+1),tarfile.REGTYPE)])
  self.rejected(BASE+[(ROOT+'/n%03d'%n,b'x',tarfile.REGTYPE) for n in range(297)])
  self.rejected(mutate_receipt=lambda r:r.update(data_files=True))
  for cap,value in (('MAX_RAW',10),('MAX_ARCHIVE',10),('MAX_TAR',10)):
   previous=getattr(V,cap);setattr(V,cap,value)
   try:self.rejected()
   finally:setattr(V,cap,previous)
 def test_duplicate_json_keys_and_result_inside_output_rejected(self):
  with self.assertRaises(ValueError):V.read_json(b'{"same":1,"same":2}')
  with self.assertRaises(ValueError):V.read_json(b'{"x":NaN}')
  ab,receipt=build(BASE);args=self.fixture(ab,receipt);args.result=args.output_dir/'result.json'
  with self.assertRaises(ValueError):V.run(args)
  self.assertFalse(args.output_dir.exists())
 def test_non_cpu_receipt_or_manifest_boundary_cannot_pass(self):
  self.rejected(mutate_receipt=lambda r:r.update(status='PASS_OTHER'))
  for key in ('GPU_runs','formal_benchmark_runs'):
   for wrong in (1,False,None):
    with self.subTest(key=key,value=wrong):
     self.rejected(mutate_receipt=lambda r:r.update({key:wrong}))
     self.rejected(mutate_manifest=lambda m:m.update({key:wrong}))

if __name__=='__main__':
 stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(BackupTests)
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 run_index=2
 while (HERE/('SELFTEST_RESULT_%02d.json'%run_index)).exists():run_index+=1
 (HERE/('SELFTEST_STDERR_%02d.log'%run_index)).write_text(stream.getvalue(),encoding='utf-8')
 document=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
  failed=len(result.failures)+len(result.errors),fixture_cases=BackupTests.sequence,gpu_runs=0,
  verifier_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),command=[sys.executable,'-B','-I','-S',str(Path(__file__).resolve())])
 (HERE/('SELFTEST_RESULT_%02d.json'%run_index)).write_text(json.dumps(document,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps(document));print(stream.getvalue());sys.exit(0 if result.wasSuccessful() else 1)
