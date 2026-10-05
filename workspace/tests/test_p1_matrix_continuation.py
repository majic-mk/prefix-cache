import importlib.util,unittest,sys,shutil
from unittest.mock import patch
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'artifacts/decoupled-v2/server31780-20260922/p1_full/common.py'
spec=importlib.util.spec_from_file_location('p1_matrix_common',p);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
mspec=importlib.util.spec_from_file_location('p1_mixed_audit',p.with_name('mixed_audit.py'))
ma=importlib.util.module_from_spec(mspec)
with patch.dict(sys.modules,{'common':c}):mspec.loader.exec_module(ma)

class MatrixContinuationTests(unittest.TestCase):
 def test_progress_event_and_action_kind_cannot_collide(self):
  r=c.progress_record('COMPLETE',time_unix=12,kind='qa',id='000')
  self.assertEqual(r['kind'],'COMPLETE');self.assertEqual(r['action_kind'],'qa')
 def test_progress_without_action_kind(self):
  self.assertEqual(c.progress_record('START',time_unix=12,id='000')['kind'],'START')
 def graph(self):
  return dict(source_builds=[dict(build_id='E',stage='P1-E',depends_on=[]),
      dict(build_id='parent',stage='P1-M',depends_on=[]),dict(build_id='M',stage='P1-M',depends_on=['parent'])],
      consumer_actions=[dict(action_key=['P1-E','g','q','E'],depends_on=['E']),
                        dict(action_key=['P1-M','g','q','M1'],depends_on=['M'])])
 def test_dependency_order_and_reused_birth(self):
  jobs=c.schedule(self.graph(),{'E'},set())
  self.assertEqual([j['key'] for j in jobs],[['P1-E','g','q','E'],'parent','M',['P1-M','g','q','M1']])
 def test_completed_qa_not_rerun(self):
  jobs=c.schedule(self.graph(),{'E'},{('P1-E','g','q','E')})
  self.assertEqual(len(jobs),3)
 def test_unknown_completion_rejected(self):
  with self.assertRaises(ValueError):c.schedule(self.graph(),{'invented'},set())
 def test_cycle_rejected(self):
  g=self.graph();g['source_builds'][1]['depends_on']=['M']
  with self.assertRaises(ValueError):c.schedule(g,set(),set())
 def test_all_wrong_ties_no_headroom(self):
  r=c.summarize_exact({'g':[dict(F1=[0]*4,dense_F1=0)]})[0]
  self.assertEqual(r['headroom'],0);self.assertFalse(r['complementary']);self.assertTrue(r['singleton_not_identifiable'])
 def test_single_request_difference_not_complementarity(self):
  self.assertFalse(c.summarize_exact({'g':[dict(F1=[1,0,0,0],dense_F1=1)]})[0]['complementary'])
 def test_cross_request_complementarity(self):
  r=c.summarize_exact({'g':[dict(F1=[1,0,0,0],dense_F1=1),dict(F1=[0,1,0,0],dense_F1=1)]})[0]
  self.assertEqual(r['headroom'],.5);self.assertEqual(r['coverage_space'],.5);self.assertTrue(r['complementary'])
 def test_constant_best_no_headroom(self):
  r=c.summarize_exact({'g':[dict(F1=[1,0,0,0],dense_F1=1),dict(F1=[1,.5,0,0],dense_F1=1)]})[0]
  self.assertFalse(r['complementary'])
 def test_missing_and_nonfinite_not_success(self):
  for v in [None,float('nan')]:
   with self.assertRaises((ValueError,TypeError)):c.summarize_exact({'g':[dict(F1=[v]*4,dense_F1=1)]})

class MixedArtifactAuditTests(unittest.TestCase):
 def setUp(self):
  # Real tiny CPU tensors/files only; this does not certify a GPU mixed Source.
  from test_source_store_v2 import TargetSourceStoreTests
  self.f=TargetSourceStoreTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
  self.graph=dict(source_builds=[dict(build_id='E',role='exact_E',group_id='g'),
                               dict(build_id='M',role='mixed_M1',group_id='g')])
 def births(self):
  out={}
  for bid,base,mixed in [('E',0,False),('M',20,True)]:
   pool=self.f.root/bid;store=self.f.store(bid+'/store')
   result=self.f.publish(store,self.f.candidate('birth-'+bid,base=base,mixed=mixed))
   row=store._catalog['rows'][result['source_id']];store.close()
   shutil.copytree(self.f.root/'manifests',pool/'registry')
   receipt=pool/'receipt.json';c.write(receipt,{k:row[k] for k in ('source_id','generation','artifact_digest')})
   c.write(pool/'operation.json',dict(request=dict(token_ids=list(self.f.tokens))))
   resultfile=pool/'result.json';c.write(resultfile,{'fixture':'CPU_ONLY_NOT_GPU_EVIDENCE'})
   out[bid]=dict(pool=str(pool),receipt=c.ref(receipt),result=c.ref(resultfile))
  return out
 def test_missing_births_remain_pending(self):
  r=ma.audit_pairs(self.graph,{},self.f.root/'audit')
  self.assertEqual(r['pairs'],[]);self.assertEqual(r['pending_groups'],['g'])
  self.assertFalse(r['full_P1_M_complete'])
 def test_detached_mixed_target_real_files_not_parent_tensor(self):
  births=self.births();before=c.sha(Path(births['M']['pool'])/'store/catalog.json')
  r=ma.audit_pairs(self.graph,births,self.f.root/'audit');pair=r['pairs'][0]
  self.assertTrue(pair['detached_target_CPU_reload_passed'])
  self.assertTrue(pair['distinct_deep_artifact']);self.assertFalse(pair['d1_exact_collision'])
  self.assertEqual(pair['parent_tensor_files_copied'],0)
  self.assertEqual(pair['parent_absent_GPU_consumption'],'NOT_EXECUTED_BY_THIS_CPU_AUDIT')
  self.assertEqual(before,c.sha(Path(births['M']['pool'])/'store/catalog.json'))
  self.assertFalse(r['full_P1_M_complete'])
 def test_receipt_tamper_rejected(self):
  births=self.births();births['M']['receipt']['sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'receipt changed'):
   ma.audit_pairs(self.graph,births,self.f.root/'audit')

if __name__=='__main__':unittest.main()
