"""Finite scheduling and actual CPU target-file isolation, not GPU evidence."""
import importlib.util
import sys
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]/'artifacts/decoupled-v2/server31780-20260922/layer3_matrix'
def load(name):
    spec=importlib.util.spec_from_file_location('layer3_'+name,ROOT/(name+'.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
c=load('common'); scheduling=load('schedule')
with patch.dict(sys.modules, {'common':c}): copying=load('pool_copy')

class Layer3ScheduleTests(unittest.TestCase):
    def graph(self):
        return dict(source_builds=[dict(build_id='H',stage='P1-E',depends_on=[]),
            dict(build_id='parent',stage='P1-M',depends_on=[]),
            dict(build_id='M',stage='P1-M',depends_on=['parent'])],
            consumer_actions=[dict(action_key=[phase,'g',q,arm],request={'partition_role':role},depends_on=[] if arm=='dense' else [birth])
                for phase,birth,source in [('P1-E','H','historical_1'),('P1-M','M','M1')]
                for q,role in [('q1','fit'),('q2','validation')] for arm in ['dense',source]])

    def test_each_phase_has_its_own_gate_before_any_qa(self):
        graph=self.graph(); jobs=scheduling.schedule_matrix(graph, {'H'})
        self.assertEqual([j['key'] for j in jobs if j['kind']=='qa'],[r['action_key'] for r in graph['consumer_actions']])
        self.assertEqual([j['key'] for j in jobs if j['kind']=='build'],['parent','M'])
        for phase in ('P1-E','P1-M'):
            rows=[j for j in jobs if j['stage']==phase and j['kind']!='build']
            self.assertEqual([r['mode'] for r in rows[:4]],['dense_teacher','source_teacher_r1','dense_qa','source_greedy_r1'])
            self.assertTrue(all(j['key'][2]=='q1' for j in rows[:4]))

    def test_no_fit_sentinel_cannot_promote_validation(self):
        graph=self.graph()
        for row in graph['consumer_actions']: row['request']['partition_role']='validation'
        with self.assertRaisesRegex(ValueError,'fit numerical'): scheduling.schedule_matrix(graph, set())

    def test_unknown_or_cyclic_birth_fails(self):
        with self.assertRaises(ValueError): scheduling.schedule_matrix(self.graph(), {'invented'})
        graph=self.graph(); graph['source_builds'][1]['depends_on']=['M']
        with self.assertRaises(ValueError): scheduling.schedule_matrix(graph, set())

    def test_existing_builds_do_not_skip_new_numerics_or_qa(self):
        jobs=scheduling.schedule_matrix(self.graph(),{'H','M','parent'})
        self.assertEqual(sum(j['kind']=='build' for j in jobs),0)
        self.assertEqual(sum(j['kind']=='qa' for j in jobs),8)
        self.assertEqual(sum(j['kind']=='sentinel' for j in jobs),8)

class TargetPoolCopyTests(unittest.TestCase):
    def setUp(self):
        from tests.test_source_store_v2 import TargetSourceStoreTests
        self.f=TargetSourceStoreTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.source=self.f.root/'source'; store=self.f.store('source/store')
        self.f.publish(store,self.f.candidate('other',base=0))
        receipt=self.f.publish(store,self.f.candidate('target',base=20,mixed=True))
        self.row=store._catalog['rows'][receipt['source_id']]; store.close()
        shutil.copytree(self.f.root/'manifests',self.source/'registry')
        self.receipt={k:self.row[k] for k in ('source_id','generation','artifact_digest')}

    def test_only_target_files_copy_original_unchanged(self):
        from probekv.source_store_v2 import parse_target_catalog_v2,TargetSourceStoreV2
        from probekv.source_manifest_v2 import RequestManifestRegistry
        original=c.sha(self.source/'store/catalog.json'); dest=self.f.root/'isolated'
        audit=copying.copy_target_pool(self.source,dest,self.receipt)
        catalog=parse_target_catalog_v2(c.read(dest/'store/catalog.json'))
        self.assertEqual(set(catalog['rows']),{self.receipt['source_id']})
        self.assertEqual(audit['parent_KV_files_copied'],0)
        self.assertEqual(original,c.sha(self.source/'store/catalog.json'))
        registry=RequestManifestRegistry(persistent_root=dest/'registry',max_bytes=2000000,max_manifest_bytes=200000)
        with TargetSourceStoreV2(dest/'store',registry=registry,**{k:v for k,v in catalog['config'].items() if k!='purpose'}) as store:
            store._verify_row(store._catalog['rows'][self.receipt['source_id']],full=True)

    def test_corrupt_target_fails_before_destination_is_created(self):
        path=self.source/'store'/self.row['kv_file']
        with path.open('ab') as stream: stream.write(b'corrupt CPU fixture')
        dest=self.f.root/'invalid'
        with self.assertRaisesRegex(ValueError,'backing changed'):
            copying.copy_target_pool(self.source,dest,self.receipt)
        self.assertFalse(dest.exists())

if __name__=='__main__': unittest.main()
