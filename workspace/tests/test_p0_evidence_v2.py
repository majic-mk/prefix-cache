import json
from pathlib import Path
import tempfile
import unittest

import torch

from probekv.p0_evidence_v2 import P0EvidenceWriter, read_p0_events, compare_logit_files
from probekv.v8_schema10_storage import file_digest


class P0EvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)/'new'
        self.binding = {'cpu_fixture':True}

    def writer(self):
        return P0EvidenceWriter(self.root,binding=self.binding,manifest={'test_only':True})

    def record(self,w,name,logits):
        return w.write_action(name,audit={'status':'COMPLETED','answer':{'token_ids':[3]}},
                              logits=logits,origin='cpu_fixture')

    def test_completed_action_is_not_numerical_pass_or_gpu_qualification(self):
        w=self.writer(); r=self.record(w,'one',[torch.ones(1,7)])
        self.assertEqual(r['numerical_verdict'],'NOT_EVALUATED')
        self.assertEqual(r['evidence_origin'],'cpu_fixture')
        self.assertFalse(r['gpu_runtime_qualified'])
        result=w.finalize({'status':'COMPLETED'})
        self.assertFalse(result['P1_execution_allowed'])
        rows=read_p0_events(w.log_path,binding=self.binding)
        self.assertEqual(len(rows),2)
        self.assertEqual(result['raw_event_sha256'],file_digest(w.log_path))

    def test_no_overwrite_duplicate_action_or_resume(self):
        w=self.writer(); self.record(w,'one',[])
        with self.assertRaises(FileExistsError): self.record(w,'one',[])
        with self.assertRaises(FileExistsError): self.writer()
        w.finalize({'status':'FAILED'})
        with self.assertRaises(RuntimeError):w.append('pretend_resume','one',{})

    def test_rejects_traversal_and_unconverted_logits_without_hiding_answer(self):
        w=self.writer()
        with self.assertRaises(ValueError):self.record(w,'../outside',[])
        with self.assertRaises(ValueError):self.record(w,'bad',[torch.ones(1,2,dtype=torch.bfloat16)])
        self.assertTrue((self.root/'bad'/'request.json').is_file())
        self.assertFalse((self.root/'bad'/'record.json').exists())

    def test_corrupted_or_torn_event_is_rejected(self):
        w=self.writer();w.append('started','a',{})
        original=w.log_path.read_bytes()
        w.log_path.write_bytes(original[:-1])
        with self.assertRaises(ValueError):read_p0_events(w.log_path,binding=self.binding)
        value=json.loads(original);value['kind']='forged'
        w.log_path.write_text(json.dumps(value)+'\n',encoding='utf-8')
        with self.assertRaises(ValueError):read_p0_events(w.log_path,binding=self.binding)

    def test_numeric_comparison_reads_raw_files_not_claimed_pass(self):
        w=self.writer(); a=self.record(w,'ref',[torch.tensor([1.,2.]),torch.zeros(2)])
        b=self.record(w,'candidate',[torch.tensor([1.,2.01]),torch.zeros(2)])
        kw=dict(expected_hashes=(a['logits']['sha256'],b['logits']['sha256']),
                relative_l2_limit=.01,minimum_positions=2)
        result=compare_logit_files(self.root/'ref/logits.npy',self.root/'candidate/logits.npy',**kw)
        self.assertTrue(result['numeric_passed'])
        self.assertFalse(result['recipe_alignment_verified'])
        self.assertFalse(result['native_runtime_qualified'])
        kw['relative_l2_limit']=0.
        self.assertFalse(compare_logit_files(self.root/'ref/logits.npy',self.root/'candidate/logits.npy',**kw)['numeric_passed'])

    def test_missing_positions_bad_digest_nonfinite_and_zero_norm(self):
        w=self.writer(); a=self.record(w,'ref',[torch.zeros(2)])
        b=self.record(w,'candidate',[torch.ones(2)])
        paths=(self.root/'ref/logits.npy',self.root/'candidate/logits.npy')
        kw=dict(expected_hashes=(a['logits']['sha256'],b['logits']['sha256']),relative_l2_limit=1.,minimum_positions=1)
        result=compare_logit_files(*paths,**kw)
        self.assertFalse(result['numeric_passed']);self.assertEqual(result['relative_l2_by_position'],[None])
        for change in ({'minimum_positions':2},{'relative_l2_limit':float('nan')},{'expected_hashes':('bad','bad')}):
            with self.assertRaises(ValueError):compare_logit_files(*paths,**{**kw,**change})
        with self.assertRaises(ValueError):self.record(w,'nan',[torch.tensor([float('nan')])])


if __name__=='__main__':unittest.main()
