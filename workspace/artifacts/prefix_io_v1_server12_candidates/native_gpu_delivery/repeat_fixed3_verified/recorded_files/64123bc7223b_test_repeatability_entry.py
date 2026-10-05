"""CPU-only path/config/source-flow rejection checks for the new entry."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def load(filename,name):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value
C=load('control_p4_single_file.py','_repeatability_controller_cpu_tests')
R=load('run_p4_single_file_experiment.py','_repeatability_runner_cpu_tests')


class EntryRejectTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        path=self.root/C.PROTOCOL;path.parent.mkdir(parents=True)
        shutil.copyfile(HERE/'PROTOCOL.json',path)
        self.uuid='GPU-a67d2f38-8e1b-29c3-7e5b-375e1410f8ac'
    def tearDown(self):self.tmp.cleanup()

    def test_three_configs_have_fixed_distinct_labels_and_same_condition(self):
        values=[C.expected_config(self.root,index,self.uuid) for index in range(3)]
        self.assertEqual([row['diagnostic_index'] for row in values],[0,1,2])
        self.assertEqual(len({row['label'] for row in values}),3)
        for index,row in enumerate(values):
            self.assertEqual(row['mode'],'off');self.assertIsNone(row['previous_qualification_ref'])
            self.assertEqual(row['seconds_limit'],300);self.assertEqual(row['reserved_seconds'],320)
            C.validate_configuration_document(self.root,row,C.config_path(index))

    def test_index_bool_and_unbounded_rejected(self):
        for index in (True,False,-1,3,'0',None):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):C.expected_config(self.root,index,self.uuid)

    def test_config_index_alias_rejected(self):
        row=C.expected_config(self.root,0,self.uuid);row['index']=0
        with self.assertRaisesRegex(ValueError,'fixed finite configuration'):
            C.validate_configuration_document(self.root,row,C.config_path(0))

    def test_changed_protocol_reference_rejected(self):
        row=C.expected_config(self.root,0,self.uuid);row['protocol_ref']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'fixed finite configuration'):
            C.validate_configuration_document(self.root,row,C.config_path(0))

    def test_previous_qualification_cannot_promote_shadow(self):
        row=C.expected_config(self.root,0,self.uuid);row['previous_qualification_ref']={'PASS':True};row['mode']='shadow'
        with self.assertRaisesRegex(ValueError,'does not inherit or issue'):
            C.verify_previous_qualification(self.root,row,{})

    def test_relative_argparse_path_is_contained(self):
        relative=Path(C.config_path(1))
        self.assertEqual(C.config_relative(self.root,relative),relative.as_posix())

    def test_absolute_contained_path_is_exact(self):
        relative=C.config_path(2)
        self.assertEqual(C.config_relative(self.root,self.root/relative),relative)

    def test_outside_absolute_path_rejected(self):
        with self.assertRaisesRegex(ValueError,'configuration outside project'):
            C.config_relative(self.root,self.root.parent/'outside.json')

    def test_config_traversal_rejected(self):
        with self.assertRaises(ValueError):C.config_relative(self.root,'../CONFIG_off01.json')

    def test_source_same_size_drift_rejected(self):
        path=self.root/'source.py';path.write_text('fixture_only\n',encoding='utf-8')
        row=C.ref('source.py',self.root);path.write_text('FIXTURE_ONLY\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'reference changed'):C.verify_reference(row,self.root)

    def test_authority_same_size_drift_rejected(self):
        path=self.root/'authority.json';path.write_text('{"fixture_only":false}\n',encoding='utf-8')
        row=C.ref('authority.json',self.root);path.write_text('{"fixture_only":true }\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'reference changed'):C.verify_reference(row,self.root)

    def test_configuration_missing_before_proof_rejects_before_public_loader(self):
        row=C.expected_config(self.root,0,self.uuid);path=self.root/C.config_path(0)
        path.write_text(__import__('json').dumps(row),encoding='utf-8')
        with patch.object(C,'common_rows',return_value=({},{})),patch.object(C,'relevant_source_refs',return_value={}),\
             patch.object(C,'load_authority',return_value={}),patch.object(C,'verify_source_proof',side_effect=ValueError('fixture missing before proof')),\
             patch.object(C,'load_receipt') as public:
            with self.assertRaisesRegex(ValueError,'missing before proof'):C.verify_configuration(self.root,C.config_path(0))
            public.assert_not_called()

    def test_default_public_api_keeps_full_validation(self):
        with patch.object(C,'verify_configuration_with_receipt',side_effect=ValueError('full path required')) as checked:
            with self.assertRaisesRegex(ValueError,'full path required'):C.verify_configuration(self.root,C.config_path(0))
            checked.assert_called_once()

    def test_scope_before_graph_is_not_circular(self):
        tree=ast.parse((HERE/'control_p4_single_file.py').read_bytes())
        functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
        names=lambda fn:{node.func.id for node in ast.walk(fn) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
        scope_calls=names(functions['scope'])
        self.assertNotIn('verify_configuration',scope_calls);self.assertNotIn('verify_source_proof',scope_calls)
        self.assertIn('load_receipt',scope_calls);self.assertIn('common_rows',scope_calls)
        self.assertIn('verify_scope',names(functions['check_sources']))

    def test_direct_runner_import_has_no_model_import(self):
        self.assertFalse(any(name=='torch' or name.startswith(('torch.','vllm.','py_kvcache.')) for name in sys.modules))

    def test_raw_index_does_not_change_fixed_request_seed_or_prompt(self):
        self.assertEqual(R.PROMPT_FIRST,(28100,));self.assertEqual(R.SEEDS,(2829,))
        self.assertEqual(R.OPERATION_COUNT,1);self.assertEqual(R.FILE_BYTES,917504)
        self.assertEqual(R.CANONICAL,C.RECEIPT)


if __name__=='__main__':unittest.main()
