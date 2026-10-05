"""Prepare a new prospective calibration entry from the retained native entry.

Only metadata, disjoint prompt families and standing-authorization binding
change. Cache/model execution, CUDA-event meter, SDK and original guard remain.
This CPU action never creates valid live context or launches GPU operations.
"""
import ast
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
BASE = WORKSPACE / 'artifacts/prefix_io_v1_server12_candidates/native_recalibration_cpu'
OLD = 'artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004'
NEW = 'artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/calibration_v2'
OLD_LABEL = 'server12-c5-native-common-cost-gpu01'
NEW_LABEL = 'server12-i-pilot-cal01'
FILES = ('gpu_entry_binding.py', 'control_native_cost_job.py', 'run_native_cost_experiment.py',
         'native_conditional_cost.py', 'prepare_and_verify_native_cost.py',
         'p4_single_file_receipt.py', 'site_sdk_binding.py', 'test_gpu_entry_binding.py',
         'test_native_entry.py', 'test_site_sdk_binding.py')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('expected one unchanged source anchor: ' + old[:80])
    return text.replace(old, new, 1)


def functions(raw):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(raw).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def put(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('append-only candidate differs: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(raw)


def main():
    target = HERE / 'calibration_v2'
    target.mkdir(exist_ok=True)
    put(target / 'standing_gpu_authorization.py', (HERE / 'calibration' / 'standing_gpu_authorization.py').read_bytes())
    rows = []
    for name in FILES:
        old = (BASE / name).read_bytes()
        text = old.decode('utf-8').replace(OLD, NEW).replace(OLD_LABEL, NEW_LABEL)
        for first, second in (('(18100, 19100, 20100)', '(40100, 41100, 42100)'),
                              ('(1829,1830,1831)', '(4029,4030,4031)'),
                              ('(1829, 1830, 1831)', '(4029, 4030, 4031)')):
            text = text.replace(first, second)
        text = text.replace('CURRENT_CONTEXT_NATIVE_FULL_STEP_SSD_READ_DIAGNOSTIC_ONLY',
                            'PROSPECTIVE_I_PILOT_CALIBRATION_ONLY')
        if name == 'gpu_entry_binding.py':
            text = replace_once(text, 'import time\n', 'import time\nimport standing_gpu_authorization as A\n')
            text = replace_once(text, '    grant = read(root, GRANT)\n',
                                '    grant = read(root, GRANT)\n    A.validate_bound_grant(root, grant)\n')
            text = replace_once(text, 'required = [GUARD, BASE_PERMISSION, MANIFEST, ANCESTOR,',
                                "required = [GUARD, BASE_PERMISSION, MANIFEST, ANCESTOR, A.AMENDMENT, DIR + '/standing_gpu_authorization.py',")
        if name == 'control_native_cost_job.py':
            text = replace_once(text, 'import gpu_entry_binding as B\n',
                                'import gpu_entry_binding as B\nimport standing_gpu_authorization as A\n')
            text = replace_once(text, 'required = [B.GUARD, B.BASE_PERMISSION, B.ANCESTOR, B.MANIFEST, B.SDK_HELPER, *B.SDK_ASSETS]',
                                'required = [B.GUARD, B.BASE_PERMISSION, B.ANCESTOR, B.MANIFEST, B.SDK_HELPER, A.AMENDMENT, *B.SDK_ASSETS]')
            text = replace_once(text, '    context_ref, grant_ref = B.ref(root, B.CONTEXT), B.ref(root, B.GRANT)\n',
                                "    context_ref = B.ref(root, B.CONTEXT)\n"
                                "    if not B.safe(root, B.GRANT).exists():\n"
                                "        grant_document = A.bind_standing_grant(root, context_document, label=B.LABEL, gpu_uuid=uuid, revision_ref=revision_ref)\n"
                                "        grant_document['context_ref'] = context_ref\n"
                                "        put(root, B.GRANT, grant_document)\n"
                                "    grant_ref = B.ref(root, B.GRANT)\n")
            text = replace_once(text, '    grant = B.read(root, B.GRANT)\n',
                                '    grant = B.read(root, B.GRANT)\n    A.validate_bound_grant(root, grant)\n')
            text = text.replace('human_gpu_grant_required=True', 'human_gpu_grant_required=False')
            text = text.replace('human_grant_required=True', 'human_grant_required=False')
            text = text.replace('LIVE_CONTEXT_FROZEN_NEEDS_EXPLICIT_HUMAN_GPU_GRANT', 'LIVE_CONTEXT_FROZEN_STANDING_AUTHORIZATION_AVAILABLE')
            text = text.replace('bind_after_explicit_human_gpu_grant', 'bind_existing_standing_human_gpu_grant')
        if name == 'test_native_entry.py':
            text = replace_once(text, "str(HERE.parents[2] / 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate')",
                                "str(Path('/root/autodl-tmp/prefix-io-v1-handoff/project') / 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate')")
        if name == 'test_site_sdk_binding.py':
            text = replace_once(text, '        cls.root = HERE.parents[2]\n',
                                "        cls.root = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')\n")
        if name == 'test_gpu_entry_binding.py':
            text = replace_once(text, '    def test_missing_fresh_grant_blocks_bind_before_permission(self):',
                                '    def test_missing_standing_grant_blocks_bind_before_permission(self):')
            text = replace_once(text, '            with self.assertRaises(FileNotFoundError):\n                C.bind(self.root)',
                                "            with self.assertRaisesRegex(ValueError, 'STANDING_AUTHORIZATION_REJECTED'):\n                C.bind(self.root)")
        if name == 'p4_single_file_receipt.py':
            for variable, filename in (('_VERIFIER', 'native_conditional_cost.py'),
                                       ('_SERIALIZER', 'prepare_and_verify_native_cost.py')):
                data = (target / filename).read_bytes()
                replacement = variable + ' = FileRef(_D6 + ' + json.dumps(filename) + ', ' + str(len(data)) + ',\n    ' + json.dumps(hashlib.sha256(data).hexdigest()) + ')'
                text, count = re.subn(variable + r' = FileRef\(_D6 \+ "[^"]+", \d+,\s*"[0-9a-f]{64}"\)', replacement, text)
                if count != 1:
                    raise ValueError('one receipt metadata source pin required')
        raw = text.encode('utf-8')
        old_functions, new_functions = functions(old), functions(raw)
        changed = [key for key in old_functions if old_functions[key] != new_functions[key]]
        if name == 'run_native_cost_experiment.py' and changed:
            raise ValueError('production native runner function changed: ' + repr(changed))
        put(target / name, raw)
        rows.append(dict(name=name, before_bytes=len(old), before_sha256=hashlib.sha256(old).hexdigest(),
                         after_bytes=len(raw), after_sha256=hashlib.sha256(raw).hexdigest(),
                         changed_top_level_functions=changed))
    put(target / 'NATIVE_SOURCE_INHERITANCE.json', (BASE / 'NATIVE_SOURCE_INHERITANCE.json').read_bytes())
    record = dict(schema='prospective_i_calibration_candidate_build_v1', GPU_operations=0,
                  old_scope=OLD, new_scope=NEW, new_label=NEW_LABEL, files=rows,
                  calibration_prompt_first=[40100,41100,42100], calibration_seeds=[4029,4030,4031],
                  old_evidence_unchanged=True, thresholds_changed=False,
                  native_runner_function_asts_unchanged=True, cache_and_model_executor_changed=False,
                  source_freeze_needed_on_server=True, no_valid_gpu_context_created=True)
    put(HERE / 'CALIBRATION_CANDIDATE_BUILD_V2.json', (json.dumps(record, indent=2)+'\n').encode())
    print(json.dumps(record))


if __name__ == '__main__':
    main()
