"""Audit the server12 site copy without importing GPU/model libraries."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

D = 'artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004'
G = 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004'
CANONICAL = '/common_candidate/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py'
OLD_JOB = 'server11-c5-native-common-cost-gpu03'
NEW_JOB = 'server12-c5-native-common-cost-gpu01'

def require(value, reason):
    if not value:
        raise ValueError(reason)

def nodes(path):
    return {node.name: node for node in ast.parse(path.read_bytes()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

def dump(node):
    return ast.dump(node, include_attributes=False)

class RestoreSDK(ast.NodeTransformer):
    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'site_sdk'):
            require(isinstance(node.value, ast.Call) and dump(node.value) ==
                dump(ast.parse("load_source(root, refs, SITE_SDK, '_server12_site_sdk_binding')", mode='eval').body),
                'unexpected site SDK loader')
            return None
        return self.generic_visit(node)
    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'site_sdk'):
            require(node.func.attr == 'prepare_site_sdk'
                    and dump(node) == dump(ast.parse("site_sdk.prepare_site_sdk(root, refs, out)", mode='eval').body),
                    'unexpected SDK process call')
            node.func.value.id = 'common'
            node.func.attr = 'prepare_process_sdk'
        return self.generic_visit(node)

class RestoreJob(ast.NodeTransformer):
    def visit_Constant(self, node):
        if node.value == NEW_JOB:
            node.value = OLD_JOB
        return node

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    a = parser.parse_args(argv)
    root = a.root.resolve(strict=True)
    require(a.output.absolute().resolve().is_relative_to(root) and not a.output.exists(), 'new contained audit output')
    proof = []
    allowed = {
        'native_conditional_cost.py': {'validate_native_plan'},
        'prepare_and_verify_native_cost.py': set(),
        'run_native_cost_experiment.py': {'load_configuration', 'execute_window'},
        'p4_single_file_receipt.py': {'load_verified_single_file'},
    }
    changed = {}
    sources = {}
    for filename, permitted in allowed.items():
        old_path = root / (G + CANONICAL) if filename == 'p4_single_file_receipt.py' else root / G / filename
        new_path = root / D / filename
        old, new = nodes(old_path), nodes(new_path)
        require(set(old) == set(new), 'original callable set changed: ' + filename)
        changes = sorted(k for k in old if dump(old[k]) != dump(new[k]))
        require(set(changes) <= permitted, 'unapproved model/math/lifecycle change: ' + filename)
        changed[filename] = changes
        for k in sorted(set(old) - set(changes)):
            proof.append(filename + ':' + k)
        sources[filename] = {'old': old, 'new': new}
    serial = root / D / 'prepare_and_verify_native_cost.py'
    require(serial.read_bytes() == (root / G / serial.name).read_bytes(), 'original serializer bytes changed')
    runner = sources['run_native_cost_experiment.py']
    restored = RestoreSDK().visit(copy.deepcopy(runner['new']['execute_window']))
    require(dump(restored) == dump(runner['old']['execute_window']),
        'execute_window differs beyond explicit site SDK loader/call')
    canonical = sources['p4_single_file_receipt.py']
    restored = RestoreJob().visit(copy.deepcopy(canonical['new']['load_verified_single_file']))
    require(dump(restored) == dump(canonical['old']['load_verified_single_file']),
        'public receipt differs beyond finite current calibration job')
    locked = {r['path']: r for r in json.loads((root / D / 'SOURCE_REVISION.json').read_bytes())['files']}
    original_site = json.loads((root / G / 'SITE_SOURCE_LOCK.json').read_bytes())
    preserved = []
    for row in original_site['files']:
        if row['path'].startswith(G + '/common_candidate/') and row['path'].endswith('.py'):
            path = root / row['path']
            require(not path.is_symlink(), 'common source symlink')
            raw = path.read_bytes()
            require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'G common byte drift')
            require(locked.get(row['path']) == row, 'G common omitted from revision closure')
            preserved.append(row)
    require(len(preserved) == 64, 'original G common source count')
    document = dict(schema='server12_native_recalibration_source_ancestry_v1',
        status='PASS_ORIGINAL_ENGINE_NUMERICS_AND_LIFECYCLE',
        unchanged_callable_AST=proof, unchanged_callable_count=len(proof),
        metadata_changed_callables=changed, preserved_G_common_files=preserved,
        preserved_G_common_count=len(preserved), serializer_byte_identical=True,
        execute_window_AST_restored_only_explicit_site_SDK=True,
        public_canonical_AST_restored_only_current_job=True,
        GPU_jobs=0, model_loads=0, native_cost_qualified=False, strategy_effect_verified=False)
    with a.output.open('x', encoding='utf8', newline='\n') as f:
        json.dump(document, f, indent=2, sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in document.items() if k not in ('unchanged_callable_AST','preserved_G_common_files')}, sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

