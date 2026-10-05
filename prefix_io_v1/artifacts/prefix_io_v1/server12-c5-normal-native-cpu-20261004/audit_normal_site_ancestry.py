"""Read-only CPU source audit of the finite server12 normal entry migration."""
import argparse
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path


def require(value, reason):
    if not value:
        raise ValueError(reason)


def safe(root, value):
    path = Path(value)
    path = path if path.is_absolute() else root / path
    require(path.resolve(strict=True).is_relative_to(root), 'source outside actual project')
    for item in (path, *path.parents):
        require(not item.is_symlink(), 'source symlink refused')
        if item == root:
            break
    return path


def functions(raw):
    return {node.name: node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)}


def dump(node):
    return ast.dump(node, include_attributes=False)


def one_statement(source):
    return ast.parse(source).body[0]


def normalize_runner(node):
    node = deepcopy(node)
    if node.name in ('notification_adapter', 'controller_api'):
        for value in ast.walk(node):
            if isinstance(value, ast.Constant) and type(value.value) is str:
                value.value = value.value.replace('_server12_', '_server11_')
    elif node.name == 'load_configuration':
        matching = [value for value in ast.walk(node) if isinstance(value, ast.Tuple)
            and any(isinstance(item, ast.Name) and item.id == 'SITE_SDK' for item in value.elts)]
        require(len(matching) == 1 and isinstance(matching[0].elts[-1], ast.Name)
            and matching[0].elts[-1].id == 'SITE_SDK', 'one extra verified SDK source pin only')
        matching[0].elts.pop()
    elif node.name == 'execute_window':
        expected_load = one_statement("site_sdk=load_source(root,refs,SITE_SDK,'_normal_actual_site_sdk')")
        expected_prepare = one_statement('sdk=site_sdk.prepare_site_sdk(root,refs,out)')
        expected_ninja = one_statement("result['ninja_environment']=common.prepare_process_ninja(root,refs,out,sdk)")
        # Statement lists are attributes, so inspect every real compound node.
        groups = [getattr(part, key) for part in ast.walk(node) for key in ('body', 'orelse', 'finalbody')
            if hasattr(part, key) and isinstance(getattr(part, key), list)]
        count = 0
        for group in groups:
            for index, value in enumerate(list(group)):
                if isinstance(value, ast.stmt) and dump(value) == dump(expected_load):
                    require(index + 1 < len(group) and dump(group[index + 1]) == dump(expected_prepare),
                        'only exact private SDK load and original argument shape permitted')
                    group[index:index + 2] = [one_statement('sdk=common.prepare_process_sdk(root,refs,out)')]
                    count += 1
        require(count == 1 and sum(dump(part) == dump(expected_ninja) for part in ast.walk(node)
            if isinstance(part, ast.stmt)) == 1, 'one explicit SDK compatibility route and unchanged Ninja')
    return node


def normalize_verifier(node):
    node = deepcopy(node)
    if node.name in ('notification_adapter', 'controller_api'):
        for value in ast.walk(node):
            if isinstance(value, ast.Constant) and type(value.value) is str:
                value.value = value.value.replace('_server12_', '_server11_')
    return node


def audit(root, baseline, candidate):
    root = Path(root).resolve(strict=True)
    baseline, candidate = safe(root, baseline), safe(root, candidate)
    require(baseline.is_dir() and candidate.is_dir() and baseline != candidate, 'distinct actual source directories')
    refs, raw_sources = {}, {}
    names = ('run_p4_single_file_experiment.py', 'verify_p4_single_file.py', 'control_p4_single_file.py',
        'combined_runtime_contract.py', 'notification_runtime_adapter.py', 'observation_cpu_cost.py')
    for directory in (baseline, candidate):
        for name in names:
            path = safe(root, directory / name)
            raw = path.read_bytes()
            relative = path.relative_to(root).as_posix()
            refs[relative] = dict(path=relative, bytes=len(raw), sha256=sha256(raw).hexdigest())
            raw_sources[(directory, name)] = raw
    evidence = []
    for name, normalize in (('run_p4_single_file_experiment.py', normalize_runner),
        ('verify_p4_single_file.py', normalize_verifier)):
        old = functions(raw_sources[(baseline, name)])
        new = functions(raw_sources[(candidate, name)])
        require(set(old) == set(new), 'runtime/validation function surface changed: ' + name)
        for function in old:
            require(dump(old[function]) == dump(normalize(new[function])),
                'unexpected runtime/numerical/notification AST change: ' + name + ':' + function)
            evidence.append(name + ':' + function)
    metadata_changes = {}
    for name, allowed in (('control_p4_single_file.py',
        {'label', 'load_receipt', 'common_rows', 'freeze_common', 'verify_configuration'}),
        ('combined_runtime_contract.py', {'runtime_source_groups', 'metadata_source_groups'})):
        old = functions(raw_sources[(baseline, name)])
        new = functions(raw_sources[(candidate, name)])
        additions = {'metadata_reference'} if name == 'control_p4_single_file.py' else set()
        require(set(old) == set(new) - additions and additions.issubset(new),
            'metadata function surface changed: ' + name)
        changed = {function for function in old if dump(old[function]) != dump(new[function])}
        require(changed.issubset(allowed), 'unexpected authority/semantics changes: ' + name + ':' + repr(changed - allowed))
        metadata_changes[name] = sorted(changed | additions)
    for name in ('notification_runtime_adapter.py', 'observation_cpu_cost.py'):
        require(raw_sources[(baseline, name)] == raw_sources[(candidate, name)], 'unchanged observation bytes: ' + name)
    for relative, expected in refs.items():
        raw = safe(root, relative).read_bytes()
        require(len(raw) == expected['bytes'] and sha256(raw).hexdigest() == expected['sha256'],
            'source changed during CPU audit: ' + relative)
    return dict(status='PASS_SERVER12_NORMAL_CPU_SOURCE_ANCESTRY', source_refs=list(refs.values()),
        normalized_function_checks=evidence, metadata_functions_requiring_site_review=metadata_changes,
        normalization='only metadata module names, additional checked SDK source, and exact explicit SDK load/prepare pair',
        provenance_change='metadata_reference normalizes actual project absolute paths after complete byte/SHA checks, permits only the full-audited current host driver; every original calibration leaf remains fully byte/SHA checked and unmodified, newly discovered raw receipts retain BFS',
        notification_adapter_byte_identical=True, gross_cpu_meter_byte_identical=True,
        math_admission_wait_model_cache_and_lifecycle_AST_preserved=True,
        actual_GPU_runs=0, GPU_seconds=0, gpu_authority_issued=False,
        native_execution_verified=False, normal_runtime_condition_qualified=False,
        full_runtime_cost_qualified=False, strategy_effect_verified=False, P4_completed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--baseline-dir', required=True, type=Path)
    parser.add_argument('--candidate-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.project_root, args.baseline_dir, args.candidate_dir), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
