"""Freeze the prospective protocol and inspect retained runner bytes on CPU.

No model/native modules imported. Source inspection is a capability mapping,
not qualification of a new GPU wrapper or a measured effect.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from i_pilot_protocol import prospective_protocol, validate_protocol, evidence_status, file_ref, parse, require

BASE = 'artifacts/prefix_io_v1_server12_candidates/'


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def inspect_source(root, relative):
    reference = file_ref(root, relative)
    raw = (root / relative).read_bytes()
    tree = ast.parse(raw, filename=relative)
    names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if node.targets[0].id in ('ROOT', 'DELIVERY', 'LABEL', 'ORDER', 'PROMPT_FIRST', 'SEEDS',
                                      'MODES', 'FILE_BYTES', 'OPERATION_COUNT', 'GUARD', 'LEDGER'):
                constants[node.targets[0].id] = value
    return dict(ref=reference, top_level_functions_and_classes=sorted(names), bounded_constants=constants)


def prepare(root, out):
    root = Path(root).resolve(strict=True)
    out = Path(out).resolve(strict=True)
    require(out.is_relative_to(root / BASE / 'i_pilot_cpu_preparation_20261004' / 'protocol'),
            'output only in owned protocol directory')
    inventory_relative = BASE + 'fixed_evidence_cpu_audit/CPU_AUDIT_INITIAL_INVENTORY.json'
    inventory = parse((root / inventory_relative).read_bytes())
    docs = inventory['normative_docs']
    norm = []
    for name in ('02_CONTROL_AND_CONTRACTS.md', '03_IMPLEMENTATION_AND_EXPERIMENTS.md', '04_CODEX_EXECUTION.md'):
        row = docs['docs/prefix_io_v1/' + name]
        require(hashlib.sha256(row['text'].encode()).hexdigest() == row['ref']['sha256']
                and len(row['text'].encode()) == row['ref']['bytes'], 'actual normative text bytes retained')
        norm.append(row['ref'])
    sources = [inspect_source(root, BASE + relative) for relative in (
        'native_recalibration_cpu/run_native_cost_experiment.py',
        'native_recalibration_cpu/control_native_cost_job.py',
        'native_recalibration_cpu/gpu_entry_binding.py',
        'normal_repeatability_v1/run_p4_single_file_experiment.py',
        'normal_repeatability_v1/control_p4_single_file.py',
        'normal_native_cpu/control_p4_single_file.py')]
    cal = sources[0]
    single = sources[3]
    require(cal['bounded_constants']['ORDER'] == (('A', 'B'), ('B', 'A'), ('A', 'B'))
            and cal['bounded_constants']['FILE_BYTES'] == 917504
            and 'execute_window' in cal['top_level_functions_and_classes'], 'retained original calibration acquisition')
    require(single['bounded_constants']['MODES'] == ('off', 'shadow', 'on')
            and 'frontend_capture' in single['top_level_functions_and_classes'], 'retained original frontend loop')
    protocol = validate_protocol(prospective_protocol())
    capabilities = dict(schema='i_pilot_retained_runner_CPU_capabilities_v1',
                        origin='actual_local_retained_source_bytes_and_AST_no_import',
                        normative_refs=norm, inventory_ref=file_ref(root, inventory_relative), source_refs=sources,
                        original_calibration_supports_three_fresh_process_pairs=True,
                        original_calibration_uses_A_B_B_A_A_B=True,
                        original_single_request_has_off_shadow_on_source_paths=True,
                        fixed_repeatability_controller_is_off_only=True,
                        retained_calibration_prompts_and_scope_are_old_fixed_values=True,
                        retained_frontend_maps_all_tokens_from_each_cumulative_output=True,
                        token_ITL_requires_validation_of_exactly_one_new_token_per_output=True,
                        new_calibration_runner_bound_by_this_inspection=False,
                        paired_effect_runner_bound=False, formal_goodput_SLO_missing=True,
                        original_guard_execution_required=True, old_failed_gate_remains_failed=True,
                        GPU_operations=0, new_GPU_evidence=False)
    write(out / 'I_PILOT_PROTOCOL.json', protocol)
    write(out / 'RETAINED_RUNNER_CAPABILITIES.json', capabilities)
    write(out / 'CPU_PROTOCOL_DECISION.json', evidence_status(protocol))
    return dict(status='CPU_PROTOCOL_AND_RETAINED_CAPABILITY_PREPARED',
                prospective_guard_slots=9, maximum_reserved_GPU_seconds=3780,
                GPU_operations=0, new_GPU_evidence=False, runtime_binding_required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.project_root, args.output_dir), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
