"""Source-group and inert combined entry contract; never a cost receipt."""
import ast
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DELIVERY = 'artifacts/prefix_io_v1/server11-c5-combined-runtime-cpu-20261004'
NATIVE = 'artifacts/prefix_io_v1/server11-c5-native-cost-preparation-cpu-20261004'
CANDIDATE = NATIVE + '/common_candidate'
CONTROL = 'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control'
REACTOR = 'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
ADAPTER_SHA256 = '9c51a1f11fc966739ab44d9852de33ea139e28db0fa9b5f4fae6315634f594ea'
C5_REACTOR_SHA256 = 'a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47'
C5_COLLECTOR_SHA256 = 'bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf'


def require(value, message):
    if not value: raise ValueError(message)


def no_grant():
    return dict(gpu_uuid=None, actual_gpu_runs=0, gpu_launch_allowed=False,
        native_execution_verified=False, native_cost_qualified=False, full_runtime_cost_qualified=False,
        on_observation_cost_measured=False, valid_native_receipt=None, effective_cost_upper_ns=None,
        effective_step_budget_ns=None, actual_on_installed=False, performance_claim=False,
        production_qualified=False, actual_model_processes_started=0)


def source_ref(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 4 * 1024**2,
        'bounded regular source')
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(raw), sha256=sha256(raw).hexdigest())


def functions(path):
    return {node.name: node for node in ast.parse(Path(path).read_bytes()).body if isinstance(node, ast.FunctionDef)}


def assert_original_runtime_ast(stage_a):
    stage_a = Path(stage_a)
    selected = {
        'run_p4_single_file_experiment.py': ('execute_window', 'native_source_binding', 'collector_source_view',
            'frontend_capture', 'original_drain', 'install_owner_snapshot_observation', 'config_for_engine'),
        'verify_p4_single_file.py': ('migration_gate', 'accounting_validator', 'analyze_runtime'),
    }
    proof = []
    for filename, names in selected.items():
        old, new = functions(stage_a / filename), functions(HERE / filename)
        for name in names:
            require(ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False),
                'original runtime/numerical AST changed: ' + filename + ':' + name)
            proof.append(filename + ':' + name)
    return dict(status='PASS_COMBINED_ORIGINAL_RUNTIME_AST', functions=proof, **no_grant())


def metadata_source_groups(native_preparation, *, stage_a):
    """Actual file metadata only: no imports of a launcher, event, model or issuer."""
    native_preparation = Path(native_preparation)
    candidate = native_preparation / 'common_candidate'
    source = candidate / REACTOR
    collector = candidate / 'native_full_step_collector.py'
    receipt = candidate / CONTROL / 'p4_single_file_receipt.py'
    rows = {name: source_ref(path) for name, path in (
        ('reactor', source), ('collector', collector), ('canonical_receipt', receipt),
        ('runtime_identity', candidate / 'single_file_runtime_binding.py'),
        ('policy', candidate / CONTROL / 'p4_policy.py'), ('bridge', candidate / CONTROL / 'p4_bridge.py'),
        ('native_verifier', native_preparation / 'native_conditional_cost.py'),
        ('adapter', HERE / 'notification_runtime_adapter.py'),
        ('launcher', HERE / 'run_p4_single_file_experiment.py'),
        ('controller', HERE / 'control_p4_single_file.py'), ('verifier', HERE / 'verify_p4_single_file.py'))}
    require(rows['reactor']['sha256'] == C5_REACTOR_SHA256 and
        rows['collector']['sha256'] == C5_COLLECTOR_SHA256, 'actual unchanged C5 engine/collector')
    require(rows['adapter']['sha256'] == ADAPTER_SHA256 and
        (HERE / 'notification_runtime_adapter.py').read_bytes() == (Path(stage_a) / 'notification_runtime_adapter.py').read_bytes(),
        'exact unchanged Stage A normal helper')
    common = [rows[name] for name in ('reactor', 'policy', 'bridge', 'native_verifier')]
    overlay = [rows[name] for name in ('collector', 'canonical_receipt', 'runtime_identity', 'adapter', 'launcher', 'controller', 'verifier')]
    require(len({row['path'] for row in common + overlay}) == len(common + overlay), 'unique common/overlay files')
    require(sum(row == rows['collector'] for row in overlay) == 1 and all(row != rows['collector'] for row in common),
        'same C5 collector uniquely runtime overlay')
    return dict(status='PASS_COMBINED_CPU_SOURCE_METADATA_ONLY', origin='CPU_actual_source_metadata',
        source_refs=rows, runtime_common_refs=common, runtime_overlay_refs=overlay,
        source_groups_are_native_binding=False, timings_measured=0, receipt_issued=False,
        positive_on_callsite_source_audited_only=True, **no_grant())


def main(argv=None):
    print(json.dumps(dict(status='GPU_BLOCKED_C5_COMBINED_RUNTIME_CPU_PREPARATION_ONLY', **no_grant()), sort_keys=True))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
