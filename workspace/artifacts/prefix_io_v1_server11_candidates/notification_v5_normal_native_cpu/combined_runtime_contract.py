"""Source-group and inert combined entry contract; never a cost receipt."""
import ast
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DELIVERY = 'artifacts/prefix_io_v1/server11-c5-normal-native-cpu-20261004'
NATIVE = 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004'
CANDIDATE = NATIVE + '/common_candidate'
CONTROL = 'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control'
REACTOR = 'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
ADAPTER_SHA256 = '9c51a1f11fc966739ab44d9852de33ea139e28db0fa9b5f4fae6315634f594ea'
C5_REACTOR_SHA256 = 'a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47'
C5_COLLECTOR_SHA256 = 'bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf'
IDENTITY_SHA256 = '8b66b3aba1c936b85f4d82fe73c568650b712acf250aae454823cd35da060d60'
CANONICAL_SHA256 = '50cc763a191a66cb002ad1e6091f35fa2af9d413b37ca251af0487686ab73a82'
NATIVE_VERIFIER_SHA256 = '37aa952a81823243ad04bee87a5a221b971289c4d7fe24bbb305b5288b433ca3'
NORMAL_CANONICAL_SHA256 = '81c72a002b8b83d5f9ef8ea11e37967609909030d8c2241f9c72d4d73180664b'
HISTORICAL_ADAPTER_SHA256 = '2397899f31d055a86046acb373d81695097f74a8b96dbd1bae8b22dcb660b8a8'
HISTORICAL_VERIFIER_SHA256 = '84388238a8cbc757c53a7dd6d7192f59a320f4e4535d1048c325c4a09fb554bd'
MODEL_RUNNER = 'third_party/work/vllm-author-p4-02-cpu/vllm/v1/worker/gpu_model_runner.py'


def runtime_source_groups(root, plan, locked, normal_sources):
    """Build public canonical binding inputs; this metadata issues no receipt.

    All common sources come from the completed calibration lock. New normal
    sources belong only to the overlay and are frozen by the new common lock.
    The public canonical loader must independently replay all real evidence.
    """
    root = Path(root).resolve(strict=True)
    require(plan.get('job_id') == 'server11-c5-native-common-cost-gpu03' and
        plan.get('evidence_origin') == 'native_runtime_preregistered' and
        plan.get('cpu_preparation_only') is False, 'real completed G calibration required')
    common_paths = [plan[key]['path'] for key in
        ('model_plan_ref', 'model_config_ref', 'cuda_event_source_ref')]
    common_paths += [MODEL_RUNNER, CANDIDATE + '/' + REACTOR,
        CANDIDATE + '/' + CONTROL + '/p4_policy.py',
        CANDIDATE + '/' + CONTROL + '/p4_bridge.py', NATIVE + '/native_conditional_cost.py']
    overlay_paths = [CANDIDATE + '/native_full_step_collector.py',
        CANDIDATE + '/single_file_runtime_binding.py',
        CANDIDATE + '/' + CONTROL + '/p4_single_file_receipt.py']
    require(type(normal_sources) is list and all(type(row) is dict and
        set(row) == {'path', 'bytes', 'sha256'} and row['path'].startswith(DELIVERY + '/')
        for row in normal_sources), 'new normal source references')
    groups = []
    for paths in (common_paths, overlay_paths):
        rows = []
        for relative in paths:
            require(relative in locked, 'source absent from actual calibration: ' + relative)
            actual = source_ref(root / relative)
            actual['path'] = relative
            require(actual == locked[relative], 'calibration source drift: ' + relative)
            rows.append(actual)
        groups.append(rows)
    common, overlay = groups
    pins = {CANDIDATE + '/' + REACTOR: C5_REACTOR_SHA256,
        CANDIDATE + '/native_full_step_collector.py': C5_COLLECTOR_SHA256,
        CANDIDATE + '/single_file_runtime_binding.py': IDENTITY_SHA256,
        CANDIDATE + '/' + CONTROL + '/p4_single_file_receipt.py': CANONICAL_SHA256,
        NATIVE + '/native_conditional_cost.py': NATIVE_VERIFIER_SHA256,
        DELIVERY + '/notification_runtime_adapter.py': ADAPTER_SHA256,
        DELIVERY + '/p4_single_file_receipt.py': NORMAL_CANONICAL_SHA256,
        DELIVERY + '/historical_calibration_binding.py': HISTORICAL_ADAPTER_SHA256,
        DELIVERY + '/native_conditional_cost.py': HISTORICAL_VERIFIER_SHA256}
    overlay += normal_sources
    all_rows = common + overlay
    require(len({row['path'] for row in all_rows}) == len(all_rows), 'unique runtime source groups')
    refs = {row['path']: row for row in all_rows}
    require(all(path in refs and refs[path]['sha256'] == digest for path, digest in pins.items()),
        'unchanged actual G common and normal notification sources')
    required_history_overlay = {DELIVERY + '/' + name for name in
        ('p4_single_file_receipt.py', 'native_conditional_cost.py', 'prepare_and_verify_native_cost.py',
         'historical_calibration_binding.py', 'GPU03_LEDGER_SNAPSHOT.json')}
    require(required_history_overlay.issubset({row['path'] for row in overlay}),
        'real new public historical-compatible canonical, verifier, serializer, adapter and snapshot overlay')
    return dict(runtime_common_refs=common, runtime_overlay_refs=overlay,
        source_groups_are_native_binding=False, receipt_issued=False, **no_grant())


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
        'run_p4_single_file_experiment.py': ('collector_source_view',
            'frontend_capture', 'original_drain', 'install_owner_snapshot_observation', 'config_for_engine'),
        'verify_p4_single_file.py': ('migration_gate', 'accounting_validator'),
    }
    proof = []
    for filename, names in selected.items():
        old, new = functions(stage_a / filename), functions(HERE / filename)
        for name in names:
            require(ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False),
                'original runtime/numerical AST changed: ' + filename + ':' + name)
            proof.append(filename + ':' + name)
    return dict(status='PASS_COMBINED_ORIGINAL_HELPER_AST', functions=proof,
        execute_window_exact_AST_claimed=False,
        execute_window_changes='authenticated entry gate and gross complete-path CPU meter require separate audit',
        glue_and_semantics_changes='source-verified public canonical module alias and NOT_EXERCISED wait evidence require separate audit; no admission thresholds changed',
        **no_grant())


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
