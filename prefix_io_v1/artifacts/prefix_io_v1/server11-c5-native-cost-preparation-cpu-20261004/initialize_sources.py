"""Append-only copy/adapt author-route CPU source preparation; never launch."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
V6 = BASE / "native_cost_v6"
C5 = BASE / "notification_v5_cpu_delivery/server_replay/candidate"
DELIVERY = "artifacts/prefix_io_v1/server11-c5-native-cost-preparation-cpu-20261004"
CANDIDATE = DELIVERY + "/common_candidate"
LABEL = "server11-c5-native-common-cost-preparation01"
BLOCKED = "GPU_BLOCKED_C5_NATIVE_COST_CODE_PREPARATION_ONLY"
SHAS = {"reactor": "a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47",
    "collector": "bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf"}
HEADER = '''CPU_PREPARATION_BLOCKED = "GPU_BLOCKED_C5_NATIVE_COST_CODE_PREPARATION_ONLY"


def blocked_document():
    return dict(status=CPU_PREPARATION_BLOCKED, gpu_started=False, actual_gpu_runs=0,
        gpu_launch_allowed=False, native_execution_verified=False, native_cost_qualified=False,
        full_runtime_cost_qualified=False, on_observation_cost_measured=False,
        valid_native_receipt=None, effective_cost_upper_ns=None, effective_step_budget_ns=None)


def block_gpu_start():
    raise RuntimeError(CPU_PREPARATION_BLOCKED)


'''


def put(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def guard_functions(text, names, *, main=False):
    tree = ast.parse(text)
    inserts = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                line = first.end_lineno
            else: line = first.lineno - 1
            lines = ["    block_gpu_start()\n"]
            if main and node.name == "main":
                lines = ["    print(json.dumps(blocked_document(), sort_keys=True))\n", "    return 2\n"]
            inserts.append((line, lines))
    found = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if not set(names).issubset(found): raise ValueError("missing original function")
    lines = text.splitlines(keepends=True)
    for line, addition in sorted(inserts, reverse=True): lines[line:line] = addition
    return "".join(lines)


def helper(text, before):
    if before not in text: raise ValueError("helper insertion marker")
    return text.replace(before, HEADER + before, 1)


def main():
    reactor = C5 / "source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"
    collector = C5 / "native_full_step_collector.py"
    if sha256(reactor.read_bytes()).hexdigest() != SHAS["reactor"] or sha256(collector.read_bytes()).hexdigest() != SHAS["collector"]:
        raise ValueError("frozen C5 source drift")
    original_refs = []
    for path in sorted(C5.rglob("*.py")):
        if "__pycache__" in path.parts: continue
        relative = path.relative_to(C5)
        raw = path.read_bytes()
        original_refs.append(dict(path=relative.as_posix(), bytes=len(raw), sha256=sha256(raw).hexdigest()))
        # Receipt is the one new module below; policy, bridge and engine bytes stay.
        if relative.as_posix().endswith("/p4_single_file_receipt.py"): continue
        put(HERE / "common_candidate" / relative, raw)
    native_text = (V6 / "native_conditional_cost.py").read_text(encoding="utf-8")
    native_text = helper(native_text, "ORIGINAL_BYTES =")
    native_text = guard_functions(native_text, ["verify_native_cell"])
    native_text += '\n\ndef main(argv=None):\n    print(json.dumps(blocked_document(), sort_keys=True))\n    return 2\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
    put(HERE / "native_conditional_cost.py", native_text.encode())
    serializer = (V6 / "prepare_and_verify_native_cost.py").read_text(encoding="utf-8")
    serializer = helper(serializer, "_spec =")
    serializer = guard_functions(serializer, ["create_plan", "main"], main=True)
    serializer = serializer.replace('def serialize_runtime_record(root, *, runtime_relative, plan_ref, source_verification_refs):',
        'def serialize_runtime_record(root, *, runtime_relative, plan_ref, source_verification_refs, synthetic_cpu=False):')
    serializer = serializer.replace('def verify_raw_window(root, *, plan_ref, child_relative):',
        'def verify_raw_window(root, *, plan_ref, child_relative, synthetic_cpu=False):')
    for name in ("serialize_runtime_record", "verify_raw_window"):
        tree = ast.parse(serializer); node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        first = node.body[0]; line = first.end_lineno if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str) else first.lineno - 1
        lines = serializer.splitlines(keepends=True)
        lines[line:line] = ['    require(synthetic_cpu is True, "explicit synthetic CPU reserialization only")\n']
        serializer = "".join(lines)
    marker = "    native = plan['native_source_ref']\n"
    serializer = serializer.replace(marker, '''    require(plan.get('cpu_preparation_only') is True and plan.get('evidence_origin') == 'synthetic_cpu_contract',
            'new C5 explicitly synthetic plan required')
    require(plan.get('job_id') == 'server11-c5-native-common-cost-preparation01' and
            plan.get('common_overlay_relative') == 'artifacts/prefix_io_v1/server11-c5-native-cost-preparation-cpu-20261004/common_candidate/source',
            'new C5 common calibration identity; old C4/v6 refused')
    native = plan['native_source_ref']
    require(native.get('sha256') == 'a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47',
            'actual unchanged C5 native source required')
    collector = plan['collector_source_ref']
    require(collector.get('path') == 'artifacts/prefix_io_v1/server11-c5-native-cost-preparation-cpu-20261004/common_candidate/native_full_step_collector.py' and
            collector.get('sha256') == 'bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf' and ref(root, collector['path']) == collector,
            'actual unchanged C5 plan collector required')
''', 1)
    serializer = serializer.replace('return dict(scope="server11_native_paired_measurements_v1", origin="native_gpu_recording", plan_ref=plan_ref,',
        'return dict(scope="server11_c5_synthetic_raw_reconstruction_v1", origin="synthetic_cpu_contract",\n        native_execution_verified=False, conditional_cost_cell_qualified=False, native_cost_qualified=False,\n        full_runtime_cost_qualified=False, on_observation_cost_measured=False, valid_native_receipt=None,\n        effective_cost_upper_ns=None, effective_step_budget_ns=None, gpu_launch_allowed=False, plan_ref=plan_ref,')
    serializer = serializer.replace('return dict(status="RAW_WINDOW_SEMANTICS_PASS_FINAL_GUARD_PENDING", window_index=index,',
        'return dict(status="PASS_C5_SYNTHETIC_RAW_WINDOW_ONLY", origin="synthetic_cpu_contract",\n        native_execution_verified=False, native_cost_qualified=False, valid_native_receipt=None,\n        full_runtime_cost_qualified=False, on_observation_cost_measured=False, gpu_launch_allowed=False,\n        effective_cost_upper_ns=None, effective_step_budget_ns=None, window_index=index,')
    put(HERE / "prepare_and_verify_native_cost.py", serializer.encode())
    calibrator = (V6 / "run_native_cost_experiment.py").read_text(encoding="utf-8")
    calibrator = calibrator.replace("artifacts/prefix_io_v1/server11-native-cost-v6-20261003", DELIVERY)
    calibrator = calibrator.replace("artifacts/prefix_io_v1/server11-p4-single-file-candidate-v4-20261003", CANDIDATE)
    calibrator = calibrator.replace("GPU_UUID = 'GPU-38c1d1c7-2aba-3fa9-e1cd-fe4d3ae569c9'", "GPU_UUID = None")
    calibrator = calibrator.replace("LABEL = 'server11-native-cost-six-window-06'", "LABEL = '" + LABEL + "'")
    calibrator = helper(calibrator, "ROOT =")
    calibrator = guard_functions(calibrator, ["load_configuration", "verify_guard", "execute_window", "execute_parent", "main"], main=True)
    put(HERE / "run_native_cost_experiment.py", calibrator.encode())
    controller = (V6 / "control_native_cost_job.py").read_text(encoding="utf-8")
    controller = controller.replace("artifacts/prefix_io_v1/server11-native-cost-v6-20261003", DELIVERY)
    controller = controller.replace("artifacts/prefix_io_v1/server11-p4-single-file-candidate-v4-20261003", CANDIDATE)
    controller = controller.replace("GPU = 'GPU-38c1d1c7-2aba-3fa9-e1cd-fe4d3ae569c9'", "GPU = None")
    controller = controller.replace("LABEL = 'server11-native-cost-six-window-06'", "LABEL = '" + LABEL + "'")
    controller = helper(controller, "ROOT =")
    controller = guard_functions(controller, ["prepare", "scope", "launch", "status", "check_sources", "main"], main=True)
    # Never replay or retain the old human authorization text as a new scope.
    tree = ast.parse(controller); node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "scope")
    lines = controller.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = ["def scope(plan_relative):\n    block_gpu_start()\n"]
    controller = "".join(lines)
    put(HERE / "control_native_cost_job.py", controller.encode())
    old_receipt = C5 / "source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py"
    receipt = old_receipt.read_text(encoding="utf-8")
    receipt = receipt.replace("artifacts/prefix_io_v1/server11-native-cost-v6-20261003/", DELIVERY + "/")
    receipt = receipt.replace("artifacts/prefix_io_v1/server11-p4-single-file-candidate-v4-20261003/source", CANDIDATE + "/source")
    receipt = receipt.replace('"server11-native-cost-six-window-06"', '"' + LABEL + '"')
    receipt = receipt.replace('"_p4_single_file_v6_serializer_"', '"_p4_single_file_c5_preparation_serializer_"')
    receipt = helper(receipt, "_D6 =")
    receipt = guard_functions(receipt, ["load_verified_single_file"])
    for name, filename in (("_VERIFIER", "native_conditional_cost.py"), ("_SERIALIZER", "prepare_and_verify_native_cost.py")):
        raw = (HERE / filename).read_bytes()
        pattern = name + r" = FileRef\(_D6 \+ \"[^\"]+\", \d+,\n    \"[0-9a-f]+\"\)"
        replacement = name + ' = FileRef(_D6 + "' + filename + '", ' + str(len(raw)) + ',\n    "' + sha256(raw).hexdigest() + '")'
        receipt, count = re.subn(pattern, replacement, receipt)
        if count != 1: raise ValueError("new dependency pin marker")
    receipt = receipt.replace('    common, overlay = result\n', '''    common, overlay = result
    collector = FileRef.from_mapping(plan['collector_source_ref'])
    _require(collector.path == _COMMON_OVERLAY.removesuffix('/source') + '/native_full_step_collector.py' and
        collector.sha256 == 'bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf' and
        locked.get(collector.path) == collector.mapping() and
        sum(ref.mapping() == collector.mapping() for ref in overlay) == 1 and
        all(ref.path != collector.path for ref in common), 'same C5 plan collector uniquely in runtime overlay')
''')
    receipt = receipt.replace('    native.read(project)\n', '''    _require(native.sha256 == 'a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47',
        'actual C5 native common source; old C4/v6 refused')
    native.read(project)
''', 1)
    receipt += '\n\ndef main(argv=None):\n    print(json.dumps(blocked_document(), sort_keys=True))\n    return 2\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
    put(HERE / "common_candidate/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py", receipt.encode())
    raw = json.dumps(dict(status="CPU_NATIVE_CODE_PREPARATION_ONLY", actual_gpu_runs=0,
        gpu_launch_allowed=False, source_parent_files=original_refs,
        numeric_functions_changed=False, policy_bridge_reactor_collector_bytes_unchanged=True,
        canonical_receipt_replaced_in_new_copy_only=True), indent=2, sort_keys=True).encode() + b"\n"
    put(HERE / "SOURCE_INHERITANCE.json", raw)
    print(json.dumps(dict(copied_candidate_python_files=len(original_refs), new_native_preparation=True,
        actual_gpu_runs=0, gpu_launch_allowed=False)))


if __name__ == "__main__":
    main()
