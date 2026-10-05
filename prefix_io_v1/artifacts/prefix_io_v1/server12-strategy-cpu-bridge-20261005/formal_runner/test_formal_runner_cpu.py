"""CPU dispatch and identity tests; fixtures never become formal input evidence."""
from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent / "gpu_prerental_preparation_20261004"
if not BASE.is_dir():
    BASE = HERE.parent.parent / "server12-gpu-prerental-preparation-20261004"
RUNNER = HERE / "strong_trace_runner_v3.py"
if not RUNNER.is_file():
    RUNNER = BASE / "runner/strong_trace_runner_v3.py"
RUNTIME = HERE.parent / "formal_runtime/native_runtime_v3.py"
if not RUNTIME.is_file():
    RUNTIME = BASE / "runner/native_runtime_v3.py"
spec = importlib.util.spec_from_file_location("_formal_runner_cpu_candidate", RUNNER)
R = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R
spec.loader.exec_module(R)


def record(index=2):
    return dict(request_id=index, prompt_sha256="a" * 64, prompt_token_ids=[index + 9, 17],
                scheduled_ns=index * 100, max_tokens=128, min_tokens=128, seed=3,
                split="development", prefix_family="CPU_FIXTURE_FAMILY_" + str(index))


def pair():
    return {arm: dict(engine=dict(skip_tokenizer_init=True, kv_transfer_config=dict(
        kv_connector_extra_config=dict(shared_storage_path="/CPU_FIXTURE/" + arm,
            prefix_io_p4_policy={}, prefix_io_parent_admission=dict(run_id="CPU_FIXTURE_" + arm)))))
            for arm in ("U", "I")}


def metadata_fixture(role=("development", "shadow", "I")):
    phase, mode, arm = role
    cfg = dict(phase=phase, mode=mode, arm=arm, run_id="CPU_FIXTURE_" + arm, gpu_uuid="GPU-CPU_FIXTURE",
        workload_ref={"CPU_FIXTURE": True}, formal_trace_binding_ref={"CPU_FIXTURE": True},
        activation_ref={"CPU_FIXTURE": True}, runtime_ref=dict(path="prep/runner/native_runtime_v3.py"))
    cfg["formal_peer_closed_ref"] = None if arm == "U" else dict(CPU_FIXTURE=True)
    p = pair()
    records = [record(), record(3)]
    partition = "evaluation" if phase == "effect" else "development"
    for row in records:
        row["split"] = partition
    original = dict(schema=R.FORMAL_SCHEMA, records=[record(0), record(1)] + deepcopy(records),
                    model_manifest_sha256="b" * 64, workload_sha256="c" * 64, max_concurrency=1)
    binding = dict(schema="formal_natural_token_id_partition_CPU_binding_v1", partition=partition,
        records=deepcopy(records), max_concurrency=1, complete_selected_record_count=2,
        source_request_ids=[2, 3], common_runtime_domain_sha256=R.common_domain_sha(p),
        independent_deadline_ref=dict(CPU_FIXTURE=True), skip_tokenizer_init=True, gpu_eligible=False,
        original_manifest_workload_sha256="c" * 64)
    selected = deepcopy(original)
    selected["records"] = [dict(deepcopy(row), prompt=None) for row in records]
    refs = {}
    finite = dict(phase=phase, actual_table_issued=False, formal_effect_qualified=False, runtime_refs=refs,
        descriptor=dict(independent_deadline_ref=binding["independent_deadline_ref"]),
        calibration_plan=dict(common_runtime_domain_sha256=R.common_domain_sha(p), gpu_uuid=cfg["gpu_uuid"]),
        independent_budget=dict(observation_only=True, ordinary_I_authorized=False, reserve_not_measured_yet=True))
    gates = dict(config=cfg, refs=refs, pair=p, workload=selected, formal_original_workload=original,
        formal_workload_binding=binding, finite_activation=finite)
    return gates


def phase_summary(config):
    return dict(schema="formal_trace_phase_CPU_preflight_v1", role=[config[k] for k in ("phase", "mode", "arm")],
        input_bindings_validated=True, gpu_eligible=False, formal_goodput_allowed=False)


def frontend_fixture(records):
    rows = []
    for rec in records:
        times = list(range(1000, 1128))
        rows.append(dict(request_id=str(rec["request_id"]), state="COMPLETED", prompt_sha256=rec["prompt_sha256"],
            actual_prompt_token_ids=rec["prompt_token_ids"].copy(), output_token_ids=list(range(128)),
            token_return_ns=times, itl_ns=[1] * 127))
    return dict(status="PASS_COMPLETE_ORIGINAL_REQUEST_OUTPUTS", actual_request_stream_completed=True,
        planned_requests=len(rows), successful_requests=len(rows), failure_or_unsubmitted_requests=0, rows=rows)


class DispatchTests(unittest.TestCase):
    def test_metadata_import_and_help_have_no_runtime_dependency(self):
        code = """import sys,importlib.util
class RejectRuntime:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','vllm','py_kvcache','numpy','ctypes'):
   raise AssertionError('forbidden CPU import: '+fullname)
sys.meta_path.insert(0,RejectRuntime())
s=importlib.util.spec_from_file_location('cpu_formal',%r)
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
try:m.main(['--help'])
except SystemExit as e:assert e.code==0
print('PASS_METADATA_NO_RUNTIME_IMPORT')
""" % str(RUNNER)
        proc = subprocess.run([sys.executable, "-B", "-I", "-S", "-c", code], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("PASS_METADATA_NO_RUNTIME_IMPORT", proc.stdout)

    def test_original_executor_drive_function_is_byte_identical(self):
        old = (BASE / "runner/strong_trace_runner_v2.py").read_text(encoding="utf-8")
        new = RUNNER.read_text(encoding="utf-8")
        def function(text):
            return ast.get_source_segment(text, next(node for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name == "drive_original_engine"))
        self.assertEqual(function(old), function(new))

    def test_formal_roles_are_bounded_and_old_qualification_is_separate(self):
        for role in (("development", "off", "U"), ("development", "shadow", "I"),
                     ("effect", "off", "U"), ("effect", "on", "I")):
            cfg = metadata_fixture(role)["config"]
            self.assertEqual(R.formal_role(cfg), "evaluation" if role[0] == "effect" else "development")
            self.assertEqual(R.check_phase(cfg, gap={}, formal=True), {})
        for role in (("qualification", "off", "U"), ("calibration", "off", "U"),
                     ("development", "on", "I"), ("effect", "shadow", "I")):
            with self.assertRaises(ValueError):
                R.formal_role(metadata_fixture(role)["config"])
        self.assertEqual(R.check_phase(dict(phase="qualification", mode="off", arm="U"), gap={}), {})

    def test_missing_actual_refs_cannot_be_accepted_as_formal_role(self):
        cfg = metadata_fixture()["config"]
        for key in ("formal_trace_binding_ref", "activation_ref"):
            changed = dict(cfg, **{key: None})
            with self.assertRaisesRegex(ValueError, "closed real formal"):
                R.formal_role(changed)

    def test_legacy_validator_rejects_formal_schema(self):
        with self.assertRaisesRegex(ValueError, "qualification workload"):
            R.validate_workload(metadata_fixture()["formal_original_workload"])

    def test_actual_formal_module_pin_is_checked_without_importing_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = "prep/formal_trace_binding/formal_trace_binding.py"
            bridge = "prep/formal_runtime_bridge/namespace_bridge.py"
            for name, original in ((relative, BASE / "formal_trace_binding/formal_trace_binding.py"),
                                   (bridge, HERE / "namespace_bridge.py")):
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(original.read_bytes())
            row = R.ref(root, relative)
            refs = {name: R.ref(root, name) for name in (relative, bridge)}
            cfg = dict(metadata_fixture()["config"], runtime_ref=dict(path="prep/runner/native_runtime_v3.py"))
            module = R.formal_binding_module(root, cfg, refs)
            self.assertEqual(module.PROTOCOL_SHA, "7fbc544597c9abe334122d354e7494b590e810ab0682f1d2d155408cc4f618eb")
            with self.assertRaisesRegex(ValueError, "exact unchanged"):
                R.formal_binding_module(root, cfg, dict(refs, **{relative: dict(row, sha256="0" * 64)}))

    def test_select_preserves_ids_order_arrivals_and_full_manifest_without_freeze(self):
        gates = metadata_fixture()
        original = deepcopy(gates["formal_original_workload"])
        api = NS(validate_formal_workload=mock.Mock(return_value=gates["formal_workload_binding"]))
        with mock.patch.object(R, "formal_binding_module", return_value=api):
            selected, binding = R.select_formal_workload(Path.cwd(), gates["config"], {}, gates["pair"], original)
        self.assertEqual(original, gates["formal_original_workload"])
        self.assertEqual(selected["records"], gates["workload"]["records"])
        self.assertEqual([row["scheduled_ns"] for row in selected["records"]], [200, 300])
        self.assertEqual(selected["workload_sha256"], original["workload_sha256"])
        self.assertTrue(all(row["prompt"] is None for row in selected["records"]))
        self.assertFalse(binding["gpu_eligible"])
        self.assertEqual(api.validate_formal_workload.call_args.kwargs["partition"], "development")

    def test_more_than128_outputs_is_rejected_before_costly_original_capture(self):
        gates = metadata_fixture()
        gates["formal_workload_binding"]["records"][0].update(min_tokens=256, max_tokens=256)
        api = NS(validate_formal_workload=mock.Mock(return_value=gates["formal_workload_binding"]))
        with mock.patch.object(R, "formal_binding_module", return_value=api):
            with self.assertRaisesRegex(ValueError, "complete128 outputs before GPU"):
                R.select_formal_workload(Path.cwd(), gates["config"], {}, gates["pair"], gates["formal_original_workload"])

    def test_original_token_id_branch_gets_actual_selected_lists(self):
        records = metadata_fixture()["workload"]["records"]
        class Engine:
            def __init__(self): self.active = {}; self.submitted = []
            def has_unfinished_requests(self): return bool(self.active)
            def add_request(self, rid, prompt, sampling):
                self.submitted.append(deepcopy(prompt)); self.active[rid] = (prompt, [])
                return rid + "-CPU_FIXTURE_NATIVE"
            def step(self):
                outputs = []
                for rid, (prompt, ids) in list(self.active.items()):
                    ids.append(len(ids)); done = len(ids) == 128
                    outputs.append(NS(request_id=rid, prompt_token_ids=prompt["prompt_token_ids"],
                        outputs=[NS(token_ids=ids.copy(), finish_reason="length" if done else None)],
                        finished=done, num_cached_tokens=0))
                    if done: self.active.pop(rid)
                return outputs
        engine = Engine()
        ticks = iter(range(100000, 10000000, 1000))
        result = R.drive_original_engine(engine, lambda rec: None, records,
            max_concurrency=1, deadline_seconds=5, clock=lambda: next(ticks), sleeper=lambda sec: None)
        self.assertEqual(result["status"], "PASS_COMPLETE_ORIGINAL_REQUEST_OUTPUTS")
        self.assertEqual(engine.submitted, [{"prompt_token_ids": row["prompt_token_ids"]} for row in records])
        self.assertEqual(len(result["rows"]), 2)
        self.assertFalse(result["formal_goodput_allowed"])

    def test_development_calls_original_formal_api_and_no_native_replay(self):
        gates = metadata_fixture()
        summary = phase_summary(gates["config"])
        api = NS(validate_formal_phase=mock.Mock(return_value=summary))
        with mock.patch.object(R, "select_formal_workload", return_value=(gates["workload"], gates["formal_workload_binding"])), \
             mock.patch.object(R, "formal_binding_module", return_value=api), mock.patch.object(R, "load") as load:
            self.assertEqual(R.preflight_formal_runtime(Path.cwd(), gates), summary)
        self.assertFalse(load.called)
        self.assertIsNone(api.validate_formal_phase.call_args.kwargs["development_replay"])
        self.assertFalse(gates["formal_phase_preflight"]["gpu_eligible"])

    def test_effect_both_arms_call_original_actual_replay_before_phase_api(self):
        for role in (("effect", "off", "U"), ("effect", "on", "I")):
            gates = metadata_fixture(role)
            replay = dict(CPU_DISPATCH_FIXTURE_ONLY=True)
            calls = []
            runtime = NS(preflight_formal_reserve_replay=lambda *a, **k: (calls.append("original_replay") or replay))
            api = NS(validate_formal_phase=lambda *a, **k: (calls.append("phase_after_replay") or phase_summary(gates["config"])))
            with mock.patch.object(R, "select_formal_workload", return_value=(gates["workload"], gates["formal_workload_binding"])), \
                 mock.patch.object(R, "formal_binding_module", return_value=api), mock.patch.object(R, "load", return_value=runtime):
                R.preflight_formal_runtime(Path.cwd(), gates)
            self.assertEqual(calls, ["original_replay", "phase_after_replay"])
            self.assertEqual(gates["development_reserve_replay"], replay)
            self.assertFalse(gates["formal_phase_preflight"]["gpu_eligible"])

    def test_development_missing_finite_or_changed_deadline_domain_rejected(self):
        for kind in ("missing", "phase", "deadline", "domain", "device", "fabricated_reserve"):
            gates = metadata_fixture()
            if kind == "missing": gates["finite_activation"] = None
            elif kind == "phase": gates["finite_activation"]["phase"] = "effect"
            elif kind == "deadline": gates["finite_activation"]["descriptor"]["independent_deadline_ref"] = {}
            elif kind == "domain": gates["finite_activation"]["calibration_plan"]["common_runtime_domain_sha256"] = "0" * 64
            elif kind == "device": gates["finite_activation"]["calibration_plan"]["gpu_uuid"] = "OTHER_CPU_FIXTURE"
            else: gates["finite_activation"]["independent_budget"]["reserve_not_measured_yet"] = False
            with mock.patch.object(R, "select_formal_workload", return_value=(gates["workload"], gates["formal_workload_binding"])):
                with self.assertRaises(ValueError): R.preflight_formal_runtime(Path.cwd(), gates)

    def test_effect_original_replay_failure_is_not_a_successful_preflight(self):
        gates = metadata_fixture(("effect", "off", "U"))
        runtime = NS(preflight_formal_reserve_replay=mock.Mock(side_effect=ValueError("actual reserve missing")))
        with mock.patch.object(R, "select_formal_workload", return_value=(gates["workload"], gates["formal_workload_binding"])), \
             mock.patch.object(R, "load", return_value=runtime):
            with self.assertRaisesRegex(ValueError, "actual reserve missing"):
                R.preflight_formal_runtime(Path.cwd(), gates)
        self.assertNotIn("formal_phase_preflight", gates)

    def test_selection_tampering_is_rejected_before_replay(self):
        gates = metadata_fixture()
        actual = deepcopy(gates["workload"])
        gates["workload"]["records"][0]["prompt_token_ids"] = [99]
        with mock.patch.object(R, "select_formal_workload", return_value=(actual, gates["formal_workload_binding"])):
            with self.assertRaisesRegex(ValueError, "forged selected"):
                R.preflight_formal_runtime(Path.cwd(), gates)

    def test_full_frontend_id_closure_preserves_all_requests(self):
        gates = metadata_fixture()
        result = R.validate_formal_frontend_records(gates, frontend_fixture(gates["formal_workload_binding"]["records"]))
        self.assertEqual(result["successful_requests"], 2)
        self.assertTrue(result["actual_original_prompt_ids_verified"])
        self.assertFalse(result["formal_goodput_allowed"])

    def test_wrong_ids_order_partial_outputs_and_clock_loss_are_rejected(self):
        gates = metadata_fixture()
        for kind in ("prompt", "order", "drop", "tokens", "clock", "ITL", "failed", "bool_count"):
            frontend = frontend_fixture(gates["formal_workload_binding"]["records"])
            if kind == "prompt": frontend["rows"][0]["actual_prompt_token_ids"] = [99]
            elif kind == "order": frontend["rows"].reverse()
            elif kind == "drop": frontend["rows"].pop()
            elif kind == "tokens": frontend["rows"][0]["output_token_ids"].pop()
            elif kind == "clock": frontend["rows"][0]["token_return_ns"][3] = 1
            elif kind == "ITL": frontend["rows"][0]["itl_ns"][3] = 2
            elif kind == "failed": frontend["failure_or_unsubmitted_requests"] = 1
            else: frontend["successful_requests"] = True
            with self.assertRaises(ValueError): R.validate_formal_frontend_records(gates, frontend)

    def test_old_qualification_frontend_skips_formal_hook(self):
        self.assertIsNone(R.validate_formal_frontend_records(dict(config={}), {}))
        self.assertIsNone(R.preflight_formal_runtime(Path.cwd(), dict(config={}, refs={}, pair={})))

    def test_old_p3_status_and_unclosed_formal_child_cannot_be_U_prerequisite(self):
        gates = metadata_fixture()
        for document in (dict(status="PASS_STRONG_NATIVE_OFF_WORKLOAD_LIFECYCLE"),
                         dict(status="PASS_FORMAL_DEVELOPMENT_U_WORKLOAD_ONLY_REQUIRES_GUARD_CLOSURE",
                              phase="development", mode="off", arm="U", os_session_drained=False)):
            cfg = dict(gates["config"], off_qualification_ref={"CPU_FIXTURE": True})
            with mock.patch.object(R, "read", return_value=document), mock.patch.object(R, "check_ref", return_value=Path("CPU_FIXTURE")):
                with self.assertRaisesRegex(ValueError, "actual same-formal-input U development"):
                    R.verify_formal_off_prerequisite(Path.cwd(), cfg, {}, gates["pair"], gates["formal_workload_binding"])

    def test_incomplete_real_config_fails_without_any_module_load_or_GPU_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "config.json"
            path.write_text(json.dumps(dict(schema=R.SCHEMA, phase="effect", mode="off", arm="U")))
            with mock.patch.object(R, "load") as load:
                with self.assertRaisesRegex(ValueError, "strict bounded run config"):
                    R.verify_configuration(root, path)
            self.assertFalse(load.called)

    def test_actual_runtime_header_and_status_expressions_match_peer_consumer(self):
        # Compile only the actual source's bounded dict/status expressions;
        # never call execute or manufacture a runtime/guard qualification.
        runtime_path = RUNTIME
        module = ast.parse(runtime_path.read_text(encoding="utf-8"))
        execute = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
        header = next(node.value for node in execute.body if isinstance(node, ast.Assign) and
                      any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets))
        statuses = next(node.value for node in ast.walk(execute) if isinstance(node, ast.Assign) and
                        any(isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and
                            target.value.id == "result" and isinstance(target.slice, ast.Constant) and
                            target.slice.value == "status" for target in node.targets) and isinstance(node.value, ast.IfExp))
        expected = {("development", "off", "U"): "PASS_FORMAL_DEVELOPMENT_U_WORKLOAD_ONLY_REQUIRES_GUARD_CLOSURE",
                    ("development", "shadow", "I"): "PASS_FINITE_DEVELOPMENT_SHADOW_REQUIRES_RESERVE_GUARD_JOIN",
                    ("effect", "off", "U"): "PASS_FORMAL_EFFECT_U_WORKLOAD_ONLY_REQUIRES_GUARD_CLOSURE",
                    ("effect", "on", "I"): "PASS_FINITE_QUALIFIED_I_WORKLOAD_LIFECYCLE_REQUIRES_EFFECT_ANALYSIS"}
        for role, status in expected.items():
            gates = metadata_fixture(role)
            config = gates["config"]
            for key in ("runner_ref", "source_lock_ref", "pair_config_ref"):
                config[key] = dict(CPU_SOURCE_EXPRESSION_ONLY=True)
            gates.update(common_runtime_domain_sha256=R.common_domain_sha(gates["pair"]),
                         standing_authorization_ref=dict(CPU_SOURCE_EXPRESSION_ONLY=True),
                         actual_run_config_ref=dict(path="CPU_SOURCE_EXPRESSION_ONLY/config.json", bytes=0, sha256="0" * 64))
            env = dict(config=config, gates=gates, route=dict(formal=True), storage=Path("CPU_SOURCE_EXPRESSION_ONLY/storage"),
                       reservation=dict(id="CPU_SOURCE_EXPRESSION_ONLY"))
            actual = eval(compile(ast.Expression(header), str(runtime_path), "eval"), {}, env)
            self.assertEqual(actual["actual_run_config_ref"], gates["actual_run_config_ref"])
            self.assertEqual(actual["formal_partition"], gates["formal_workload_binding"]["partition"])
            self.assertEqual(actual["formal_initial_namespace"]["arm"], role[2])
            self.assertFalse(actual["os_session_drained"])
            self.assertFalse(actual["formal_effect_qualified"])
            self.assertEqual(eval(compile(ast.Expression(statuses), str(runtime_path), "eval"), {}, env), status)

    def test_all_new_bridge_runtime_calls_are_actual_candidate_exports(self):
        runtime = ast.parse(RUNTIME.read_text(encoding="utf-8"))
        exports = {node.name for node in runtime.body if isinstance(node, ast.FunctionDef)}
        bridge = ast.parse((HERE / "namespace_bridge.py").read_text(encoding="utf-8"))
        runner = ast.parse(RUNNER.read_text(encoding="utf-8"))
        calls = {node.func.attr for tree in (bridge, runner) for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                 isinstance(node.func.value, ast.Name) and node.func.value.id == "runtime"}
        self.assertTrue(calls)
        self.assertLessEqual(calls, exports)


class NamespaceTests(unittest.TestCase):
    """Lifetime dispatch checks; mocked peer approval is no native evidence."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve(strict=True)
        self.config = metadata_fixture(("development", "off", "U"))["config"]
        self.names = {partition: {arm: (self.root / "experiments/prefix_io_v1/runs" /
            ("CPU_NAMESPACE_FIXTURE_" + partition + "_" + arm) / "storage").as_posix()
            for arm in ("U", "I")} for partition in ("calibration", "development", "evaluation")}
        self.pair = pair()
        for arm in ("U", "I"):
            self.pair[arm]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["shared_storage_path"] = self.names["development"][arm]
        self.manifest = dict(workload_sha256="a" * 64, model_manifest_sha256="b" * 64,
                             tokenizer_receipt_digest="c" * 64)
        self.contract = dict(schema="formal_trace_namespace_contract_v1", **self.manifest,
            common_runtime_domain_sha256=R.common_domain_sha(self.pair),
            initial_cache_state="fresh_equal_namespace_preserved_through_whole_partition",
            no_per_request_reset=True, partition_namespaces=self.names)
        spec = importlib.util.spec_from_file_location("_cpu_namespace_bridge_" + str(id(self)), HERE / "namespace_bridge.py")
        self.bridge = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.bridge
        spec.loader.exec_module(self.bridge)

    def tearDown(self):
        self.temporary.cleanup()

    def module(self, config=None):
        name = "_cpu_original_formal_" + str(id(self)) + "_" + str(len(sys.modules))
        spec = importlib.util.spec_from_file_location(name, BASE / "formal_trace_binding/formal_trace_binding.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return self.bridge.bind(module, config=config or self.config, refs={}, pair=self.pair, driver=R)

    def validate(self, module):
        return module.validate_namespace(self.contract, root=self.root, manifest=self.manifest,
                                         pair=self.pair, partition="development")

    def test_fresh_U_arm_keeps_original_full_partition_namespace_contract(self):
        module = self.module()
        self.assertEqual(self.validate(module), self.names["development"])
        self.assertTrue(all(not Path(path).exists() for names in self.names.values() for path in names.values()))

    def test_consumer_loads_closed_parent_then_delegates_same_pure_document_api(self):
        row = dict(path="CPU_DISPATCH_FIXTURE_ONLY.json", bytes=1, sha256="0" * 64)
        config = dict(self.config, formal_peer_closed_ref=row)
        parent = dict(CPU_DISPATCH_FIXTURE_ONLY=True)
        with mock.patch.object(R, "read", return_value=parent), \
             mock.patch.object(R, "check_ref", return_value=Path("CPU_DISPATCH_FIXTURE_ONLY")), \
             mock.patch.object(self.bridge, "verify_closed_peer_document", return_value=parent) as verify:
            result = self.bridge.verify_closed_peer(self.root, config, {row["path"]: row}, self.pair,
                partition="development", peer_arm="U", peer_storage=self.names["development"]["U"],
                manifest=self.manifest, driver=R)
        self.assertIs(result, parent)
        self.assertIs(verify.call_args.args[0], parent)
        self.assertEqual(verify.call_args.kwargs["peer_arm"], "U")
        self.assertEqual(verify.call_args.kwargs["partition"], "development")

    def test_pure_document_api_cannot_omit_actual_child_guard_or_native_evidence(self):
        with self.assertRaisesRegex(ValueError, "actual U lifecycle/domain/guard closure"):
            self.bridge.verify_closed_peer_document({}, self.root, self.config, {}, self.pair,
                partition="development", peer_arm="U", peer_storage=self.names["development"]["U"],
                manifest=self.manifest, driver=R)

    def test_completed_U_peer_without_actual_guard_raw_closure_is_rejected(self):
        Path(self.names["development"]["U"]).mkdir(parents=True)
        cfg = dict(self.config, arm="I", mode="shadow", formal_peer_closed_ref=dict(CPU_FIXTURE=True))
        with self.assertRaisesRegex(ValueError, "actual source-closed U peer"):
            self.validate(self.module(cfg))
        self.assertTrue(Path(self.names["development"]["U"]).exists())

    def test_formal_I_cannot_treat_missing_U_namespace_as_fresh_pair(self):
        cfg = dict(self.config, arm="I", mode="shadow", formal_peer_closed_ref=dict(CPU_FIXTURE=True))
        with self.assertRaisesRegex(ValueError, "peer namespace must be retained"):
            self.validate(self.module(cfg))

    def test_effect_AB_and_BA_first_arm_both_retain_fresh_peer_rule(self):
        for arm, mode in (("U", "off"), ("I", "on")):
            cfg = dict(self.config, phase="effect", arm=arm, mode=mode, formal_peer_closed_ref=None)
            R.formal_role(cfg)
            for name in ("U", "I"):
                self.pair[name]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["shared_storage_path"] = self.names["evaluation"][name]
            module = self.module(cfg)
            result = module.validate_namespace(self.contract, root=self.root, manifest=self.manifest,
                pair=self.pair, partition="evaluation")
            self.assertEqual(result, self.names["evaluation"])
            self.assertTrue(all(not Path(path).exists() for path in result.values()))

    def test_effect_BA_second_U_consumes_closed_I_peer_without_deleting_it(self):
        peer = Path(self.names["evaluation"]["I"])
        peer.mkdir(parents=True)
        cfg = dict(self.config, phase="effect", arm="U", mode="off", formal_peer_closed_ref=dict(CPU_FIXTURE=True))
        R.formal_role(cfg)
        for name in ("U", "I"):
            self.pair[name]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["shared_storage_path"] = self.names["evaluation"][name]
        with mock.patch.object(self.bridge, "verify_closed_peer", return_value=dict(CPU_DISPATCH_FIXTURE_ONLY=True)) as gate:
            module = self.module(cfg)
            result = module.validate_namespace(self.contract, root=self.root, manifest=self.manifest,
                pair=self.pair, partition="evaluation")
        self.assertEqual(result, self.names["evaluation"])
        self.assertEqual(gate.call_args.kwargs["peer_arm"], "I")
        self.assertEqual(gate.call_args.kwargs["partition"], "evaluation")
        self.assertTrue(peer.exists())
        self.assertFalse(Path(self.names["evaluation"]["U"]).exists())

    def test_peer_dispatch_uses_exact_partition_path_and_preserves_all_paths(self):
        peer = Path(self.names["development"]["U"])
        peer.mkdir(parents=True)
        cfg = dict(self.config, arm="I", mode="shadow", formal_peer_closed_ref=dict(CPU_FIXTURE=True))
        with mock.patch.object(self.bridge, "verify_closed_peer", return_value=dict(CPU_DISPATCH_FIXTURE_ONLY=True)) as gate:
            self.assertEqual(self.validate(self.module(cfg)), self.names["development"])
        self.assertEqual(gate.call_args.kwargs["partition"], "development")
        self.assertEqual(gate.call_args.kwargs["peer_storage"], peer.as_posix())
        self.assertTrue(peer.exists())
        self.assertFalse(Path(self.names["development"]["I"]).exists())

    def test_existing_current_arm_always_fails_even_with_peer_gate(self):
        for path in self.names["development"].values(): Path(path).mkdir(parents=True)
        cfg = dict(self.config, arm="I", mode="shadow", formal_peer_closed_ref=dict(CPU_FIXTURE=True))
        with mock.patch.object(self.bridge, "verify_closed_peer", return_value=dict(CPU_DISPATCH_FIXTURE_ONLY=True)):
            with self.assertRaisesRegex(ValueError, "fresh selected current-arm"):
                self.validate(self.module(cfg))

    def test_alias_reset_and_foreign_domain_are_not_relaxed_for_peer_lifetime(self):
        module = self.module()
        for kind in ("alias", "reset", "domain"):
            old = deepcopy(self.contract)
            if kind == "alias": self.contract["partition_namespaces"]["evaluation"]["I"] = self.names["calibration"]["U"]
            elif kind == "reset": self.contract["no_per_request_reset"] = False
            else: self.contract["common_runtime_domain_sha256"] = "0" * 64
            with self.assertRaises(ValueError): self.validate(module)
            self.contract = old

    def test_development_None_SLO_is_diagnostic_only_and_original_source_stays_strict(self):
        diagnostic = self.module()
        self.assertIsNone(diagnostic.require_independent_service_SLO(None))
        with self.assertRaises(ValueError): diagnostic.require_independent_service_SLO({})
        effect = self.module(dict(self.config, phase="effect"))
        with self.assertRaisesRegex(ValueError, "SLO missing"):
            effect.require_independent_service_SLO(None)
        self.assertIsNot(diagnostic, effect)
        service = dict(schema="independent_service_SLO_v1", origin="independent_requirement_before_development",
                       TTFT_ns=100, request_ITL_P95_ns=10)
        self.assertEqual(diagnostic.require_independent_service_SLO(service), service)
        self.assertEqual(effect.require_independent_service_SLO(service), service)


if __name__ == "__main__":
    unittest.main()
