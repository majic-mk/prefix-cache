"""CPU interface fixtures only; none are GPU runtime or performance evidence."""
from __future__ import annotations
import ast
import __future__
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import cast
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = load("_cpu_strong_trace_driver", "strong_trace_runner.py")
N = load("_cpu_strong_trace_native_adapter", "native_runtime.py")


def fixture_record(identifier=0, arrival=0):
    text = "cpu fixture prompt " + str(identifier)
    return dict(request_id=identifier, prompt=text, prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                scheduled_time_s=arrival / 1e9, scheduled_ns=arrival, min_tokens=128, max_tokens=128,
                seed=0, split="qualification", prefix_family=None, prompt_token_ids=None)


def natural_document(records=None):
    records = records or [fixture_record()]
    result = dict(schema="natural_off_qualification_workload_v1", fit_or_evaluation_input_allowed=False,
                  initial_cache_state="fresh_equal_namespace_preserved_through_whole_partition",
                  records=records, selected_prompt_count=len(records), unselected_tail_count=0,
                  max_concurrency=2, dataset_sha256="a" * 64, model_manifest_sha256="b" * 64)
    result["workload_sha256"] = R.canonical_sha(result)
    return result


def refresh(document):
    document["workload_sha256"] = R.canonical_sha({key: val for key, val in document.items() if key != "workload_sha256"})
    return document


class FixtureClock:
    def __init__(self):
        self.value = 10_000_000

    def __call__(self):
        self.value += 1000
        return self.value

    def sleep(self, seconds):
        self.value += int(seconds * 1e9)


class FixtureOriginalEngineAPI:
    """Explicit API fixture. It contains no model, cache or backend implementation."""
    def __init__(self, *, chunk=False, retract=False, short=False, bad_id=False, fail_step=False):
        self.active, self.calls, self.prompt_inputs = {}, [], []
        self.chunk, self.retract, self.short, self.bad_id, self.fail_step = chunk, retract, short, bad_id, fail_step

    def has_unfinished_requests(self):
        self.calls.append("has_unfinished_requests")
        return bool(self.active)

    def add_request(self, request_id, prompt, sampling):
        self.calls.append("add_request")
        self.prompt_inputs.append(prompt)
        self.active[request_id] = []
        return request_id + "-cpu-fixture-native"

    def step(self):
        self.calls.append("step")
        if self.fail_step:
            raise RuntimeError("CPU fixture step failure")
        rows = []
        for rid, tokens in list(self.active.items()):
            tokens.append(len(tokens) + 1)
            if self.chunk and len(tokens) == 1:
                tokens.append(2)
            if self.retract and len(tokens) == 2:
                tokens[0] = 100
            finished = len(tokens) == (127 if self.short else 128)
            rows.append(NS(request_id="wrong" if self.bad_id else rid, prompt_token_ids=[1, 2, 3],
                           outputs=[NS(token_ids=tokens.copy(), finish_reason="length" if finished else None)],
                           finished=finished, num_cached_tokens=0))
            if finished:
                self.active.pop(rid)
        return rows


def drive(engine, records=None, **kwargs):
    clock = FixtureClock()
    return R.drive_original_engine(engine, lambda record: NS(cpu_fixture=True), records or [fixture_record()],
        max_concurrency=kwargs.pop("max_concurrency", 2), deadline_seconds=kwargs.pop("deadline_seconds", 5),
        clock=kwargs.pop("clock", clock), sleeper=clock.sleep, **kwargs)


class StrongTraceCPU(unittest.TestCase):
    def test_import_does_not_touch_framework_backend_or_gpu(self):
        script = """import sys,importlib.util
class NoRuntime:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','vllm','py_kvcache','ctypes','numpy'):
   raise AssertionError('runtime import in CPU mode: '+fullname)
sys.meta_path.insert(0,NoRuntime())
for filename in ('strong_trace_runner.py','native_runtime.py'):
 spec=importlib.util.spec_from_file_location('cpu_'+filename.replace('.','_'),%s+'/'+filename)
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
print('PASS_NO_RUNTIME_IMPORT')
""" % repr(HERE.as_posix())
        proc = subprocess.run([sys.executable, "-B", "-I", "-S", "-c", script], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("PASS_NO_RUNTIME_IMPORT", proc.stdout)

    def test_valid_natural_text_manifest_remains_qualification(self):
        self.assertEqual(R.validate_workload(natural_document()), 2)

    def test_workload_content_mutation_is_rejected(self):
        doc = natural_document()
        doc["records"][0]["prompt"] = "different"
        with self.assertRaisesRegex(ValueError, "workload digest"):
            R.validate_workload(doc)

    def test_prompt_mutation_even_with_new_manifest_digest_rejected(self):
        doc = natural_document()
        doc["records"][0]["prompt"] = "different"
        with self.assertRaisesRegex(ValueError, "unchanged author prompt"):
            R.validate_workload(refresh(doc))

    def test_author_arrival_or_order_drift_rejected(self):
        doc = natural_document([fixture_record(0, 5000), fixture_record(1, 1000)])
        with self.assertRaisesRegex(ValueError, "order unchanged"):
            R.validate_workload(doc)
        doc = natural_document()
        doc["records"][0]["scheduled_time_s"] = 1
        with self.assertRaisesRegex(ValueError, "same author arrival"):
            R.validate_workload(refresh(doc))

    def test_qualification_data_cannot_be_promoted_to_fitting(self):
        doc = natural_document()
        doc["fit_or_evaluation_input_allowed"] = True
        with self.assertRaisesRegex(ValueError, "cannot become fitting"):
            R.validate_workload(refresh(doc))

    def test_no_midstream_cache_reset_or_prefix_family_fabrication(self):
        doc = natural_document()
        doc["initial_cache_state"] = "reset_every_request"
        with self.assertRaisesRegex(ValueError, "no midstream reset"):
            R.validate_workload(refresh(doc))
        doc = natural_document()
        doc["records"][0]["prefix_family"] = "guessed"
        with self.assertRaisesRegex(ValueError, "qualification IDs"):
            R.validate_workload(refresh(doc))

    def test_outputs_at_least_full128_no_eos_shortcut(self):
        doc = natural_document()
        doc["records"][0]["max_tokens"] = doc["records"][0]["min_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "full min output"):
            R.validate_workload(refresh(doc))

    def test_current_actual_controlled_p3_manifest_is_consumable(self):
        path = HERE.parent / "protocol/CONTROLLED_P3_OFF_QUALIFICATION_WORKLOAD.json"
        document = json.loads(path.read_bytes())
        self.assertEqual(R.validate_workload(document), 4)
        self.assertEqual(len(document["records"]), 12)
        self.assertTrue(all(len(r["prompt_token_ids"]) == 512 for r in document["records"]))
        self.assertFalse(document["natural_trace_bound"])
        self.assertFalse(document["fit_or_evaluation_input_allowed"])

    def test_cpu_fixture_complete_tokens_and_every_itl(self):
        engine = FixtureOriginalEngineAPI()
        result = drive(engine)
        self.assertEqual(result["status"], "PASS_COMPLETE_ORIGINAL_REQUEST_OUTPUTS")
        row = result["rows"][0]
        self.assertEqual(len(row["output_token_ids"]), 128)
        self.assertEqual(len(row["token_return_ns"]), 128)
        self.assertEqual(len(row["itl_ns"]), 127)
        self.assertGreater(row["ttft_ns"], 0)
        self.assertFalse(result["kernel_cost_measurement"])
        self.assertFalse(result["formal_goodput_allowed"])
        self.assertEqual(set(engine.calls), {"step", "has_unfinished_requests", "add_request"})

    def test_cpu_fixture_parallel_requests_keep_request_denominator(self):
        result = drive(FixtureOriginalEngineAPI(), [fixture_record(0), fixture_record(1), fixture_record(2)])
        self.assertEqual(result["planned_requests"], 3)
        self.assertEqual(result["successful_requests"], 3)
        self.assertEqual(result["failure_or_unsubmitted_requests"], 0)
        self.assertEqual({r["request_id"] for r in result["rows"]}, {"0", "1", "2"})

    def test_original_token_ids_forwarded_without_text_retokenization(self):
        record = fixture_record()
        record["prompt"] = None
        record["prompt_token_ids"] = [10, 20, 30]
        engine = FixtureOriginalEngineAPI()
        result = drive(engine, [record])
        self.assertEqual(engine.prompt_inputs, [{"prompt_token_ids": [10, 20, 30]}])
        self.assertEqual(result["status"], "PASS_COMPLETE_ORIGINAL_REQUEST_OUTPUTS")
        self.assertEqual(record["prompt_token_ids"], [10, 20, 30])

    def test_chunk_is_rejected_instead_of_fabricated_per_token_itl(self):
        result = drive(FixtureOriginalEngineAPI(chunk=True))
        self.assertEqual(result["status"], "FAILED_ORIGINAL_REQUEST_OUTPUTS")
        self.assertIn("one actual new token", result["error"]["message"])
        self.assertEqual(result["rows"][0]["token_return_ns"], [])

    def test_retracted_token_or_wrong_request_rejected(self):
        for engine in (FixtureOriginalEngineAPI(retract=True), FixtureOriginalEngineAPI(bad_id=True)):
            self.assertEqual(drive(engine)["status"], "FAILED_ORIGINAL_REQUEST_OUTPUTS")

    def test_short_or_failed_output_is_counted_in_denominator(self):
        for engine in (FixtureOriginalEngineAPI(short=True), FixtureOriginalEngineAPI(fail_step=True)):
            result = drive(engine, [fixture_record(0), fixture_record(1, arrival=1_000_000_000)])
            self.assertEqual(result["planned_requests"], 2)
            self.assertEqual(result["failure_or_unsubmitted_requests"], 2)
            self.assertTrue(all(row["state"] == "FAILED_OR_UNSUBMITTED" for row in result["rows"]))

    def test_timeout_keeps_unsubmitted_and_accepted_failures(self):
        clock = FixtureClock()
        record = fixture_record(0, 10_000_000_000)
        result = drive(FixtureOriginalEngineAPI(), [record], clock=clock, deadline_seconds=.002)
        self.assertEqual(result["status"], "FAILED_ORIGINAL_REQUEST_OUTPUTS")
        self.assertEqual(result["failure_or_unsubmitted_requests"], 1)

    def test_progress_is_bounded_actual_output_values_only(self):
        captured = []
        result = drive(FixtureOriginalEngineAPI(), on_progress=lambda rows, step: captured.append(
            (step, len(rows[0]["output_token_ids"]))))
        self.assertEqual(result["status"], "PASS_COMPLETE_ORIGINAL_REQUEST_OUTPUTS")
        self.assertEqual(captured[-1], (128, 128))

    def test_generalized_i_cannot_silently_run_with_table_none(self):
        config = dict(phase="effect", mode="on", arm="I")
        with self.assertRaisesRegex(ValueError, "effect interface blocked"):
            R.check_phase(config, gap={"table": None})
        with self.assertRaises(ValueError):
            R.check_phase(dict(phase="qualification", mode="on", arm="I"), gap={})

    def test_shadow_requires_actual_off_and_does_not_mean_effect(self):
        with self.assertRaisesRegex(ValueError, "completed same-domain original off"):
            R.check_phase(dict(phase="shadow", mode="shadow", arm="I", off_qualification_ref=None), gap={})
        self.assertEqual(R.check_phase(dict(phase="qualification", mode="off", arm="U"), gap={"status": "closed"}),
                         {"status": "closed"})

    def test_original_guard_timeout_sessions_and_source_refs_remain_bound(self):
        root = Path("/root/cpu-fixture")
        config = dict(permissions_ref={"path": "scopes/private.yaml"}, run_id="cpu-fixture-off",
                      seconds_limit=300, runner_ref={"path": "runner/strong_trace_runner.py"})
        command = R.guard_command(root, "config.json", config)
        self.assertEqual(command[:4], [".venv/bin/python", "-B", R.GUARD, "--permissions-path"])
        self.assertIn("300", command)
        self.assertEqual(command[-1], "--execute")
        self.assertNotIn("sudo", command)

    def test_source_safe_json_and_byte_drift_rejection(self):
        with tempfile.TemporaryDirectory(dir=HERE, prefix="cpu_fixture_") as directory:
            root = Path(directory).resolve()
            path = root / "input.json"
            path.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                R.read(path)
            with self.assertRaises(ValueError):
                R.safe(root, "../escape")
            original = R.ref(root, "input.json")
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "byte drift"):
                R.check_ref(root, original)

    def test_native_drain_unknown_never_claims_release(self):
        good = dict(owner_capture=True, native={"active_parents": 0},
                    aio=dict(outstanding=0, pending=0, ready=0, unreaped=0), stage_accounting=None)
        self.assertEqual(N.validate_drained_snapshot(good), good)
        for key in ("pending", "ready", "unreaped", "outstanding"):
            changed = deepcopy(good)
            changed["aio"][key] = 1
            with self.assertRaisesRegex(ValueError, "not drained"):
                N.validate_drained_snapshot(changed)
        good["owner_capture"] = False
        with self.assertRaises(ValueError):
            N.validate_drained_snapshot(good)

    def test_original_shutdown_tail_must_be_empty_not_only_returned(self):
        good = dict(actual_aio=dict(outstanding=0, pending=0, ready=0, unreaped=0),
                    actual_native=dict(active_parents=0), actual_handler_active=0, observation_failures=0)
        self.assertTrue(N.verify_original_tail(good))
        for key in ("actual_handler_active", "observation_failures"):
            changed = deepcopy(good)
            changed[key] = 1
            with self.assertRaises(ValueError):
                N.verify_original_tail(changed)

    def test_runtime_member_chain_is_present_in_actual_original_api_sources(self):
        inputs = HERE.parent / "source_inputs"
        expected = {"llm_engine.py": "self.engine_core", "core_client.py": "self.engine_core = EngineCore",
                    "scheduler.py": "self.connector", "offloading_connector.py": "self.connector_scheduler"}
        for filename, substring in expected.items():
            text = (inputs / filename).read_text(encoding="utf-8")
            self.assertIn(substring, text)
            ast.parse(text)
        self.assertEqual(hashlib.sha256((inputs / "offloading_connector.py").read_bytes()).hexdigest(),
                         "02969dfa19fe6e9d7e9856e587fd115e67e614b09ce3cc50f5b009d179d05177")

    def test_actual_original_id_methods_roundtrip_external_and_native_identity(self):
        # Complete original methods, unchanged AST. Imported dependencies are
        # explicit CPU value fixtures, not installed vLLM runtime capability.
        inputs = HERE.parent / "source_inputs"
        input_tree = ast.parse((inputs / "input_processor.py").read_text(encoding="utf-8"))
        assign = next(node for node in ast.walk(input_tree) if isinstance(node, ast.FunctionDef) and
                      node.name == "assign_request_id")
        output_tree = ast.parse((inputs / "output_processor.py").read_text(encoding="utf-8"))
        make = next(node for node in ast.walk(output_tree) if isinstance(node, ast.FunctionDef) and
                    node.name == "_new_request_output")
        input_class = ast.ClassDef(name="OriginalCPUInput", bases=[], keywords=[], body=[assign], decorator_list=[])
        output_class = ast.ClassDef(name="OriginalCPURequestState", bases=[], keywords=[], body=[make], decorator_list=[])
        values = dict(envs=NS(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=False), random_uuid=lambda: "1234abcdffffffff",
                      logger=NS(warning_once=lambda value: None), PoolingOutput=type("CpuPoolingOutput", (), {}),
                      RequestOutputKind=NS(DELTA="delta"), RequestOutput=lambda **fields: NS(**fields), cast=cast,
                      CompletionOutput=type("CPUCompletionOutput", (), {}))
        tree = ast.fix_missing_locations(ast.Module(body=[input_class, output_class], type_ignores=[]))
        exec(compile(tree, "actual_source_cpu_id_methods", "exec", flags=__future__.annotations.compiler_flag), values)
        request = NS(request_id="outside-0", external_req_id=None)
        values["OriginalCPUInput"].assign_request_id(request)
        self.assertEqual(request.external_req_id, "outside-0")
        self.assertEqual(request.request_id, "outside-0-1234abcd")
        state = values["OriginalCPURequestState"]()
        state.prompt_token_ids, state.prompt_embeds = [10, 20, 30], None
        state.num_cached_tokens, state.output_kind = 16, "cumulative"
        state.logprobs_processor = NS(prompt_logprobs=None)
        state.lora_request, state.prompt, state.stats = None, "cpu fixture text", NS(cpu_fixture=True)
        output = state._new_request_output(request.external_req_id, [NS(token_ids=[40])], False)
        self.assertEqual(output.request_id, "outside-0")
        self.assertNotEqual(output.request_id, request.request_id)
        self.assertEqual(output.prompt_token_ids, [10, 20, 30])


if __name__ == "__main__":
    unittest.main(verbosity=2)
