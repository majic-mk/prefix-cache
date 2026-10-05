"""CPU evidence-file serializer counterexamples, never native execution claims."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("server11_cpu_prepare", Path(__file__).with_name("prepare_and_verify_native_cost.py"))
P = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = P
spec.loader.exec_module(P)
fixture_spec = importlib.util.spec_from_file_location("server11_cost_fixture", Path(__file__).with_name("test_native_conditional_cost.py"))
F = importlib.util.module_from_spec(fixture_spec)
sys.modules[fixture_spec.name] = F
fixture_spec.loader.exec_module(F)


class SerializerContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        plan, windows, _, _ = F.fixture()
        plan["model_plan_ref"] = dict(path="model-plan.json", bytes=1, sha256="e" * 64)
        plan["external_warmup_output_tokens"] = 128
        plan["external_flush_output_tokens"] = 1
        self.plan_ref = P.write(self.root, "plan.json", plan)
        self.report = dict(status="PASS_NATIVE_SIX_PROCESS_RAW_CAPTURE_REQUIRES_VALIDATION",
            origin="native_gpu_recording", label=plan["job_id"], gpu_uuid=plan["gpu_uuid"],
            original_model_subprocesses_started=6, actual_guarded_gpu_job_count=1,
            input_template_unchanged=True, original_engine_shutdown_returned=True, load_planner="off",
            guard=dict(session_id=123), windows=[], children=[])
        self.children = []
        for index, window in enumerate(windows):
            external_id = "external-" + str(index)
            capture = deepcopy(window["capture"])
            capture["run_id"] = external_id
            raw = dict(pair_index=index // 2, condition="A" if window["arm"] == "baseline" else "B",
                prompt_token_ids=window["prompt_token_ids"], seed=window["seed"], request_id=external_id,
                frontend=dict(output=dict(num_cached_tokens=128, output_token_ids=window["output_token_ids"],
                    request_id=external_id, native_request_id=window["request_id"])),
                capture=capture, independent_payload=window["independent_payload"])
            child = dict(status="PASS_NATIVE_FULL_STEP_RAW_CAPTURE_REQUIRES_VALIDATION", window_index=index,
                fresh_original_process=True, original_engine_shutdown_returned=True, gpu_uuid=plan["gpu_uuid"],
                source_lock=plan["source_lock_ref"]["path"], model=dict(manifest_sha256="e"*64, plan_sha256="e"*64),
                subprocess_pid=1000+index, private_storage="private-"+str(index), subprocess_sid=123,
                windows=[raw], native_journal={}, native_post_shutdown={}, native_tail_assertions={},
                warmups=[dict(output_token_ids=list(range(128)), prompt_token_ids=window["prompt_token_ids"],
                    seed=window["seed"], flush=dict(output_token_ids=[1]))])
            self.children.append(child)
            self.report["windows"].append(deepcopy(raw) | dict(native_journal={}, native_post_shutdown={}))
            self.report["children"].append(dict(window_index=index, exit=0))
        self.version = 0

    def serialize(self):
        self.version += 1
        for index, child in enumerate(self.children):
            self.report["windows"][index]["child_receipt_ref"] = P.write(self.root,
                "%d/child-%d.json" % (self.version, index), child)
        relative = "%d/runtime.json" % self.version
        P.write(self.root, relative, self.report)
        return P.serialize_runtime_record(self.root, runtime_relative=relative,
            plan_ref=self.plan_ref, source_verification_refs={})

    def test_six_fresh_children_preserve_real_file_refs(self):
        result = self.serialize()
        self.assertEqual(result["actual_subprocess_count"], 6)
        self.assertEqual(len(result["windows"]), 6)
        self.assertFalse(result["production_qualified"])
        self.assertTrue(all("raw_child_ref" in w for w in result["windows"]))
        self.assertEqual(result["windows"][0]["request_id"], "request-0")
        self.assertEqual(result["windows"][0]["external_request_id"], "external-0")
        self.assertEqual(result["windows"][0]["capture"]["frames"][0]["outputs"][0][0], "request-0")

    def test_shared_pid_rejected(self):
        self.children[1]["subprocess_pid"] = self.children[0]["subprocess_pid"]
        with self.assertRaisesRegex(ValueError, "fresh original process"):
            self.serialize()

    def test_original_child_failure_rejected(self):
        self.report["children"][3]["exit"] = 1
        with self.assertRaisesRegex(ValueError, "child exit"):
            self.serialize()

    def test_parent_modified_output_rejected(self):
        self.report["windows"][1]["frontend"]["output"]["output_token_ids"][5] = 1234
        with self.assertRaisesRegex(ValueError, "parent/child raw"):
            self.serialize()

    def test_model_identity_drift_rejected(self):
        self.children[2]["model"]["manifest_sha256"] = "8" * 64
        with self.assertRaisesRegex(ValueError, "child source/model"):
            self.serialize()

    def test_short_external_warmup_rejected(self):
        self.children[0]["warmups"][0]["output_token_ids"] = [1]
        with self.assertRaisesRegex(ValueError, "full-output warmup"):
            self.serialize()

    def test_ast_constants_never_execute_source(self):
        path = self.root / "source.py"
        path.write_text("A='safe'\nB=A+'/file'\nC=(1,2)\nraise RuntimeError('never execute')\n", encoding="utf-8")
        self.assertEqual(P.constants(path), dict(A="safe", B="safe/file", C=(1, 2)))

    def test_append_only_output_and_traversal(self):
        P.write(self.root, "small.json", dict(cpu=True))
        with self.assertRaises(FileExistsError):
            P.write(self.root, "small.json", dict(cpu=False))
        with self.assertRaisesRegex(ValueError, "relative path"):
            P.relative(self.root, "../escape.json")


if __name__ == "__main__":
    unittest.main()
