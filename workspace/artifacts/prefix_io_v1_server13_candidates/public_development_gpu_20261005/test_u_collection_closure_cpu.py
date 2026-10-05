"""Four bounded negative cases; CPU fixtures never stand for GPU evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


C = source_module(ROOT / "u_collection_closure_cpu.py", "_fixture_U_closure")
R = source_module(ROOT / "u_collection_runtime/strong_trace_runner_v6.py", "_fixture_original_runner")
N = source_module(ROOT / "u_collection_runtime/native_runtime_v6.py", "_fixture_original_runtime")


class RejectIncompleteOrEscalatedEvidence(unittest.TestCase):
    def test_original_frontend_rejects_missing_actual_token(self):
        raw = json.loads((ROOT / "actual_uoff02/actual-request-outputs.json").read_text(encoding="utf-8"))
        native = json.loads((ROOT / "actual_uoff02/strong-native-workload-result.json").read_text(encoding="utf-8"))
        records = [dict(request_id=int(row["request_id"]), prompt_sha256=row["prompt_sha256"],
            prompt_token_ids=row["actual_prompt_token_ids"], max_tokens=128) for row in raw["rows"]]
        gates = dict(config=dict(formal_trace_binding_ref={}), formal_workload_binding=dict(records=records,
            partition="development", original_manifest_workload_sha256=native["formal_frontend_identity_closure"]["original_manifest_workload_sha256"]))
        R.validate_formal_frontend_records(gates, raw)
        changed = deepcopy(raw)
        changed["rows"][0]["output_token_ids"].pop()
        with self.assertRaisesRegex(ValueError, "complete actual outputs"):
            R.validate_formal_frontend_records(gates, changed)

    def test_actual_failed_capture_cannot_pass_original_join(self):
        raw = json.loads((ROOT / "actual_uoff02/actual-original-full-step-capture.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "whole-stream CUDA observation UNKNOWN"):
            N.validate_formal_capture(ROOT, {}, raw, {}, source_ref={}, driver=R)

    def test_U_diagnostic_cannot_promote_cost_table(self):
        native = json.loads((ROOT / "actual_uoff02/strong-native-workload-result.json").read_text(encoding="utf-8"))
        descriptor = json.loads((ROOT / "UNCALIBRATED_U_COLLECTION_DESCRIPTOR_03.json").read_text(encoding="utf-8"))
        C.validate_no_authority(native, descriptor)
        native["cost_qualified"] = True
        with self.assertRaisesRegex(ValueError, "cannot promote collection"):
            C.validate_no_authority(native, descriptor)

    def test_duplicate_token_event_is_rejected(self):
        row = dict(request_id="CPU_fixture", native_request_id="CPU_fixture_native", finished_ns=300,
            output_token_ids=list(range(128)), token_return_ns=list(range(100, 228)))
        events = [dict(request_id=row["request_id"], native_request_id=row["native_request_id"], token_ordinal=i,
            token_id=i, return_ns=100+i, original_step=i+1) for i in range(128)]
        events.append(dict(request_id=row["request_id"], state="COMPLETED", finish_ns=300, total_tokens=128))
        C.validate_token_events(events, dict(rows=[row]))
        events.insert(1, deepcopy(events[0]))
        with self.assertRaisesRegex(ValueError, "token event matches complete"):
            C.validate_token_events(events, dict(rows=[row]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
