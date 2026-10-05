"""CPU rejection tests; fixture records are never called native successes."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name("analyze_cost_evidence_cpu.py")
SPEC = importlib.util.spec_from_file_location("_cost_field_audit_cpu_tests", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fixture():
    # Minimal deterministic CPU metadata for checking the rejection boundary.
    frames, witnesses, steps, ids = [], [], [], list(range(128))
    for i in range(128):
        start = 1000+100*i
        ordinal = 130+i
        frames.append(dict(native_step_ordinal=ordinal,start_ns=start+10,end_ns=start+40,
            prepared=dict(native_step_ordinal=ordinal,batch=1,active_decode=0 if i == 0 else 1,
                prefill_tokens=1 if i == 0 else 0,context_length=128+i,
                step_kind="prefill" if i == 0 else "decode",context_basis="pre_computed_tokens",
                rows=[dict(request_id="CPU_FIXTURE",pre_context=128+i,prompt_tokens=129,scheduled_tokens=1)],
                input_seq_lens_from_cpu_inputs=[129 if i == 0 else 129+i]),
            outputs=[["CPU_FIXTURE",[ids[i]]]],intended_timing_scope="full_decode_step",gpu_elapsed_ns=None))
        witnesses.append(dict(native_step_ordinal=ordinal,start_record_before_ns=start,
            start_record_after_ns=start+5,start_completed_query_ns=start+15,
            end_record_before_ns=start+45,end_record_after_ns=start+50,end_completed_query_ns=start+60,
            gpu_elapsed_ns=30,event_elapsed_source="torch.cuda.Event.elapsed_time",cross_clock_absolute_mapping=False))
        steps.append(dict(before_ns=start+8,after_ns=start+42,cumulative_output_counts=[i+1]))
    prompt=[28100]+list(range(1001,1129))
    capture=dict(origin="native_gpu_recording",valid=True,failures=[],pending_event_pairs=0,
        open_event_pair=False,selected_offsets=[16],gpu_duration_scope="execute_model_through_sample_tokens_current_stream",
        host_window_scope="original_execute_through_original_sample_host_calls",cross_clock_absolute_mapping=False,
        no_added_synchronization=True,event_source=dict(sha256="0"*64,class_name="Event",module="torch.cuda.streams",
            path="CPU_FIXTURE_METADATA_ONLY",bytes=1),frames=frames,event_witnesses=witnesses)
    window=dict(prompt_token_ids=prompt,frontend=dict(output=dict(native_request_id="CPU_FIXTURE",
        prompt_token_ids=prompt,output_token_ids=ids),steps=steps))
    return capture,window


class InputBoundaryTests(unittest.TestCase):
    def test_duplicate_json(self):
        with self.assertRaises(M.AuditRejected):M.parse_json(b'{"x":1,"x":1}')

    def test_nonfinite_json(self):
        with self.assertRaises(M.AuditRejected):M.parse_json(b'{"x":NaN}')

    def test_boolean_reference_bytes(self):
        with self.assertRaises(M.AuditRejected):M.canonical_ref(dict(path="x.json",bytes=True,sha256="0"*64))

    def test_path_traversal(self):
        with self.assertRaises(M.AuditRejected):M.canonical_ref(dict(path="a/../x.json",bytes=0,sha256="0"*64))

    def test_absolute_reference(self):
        with self.assertRaises(M.AuditRejected):M.canonical_ref(dict(path="/tmp/x.json",bytes=0,sha256="0"*64))

    def test_real_manifest_sha_drift(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve(); data=root/"x.json";data.write_bytes(b'{}')
            manifest=root/"manifest.json";manifest.write_text(json.dumps(dict(files=[M.reference("x.json",b'{}')])))
            evidence=M.Evidence(root);evidence.add_manifest(manifest);self.assertEqual(evidence.json("x.json"),{})
            data.write_bytes(b'{"changed":true}')
            with self.assertRaises(M.AuditRejected):evidence.recheck()

    def test_unknown_reference(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(M.AuditRejected):M.Evidence(Path(d)).bytes("missing.json")

    def test_duplicated_manifest_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();ref=M.reference("x.json",b'{}');m=root/"manifest.json"
            m.write_text(json.dumps(dict(files=[ref,ref])))
            with self.assertRaises(M.AuditRejected):M.Evidence(root).add_manifest(m)

    def test_symlink_boundary_cpu(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();f=root/"x";f.write_bytes(b"x")
            with patch.object(Path,"is_symlink",return_value=True):
                with self.assertRaises(M.AuditRejected):M.contained_file(root,f)


class CaptureRejectionTests(unittest.TestCase):
    def rejects(self,change):
        capture,window=fixture()
        self.assertEqual(M.audit_capture(capture,window,expected_event_sha="0"*64)["full_frames"],128)
        change(capture,window)
        with self.assertRaises(M.AuditRejected):M.audit_capture(capture,window,expected_event_sha="0"*64)

    def test_wrong_witness_ordinal(self):
        self.rejects(lambda c,w:c["event_witnesses"][16].update(native_step_ordinal=145))

    def test_missing_last_frame(self):
        self.rejects(lambda c,w:c["frames"].pop())

    def test_changed_selected_offset(self):
        self.rejects(lambda c,w:c.update(selected_offsets=[17]))

    def test_changed_selected_context(self):
        self.rejects(lambda c,w:c["frames"][16]["prepared"].update(context_length=145))

    def test_boolean_witness_duration(self):
        self.rejects(lambda c,w:c["event_witnesses"][16].update(gpu_elapsed_ns=True))

    def test_broken_witness_enclosure(self):
        self.rejects(lambda c,w:c["event_witnesses"][16].update(end_record_before_ns=1))

    def test_output_changed(self):
        self.rejects(lambda c,w:c["frames"][16].update(outputs=[["CPU_FIXTURE",[999]]]))

    def test_bad_event_source(self):
        self.rejects(lambda c,w:c["event_source"].update(sha256="1"*64))

    def test_raw_duration_disagrees(self):
        self.rejects(lambda c,w:c["frames"][16].update(gpu_elapsed_ns=31))

    def test_gpu_absolute_clock_claim(self):
        self.rejects(lambda c,w:c.update(cross_clock_absolute_mapping=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
