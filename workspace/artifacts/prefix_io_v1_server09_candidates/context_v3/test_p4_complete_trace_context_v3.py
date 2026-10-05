"""Source-bound stdlib CPU schema 3 tests; all GPU-looking data are fixtures."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--control-source", type=Path, default=
    Path(__file__).resolve().parents[2] /
    "prefix_io_v1_server08_p4_cpu_extended_20261001/verified-v3/source/third_party/work/prefix-io-p4-02-cpu/src")
args, unittest_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unittest_args
path = Path(__file__).with_name("p4_complete_trace_context_v3.py")
spec = importlib.util.spec_from_file_location("server09_context_v3_candidate", path)
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)
CONTROL = args.control_source.resolve(strict=True)


def io(ops=0, nbytes=0, stage=2):
    result = [{"ops": 0, "bytes": 0} for _ in range(4)]
    result[stage] = {"ops": ops, "bytes": nbytes}
    return result


class Fixture:
    def __init__(self, root, verifier, *, arm="baseline", origin="cpu_fixture"):
        self.root = Path(root).resolve(strict=True)
        self.v = verifier
        self.b = verifier.base
        self.arm, self.origin = arm, origin
        self.context = self.b.TableContext("cpu-context-v3", "a"*64, "GPU-CPU-FIXTURE", "b"*64,
            "c"*64, "d"*64, "original-execute-sample-sync", "torch-fixture", "cuda-fixture",
            "driver-fixture", 8, 128, 64, 200, "existing_io_plus_delta", "paired_residual_margin")
        self.source_ref = self.dump("cpu-event-source.py",
            "# CPU fixture only: no CUDA event, model execution or GPU authenticity.\n", raw=True)
        self.timing = dict(timing_scope="full_decode_step", clock_domain="cuda_event_elapsed",
            reference_source_ref=self.source_ref, reference_valid=True,
            fallback_used=False, clock_domain_valid=True)
        self.entries, self.runs, self.traces = [], [], []
        for index, split in enumerate(("calibration", "calibration", "validation")):
            entry = dict(cell_id="cold-cell",
                trace_sha256=sha256(("trace-"+str(index)).encode()).hexdigest(),
                prefix_family_sha256=sha256(("prefix-"+str(index)).encode()).hexdigest(),
                seed=index, split=split,
                workload_sha256=sha256(("workload-"+str(index)).encode()).hexdigest(),
                input_tokens=128, output_tokens=128, request_count=1)
            self.entries.append(entry)
            start = 1000 + index * 2000000
            first = 100 + index * 128
            added = io(1, 8) if arm == "action" else io()
            run = dict(pair_id="pair-"+str(index), **{k: v for k, v in entry.items() if k != "cell_id"},
                arm_order="BA" if index == 1 else "AB",
                warmup_windows=1, measured_windows=1, start_ns=start,
                end_ns=start+128*1000+1000, exit_code=0, accepted_io_drained=True,
                completed_new_io=deepcopy(added), accepted_new_io=deepcopy(added),
                first_step_ordinal=first, full_steps=128, full_output_tokens=128,
                selected_output_tokens=1, complete_trace_ref={})
            trace = dict(schema_version=3, scope="p4_complete_step_output_trace", origin=origin,
                context=asdict(self.context), arm=arm, cell_id="cold-cell", pair_id=run["pair_id"],
                steps=[], outputs=[{"request_id":"actual-native-0","token_ids":list(range(128))}],
                outside_window_new_io=io(), accepted_new_io=deepcopy(added),
                completed_new_io=deepcopy(added), accepted_io_drained=True)
            for offset in range(128):
                load = (dict(active_decode=0,batch=1,prefill_tokens=128,context_length=0)
                        if offset == 0 else
                        dict(active_decode=1,batch=1,prefill_tokens=0,context_length=127+offset))
                row = dict(step_offset=offset, native_step_ordinal=first+offset,
                    start_ns=start+offset*1000+1, end_ns=start+offset*1000+501,
                    load=load, existing_io=io(),
                    new_io=deepcopy(added) if offset == 2 else io(),
                    outputs=[{"request_id":"actual-native-0","token_ids":[offset]}],
                    timing=dict(self.timing, gpu_elapsed_ns=100+offset))
                if offset == 0:
                    row["step_kind"] = "prefill"
                trace["steps"].append(row)
            self.runs.append(run)
            self.traces.append(trace)
        self.plan = dict(schema_version=2,scope="p4_paired_semantic_plan",
            context=asdict(self.context), workload_split_ref={},
            action_operations={"cold-cell":1}, min_calibration_pairs=2,
            min_validation_pairs=1,max_pairs=64,max_windows=4096,formula=self.b.FORMULA,
            selections={"cold-cell":{"load":dict(active_decode=1,batch=1,prefill_tokens=0,context_length=129),
                                      "warmup_step_offsets":[1],"measured_step_offsets":[2]}},
            timing_contract=deepcopy(self.timing))
        self.wrapper = dict(schema_version=3,scope="paired_measurement_wrapper",origin=origin,
            context=asdict(self.context),arm=arm,cell_id="cold-cell",runs=self.runs)
        self.refresh()

    def ref(self, name):
        data = (self.root/name).read_bytes()
        return dict(path=name,bytes=len(data),sha256=sha256(data).hexdigest())

    def dump(self, name, value, *, raw=False):
        data = value.encode() if raw else json.dumps(value,sort_keys=True,allow_nan=False).encode()
        (self.root/name).write_bytes(data)
        return self.ref(name)

    def refresh(self):
        split = dict(schema_version=1,scope="p4_frozen_measurement_split",entries=self.entries)
        self.plan["workload_split_ref"] = self.dump("split.json",split)
        self.plan_ref = self.dump("plan.json",self.plan)
        for index, trace in enumerate(self.traces):
            reference = self.dump("trace-"+str(index)+".json",trace)
            if index < len(self.runs):
                self.runs[index]["complete_trace_ref"] = reference
        self.wrapper_ref = self.dump("wrapper.json",self.wrapper)

    def load(self):
        return self.v.verify(self.root,wrapper_ref=self.wrapper_ref,selection_plan_ref=self.plan_ref,
                             arm=self.arm,cell_id="cold-cell")

    def loaded_plan(self):
        ref = self.b.EvidenceRef.from_mapping(self.plan_ref)
        return self.b.load_verification_plan(self.root,ref.path,expected_plan_ref=ref)


class CompleteTraceV3Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = candidate.CompleteTraceContextV3Verifier(CONTROL)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.f = Fixture(self.temp.name,self.v)

    def tearDown(self):
        self.temp.cleanup()

    def reject(self):
        with self.assertRaises((ValueError,self.v.base.TableContractError)):
            self.f.load()

    def test_all_real_cold_frames_and_128_tokens_remain_in_complete_trace(self):
        result = self.f.load()
        self.assertEqual((result.schema_version,result.runs,result.steps,result.full_output_tokens),(3,3,384,384))
        self.assertEqual(result.context_basis,"pre_computed_tokens")
        self.assertEqual(len(result.cold_prefill_rows),3)
        self.assertEqual(result.cold_prefill_rows[0],("pair-0",0,100))
        self.assertFalse(result.production_qualified)
        self.assertFalse(result.gpu_verified)
        self.assertFalse(result.paired_cost_verification_performed)
        self.assertEqual(result.runtime_hook_status,"not_installed")
        self.assertEqual(self.f.traces[0]["steps"][0]["load"]["context_length"],0)
        self.assertEqual(self.f.traces[0]["outputs"][0]["token_ids"],list(range(128)))

    def test_native_looking_origin_remains_gpu_unverified(self):
        other = Fixture(self.temp.name,self.v,origin="native_gpu_recording")
        result = other.load()
        self.assertEqual(result.origin,"native_gpu_recording")
        self.assertFalse(result.production_qualified)
        self.assertFalse(result.gpu_verified)

    def test_action_full_extra_physical_io_totals_and_drain_are_reused(self):
        other = Fixture(self.temp.name,self.v,arm="action")
        self.assertEqual(other.load().full_output_tokens,384)
        for trace in other.traces:
            self.assertEqual(trace["accepted_new_io"],io(1,8))

    def test_zero_requires_explicit_prefill_kind_and_actual_prefill_load(self):
        cases = ("no_kind","wrong_kind","decode","active_decode","no_prefill","negative","bool",
                 "zero_batch","unknown_key")
        for case in cases:
            with self.subTest(case=case):
                self.f = Fixture(self.temp.name,self.v)
                row = self.f.traces[0]["steps"][0]
                if case == "no_kind": row.pop("step_kind")
                elif case == "wrong_kind": row["step_kind"]="decode"
                elif case == "decode":
                    row.pop("step_kind")
                    row["load"].update(active_decode=1,prefill_tokens=0)
                elif case == "active_decode": row["load"]["active_decode"]=1
                elif case == "no_prefill": row["load"]["prefill_tokens"]=0
                elif case == "negative": row["load"]["context_length"]=-1
                elif case == "bool": row["load"]["context_length"]=False
                elif case == "zero_batch": row["load"]["batch"]=0
                elif case == "unknown_key": row["load"]["post_context"]=128
                self.f.refresh()
                self.reject()

    def test_positive_prefill_keeps_original_load_rules(self):
        self.f.traces[0]["steps"][0]["load"].update(context_length=5,prefill_tokens=123)
        self.f.refresh()
        result = self.f.load()
        self.assertEqual(len(result.cold_prefill_rows),2)
        self.assertEqual(result.full_output_tokens,384)

    def test_decode_zero_negative_or_boolean_remains_rejected(self):
        for value in (0,-1,False):
            with self.subTest(value=value):
                self.f = Fixture(self.temp.name,self.v)
                self.f.traces[0]["steps"][1]["load"]["context_length"]=value
                self.f.refresh()
                self.reject()

    def test_zero_cannot_enter_original_frozen_selection_load(self):
        self.f.plan["selections"]["cold-cell"]["load"]["context_length"]=0
        self.f.refresh()
        self.reject()

    def test_cold_first_frame_cannot_be_selected_as_warmup(self):
        self.f.plan["selections"]["cold-cell"]["warmup_step_offsets"]=[0]
        self.f.refresh()
        self.reject()

    def test_positive_prefill_cannot_be_selected_as_pure_decode_warmup(self):
        self.f.traces[0]["steps"][1]["load"].update(active_decode=0,prefill_tokens=1)
        self.f.traces[0]["steps"][1]["step_kind"]="prefill"
        self.f.refresh()
        self.reject()

    def test_measured_exact_load_context_is_not_bucketted_or_renamed(self):
        self.f.traces[0]["steps"][2]["load"]["context_length"]=130
        self.f.refresh()
        self.reject()

    def test_schema_two_trace_is_rejected_by_v3(self):
        self.f.traces[0]["schema_version"]=2
        self.f.refresh()
        self.reject()

    def test_schema_three_trace_is_rejected_by_original_v2(self):
        plan = self.f.loaded_plan()
        with self.assertRaises(self.v.base.TableContractError):
            self.v.base._v2_complete_trace(self.f.root,self.f.runs[0],self.f.wrapper,plan)

    def test_original_v2_still_rejects_real_zero_prefill_context(self):
        self.f.traces[0]["schema_version"]=2
        self.f.refresh()
        plan = self.f.loaded_plan()
        with self.assertRaisesRegex(self.v.base.TableContractError,"geometry"):
            self.v.base._v2_complete_trace(self.f.root,self.f.runs[0],self.f.wrapper,plan)

    def test_old_source_globals_function_identity_and_strict_load_remain_unchanged(self):
        original = self.v.base._v2_complete_trace
        old_load = self.v.base._valid_load
        source_before = (CONTROL/"prefix_io_control/p4_paired_measurement_verifier.py").read_bytes()
        self.f.load()
        self.assertIs(self.v.base._v2_complete_trace,original)
        self.assertIs(self.v.base._valid_load,old_load)
        self.assertIsNot(self.v._complete.__globals__,self.v.base.__dict__)
        self.assertIs(self.v._complete.__globals__["_valid_load"],old_load)
        self.assertNotIn("_valid_complete_trace_load_v3",self.v.base.__dict__)
        self.assertEqual(self.v.ast_patch["schema_comparison"],1)
        self.assertEqual(self.v.ast_patch["complete_row_load_call"],1)
        self.assertEqual((CONTROL/"prefix_io_control/p4_paired_measurement_verifier.py").read_bytes(),source_before)
        with self.assertRaises(self.v.base.TableContractError):
            old_load(dict(active_decode=0,batch=1,prefill_tokens=128,context_length=0))

    def test_wrapper_schema_scope_origin_or_context_drift_is_rejected(self):
        cases = ("schema","scope","origin","context","arm","cell","extra")
        for case in cases:
            with self.subTest(case=case):
                self.f = Fixture(self.temp.name,self.v)
                if case=="schema": self.f.wrapper["schema_version"]=2
                elif case=="scope": self.f.wrapper["scope"]="loose_trace"
                elif case=="origin": self.f.wrapper["origin"]="unknown"
                elif case=="context": self.f.wrapper["context"]["gpu_uuid"]="GPU-OTHER"
                elif case=="arm": self.f.wrapper["arm"]="action"
                elif case=="cell": self.f.wrapper["cell_id"]="other"
                elif case=="extra": self.f.wrapper["claim_gpu"]=True
                self.f.refresh()
                self.reject()

    def test_pinned_bytes_cannot_hide_deletion_or_context_zero_to_one_conversion(self):
        for change in ("zero_to_one","remove_first","truncate_output"):
            with self.subTest(change=change):
                self.f = Fixture(self.temp.name,self.v)
                trace = deepcopy(self.f.traces[0])
                if change=="zero_to_one": trace["steps"][0]["load"]["context_length"]=1
                elif change=="remove_first": trace["steps"].pop(0)
                else: trace["outputs"][0]["token_ids"].pop()
                self.f.dump("trace-0.json",trace)
                # Deliberately retain the independently frozen original byte reference.
                self.reject()

    def test_missing_reordered_duplicate_or_invented_native_ordinal_is_rejected(self):
        for change in ("missing","reordered","duplicate","invented"):
            with self.subTest(change=change):
                self.f = Fixture(self.temp.name,self.v)
                rows = self.f.traces[0]["steps"]
                if change=="missing": rows.pop(0)
                elif change=="reordered": rows[0],rows[1]=rows[1],rows[0]
                elif change=="duplicate": rows[1]=deepcopy(rows[0])
                else: rows[1]["native_step_ordinal"]+=1
                self.f.refresh()
                self.reject()

    def test_truncated_full_outputs_hidden_token_or_wrong_request_is_rejected(self):
        cases = ("truncate","wrong_token","unknown_request","missing_decode_output","multi_token")
        for case in cases:
            with self.subTest(case=case):
                self.f = Fixture(self.temp.name,self.v)
                trace = self.f.traces[0]
                if case=="truncate": trace["outputs"][0]["token_ids"].pop()
                elif case=="wrong_token": trace["steps"][1]["outputs"][0]["token_ids"]=[999]
                elif case=="unknown_request": trace["steps"][1]["outputs"][0]["request_id"]="other"
                elif case=="missing_decode_output": trace["steps"][1]["outputs"][0]["token_ids"]=[]
                else: trace["steps"][1]["outputs"][0]["token_ids"]=[1,2]
                self.f.refresh()
                self.reject()

    def test_host_order_run_membership_or_exit_and_drain_is_rejected(self):
        cases = ("overlap","outside","backwards","failed_exit","undrained_run","undrained_trace")
        for case in cases:
            with self.subTest(case=case):
                self.f = Fixture(self.temp.name,self.v)
                trace,run = self.f.traces[0],self.f.runs[0]
                if case=="overlap": trace["steps"][1]["start_ns"]=trace["steps"][0]["end_ns"]-1
                elif case=="outside": trace["steps"][-1]["end_ns"]=run["end_ns"]+1
                elif case=="backwards": trace["steps"][0]["end_ns"]=trace["steps"][0]["start_ns"]
                elif case=="failed_exit": run["exit_code"]=1
                elif case=="undrained_run": run["accepted_io_drained"]=False
                else: trace["accepted_io_drained"]=False
                self.f.refresh()
                self.reject()

    def test_timing_source_scope_domain_reference_or_invalid_elapsed_is_rejected(self):
        cases = ("source","scope","clock","fallback","reference","valid","zero","boolean","negative")
        for case in cases:
            with self.subTest(case=case):
                self.f = Fixture(self.temp.name,self.v)
                timing=self.f.traces[0]["steps"][0]["timing"]
                if case=="source":
                    timing["reference_source_ref"]=self.f.dump("other-source.py","# other",raw=True)
                elif case=="scope": timing["timing_scope"]="model_forward"
                elif case=="clock": timing["clock_domain"]="host_monotonic"
                elif case=="fallback": timing["fallback_used"]=True
                elif case=="reference": timing["reference_valid"]=False
                elif case=="valid": timing["clock_domain_valid"]=False
                elif case=="zero": timing["gpu_elapsed_ns"]=0
                elif case=="boolean": timing["gpu_elapsed_ns"]=True
                else: timing["gpu_elapsed_ns"]=-1
                self.f.refresh()
                self.reject()

    def test_real_timing_source_bytes_must_still_match_pinned_reference(self):
        (self.f.root/self.f.source_ref["path"]).write_text("# changed")
        self.reject()

    def test_forward_only_plan_and_rows_cannot_be_relabelled_as_full_step(self):
        self.f.plan["timing_contract"]["timing_scope"] = "model_forward"
        for trace in self.f.traces:
            for row in trace["steps"]:
                row["timing"]["timing_scope"] = "model_forward"
        self.f.refresh()
        self.reject()

    def test_four_stage_physical_totals_unknown_stages_and_completion_mismatch_are_rejected(self):
        cases = ("baseline_added","outside_added","bad_shape","ops_bytes","uncompleted","run_disagrees")
        for case in cases:
            with self.subTest(case=case):
                self.f = Fixture(self.temp.name,self.v)
                trace=self.f.traces[0]
                if case=="baseline_added": trace["steps"][0]["new_io"]=io(1,8)
                elif case=="outside_added": trace["outside_window_new_io"]=io(1,8)
                elif case=="bad_shape": trace["steps"][0]["new_io"].append({"ops":0,"bytes":0})
                elif case=="ops_bytes": trace["steps"][0]["existing_io"]=io(0,8)
                elif case=="uncompleted": trace["completed_new_io"]=io(1,8)
                else: self.f.runs[0]["accepted_new_io"]=io(1,8)
                self.f.refresh()
                self.reject()

    def test_independent_frozen_workload_split_cannot_be_relabelled_or_reused(self):
        cases = ("trace","seed","prefix","tokens","order","omit_run","duplicate_run","overlap")
        for case in cases:
            with self.subTest(case=case):
                self.f = Fixture(self.temp.name,self.v)
                run=self.f.runs[0]
                if case=="trace": run["trace_sha256"]="f"*64
                elif case=="seed": run["seed"]=99
                elif case=="prefix": run["prefix_family_sha256"]="f"*64
                elif case=="tokens": run["input_tokens"]=129
                elif case=="order": run["arm_order"]="BA"
                elif case=="omit_run": self.f.runs.pop()
                elif case=="duplicate_run": self.f.runs[1]=deepcopy(run)
                else: self.f.runs[1]["start_ns"]=run["start_ns"]
                self.f.refresh()
                self.reject()

    def test_selected_counts_remain_actual_and_cannot_replace_full_128_counts(self):
        self.f.runs[0]["selected_output_tokens"]=2
        self.f.refresh()
        self.reject()

    def test_original_base_source_sha_drift_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as location:
            folder=Path(location)/"prefix_io_control"
            folder.mkdir()
            for name in candidate.BASE_REFS:
                shutil.copyfile(CONTROL/"prefix_io_control"/name,folder/name)
            changed=folder/"p4_paired_measurement_verifier.py"
            changed.write_bytes(changed.read_bytes()+b"\n# source drift\n")
            with self.assertRaisesRegex(ValueError,"source bytes/SHA drift"):
                candidate.CompleteTraceContextV3Verifier(Path(location))

    def test_import_and_verification_do_not_import_gpu_or_executor_modules(self):
        self.f.load()
        self.assertNotIn("torch",sys.modules)
        self.assertNotIn("vllm",sys.modules)
        self.assertNotIn("pynvml",sys.modules)
        self.assertFalse(hasattr(candidate,"execute_model"))
        self.assertFalse(hasattr(candidate,"load_cost_table"))


if __name__=="__main__":
    unittest.main(verbosity=2)

