"""Meaningful stdlib CPU paired-boundary tests; CUDA metadata are fixtures."""
import argparse
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--control-source", type=Path, default=Path(__file__).resolve().parents[2] /
    "prefix_io_v1_server08_p4_cpu_extended_20261001/verified-v3/source/third_party/work/prefix-io-p4-02-cpu/src")
parser.add_argument("--context-source", type=Path, default=Path(__file__).resolve().parent.parent / "context_v3")
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining
spec = importlib.util.spec_from_file_location("server09_paired_v3_candidate",
    Path(__file__).with_name("p4_paired_context_v3.py"))
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)
CONTROL = args.control_source.resolve(strict=True)
CONTEXT = args.context_source.resolve(strict=True)


def io(ops=0, size=0, stage=2):
    out = [{"ops": 0, "bytes": 0} for _ in range(4)]
    out[stage] = {"ops": ops, "bytes": size}
    return out


def add_io(*vectors):
    return [{"ops":sum(v[n]["ops"] for v in vectors),
             "bytes":sum(v[n]["bytes"] for v in vectors)} for n in range(4)]


class Fixture:
    def __init__(self, root, verifier):
        self.root, self.v, self.b = Path(root), verifier, verifier.base
        self.context = self.b.TableContext("cpu-paired-context-v3", "a"*64, "GPU-CPU-FIXTURE", "b"*64,
            "c"*64, "d"*64, "original-execute-sample-sync", "torch-fixture", "cuda-fixture",
            "driver-fixture", 8, 128, 64, 200, "existing_io_plus_delta", "paired_residual_margin")
        self.source_ref = self.dump("cpu-event-source.py", "# CPU fixture: no real CUDA event or model.\n", raw=True)
        self.timing = dict(timing_scope="full_decode_step", clock_domain="cuda_event_elapsed",
            reference_source_ref=self.source_ref, reference_valid=True, fallback_used=False, clock_domain_valid=True)
        self.geometry = dict(cell_id="cold-cell", load=dict(active_decode=1,batch=1,prefill_tokens=0,context_length=129),
            existing_io=io(1,8,0), stage="h2d", physical_bytes=16)
        self.entries, self.runs, self.traces, self.wrappers, self.observations = [], {}, {}, {}, {}
        for n, split in enumerate(("calibration", "calibration", "validation")):
            self.entries.append(dict(cell_id="cold-cell", trace_sha256=sha256(("trace"+str(n)).encode()).hexdigest(),
                prefix_family_sha256=sha256(("prefix"+str(n)).encode()).hexdigest(), seed=n, split=split,
                workload_sha256=sha256(("workload"+str(n)).encode()).hexdigest(),
                input_tokens=128, output_tokens=128, request_count=1))
        for arm in ("baseline", "action"):
            self.runs[arm], self.traces[arm], windows = [], [], []
            for n, entry in enumerate(self.entries):
                order = "BA" if n == 1 else "AB"
                first_arm = "action" if order == "BA" else "baseline"
                shift = 0 if arm == first_arm else 500000
                start, ordinal = 1000+n*2000000+shift, 100+n*256+(0 if shift==0 else 128)
                run = dict(pair_id="pair-"+str(n), **{k:v for k,v in entry.items() if k!="cell_id"},
                    arm_order=order, warmup_windows=1, measured_windows=1, start_ns=start,
                    end_ns=start+129000, exit_code=0, accepted_io_drained=True,
                    completed_new_io=io(), accepted_new_io=io(), first_step_ordinal=ordinal,
                    full_steps=128, full_output_tokens=128, selected_output_tokens=1, complete_trace_ref={})
                trace = dict(schema_version=3,scope="p4_complete_step_output_trace",origin="cpu_fixture",
                    context=asdict(self.context),arm=arm,cell_id="cold-cell",pair_id=run["pair_id"],steps=[],
                    outputs=[dict(request_id="actual-native-0",token_ids=list(range(128)))],
                    outside_window_new_io=io(),accepted_new_io=io(),completed_new_io=io(),accepted_io_drained=True)
                for offset in range(128):
                    load = (dict(active_decode=0,batch=1,prefill_tokens=128,context_length=0) if offset==0 else
                            dict(active_decode=1,batch=1,prefill_tokens=0,context_length=127+offset))
                    duration = ((100,102,104) if arm=="baseline" else (110,116,119))[n] if offset==2 else 9999
                    row = dict(step_offset=offset,native_step_ordinal=ordinal+offset,start_ns=start+offset*1000+1,
                        end_ns=start+offset*1000+501,load=load,existing_io=io(1,8,0),
                        new_io=io(2,16) if arm=="action" and offset==2 else io(),
                        outputs=[dict(request_id="actual-native-0",token_ids=[offset])],
                        timing=dict(self.timing,gpu_elapsed_ns=duration))
                    if offset==0: row["step_kind"]="prefill"
                    trace["steps"].append(row)
                    if offset in (1,2):
                        window = {k:deepcopy(row[k]) for k in
                            ("start_ns","end_ns","load","existing_io","new_io","timing","native_step_ordinal")}
                        window.update(pair_id=run["pair_id"],window_id="step-"+str(offset),
                            phase="warmup" if offset==1 else "measured",step_offset=offset,output_tokens=1)
                        windows.append(window)
                self.runs[arm].append(run)
                self.traces[arm].append(trace)
            self.wrappers[arm]=dict(schema_version=3,scope="paired_measurement_wrapper",origin="cpu_fixture",
                context=asdict(self.context),arm=arm,cell_id="cold-cell",runs=self.runs[arm])
            self.observations[arm]=dict(schema_version=2,scope="paired_window_observations",origin="cpu_fixture",
                context=asdict(self.context),arm=arm,cell_id="cold-cell",windows=windows)
        self.plan=dict(schema_version=2,scope="p4_paired_semantic_plan",context=asdict(self.context),workload_split_ref={},
            action_operations={"cold-cell":2},min_calibration_pairs=2,min_validation_pairs=1,max_pairs=64,max_windows=4096,
            formula=self.b.FORMULA,selections={"cold-cell":{"load":deepcopy(self.geometry["load"]),
                "warmup_step_offsets":[1],"measured_step_offsets":[2]}},timing_contract=deepcopy(self.timing))
        self.analysis=dict(schema_version=2,scope="p4_paired_analysis_candidate",origin="cpu_fixture",
            context=asdict(self.context),cell_id="cold-cell",plan_ref={},evidence_refs={},formula=self.b.FORMULA,
            baseline_ns=101,incremental_or_joint_ns=12,uncertainty_ns=6,calibration_pairs=2,validation_pairs=1,
            paired_runs_reported=3,windows_reported=3)
        self.refresh()

    def dump(self, name, value, raw=False):
        data=value.encode() if raw else json.dumps(value,sort_keys=True,allow_nan=False).encode()
        (self.root/name).write_bytes(data)
        return dict(path=name,bytes=len(data),sha256=sha256(data).hexdigest())

    def refresh(self, recalculate_totals=True):
        self.plan["workload_split_ref"]=self.dump("split.json",dict(schema_version=1,
            scope="p4_frozen_measurement_split",entries=self.entries))
        self.plan_ref=self.dump("plan.json",self.plan)
        self.raw_refs={}
        for arm in ("baseline","action"):
            for n,trace in enumerate(self.traces[arm]):
                if recalculate_totals:
                    total=add_io(trace["outside_window_new_io"],*(r["new_io"] for r in trace["steps"]))
                    trace["accepted_new_io"]=deepcopy(total)
                    trace["completed_new_io"]=deepcopy(total)
                    if n<len(self.runs[arm]):
                        self.runs[arm][n]["accepted_new_io"]=deepcopy(total)
                        self.runs[arm][n]["completed_new_io"]=deepcopy(total)
                ref=self.dump(arm+"-trace-"+str(n)+".json",trace)
                if n<len(self.runs[arm]): self.runs[arm][n]["complete_trace_ref"]=ref
            self.raw_refs[arm+"_wrapper"]=self.dump(arm+"-wrapper.json",self.wrappers[arm])
            self.raw_refs[arm+"_observations"]=self.dump(arm+"-observations.json",self.observations[arm])
        self.analysis["plan_ref"]=deepcopy(self.plan_ref)
        self.analysis["evidence_refs"]=deepcopy(self.raw_refs)
        self.raw_refs["pair_analysis"]=self.dump("analysis.json",self.analysis)

    def load(self):
        return self.v.verify_cell(self.root,cell_geometry=self.geometry,raw_refs=self.raw_refs,
            selection_plan_ref=self.plan_ref)

    def row(self,arm="action",pair=0,offset=2):
        return self.traces[arm][pair]["steps"][offset]

    def observation(self,arm="action",pair=0,offset=2):
        return next(w for w in self.observations[arm]["windows"] if
                    w["pair_id"]=="pair-"+str(pair) and w["step_offset"]==offset)

    def sync_projection(self,arm="action",pair=0,offset=2):
        row,window=self.row(arm,pair,offset),self.observation(arm,pair,offset)
        for k in ("start_ns","end_ns","load","existing_io","new_io","timing","native_step_ordinal"):
            window[k]=deepcopy(row[k])


class PairedV3Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v=candidate.PairedContextV3CPUVerifier(CONTROL,CONTEXT)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.f=Fixture(self.temp.name,self.v)

    def tearDown(self):
        self.temp.cleanup()

    def reject(self):
        with self.assertRaises((ValueError,self.v.base.TableContractError)):
            self.f.load()

    def test_raw_cost_recomputation_excludes_warmup_and_preserves_full_outputs(self):
        r=self.f.load()
        self.assertEqual((r.baseline_ns,r.incremental_or_joint_ns,r.empirical_residual_ns),(101,12,6))
        self.assertEqual((r.calibration_pairs,r.validation_pairs,r.paired_runs,r.measured_windows,r.warmup_windows),
                         (2,1,3,3,3))
        self.assertEqual((r.full_steps,r.full_output_tokens,len(r.cold_prefill_rows)),(768,768,6))
        self.assertEqual(r.context_basis,"pre_computed_tokens")
        self.assertEqual(r.cold_prefill_rows[0],("baseline_wrapper","pair-0",0,100))
        self.assertTrue(r.paired_cost_verification_performed)
        self.assertFalse(r.gpu_verified or r.production_qualified or r.effect_verified or r.P4_complete)

    def test_result_is_immutable_primitives_and_never_exposes_original_table_object(self):
        r=self.f.load()
        with self.assertRaises(FrozenInstanceError): r.baseline_ns=42
        with self.assertRaises(FrozenInstanceError): r.production_qualified=True
        self.assertFalse(hasattr(r,"cost") or hasattr(r,"candidate") or hasattr(r,"context"))
        self.assertTrue(all(type(x) is tuple for x in r.evidence_refs+r.source_refs))
        self.assertEqual(r.origin,"cpu_fixture")

    def test_original_globals_functions_bytes_and_loader_are_unchanged(self):
        b=self.v.base
        names=("_verify_cell","_v2_windows","_v2_complete_trace","_runs","_valid_load")
        before={n:getattr(b,n) for n in names}
        old_globals=dict(b.__dict__)
        sources={p:p.read_bytes() for p in (CONTROL/"prefix_io_control").glob("*.py")}
        self.f.load()
        self.assertEqual(old_globals,b.__dict__)
        for n in names: self.assertIs(before[n],getattr(b,n))
        for p,data in sources.items(): self.assertEqual(p.read_bytes(),data)
        g=self.v._cell.__globals__
        self.assertIsNot(g,b.__dict__)
        self.assertIs(g["_v2_windows"].__globals__,g)
        self.assertIs(g["_complete_trace_context_v3"],self.v.complete._complete)
        self.assertIs(g["_runs"],b._v2_runs)
        self.assertEqual((self.v.clone_proof["role_wrapper_version"],self.v.clone_proof["complete_trace_call"]),(1,1))
        with self.assertRaises(b.TableContractError):
            b._valid_load(dict(active_decode=0,batch=1,prefill_tokens=128,context_length=0))

    def test_verification_is_read_only_and_does_not_publish_or_import_backends(self):
        before={p.relative_to(self.f.root):p.read_bytes() for p in self.f.root.rglob("*") if p.is_file()}
        self.f.load()
        self.assertEqual(before,{p.relative_to(self.f.root):p.read_bytes() for p in self.f.root.rglob("*") if p.is_file()})
        for name in ("torch","vllm","pynvml","numpy"):
            self.assertNotIn(name,sys.modules)
        self.assertFalse(hasattr(candidate,"load_cost_table") or hasattr(candidate,"execute_model"))

    def test_gpu_entry_rejects_even_native_looking_authorization(self):
        for auth in (None,{"approved":True,"gpu_uuid":"GPU-CPU-FIXTURE","seconds":400}):
            with self.assertRaisesRegex(ValueError,"CPU-only"):
                self.v.require_gpu_launch(auth)

    def test_native_origin_cannot_attest_gpu_even_when_consistent(self):
        for obj in list(self.f.wrappers.values())+list(self.f.observations.values())+[self.f.analysis]:
            obj["origin"]="native_gpu_recording"
        for traces in self.f.traces.values():
            for trace in traces: trace["origin"]="native_gpu_recording"
        self.f.refresh()
        self.reject()

    def test_only_exact_wrapper_roles_use_schema_three(self):
        for role in self.v.base.MEASUREMENT_ROLES:
            with self.subTest(role=role):
                self.f=Fixture(self.temp.name,self.v)
                arm,kind=role.split("_")
                obj=self.f.wrappers[arm] if kind=="wrapper" else self.f.observations[arm]
                obj["schema_version"]=2 if kind=="wrapper" else 3
                self.f.refresh(); self.reject()

    def test_analysis_and_plan_remain_exact_schema_two(self):
        for which in ("analysis","plan"):
            with self.subTest(which=which):
                self.f=Fixture(self.temp.name,self.v)
                getattr(self.f,which)["schema_version"]=3
                self.f.refresh(); self.reject()

    def test_role_scope_arm_cell_context_extra_key_are_rejected(self):
        cases=(("scope","paired_measurement_wrapper"),("arm","baseline"),("cell_id","other"),
               ("extra",True),("origin","unknown"))
        for k,val in cases:
            with self.subTest(field=k):
                self.f=Fixture(self.temp.name,self.v)
                self.f.observations["action"][k]=val
                self.f.refresh(); self.reject()

    def test_exact_raw_role_set_does_not_allow_missing_or_unrecognized_role(self):
        for mode in ("missing","extra"):
            with self.subTest(mode=mode):
                self.f=Fixture(self.temp.name,self.v)
                if mode=="missing": self.f.raw_refs.pop("baseline_observations")
                else: self.f.raw_refs["native_claim"]=self.f.raw_refs["baseline_observations"]
                self.reject()

    def test_pinned_bytes_ref_rejects_zero_to_one_and_dropped_frame_without_rehash(self):
        for mode in ("zero_to_one","drop"):
            with self.subTest(mode=mode):
                self.f=Fixture(self.temp.name,self.v)
                trace=deepcopy(self.f.traces["baseline"][0])
                if mode=="zero_to_one": trace["steps"][0]["load"]["context_length"]=1
                else: trace["steps"].pop(0)
                self.f.dump("baseline-trace-0.json",trace)
                self.reject()

    def test_rehashed_drop_duplicate_reorder_and_invented_ordinal_are_rejected(self):
        for mode in ("drop","duplicate","reorder","ordinal"):
            with self.subTest(mode=mode):
                self.f=Fixture(self.temp.name,self.v)
                rows=self.f.traces["baseline"][0]["steps"]
                if mode=="drop": rows.pop(0)
                elif mode=="duplicate": rows[1]=deepcopy(rows[0])
                elif mode=="reorder": rows[0],rows[1]=rows[1],rows[0]
                else: rows[1]["native_step_ordinal"]+=1
                self.f.refresh(); self.reject()

    def test_zero_needs_explicit_prefill_and_cannot_enter_selected_decode(self):
        for mode in ("kind","prefill","decode","selected"):
            with self.subTest(mode=mode):
                self.f=Fixture(self.temp.name,self.v)
                row=self.f.row("baseline",0,0)
                if mode=="kind": row.pop("step_kind")
                elif mode=="prefill": row["load"]["prefill_tokens"]=0
                elif mode=="decode": row["load"]["active_decode"]=1
                else:
                    self.f.row("baseline")["load"]["context_length"]=0
                    self.f.sync_projection("baseline")
                self.f.refresh(); self.reject()

    def test_positive_prefill_remains_valid_without_claiming_cold_prompt_provenance(self):
        self.f.row("baseline",0,0)["load"].update(context_length=5,prefill_tokens=123)
        self.f.refresh()
        self.assertEqual(len(self.f.load().cold_prefill_rows),5)

    def test_selected_warmup_cannot_be_cold_or_positive_prefill(self):
        for mode in ("cold","positive_prefill"):
            with self.subTest(mode=mode):
                self.f=Fixture(self.temp.name,self.v)
                if mode=="cold": self.f.plan["selections"]["cold-cell"]["warmup_step_offsets"]=[0]
                else:
                    row=self.f.row("baseline",0,1)
                    row["load"].update(active_decode=0,prefill_tokens=1)
                    row["step_kind"]="prefill"
                    self.f.sync_projection("baseline",0,1)
                self.f.refresh(); self.reject()

    def test_complete_outputs_truncation_hidden_request_and_token_are_rejected(self):
        for mode in ("truncate","wrong_request","wrong_token","missing"):
            with self.subTest(mode=mode):
                self.f=Fixture(self.temp.name,self.v)
                trace=self.f.traces["action"][0]
                if mode=="truncate": trace["outputs"][0]["token_ids"].pop()
                elif mode=="wrong_request": trace["steps"][1]["outputs"][0]["request_id"]="hidden"
                elif mode=="wrong_token": trace["steps"][1]["outputs"][0]["token_ids"]=[777]
                else: trace["steps"][1]["outputs"]=[]
                self.f.refresh(); self.reject()

    def test_individually_valid_but_different_paired_outputs_are_rejected(self):
        trace=self.f.traces["action"][0]
        for row in trace["steps"]: row["outputs"][0]["token_ids"][0]+=1000
        trace["outputs"][0]["token_ids"]=[x+1000 for x in trace["outputs"][0]["token_ids"]]
        self.f.refresh(); self.reject()

    def test_selected_projection_fields_cannot_drift_from_complete_trace(self):
        fields=("start_ns","end_ns","native_step_ordinal","load","existing_io","new_io","timing","output_tokens")
        for field in fields:
            with self.subTest(field=field):
                self.f=Fixture(self.temp.name,self.v)
                w=self.f.observation()
                if field in ("start_ns","end_ns","native_step_ordinal","output_tokens"): w[field]+=1
                elif field=="load": w[field]["context_length"]+=1
                elif field=="timing": w[field]["gpu_elapsed_ns"]+=1
                else: w[field]=io(3,24)
                self.f.refresh(); self.reject()

    def test_preregistered_window_identity_phase_duplicates_and_coverage_are_rejected(self):
        for mode in ("phase","id","offset","duplicate","missing","unselected"):
            with self.subTest(mode=mode):
                self.f=Fixture(self.temp.name,self.v)
                w=self.f.observation()
                windows=self.f.observations["action"]["windows"]
                if mode=="phase": w["phase"]="warmup"
                elif mode=="id": w["window_id"]="other"
                elif mode=="offset": w["step_offset"]=3
                elif mode=="duplicate": windows.append(deepcopy(w))
                elif mode=="missing": windows.remove(w)
                else:
                    w["step_offset"]=3
                    row=self.f.row("action",0,3)
                    for field in ("start_ns","end_ns","native_step_ordinal","load","existing_io","new_io","timing"):
                        w[field]=deepcopy(row[field])
                    w["window_id"]="step-3"
                self.f.refresh(); self.reject()

    def test_selected_single_stage_io_geometry_rejects_extra_stage_wrong_bytes_or_ops(self):
        for mode in ("extra_stage","bytes","ops","baseline","existing"):
            with self.subTest(mode=mode):
                self.f=Fixture(self.temp.name,self.v)
                arm="baseline" if mode=="baseline" else "action"
                row=self.f.row(arm)
                if mode=="extra_stage": row["new_io"][0]={"ops":1,"bytes":8}
                elif mode=="bytes": row["new_io"]=io(2,24)
                elif mode=="ops": row["new_io"]=io(1,16)
                elif mode=="baseline": row["new_io"]=io(2,16)
                else: row["existing_io"]=io(2,16,0)
                self.f.sync_projection(arm)
                self.f.refresh(); self.reject()

    def test_physical_quantum_stage_and_selected_load_geometry_are_not_widened(self):
        for mode in ("unknown_stage","unaligned","illegal_units","ops_exceed","context","mixed","extra_key"):
            with self.subTest(mode=mode):
                self.f=Fixture(self.temp.name,self.v)
                g=self.f.geometry
                if mode=="unknown_stage": g["stage"]="logical_copy"
                elif mode=="unaligned": g["physical_bytes"]=17
                elif mode=="illegal_units": g["physical_bytes"]=24
                elif mode=="ops_exceed": self.f.plan["action_operations"]["cold-cell"]=3
                elif mode=="context": g["load"]["context_length"]=130
                elif mode=="mixed": g["load"]["prefill_tokens"]=1
                else: g["ignored_qualification"]=True
                self.f.refresh(); self.reject()

    def test_nonselected_and_outside_io_must_remain_in_full_totals(self):
        self.f.row("action",0,9)["new_io"]=io(1,8,3)
        self.f.traces["action"][0]["outside_window_new_io"]=io(1,8,1)
        self.f.refresh()
        self.assertEqual(self.f.load().incremental_or_joint_ns,12)
        self.f.traces["action"][0]["completed_new_io"]=io(2,16)
        self.f.refresh(recalculate_totals=False); self.reject()

    def test_drain_completion_and_exit_failures_cannot_be_hidden(self):
        for mode in ("run_drain","trace_drain","exit","run_completed","trace_completed"):
            with self.subTest(mode=mode):
                self.f=Fixture(self.temp.name,self.v)
                run,trace=self.f.runs["action"][0],self.f.traces["action"][0]
                if mode=="run_drain": run["accepted_io_drained"]=False
                elif mode=="trace_drain": trace["accepted_io_drained"]=False
                elif mode=="exit": run["exit_code"]=1
                elif mode=="run_completed": run["completed_new_io"]=io()
                else: trace["completed_new_io"]=io()
                self.f.refresh(recalculate_totals=False); self.reject()

    def test_abba_arm_order_pair_matching_and_nonoverlap_remain_required(self):
        for mode in ("unbalanced","mismatch","wrong_time","pair_missing","global_overlap"):
            with self.subTest(mode=mode):
                self.f=Fixture(self.temp.name,self.v)
                if mode=="unbalanced":
                    for arm in ("baseline","action"): self.f.runs[arm][1]["arm_order"]="AB"
                elif mode=="mismatch": self.f.runs["action"][0]["seed"]=88
                elif mode=="wrong_time":
                    for arm in ("baseline","action"): self.f.runs[arm][0]["arm_order"]="BA"
                elif mode=="pair_missing": self.f.runs["action"].pop()
                else: self.f.runs["baseline"][1]["start_ns"]=self.f.runs["baseline"][0]["start_ns"]
                self.f.refresh(); self.reject()

    def test_frozen_split_seed_prefix_workload_and_reported_counts_are_rejected(self):
        for mode in ("seed","prefix","workload","token_count","full_count","selected_count","split_leak"):
            with self.subTest(mode=mode):
                self.f=Fixture(self.temp.name,self.v)
                run=self.f.runs["baseline"][0]
                if mode=="seed": run["seed"]+=1
                elif mode=="prefix": run["prefix_family_sha256"]="f"*64
                elif mode=="workload": run["workload_sha256"]="f"*64
                elif mode=="token_count": run["output_tokens"]+=1
                elif mode=="full_count": run["full_output_tokens"]+=1
                elif mode=="selected_count": run["selected_output_tokens"]+=1
                else: self.f.entries[2]["prefix_family_sha256"]=self.f.entries[0]["prefix_family_sha256"]
                self.f.refresh(); self.reject()

    def test_actual_event_source_scope_domain_and_flags_are_not_relabelled(self):
        for mode in ("source","scope","clock","fallback","reference","domain","zero","bool"):
            with self.subTest(mode=mode):
                self.f=Fixture(self.temp.name,self.v)
                t=self.f.row()["timing"]
                if mode=="source": t["reference_source_ref"]=self.f.dump("other.py","#other",raw=True)
                elif mode=="scope": t["timing_scope"]="model_forward"
                elif mode=="clock": t["clock_domain"]="host_monotonic"
                elif mode=="fallback": t["fallback_used"]=True
                elif mode=="reference": t["reference_valid"]=False
                elif mode=="domain": t["clock_domain_valid"]=False
                elif mode=="zero": t["gpu_elapsed_ns"]=0
                else: t["gpu_elapsed_ns"]=True
                self.f.sync_projection(); self.f.refresh(); self.reject()

    def test_forward_only_plan_and_all_frames_cannot_be_accepted_as_full_step(self):
        self.f.plan["timing_contract"]["timing_scope"]="model_forward"
        for traces in self.f.traces.values():
            for trace in traces:
                for row in trace["steps"]: row["timing"]["timing_scope"]="model_forward"
        for obj in self.f.observations.values():
            for window in obj["windows"]: window["timing"]["timing_scope"]="model_forward"
        self.f.refresh(); self.reject()

    def test_timer_source_bytes_pinned_independently_and_rechecked(self):
        (self.f.root/self.f.source_ref["path"]).write_bytes(b"#drift")
        self.reject()

    def test_analysis_must_independently_bind_original_arithmetic_and_all_refs(self):
        for mode in ("cost","counts","formula","evidence","scope","extra"):
            with self.subTest(mode=mode):
                self.f=Fixture(self.temp.name,self.v)
                a=deepcopy(self.f.analysis)
                if mode=="cost": a["incremental_or_joint_ns"]+=1
                elif mode=="counts": a["paired_runs_reported"]+=1
                elif mode=="formula": a["formula"]="reported_pass"
                elif mode=="evidence": a["evidence_refs"]["baseline_wrapper"]=a["evidence_refs"]["action_wrapper"]
                elif mode=="scope": a["scope"]="production_table"
                else: a["gpu_verified"]=True
                self.f.raw_refs["pair_analysis"]=self.f.dump("analysis.json",a)
                self.reject()

    def test_context_and_control_source_sha_drift_rejected_before_reuse(self):
        for mode in ("context","control","loader"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                dst=Path(temp)
                shutil.copytree(CONTROL/"prefix_io_control",dst/"src/prefix_io_control")
                shutil.copytree(CONTEXT,dst/"context")
                changed=(dst/"context/p4_complete_trace_context_v3.py" if mode=="context" else
                    dst/"src/prefix_io_control"/("p4_verified_cost_loader.py" if mode=="loader" else
                                                "p4_paired_measurement_verifier.py"))
                changed.write_bytes(changed.read_bytes()+b"\n# drift\n")
                with self.assertRaisesRegex(ValueError,"bytes/SHA drift"):
                    candidate.PairedContextV3CPUVerifier(dst/"src",dst/"context")

    def test_original_v2_entry_rejects_v3_without_mutating_its_dispatch(self):
        b=self.v.base
        pref=b.EvidenceRef.from_mapping(self.f.plan_ref)
        plan=b.load_verification_plan(self.f.root,pref.path,expected_plan_ref=pref)
        with self.assertRaises(b.TableContractError):
            b._measurement(self.f.root,b.EvidenceRef.from_mapping(self.f.raw_refs["baseline_wrapper"]),
                scope="paired_measurement_wrapper",arm="baseline",cell_id="cold-cell",context=plan.context,version=2)

    def test_original_cost_table_production_lookup_stays_closed_and_loader_rejects_new_result(self):
        b=self.v.base
        result=self.f.load()
        cost_module=__import__(b.__package__+".p4_cost_table",fromlist=["CostTable"])
        loader=__import__(b.__package__+".p4_verified_cost_loader",fromlist=["PreparedCostTable"])
        load=self.f.geometry["load"]
        signature=(self.f.context.model_sha256,self.f.context.gpu_uuid,self.f.context.kv_layout_sha256,
            self.f.context.kernel_mode,load["active_decode"],load["batch"],load["prefill_tokens"],
            load["context_length"],self.f.context.transfer_quantum_bytes)
        cell=b.CostCell(signature,b._mapping_vector(self.f.geometry["existing_io"]),"h2d",16,
            self.f.context.cost_basis,result.baseline_ns,result.incremental_or_joint_ns,result.empirical_residual_ns)
        table=cost_module.CostTable((cell,),scope="mock_only",source_sha256="e"*64)
        self.assertIsNone(table.lookup(*cell.key,execution="production"))
        self.assertEqual(table.lookup(*cell.key,execution="cpu_mock").total_ns,119)
        self.assertFalse(table.production_qualified)
        with self.assertRaisesRegex(TypeError,"exact semantic verification"):
            loader.PreparedCostTable(result,table)


if __name__=="__main__":
    unittest.main(verbosity=2)
