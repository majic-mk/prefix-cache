"""Explicit synthetic raw shapes copied from original finite contract tests.
No event/model/physical I/O/GPU process is executed by these helpers.
"""
from copy import deepcopy
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys
_spec = importlib.util.spec_from_file_location('_c5_synthetic_shape_original_math', Path(__file__).with_name('native_conditional_cost.py'))
M = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = M; _spec.loader.exec_module(M)

def amounts(ops=0, size=0):
    return [dict(ops=ops, nbytes=size)] + [dict(ops=0, nbytes=0) for _ in range(3)]

def fixture():
    entries = []
    for index in range(3):
        prompt = [18100 + index * 1000] + list(range(1001, 1129))
        seed = 1829 + index
        entries.append(dict(pair_id="pair-" + str(index), split="calibration" if index < 2 else "validation",
            arm_order="BA" if index == 1 else "AB", seed=seed, prompt_token_ids=prompt,
            prompt_sha256=M.canonical_hash(prompt), prefix_family_sha256=M.canonical_hash(prompt[:16]),
            trace_sha256=M.canonical_hash(dict(prompt_token_ids=prompt, seed=seed, output_tokens=128)),
            workload_sha256=M.canonical_hash(dict(prompt_token_ids=prompt, output_tokens=128,
                                                  temperature=0, ignore_eos=True))))
    plan = dict(scope="server11_preregistered_native_conditional_cell_v1",
        qualification_rule="zero_observed_holdout_underprediction_no_refit_v1",
        gpu_uuid="GPU-CPU-TEST", job_id="cpu-fixture", source_lock_ref=dict(path="lock.json", bytes=1, sha256="a" * 64),
        collector_source_ref=dict(path="collector.py", bytes=1, sha256="b" * 64),
        wrapper_source_ref=dict(path="wrapper.py", bytes=1, sha256="c" * 64),
        project_root="/CPU-FIXTURE", cuda_event_source_ref=dict(path="torch/cuda/streams.py", bytes=1, sha256="d" * 64),
        model_sha256="e" * 64, kv_layout_sha256="f" * 64,
        native_source_sha256="1" * 64, journal_run_id="cpu-shared-journal", source_lock_files=2,
        stage="ssd_read", transfer_quantum_bytes=8, units=1, operations=1,
        cached_prompt_tokens=128, prompt_tokens=129, output_tokens=128, measured_offset=16,
        warmup_offsets=[1], entries=entries)
    windows = []
    journal = dict(events=[], frames=[], valid=True, lost=0)
    accepted_count = completed_count = 0
    def owner(at):
        journal["frames"].append(dict(sequence=len(journal["events"]), captured_ns=at,
            run_id=plan["journal_run_id"], source_sha256=plan["native_source_sha256"],
            clock_domain="monotonic_ns", valid=True, journal_complete=True, failed_ops=[0] * 4,
            accepted=amounts(accepted_count, accepted_count * 8),
            completed=amounts(completed_count, completed_count * 8),
            inflight=amounts(accepted_count - completed_count, (accepted_count - completed_count) * 8)))
    for pair_index, entry in enumerate(entries):
        for arm in (("action", "baseline") if pair_index == 1 else ("baseline", "action")):
            index = len(windows)
            start = 1000000 + index * 200000000
            request = "request-" + str(index)
            owner(start - 10)
            output = list(range(128))
            capture = dict(scope="server11_full_step_native_capture_v1", origin="native_gpu_recording",
                run_id="capture-" + str(index), valid=True, frames=[], event_witnesses=[], actions=[],
                failures=[], pending_event_pairs=0, open_event_pair=False, no_added_synchronization=True,
                cross_clock_absolute_mapping=False, selected_offsets=[16], event_source=dict(
                    origin="native_gpu_recording", module="torch.cuda.streams", class_name="Event", sha256="d" * 64,
                    bytes=1, path="/CPU-FIXTURE/torch/cuda/streams.py"))
            for offset in range(128):
                ordinal = index * 128 + offset
                now = start + offset * 1000000
                context = 128 if offset == 0 else 128 + offset
                capture["frames"].append(dict(native_step_ordinal=ordinal, start_ns=now + 30,
                    end_ns=now + 9000, prepared=dict(native_step_ordinal=ordinal,
                        rows=[dict(request_id=request, pre_context=context, prompt_tokens=129,
                                   scheduled_tokens=1)], batch=1,
                        active_decode=0 if offset == 0 else 1, prefill_tokens=1 if offset == 0 else 0,
                        context_length=context, step_kind="prefill" if offset == 0 else "decode",
                        input_seq_lens_from_cpu_inputs=[129 if offset == 0 else context + 1],
                        context_basis="pre_computed_tokens"), outputs=[[request, [output[offset]]]],
                    intended_timing_scope="full_decode_step", gpu_elapsed_ns=None, existing_io=None, new_io=None))
                duration = 100 if arm == "baseline" else [110, 120, 118][pair_index]
                capture["event_witnesses"].append(dict(native_step_ordinal=ordinal,
                    start_record_before_ns=now + 10, start_record_after_ns=now + 20,
                    start_completed_query_ns=now + 50, end_record_before_ns=now + 9100,
                    end_record_after_ns=now + 9200, end_completed_query_ns=now + 9300,
                    gpu_elapsed_ns=duration, event_elapsed_source="torch.cuda.Event.elapsed_time"))
            selected = start + 16 * 1000000
            capture["actions"].append(dict(step_offset=16, native_step_ordinal=index * 128 + 16,
                trigger_before_ns=selected + 60, trigger_after_ns=selected + 70,
                action_enabled=arm == "action", trigger_result={} if arm == "action" else None))
            payload = None
            if arm == "action":
                payload = dict(ssd_only=True, cache_miss_before_submit=True, physical_bytes=8, operations=1,
                    preload_key_sha256=[sha256((str(pair_index) + ":" + str(j)).encode()).hexdigest() for j in range(1)],
                    request_prefix_key_sha256=["a" * 64])
                for j in range(1):
                    seq = len(journal["events"]) + 1
                    journal["events"].append(dict(sequence=seq, operation_sequence=seq,
                        at_ns=selected + 80 + 2*j, kind="accepted", stage="ssd_read", physical_bytes=8, result=None))
                    accepted_count += 1
                    owner(selected + 80 + 2*j)
                    journal["events"].append(dict(sequence=seq+1, operation_sequence=seq,
                        at_ns=selected + 81 + 2*j, kind="completed", stage="ssd_read", physical_bytes=8, result=8))
                    completed_count += 1
                    owner(selected + 81 + 2*j)
            owner(start + 128000000)
            windows.append(dict(pair_id=entry["pair_id"], arm=arm, seed=entry["seed"],
                prompt_token_ids=entry["prompt_token_ids"], output_token_ids=output, request_id=request,
                run_id=capture["run_id"], gpu_uuid=plan["gpu_uuid"], source_lock_sha256="a" * 64,
                model_sha256="e" * 64, kv_layout_sha256="f" * 64, capture=capture, independent_payload=payload))
    drain = dict(run_id=plan["journal_run_id"], native_source_sha256=plan["native_source_sha256"],
        boundary="after_original_handler_shutdown", accounting_valid=True, worker_joined=True,
        ring_closed=True, ring_drained=True, outstanding_records=0, ring_outstanding=0, pending_copies=0,
        active_native=0, inflight_parents=0, observation_failures=0, accepted=amounts(3, 24),
        completed=amounts(3, 24), failed_ops=[0] * 4, transferred_bytes=[24, 0, 0, 0])
    return plan, windows, journal, drain
