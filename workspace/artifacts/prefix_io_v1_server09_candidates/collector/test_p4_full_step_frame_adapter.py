"""Standalone stdlib CPU contracts; no CUDA/runtime import or authenticity claim."""
import gc
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import weakref
from dataclasses import replace

path = Path(__file__).with_name("p4_full_step_frame_adapter.py")
spec = importlib.util.spec_from_file_location("server09_frame_adapter", path)
adapter_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter_module
spec.loader.exec_module(adapter_module)
FullStepFrameAdapter = adapter_module.FullStepFrameAdapter
DrainEvidence = adapter_module.DrainEvidence
StageTotal = adapter_module.StageTotal

RUN = "server09-cpu-frame-fixture"
SOURCE = "a" * 64
NATIVE = "b" * 64
ZERO = tuple(StageTotal(0, 0) for _ in range(4))


def adapter(**kwargs):
    return FullStepFrameAdapter(RUN, SOURCE, NATIVE, enabled=True, **kwargs)


def runner(ordinal=20, context=128, prompt=128, ids=("native-0",)):
    return NS(speculative_config=None, use_async_scheduling=False, is_pooling_model=False,
              _profile_step=ordinal + 1,
              parallel_config=NS(pipeline_parallel_size=1, data_parallel_size=1,
                                 tensor_parallel_size=1),
              input_batch=NS(num_reqs=len(ids), req_ids=list(ids),
                             num_computed_tokens_cpu=[context] * len(ids),
                             num_prompt_tokens=[prompt] * len(ids)))


def scheduler(ids=("native-0",), tokens=1):
    return NS(num_scheduled_tokens={rid: tokens for rid in ids},
              total_num_scheduled_tokens=len(ids) * tokens,
              scheduled_spec_decode_tokens={})


def output(ids=("native-0",), tokens=None):
    if tokens is None:
        tokens = [[7] for _ in ids]
    return NS(req_ids=list(ids), sampled_token_ids=tokens,
              req_id_to_index={rid: i for i, rid in enumerate(ids)})


def drain(**changes):
    values = dict(run_id=RUN, native_source_sha256=NATIVE, accepted=ZERO, completed=ZERO,
                  transferred_bytes=(0, 0, 0, 0), failed_ops=(0, 0, 0, 0),
                  accounting_valid=True, outstanding_records=0, worker_joined=True,
                  ring_closed=True, ring_drained=True, ring_outstanding=0, pending_copies=0,
                  active_native=0, inflight_parents=0, observation_failures=0)
    values.update(changes)
    return DrainEvidence(**values)


def one_step(a, *, ordinal=20, context=128, prompt=128, ids=("native-0",), tokens=1,
             samples=None, start=100, end=200):
    a.begin(ordinal, start)
    prepared = a.prepared(runner(ordinal, context, prompt, ids), scheduler(ids, tokens))
    a.executed(returned_none=True)
    closed = a.sampled(output(ids, samples), end)
    return prepared, closed


class AdapterContracts(unittest.TestCase):
    def test_real_cold_prefill_zero_context_and_full_128_outputs_are_preserved(self):
        a = adapter()
        prepared, first = one_step(a, context=0, prompt=128, tokens=128, samples=[[0]])
        self.assertEqual(prepared.rows[0].pre_context, 0)
        self.assertEqual(prepared.context_length, 0)
        self.assertEqual(prepared.context_basis, "pre_computed_tokens")
        self.assertEqual(prepared.input_seq_lens_from_cpu_inputs, (128,))
        self.assertEqual((prepared.active_decode, prepared.prefill_tokens, prepared.step_kind),
                         (0, 128, "prefill"))
        self.assertEqual(first.outputs, (("native-0", (0,)),))
        for offset in range(127):
            one_step(a, ordinal=21 + offset, context=128 + offset, samples=[[offset + 1]],
                     start=200 + offset * 100, end=250 + offset * 100)
        result = a.close_run({"native-0": list(range(128))}, drain())
        self.assertEqual(len(result.frames), 128)
        self.assertEqual(sum(f.prepared.step_kind == "decode" for f in result.frames), 127)
        self.assertEqual(sum(len(tokens) for f in result.frames for _, tokens in f.outputs), 128)
        self.assertEqual(result.frames[0].prepared.context_length, 0)
        self.assertFalse(result.production_qualified)
        # Existing v2 requires all complete-trace context_length values >= 1.
        # Preserve the genuine incompatible first frame instead of omitting it.
        self.assertFalse(all(f.prepared.context_length >= 1 for f in result.frames))

    def test_128_decode_forwards_after_cold_prefill_have_129_actual_outputs(self):
        a = adapter()
        one_step(a, context=0, prompt=128, tokens=128, samples=[[0]])
        for offset in range(128):
            one_step(a, ordinal=21 + offset, context=128 + offset, samples=[[offset + 1]],
                     start=200 + offset * 100, end=250 + offset * 100)
        self.assertEqual(sum(f.prepared.step_kind == "decode" for f in a.frames), 128)
        self.assertEqual(sum(len(tokens) for f in a.frames for _, tokens in f.outputs), 129)
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            a.close_run({"native-0": list(range(128))}, drain())

    def test_partial_cold_prefill_cannot_invent_output_before_prompt_completion(self):
        a = adapter()
        one_step(a, context=0, prompt=128, tokens=64, samples=[[]])
        one_step(a, ordinal=21, context=64, prompt=128, tokens=64, samples=[[5]],
                 start=200, end=300)
        one_step(a, ordinal=22, context=128, samples=[[6]], start=300, end=400)
        result = a.close_run({"native-0": [5, 6]}, drain())
        self.assertEqual([f.prepared.context_length for f in result.frames], [0, 64, 128])
        self.assertEqual(result.frames[0].prepared.input_seq_lens_from_cpu_inputs, (64,))
        bad = adapter()
        bad.begin(20, 100)
        bad.prepared(runner(context=0), scheduler(tokens=64))
        bad.executed(returned_none=True)
        with self.assertRaisesRegex(ValueError, "partial original prefill"):
            bad.sampled(output(tokens=[[5]]), 200)

    def test_original_execute_none_is_not_a_completed_sample_step(self):
        a = adapter()
        a.begin(20, 100)
        a.prepared(runner(), scheduler())
        a.executed(returned_none=True)
        self.assertEqual(a.frames, ())
        with self.assertRaisesRegex(ValueError, "incomplete"):
            a.close_run({"native-0": [7]}, drain())

    def test_original_two_call_step_closes_with_actual_cpu_token_ids(self):
        a = adapter()
        prepared, frame = one_step(a)
        self.assertEqual(prepared.context_length, 128)
        self.assertEqual((prepared.batch, prepared.active_decode, prepared.prefill_tokens), (1, 1, 0))
        self.assertEqual(frame.outputs, (("native-0", (7,)),))
        self.assertEqual(frame.native_step_ordinal, 20)
        self.assertEqual(frame.intended_timing_scope, "full_decode_step")
        self.assertIsNone(frame.gpu_elapsed_ns)
        self.assertIsNone(frame.existing_io)
        self.assertIsNone(frame.new_io)
        self.assertFalse(frame.gpu_verified)
        self.assertFalse(frame.production_qualified)

    def test_full_128_original_outputs_are_reconciled_independently(self):
        a = adapter()
        for offset in range(128):
            one_step(a, ordinal=20 + offset, context=128 + offset,
                     samples=[[offset]], start=100 + offset * 100, end=150 + offset * 100)
        result = a.close_run({"native-0": list(range(128))}, drain())
        self.assertEqual(len(result.frames), 128)
        self.assertEqual(result.outputs, (("native-0", tuple(range(128))),))
        self.assertEqual(result.runtime_hook_status, "not_installed")
        self.assertEqual(result.status, "CPU_FRAME_RECONCILIATION_ONLY")
        self.assertFalse(result.gpu_verified)
        self.assertFalse(result.production_qualified)
        with self.assertRaisesRegex(ValueError, "closed"):
            a.begin(148, 20000)

    def test_explicit_prefill_precedes_decode_and_is_not_discarded(self):
        a = adapter()
        prepared, _ = one_step(a, context=126, prompt=128, tokens=2, samples=[[5]])
        self.assertEqual((prepared.step_kind, prepared.active_decode, prepared.prefill_tokens),
                         ("prefill", 0, 2))
        one_step(a, ordinal=21, context=128, samples=[[6]], start=200, end=300)
        result = a.close_run({"native-0": [5, 6]}, drain())
        self.assertEqual([f.prepared.step_kind for f in result.frames], ["prefill", "decode"])

    def test_prefill_without_sampled_token_preserves_empty_original_output(self):
        a = adapter()
        one_step(a, context=100, prompt=128, tokens=16, samples=[[]])
        one_step(a, ordinal=21, context=116, prompt=128, tokens=12, samples=[[5]],
                 start=200, end=300)
        one_step(a, ordinal=22, context=128, samples=[[6]], start=300, end=400)
        result = a.close_run({"native-0": [5, 6]}, drain())
        self.assertEqual(result.frames[0].outputs, (("native-0", ()),))
        self.assertEqual(len(result.frames), 3)

    def test_actual_batch_two_cannot_be_relabelled_single_active(self):
        a = adapter()
        prepared, _ = one_step(a, ids=("native-0", "native-1"), samples=[[4], [9]])
        self.assertEqual((prepared.batch, prepared.active_decode), (2, 2))
        result = a.close_run({"native-0": [4], "native-1": [9]}, drain())
        self.assertEqual(len(result.outputs), 2)

    def test_off_returns_before_reading_any_runtime_or_event_object(self):
        class Poison:
            def __getattribute__(self, name):
                raise AssertionError("off must not read " + name)
        a = FullStepFrameAdapter(RUN, SOURCE, NATIVE)
        value = Poison()
        a.begin(value, value)
        a.prepared(value, value)
        a.executed(returned_none=value)
        a.sampled(value, value)
        a.close_run(value, value)
        self.assertEqual(a.frames, ())
        self.assertTrue(a.valid)

    def test_unsupported_original_geometry_is_unknown_and_never_changes_runner(self):
        cases = ("async", "async_unknown", "speculative", "spec_tokens", "pooling",
                 "tensor_parallel", "data_parallel", "pipeline_parallel", "bool_parallel",
                 "scheduled_cohort", "total", "bool_token", "negative_context", "mixed_context",
                 "mixed_prefill_decode", "multi_token_decode", "cross_prompt", "ordinal")
        for case in cases:
            with self.subTest(case=case):
                a = adapter()
                r, s = runner(), scheduler()
                if case == "async":
                    r.use_async_scheduling = True
                elif case == "async_unknown":
                    r.use_async_scheduling = None
                elif case == "speculative":
                    r.speculative_config = object()
                elif case == "spec_tokens":
                    s.scheduled_spec_decode_tokens = {"native-0": [8]}
                elif case == "pooling":
                    r.is_pooling_model = True
                elif case in ("tensor_parallel", "data_parallel", "pipeline_parallel"):
                    setattr(r.parallel_config, case + "_size", 2)
                elif case == "bool_parallel":
                    r.parallel_config.tensor_parallel_size = True
                elif case == "scheduled_cohort":
                    s.num_scheduled_tokens = {"foreign": 1}
                elif case == "total":
                    s.total_num_scheduled_tokens = 2
                elif case == "bool_token":
                    s.num_scheduled_tokens["native-0"] = True
                elif case == "negative_context":
                    r.input_batch.num_computed_tokens_cpu = [-1]
                elif case in ("mixed_context", "mixed_prefill_decode"):
                    r = runner(ids=("native-0", "native-1"))
                    s = scheduler(ids=("native-0", "native-1"))
                    if case == "mixed_context":
                        r.input_batch.num_computed_tokens_cpu[1] = 127
                    else:
                        r.input_batch.num_prompt_tokens[1] = 129
                elif case == "multi_token_decode":
                    s = scheduler(tokens=2)
                elif case == "cross_prompt":
                    r.input_batch.num_computed_tokens_cpu = [127]
                    s = scheduler(tokens=2)
                elif case == "ordinal":
                    r._profile_step = 99
                original_step = r._profile_step
                a.begin(20, 100)
                with self.assertRaises(ValueError):
                    a.prepared(r, s)
                self.assertEqual(r._profile_step, original_step)
                self.assertFalse(a.valid)
                self.assertEqual(a.frames, ())

    def test_actual_context_growth_cannot_be_bucketed_or_replayed(self):
        for context, prompt in ((128, 128), (130, 128), (129, 129)):
            with self.subTest(context=context, prompt=prompt):
                a = adapter()
                one_step(a)
                a.begin(21, 200)
                with self.assertRaisesRegex(ValueError, "progression"):
                    a.prepared(runner(21, context, prompt), scheduler())

    def test_missing_duplicate_or_reordered_original_ordinal_is_rejected(self):
        for ordinal in (20, 22):
            with self.subTest(ordinal=ordinal):
                a = adapter()
                one_step(a)
                with self.assertRaisesRegex(ValueError, "ordinal"):
                    a.begin(ordinal, 200)

    def test_overlap_or_backwards_host_interval_is_rejected(self):
        a = adapter()
        one_step(a)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            a.begin(21, 199)
        a = adapter()
        a.begin(20, 100)
        a.prepared(runner(), scheduler())
        a.executed(returned_none=True)
        with self.assertRaisesRegex(ValueError, "positive"):
            a.sampled(output(), 100)

    def test_actual_non_none_early_return_is_not_a_normal_two_call_step(self):
        a = adapter()
        a.begin(20, 100)
        a.prepared(runner(), scheduler())
        with self.assertRaisesRegex(ValueError, "await sample_tokens"):
            a.executed(returned_none=False)

    def test_sample_before_original_execute_or_twice_is_rejected(self):
        a = adapter()
        a.begin(20, 100)
        a.prepared(runner(), scheduler())
        with self.assertRaisesRegex(ValueError, "unexecuted"):
            a.sampled(output(), 200)
        a = adapter()
        one_step(a)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            a.sampled(output(), 300)

    def test_wrong_request_multitoken_tensor_or_missing_decode_output_is_rejected(self):
        variants = (output(("foreign",)), output(tokens=[[1, 2]]),
                    output(tokens=[[]]), NS(req_ids=["native-0"], sampled_token_ids=object(),
                                           req_id_to_index={"native-0": 0}),
                    NS(req_ids=["native-0"], sampled_token_ids=[[1]], req_id_to_index={"native-0": 1}),
                    output(tokens=[[True]]))
        for value in variants:
            with self.subTest(value=value):
                a = adapter()
                a.begin(20, 100)
                a.prepared(runner(), scheduler())
                a.executed(returned_none=True)
                with self.assertRaises(ValueError):
                    a.sampled(value, 200)

    def test_tampered_or_omitted_complete_output_is_rejected(self):
        for expected in ({"native-0": [8]}, {"foreign": [7]}, {"native-0": []}):
            with self.subTest(expected=expected):
                a = adapter()
                one_step(a)
                with self.assertRaisesRegex(ValueError, "reconstruct"):
                    a.close_run(expected, drain())

    def test_future_done_or_one_empty_queue_does_not_prove_actual_native_drain(self):
        cases = ("accounting_valid", "worker_joined", "ring_closed", "ring_drained",
                 "outstanding_records", "ring_outstanding", "pending_copies",
                 "active_native", "inflight_parents", "observation_failures")
        for name in cases:
            with self.subTest(name=name):
                a = adapter()
                one_step(a)
                value = False if name in ("accounting_valid", "worker_joined",
                                         "ring_closed", "ring_drained") else 1
                with self.assertRaisesRegex(ValueError, "closure"):
                    a.close_run({"native-0": [7]}, drain(**{name: value}))

    def test_wrong_drain_run_source_boundary_or_uncompleted_bytes_is_rejected(self):
        accepted = (StageTotal(1, 8),) + ZERO[1:]
        cases = (drain(run_id="foreign"), drain(native_source_sha256="c" * 64),
                 drain(boundary="future_done_only"), drain(accepted=accepted),
                 drain(ring_outstanding=False), drain(worker_joined=1))
        for value in cases:
            with self.subTest(value=value):
                a = adapter()
                one_step(a)
                with self.assertRaises(ValueError):
                    a.close_run({"native-0": [7]}, value)

    def test_all_actual_accepted_physical_stages_are_preserved_at_closure(self):
        accepted = (StageTotal(1, 8), StageTotal(2, 16), StageTotal(3, 24), StageTotal(4, 32))
        a = adapter()
        one_step(a)
        result = a.close_run({"native-0": [7]}, drain(accepted=accepted, completed=accepted,
                           transferred_bytes=(8, 16, 24, 32)))
        self.assertEqual(result.drain.accepted, accepted)
        self.assertFalse(result.production_qualified)

    def test_completed_requested_bytes_alone_cannot_hide_short_or_failed_io(self):
        accepted = (StageTotal(1, 8),) + ZERO[1:]
        cases = (drain(accepted=accepted, completed=accepted, transferred_bytes=(4, 0, 0, 0)),
                 drain(accepted=accepted, completed=accepted, transferred_bytes=(8, 0, 0, 0),
                       failed_ops=(1, 0, 0, 0)),
                 drain(accepted=accepted, completed=accepted, transferred_bytes=(True, 0, 0, 0)))
        for value in cases:
            with self.subTest(value=value):
                a = adapter()
                one_step(a)
                with self.assertRaises(ValueError):
                    a.close_run({"native-0": [7]}, value)

    def test_bounded_journal_refuses_to_silently_drop_a_full_step(self):
        a = adapter(max_steps=1)
        one_step(a)
        with self.assertRaisesRegex(ValueError, "bound"):
            a.begin(21, 200)
        self.assertFalse(a.valid)
        self.assertEqual(len(a.frames), 1)

    def test_retained_frames_are_immutable_copies_and_keep_no_runtime_owners(self):
        a = adapter()
        r, s, out = runner(), scheduler(), output()
        class Owner:
            pass
        owners = []
        for values in (r, s, out):
            owner = Owner()
            owner.__dict__.update(vars(values))
            owners.append(owner)
        r, s, out = owners
        refs = [weakref.ref(owner) for owner in owners]
        del owners, owner, values
        a.begin(20, 100)
        frame = a.prepared(r, s)
        a.executed(returned_none=True)
        closed = a.sampled(out, 200)
        r.input_batch.req_ids[0] = "mutated"
        r.input_batch.num_computed_tokens_cpu[0] = 999
        s.num_scheduled_tokens.clear()
        out.sampled_token_ids[0][0] = 999
        self.assertEqual(frame.rows[0].request_id, "native-0")
        self.assertEqual(frame.context_length, 128)
        self.assertEqual(closed.outputs, (("native-0", (7,)),))
        self.assertFalse(any(key in a.__dict__ for key in ("runner", "scheduler", "output", "future", "tensor")))
        del r, s, out
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_observer_fault_is_explicit_and_never_changes_original_exception(self):
        a = adapter()
        def original():
            raise RuntimeError("original-model-fault")
        with self.assertRaisesRegex(RuntimeError, "original-model-fault"):
            try:
                original()
            finally:
                a.invalidate("original-model-fault")
        self.assertFalse(a.valid)
        self.assertEqual(a.last_reason, "original-model-fault")

    def test_import_does_not_initialize_cuda_or_replace_executor(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("vllm", sys.modules)
        self.assertNotIn("pynvml", sys.modules)
        self.assertFalse(hasattr(adapter_module, "attach_observer"))
        self.assertFalse(hasattr(adapter_module, "execute_model"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

