# Server11 bounded full-step collector

`native_full_step_collector.py` reuses the frozen G2 worker observer, its scalar
connector and full-step adapter. The original model executor is not copied or
changed. Installation is performed only by the separately source-locked,
budget-guarded GPU driver after actual environment checks.

The root driver calls:

```python
capture = collector.install(
    worker, common=original_g2_module, root=project_root, refs=source_refs,
    run_id=unique_window_id, event_class=torch.cuda.Event,
    selected_offsets=(16,), action=preload_callback_or_none)
# Original add_request/step loop returns all 128 output tokens.
raw = capture.export()
capture.detach()
```

The callback takes `(step_offset, native_step_ordinal)` and calls only the
existing native `handler.preload_async`. It returns bounded plain metadata.
The driver owns the handler, native owner journal and final original shutdown.
This collector does not consume native completions, alter queues, grant release
credit, or perform original shutdown on the driver's behalf.

The original `_prepare_inputs` finishes before the selected callback. The
actual GPU start event must already return `query() is True`; otherwise the
capture fails and it does not issue experimental I/O. No spin, extra wait,
`synchronize`, or enlarged window is used to create an apparent overlap.
Ordinary observer exceptions invalidate evidence while the original model
continues; control signals and original errors propagate without retries.

Every original execute/sample frame remains unchanged, including zero or
positive pre-context prefill, every native ordinal and every sampled token.
The existing G2 frame's `gpu_elapsed_ns` stays `None`. A separate per-ordinal
event witness records the genuine CUDA event elapsed duration and event API
host brackets. The duration is cross-checked against the original G2 event
diagnostic, after both real events query complete. A source hash describes the
actual installed Torch Event API; root must bind that source and native runtime
to the new source lock and preserve it after execution.

There is no fabricated GPU-to-host absolute clock mapping. A real owner event
strictly after `start_completed_query_ns` and strictly before
`end_record_before_ns` is causally inside the GPU event interval. The native
validator must additionally require that it belongs to the actual original
host step, is SSD-read only, has the frozen physical amount, completes with
the exact actual bytes, and does not hide other accepted I/O. Boundary ties,
missing completion, an unready selected start, or insufficient causal coverage
invalidate the selected cost window. A successful capture alone does not
establish a qualified production cost table or a policy performance effect.

Host step boundaries, native request ID/full output reconciliation, complete
owner events, window-end idle snapshots, final original shutdown, job source
and GPU identities, and AB/BA split binding remain separate validator inputs.
The driver must use one unchanged observer configuration for all A/B windows.

The local CPU test uses Python 3.12 and the actual unchanged G2 observer source
with fake CUDA events and a minimal fake original runner. It checks full 128
step integration, exact original once-calls and results, failures, event bounds,
no added synchronization, rejection of fake native clock/class identity, and
detachment. CPU fixture success is not GPU evidence.

Server CPU replay (with actual absolute source paths):

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S \
  <collector-directory>/test_native_full_step_collector.py \
  --g2-source artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/g2_worker_observation.py \
  --scalar-source artifacts/prefix_io_v1/server09-runtime-connector-20261001/p4_runtime_scalar_connector.py \
  --frame-source artifacts/prefix_io_v1/server09-migration-20261001/collector-candidate/p4_full_step_frame_adapter.py -v
```

At this delivery: local CPU **16/16 passed**; this subtask performed no RPC,
GPU operation, driver/package change, model download, or storage deletion.
