# Native KV byte diagnosis

This helper is restricted to the existing original G3 acquisition: Qwen2.5-7B,
size 128, domain 1024, three repetitions, one canonical KV group, ordinary
synchronous single-rank generation. It does not execute models or authorize GPU
work. The existing guarded launcher retains budget, GPU UUID, source closure,
publication, original acquisition and shutdown checks.

## Adapter API

Load `production_kv_capture.py` from the new frozen source list. In the private
original acquisition `worker_control` wrapper, after the first original call
has returned and the original unique handler exists, call once:

```python
handle = capture.install_capture(
    worker, project_root, source_refs, output_dir / "kv-capture", mode,
    producer_manifest=producer_receipt_path if mode == "paired" else None,
)
```

`source_refs` is the existing path-to-ref mapping. `mode` is `populate` or
`paired`. The output directory must be new. The paired producer path is exactly
the populate job's `kv-capture/native-kv-byte-receipt.json`.

After the original acquisition has completed its original shutdown (including
when the acquisition fails), call `handle.detach()` and then `handle.finish()`.
`finish()` writes `native-kv-byte-receipt.json` once and returns its JSON object.
It requires the original reactor to be closed with no pending CUDA copies.
The caller must still inspect the original acquisition exit status and complete
shutdown evidence. A capture PASS never overrides an original failure.

Call `capture.validate_receipt(receipt, mode=mode)` before advancing. It returns
True or raises. It checks receipt closure only; the launcher must bind the actual
receipt file/ref, original acquisition rows, publication, frozen helper/source,
GPU UUID and original shutdown. Do not treat a caller-created JSON fixture as
native evidence.

## Observation and bounds

Four pinned original methods receive private AST copies with one observation
call inserted per method. Original decorators and statements remain. Methods
are restored on detach; no author file is edited. Callbacks do not consume
`get_finished`, submit transfers, change queues, wait/release ownership, complete
Futures or replace the executor.

The native CUDA callback executes only after the original successful
`end_event.query()` branch, before original pending removal or terminal handling.
The producer source is read after D2H completion while whole-parent protection
still holds, and is compared byte for byte with the original staging slot. This
is the actual order; no earlier source-capture event is fabricated. The consumer
target is likewise read after successful original H2D completion and before
publication. Its complete bytes equal staging and the actual producer payload.

Write observation follows original successful CQE validation and `finish_write`.
Read observation receives the original read completion and validates its exact
byte count; preloaded/shared reads retain their original route. Each real
consumer's model-forward boundary requires all eight loaded files, current
request block IDs, and the same complete prompt identity as its producer.

Populate must cover three producers and three staging consumers. Paired must
cover six consumers, one real-read then one staging-hit request per prefix.
Every cohort contains exactly its producer's eight unique full-file keys.
Maximum binary output is **24 × 917,504 = 22,020,096 bytes**, written only once by
populate. Paired reads those files without duplicating them. Full staging/GPU/SSD
bytes are compared, then only SHA and provenance scalars are retained. No sampling
of bytes is used. CPU byte temporaries peak at approximately 4 MiB; one canonical
layer page is copied from GPU at a time. At most 512 scalar records are retained.

Ordinary callback errors mark the diagnostic failed and allow original cleanup.
Termination signals are recorded and re-raised. Missing records, unsupported
geometry, mismatched bytes, overflow, early compute and source drift cannot PASS.

## Meaning of a PASS

`PASS_NATIVE_KV_BYTE_DIAGNOSTIC` establishes only the captured finite native byte
paths. `production_qualified`, `cost_qualified`, `effect_verified`, and
`timing_usable` remain false. LoadPlanner is off. Extra readback and blocking CPU
work perturb timings, so these timings must not be fitted as costs or compared
as performance. CPU tests do not establish GPU provenance.

Local CPU command (Python 3.12):

```text
C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe -B artifacts/prefix_io_v1_server10_candidates/kv_capture_v1/test_production_kv_capture.py
```

The real-source transformation test reads the existing local read-only mirrors.
When deployed to the server, set `KV_CAPTURE_SOURCE_ROOT` to the project root so
the same test checks actual frozen files there.
