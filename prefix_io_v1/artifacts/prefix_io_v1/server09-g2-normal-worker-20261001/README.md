# G2 normal model lifecycle CPU candidate

This is a concrete thin future GPU entry over the author's existing LLM,
SamplingParams, UniProcExecutor, add_request/step and EngineCore shutdown. It
does not implement another executor or activate a research strategy. Importing
it, printing its scope template, and running source-only preflight use stdlib.
The default has no new G2 GPU authority. G1's two native-only jobs cannot be
reused for this model phase.

Each proposed job (off, then shadow) makes cold and repeated requests with the
same 128 prompt IDs, each producing exactly128 outputs. Two jobs yield four
complete128-token arrays (512 outputs). Original Prefix Cache is enabled; cold
cached tokens must be0 and repeat112. All four output arrays must later compare
exactly equal. This is model/runtime/output lifecycle qualification only; it
cannot establish our research method's effect or SSD performance.

The shadow candidate connects the frozen scalar adapter to original execute,
prepare and sample methods, with each original method called once. End events
are recorded only after real sample return and scalar closure. Its raw current
stream Event diagnostics have no proved hostclock mapping: mapped endpoints
and gpu_elapsed_ns remain None and all cost/production flags remain False.
Pending events are queried, never synchronized or waited upon. Cold context0
is preserved. native_io is none, so journal/drain evidence is not_applicable;
the separately prepared native drain provider is not installed or consumed by
this phase. Only original model EngineCore shutdown releases its resources.

Ordinary observation exceptions fail closed to original progress. Original
exceptions/control signals propagate as the same object and are never retried.
Observer KeyboardInterrupt/SystemExit are transparent cancellation signals:
they can cancel before execute or after a sample already completed, in keeping
with the frozen scalar connector. They never justify re-executing a method.

The source-only delivery and exact commands are in G2_NORMAL_MODEL_CPU_PLAN.json.
Local CPython3.12.14 -B -I -S tests passed45/45; actual server tests and complete
new G2 source-lock preflight remain root's next step. No new GPU operation,
remote write, live model hook, cache engine edit or framework import was
performed by this CPU preparation.

The previous eight-file freeze is preserved byte-for-byte under
historical-before-runtime-cache-repair. Root found that original smoke keys
omitted FlashInfer/CUDA-driver/Torch-extension caches. Actual pinned server
sources now bind FLASHINFER_WORKSPACE_BASE, CUDA_CACHE_PATH and
TORCH_EXTENSIONS_DIR; these join the original keys under the new run's
runtime-cache before any framework import. No JIT is disabled. CPU replay of
the actual FlashInfer default expressions proves its .cache/flashinfer path
stays beneath the new workspace. Framework/GPU import attempt flags now precede
first import, so an import-time CUDA probe/failure cannot appear as unattempted.
Existing cloned-cache CPU metadata (~32.12MiB) informs the proposed128MiB
reserve and does not prove a new model's first-startup cache upper bound.

The second frozen nine-file version is preserved under
historical-before-origin-type-repair. Independent review reproduced an owner
retention risk from custom equality objects in origin/scope. These fields now
require exact builtin strings, including at connection preflight; subclass and
owner-GC counterexamples are covered. Completed frontend output evidence is
also saved before a shadow frame-reconciliation failure can occur.

The third frozen ten-file version is preserved under
historical-before-phase-export-repair. After an original request completes,
the driver immediately appends that phase and saves the complete128 output IDs
as cold-frontend.json or repeat-frontend.json in the new details directory,
before export/query or reconciliation. An export exception or later budget
timeout therefore cannot discard an already completed output array. CPU tests
replay that exact driver slice with a failing export and check both the
unchanged frontend object and the persisted complete array.

Future launch is blocked until root records actual new human approval and
freezes a distinct G2 scope with the new complete source lock, same UUID,
original eight-hour cumulative budget and exact child/session/permit binding.
The template is NOT_AUTHORIZED and is not a human record. Proposed limits are
300seconds per child plus20seconds termination reserve, two jobs/640seconds
total planned reserve, with128MiB proposed new PRIMARY storage reserve. Actual
first-startup compiler cache size is unknown on CPU; root must review the
reserve before approval and must never silently enlarge it.

Run off first only after approval. On failure preserve all logs and the original
budget ledger and perform CPU diagnosis; do not retry or launch shadow.
Successful/drained off is also checked before shadow. After shadow, root must
retain all four full output arrays, compare exact IDs and audit original
shutdown and the original guard's final session/ledger closure.
