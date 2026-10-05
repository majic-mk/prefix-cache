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
Local CPython3.12.14 -B -I -S tests passed40/40; actual server tests and complete
new G2 source-lock preflight remain root's next step. No new GPU operation,
remote write, live model hook, cache engine edit or framework import was
performed by this CPU preparation.

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
