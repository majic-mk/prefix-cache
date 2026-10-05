# Bounded existing I runtime input

Five private source copies incrementally add a development lookup over one
actual A/B SSD cost cell. Old frozen sources, native queues, model execution,
resource reservations, completion and release protocols are unchanged.

`control/p4_cost_table.py::load_gpu_development_table(root, refs,
pair_binding_ref, driver=..., expected_gpu_uuid=...,
expected_common_runtime_domain_sha256=...)` requires actual frozen files and
the source-pinned `driver.validate_actual_development_pair` callback. That
callback must replay the original completed guards, raw frontend/CUDA/native
I/O evidence and return exactly the material in the binding document. Missing
inputs or callback reject before a development table exists. The thin actual
consumer and its validator are owned by the root agent; this package does not
invent them or their results.

Binding JSON has exactly `schema="bounded_I_actual_AB_cost_input_v1"`,
`material`, and `evidence_refs`. Material has exactly:

```
gpu_uuid, common_runtime_domain_sha256, source_lock_ref, model_manifest_ref,
kv_layout_ref, native_source_ref, collector_source_ref, cuda_event_source_ref,
kernel_mode, context_length, a_ns, b_ns, budget_ns, stage, physical_bytes,
operations, existing_io, batch, active_decode, prefill_tokens
```

References are actual project-relative `{path,bytes,sha256}` rows in the current
source closure. The one cell is single decode (`batch=active_decode=1`,
`prefill_tokens=0`), actual prepared context, `stage="ssd_read"`,
`physical_bytes=917504`, `operations=1`, four ZERO existing-I/O amounts,
`kernel_mode="eager"`. `budget_ns=a_ns` is an explicitly exploratory observed
A engineering threshold. Baseline=A, delta=max(0,B-A), uncertainty=0. Thus the
total is max(A,B); these observed values have no statistical upper-bound,
holdout, SLO, or production interpretation. Other contexts, stages, byte sizes,
I/O states or hardware/common-domain identities fall back to the original path.

The public CostTable constructor cannot select `gpu_development`. Only the
actual replay loader creates that scope. Lookup estimates remain
`mock_only=False`, `production_qualified=False`; the production and CPU-mock
lookups cannot consume it. `choose_batch` and bridge batching remain unchanged.
This is a source-bound process contract, not a security boundary against
arbitrary malicious Python code mutating private objects in the same process.

`table.development_input` exposes immutable scalar provenance fields:
`gpu_uuid`, `common_runtime_domain_sha256`, `source_lock_sha256`,
`model_sha256`, `kv_layout_sha256`, `kernel_mode`, `native_source_sha256`,
`collector_source_sha256`, `cuda_event_source_sha256`, `runtime_refs`, `cells`,
`budget_ns`, `a_ns`, `b_ns`. `runtime_refs` is a tuple of `(path,bytes,sha256)`
rows. The exact signature is `(GPUuuid,commonDomainSHA,modelManifestSHA,
layoutSHA,1,1,0,actualContext,917504)`.

The existing runtime can call `verify_running_identity(...,issuer=None,...)`
and the existing `FiniteStartup` with `issuer=None`,
`budget_ns=table.development_input.budget_ns`, `observation_only=False`,
`reserve_covered_cell_signatures=None`. Hardware/model/layout/source checks and
the original owner installation/drain/detach remain. The finite binding adds
the development load capability; it never adds a production load capability.
Its eligible-preview records are **proposals**, not proof of actual blocked
native dispatch or performance benefit. Actual ordinary ready/dispatch,
submission and completion must be evidenced by the existing native boundaries.

CPU command (local):

```
python -B -I -S test_bounded_i_cpu.py
```

Server command accepts the existing unchanged control source directory:

```
python -B -I -S test_bounded_i_cpu.py --original-control-root <original-control-root>
```

Nine tests pass. Synthetic tables are explicit in-memory CPU branch fixtures,
never written as receipts or supplied to a model/GPU. The first test execution
had three fixture errors (weak-reference type, malformed-signature expectation,
and original runner path); only tests were corrected. No actual tokenizer,
model, GPU, SSH, driver/system/package mutation, download or deletion occurred.
Real cost data and GPU validation are not supplied by this CPU package.
