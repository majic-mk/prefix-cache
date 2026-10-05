# Exact optional capability query adapter

Actual off03 failed after model weights, sampler JIT and KV initialization had
progressed. The error was the locked importer rejecting the missing optional
vllm.third_party.deep_gemm probe. No request phase completed. The author's
cached _has_module expects find_spec returning no spec to mean unavailable.

This separate CPU candidate leaves the original BoundAuthorFinder and all
author files unchanged. It explicitly wraps only the loaded author's
import_utils._has_module in the current process, after the original runtime
imports and before LLM construction. It always calls the original cached
function first. Original results, external deep_gemm availability and cache
success pass through. Other errors propagate as the same original object.

Only the exact vendored optional name, original builtin missing-error fields,
three exact original traceback code objects and the same original finder can
trigger a False result. Before that result, all five original source pins,
actual absent file/package/namespace/extension/link variants, unique locked
runtime parent path, standard PathFinder and path importer contract are
rechecked. Drift cannot produce False. This does not install DeepGEMM, disable
kernels, invent a module, change numerics or alter model execution.

No finder returns None, so a later custom meta finder cannot fill the absent
module after our adapter. Direct required imports still encounter the original
hard rejection. The adapter lives only around the capability query, and detach
restores only its own wrapper while preserving any later foreign override.

verify_optional_absence(root, refs) is the source-only preflight API and returns
a plain witness with runtime/GPU qualification false. It creates no directories,
changes no environment and executes no author module. Runtime installation API:
install_capability_probe(import_utils_module, original_finder, root, refs),
returning handle.detach() and handle.evidence(). The locked original qualifier
module must already be registered by the existing loader. Runtime/source gates
and separate human authorization remain the future driver's responsibility.

Local Python3.12 explicit -B -I -S tests:13/13 PASS0.788s, zero skips. Tests use
exact original finder source and independently extracted original capability
AST in a synthetic locked parent/module. No framework, compiler, .so or GPU is
executed. A Windows symlink rejection is an explicit fault mock, not an actual
link. Server command and exact source refs are in the plan. The unused earlier
finder prototype is preserved only locally in historical-before-capability-probe-design
and is excluded from runtime/delivery refs.

Independent review and actual server source/absence/12-test verification are
next. This candidate gives no GPU authority or normal-model/output/performance
qualification, and it does not make the failed off03 eligible for retry.

V2 preserves the prior frozen five files unchanged. The only source repair
checks exact builtin exception type before reading ANY exception attribute,
then checks plain name/args/message before traceback. A legitimate subclass
with poisoned traceback still propagates as the same object with zero custom
attribute reads; one new contract records baseline/wrapper each once. Existing
12 contracts and all source/finder/model/kernel/environment behavior remain
unchanged. Final review must pass before the new normal V3 runner is assembled.
