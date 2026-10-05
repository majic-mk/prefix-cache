# Initial CPU boundary review of the optional capability repair

Status: conditional common-fix direction accepted; draft implementation issues below require repair and final independent review. This document grants no GPU permission and does not claim native runtime qualification.

The read-only evidence is under `artifacts/prefix_io_v1_server09_g2_ninja_20261001/source-readonly` and `runs/off`. The real off03 result is `FAILED_NORMAL_MODEL_LIFECYCLE`, with no completed phases. Its error is `vllm.third_party.deep_gemm has no exact locked binary fallback`; its source lock is the frozen 4,040-row candidate, SHA-256 `7b8445363e985a7dc4bff08960b31ee4cc325e3d309322f7f4b15d5b0f8aa423`. This is an importer/capability-detection failure, not evidence against the scheduling method or evidence of generated model outputs.

## Original source and failure semantics

The actual mirrored `qualify_p4_native_gpu.py` is 29,692 bytes, SHA-256 `5d9604fc07d80495a49febfcd488cb2601f74a2bef107261c8e623a4d0a1c77d`. Its `BoundAuthorFinder.find_spec` delegates every missing author module to `locked_binary_spec`; the latter raises when no exact locked binary exists.

The actual `vllm/utils/import_utils.py` is 14,589 bytes, SHA-256 `f14b07131a4d49955150016dbfa34f7d8d5f586b1545a1173e0d4f904578b18b`. `_has_module` is the original cached `find_spec(...) is not None` predicate. `has_deep_gemm` probes external `deep_gemm` first, then the vendored fullname. It does not catch the finder's `ModuleNotFoundError`. A local AST-only check confirmed both intended `None -> False` semantics and propagation of the strict missing-extension error; no author framework was imported. The later directory lookup in that grouped command returned exit 1 because the new candidate directory did not yet exist, rather than because its AST assertions failed.

The actual server CPU reproduction receipt `ACTUAL_SERVER_CPU_ORIGINAL_PROBE_REPRODUCTION.json` independently records the unchanged source chain: `has_deep_gemm` -> `_has_module` -> standard import machinery -> original `BoundAuthorFinder.find_spec` -> `locked_binary_spec`. It is a CPU reproduction with source-bound extracted functions, not a live model run. The original parent `vllm/third_party/__init__.py` is empty and SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; the actual directory inventory contains no deep_gemm module/package.

## Accepted minimal direction

Preserve the original finder and all author files. A reversible process-local adapter may wrap only the already loaded, byte-pinned `import_utils._has_module`. It must call the original cached function first. Only the exact, proved original missing-vendored-module exception may become `False`. External `deep_gemm`, positive results, unrelated names, ordinary errors, and required imports retain their original behavior.

Changing the finder to return `None` has a downstream lookup hazard: later meta finders can still return an unlocked spec. The selected exception-adapter direction avoids admitting that fallback: the original strict finder has already stopped lookup with a verified exception. It neither installs a module nor disables an environment option, changes precision/kernels/model configuration, or replaces an executor/cache engine.

For the exception conversion, require exact plain-string fullname/name/message, exact `ModuleNotFoundError` type, the source-bound original cached function and `__wrapped__` code/globals identities, and the original finder instance remaining first in `sys.meta_path`. The traceback must include the original `_has_module`, `find_spec`, and `locked_binary_spec` code identities in the expected order. Finder frame locals must identify that same instance and exact fullname. A message alone is insufficient provenance.

Revalidate the sole canonical loaded author parent path, its empty pinned initializer, all physical module/package/namespace/bytecode/extension/broken-link variants, and absence of matching author or locked-binary refs. The original standard `PathFinder` must really return `None` for that parent, under a verified standard path hook/importer contract. Present, unlocked, changed, ambiguous, cached/preloaded or unknown sources must never be hidden as `False`.

Do not clear or rebuild the original cache. Restore the module attribute only if it still equals the adapter's wrapper; preserve a foreign later override. Installation occurs after the existing guard/source checks and original imports; restoration belongs in the original runner's final cleanup. No live hook was installed by this review.

## Draft issues observed before final freeze

The first capability-wrapper draft called `_verify_absence` before calling the original function for the exact optional name. That violates the agreed original-first contract and can hide the original rejection or positive cached result. Move absence verification into the exact verified exception branch.

The same draft used `require(...)` while classifying unrecognized exceptions, which can replace the original exception object with a new `ValueError`. Nonmatching exceptions must propagate with bare `raise`; malformed or unknown absence evidence must fail closed without converting to `False`. The exception proof should additionally bind the original `__wrapped__` traceback frame and globals, not only the two finder frames. These findings were sent to root and the implementation agent; this initial document is not a final PASS for an evolving draft.

## Downstream DeepGEMM warmup gate verified on CPU

The first inspection found a possible secondary error: `_fp8_linear_may_use_deep_gemm` asks for alignment before checking whether the layer is FP8. A direct call with missing DeepGEMM can therefore reach `_missing()`. Root supplied the actual outer `kernel_warmup.py`, 6,872 bytes, SHA-256 `9353b419dfdb63e49f38d98e8ac38d9f303ac6ac9d83ca990d628d2cf1940a78`, already present in the frozen 4,040-row lock.

Its unchanged first assignment/if checks `is_deep_gemm_supported()` before reading the model or calling DeepGEMM warmup. An exact AST slice of those original statements, coupled with original cached capability/support functions, passed two CPU cases:

- With `VLLM_USE_DEEP_GEMM=True`, supported architecture metadata and warmup mode `once` rather than `skip`, both optional probes return `None`; the original support predicate is false. A worker that throws on every attribute access was never touched, and the DeepGEMM warmup callback was never called.
- With a present capability positive control, the unchanged branch reaches `worker.get_model` and raises the identical sentinel exception, proving the test did not silently conceal model access.

The actual command exited 0. Receipt: `INDEPENDENT_ORIGINAL_OUTER_GUARD_CPU_RESULT.json`, 1,778 bytes, SHA-256 `bfbbe49e4ebaaf8c6a5e38833fe4f60d01398f06e979f444e728f2f46126fcbf`. This closes the specific outer-guard concern without changing configuration. Other FlashInfer/attention branches and actual runtime compatibility are outside this CPU proof.

## Required next evidence

Final helper tests should cover the real original finder/function AST, exact absence, positive cached/uncached returns, external probes, discovered unlocked source/namespace/link, wrong parent paths, forged message and traceback, another finder instance, altered path hooks/importer cache, source drift, exception identity, cache preservation and restoration/foreign override. Freeze the actual helper/plan refs and independently review them before preparing a new runner, source lock or concrete GPU proposal. Current failed names/scopes and the single accumulated ledger remain unchanged; no retry or slot reuse is authorized here.

This review performed local reads, two original outer-guard CPU cases and append-new review artifacts only. Server operations, framework imports, GPU operations, compiler executions, source edits, authorization edits and budget edits were zero.
