# Final CPU source freeze supplement

Latest owned source refs: `gpu-next-day/owned-source-lock-v6.json`. It contains
the last seven modified owned files. Earlier unchanged files are in v5.
The raw-pair CLI is now verifier-agent owned; its V2 source/test receipts are
separate. All source editing by this agent has stopped.

## Independent red evidence and precise fixes

ETA's actual independent counterexamples are preserved at
`eta/review-integration/observer-counterexamples-02/result.json`.
The initial reviewer script typo in run-01 also remains as historical evidence.

1. GPU/native boundary timestamp ties are conservatively `UNKNOWN`.
   The raw accepted/completed counters remain intact; an operation cannot be
   counted in both existing I/O and added I/O.
2. Original `StageAccounting.invalidate()` executes first. The journal then
   publishes a sticky invalid flag without worker-side native-stat reads.
   The original unresolved inflight counters remain preserved.
3. Reversed/equal/unknown host call times do not supply a valid host clock or
   mapped native window. GPU event elapsed may remain a diagnostic, while
   host/native attribution stays unknown. Full-step and production qualifications
   remain false.

Targeted final regression: **3 passed, 42 deselected**, saved at
`observation-cpu/run-03-review-fixes/{tests.log,tests.xml,CPU_GUARD.json}`.
The previous full 42-case CPU set remains in run-02; root's unified regression
will cover all 45 cases after the final source freeze.

## Exact binary metadata and module-path preparation

The existing pure-stdlib source-ref helper now allows 512 MiB only for exact
`third_party/work/vllm-author-build/vllm/... .so` paths after the same
canonical/symlink checks. Ordinary sources and every other path retain the
64 MiB bound. Hashing streams 1 MiB blocks; source metadata is not GPU permission.

The author finder can prepare a nested old binary module spec when the new
Python package search path has no extension: it matches the fullname and
platform EXTENSION_SUFFIXES against individually locked refs, requires a
single exact candidate, validates path/bytes/SHA and constructs an
ExtensionFileLoader spec. Unknown, ambiguous and changed candidates fail.
It never scans or imports old Python as a fallback. CPU tests construct fake
extension specs without module construction/execution or binary loading.

Targeted result: **77 passed** at
`gpu-next-day/cpu-tests-09-binary-path.{log,xml}`; guard receipt is
`cpu-test-guard-09-binary-path.json`. Tests include actual sparse CPU files
at source/binary size boundaries, streaming-read enforcement, wrong-tree paths,
nested known module lookup, no unknown old-file reads, ambiguous suffixes,
source priority and metadata/path drift.

All targeted runs imported zero GPU backends, made zero GPU runs and left the
GPU ledger byte hash unchanged. The exact commands are
`cpu-test-command-09-binary-path.py` and `cpu-test-command-observer-03.py`.

The seven old-binary metadata refs are
`observation-cpu/source-lock-v1.json` → `old_binary_inventory`.
Root may pin all seven byte refs in the final source metadata lock. That is
not ABI validation or broader GPU scope. G1 still loads no model and remains
off/shadow primitive qualification only. Actual module origins, binary ABI,
runtime seam installation, complete sampling/output/drain collection and
model benefit have not run today.
