# Independent observer and binary-selection CPU review

The two optional observer seams are not installed in a real worker. This review does not complete a full-decode, original-output or final-drain GPU collector. All GPU and production qualifications remain false.

The first actual independent counterexample run found boundary double counting, inherited native-accounting invalidation not reflected in published telemetry, and host-clock backwards movement still allowing a mapped native/GPU axis. Evidence is retained in observer-counterexamples-02/result.json. The failed 01 review script only used the wrong host field spelling and is preserved separately.

After owner fixes, observer-counterexamples-03/result.json passes all three directed replays. Native operation boundary ties now return UNKNOWN; invalidation runs original accounting first and makes the scalar journal sticky-invalid; reversed host call ordering retains only explicit GPU-event duration diagnostics and returns unknown mapped wall coordinates. A nonboundary normal case remains diagnostic. Invalidating an unresolved SSD read preserves its original 8-byte inflight counter and does not invent a completion or release credit.

The new forward seam wraps original prepared input and model-forward calls only. It reads a bounded actual pure-decode batch, rejects speculative/async/distributed/mixed-context geometry, preserves original return/exception behavior, and resolves bounded event pairs by query without synchronization. It reports output_tokens and post_context as unknown. Host time is only call ordering; the explicit GPU event field is the duration. Cross-clock mapping requires a separate reference qualification. These properties do not certify a CUDA execution.

The native journal extends existing StageAccounting, calls original accept/complete first, retains only bounded scalar identities/events and immutable frames, adds no work queue or resource owner, and refuses missing coverage, incomplete/overflowed observations, partial completion, cross-clock ambiguity and multiple added stages. Only SSD read and H2D diagnostic attribution are attempted. D2H and store-write attribution remain unknown.

Nested-binary-spec-01/result.json independently generated a spec for an exact byte-bound nested extension while replacing extension create/exec entry points with faults. No binary was executed. Old adjacent Python fallback, changed SHA, ambiguous extension suffixes, invalid module names and oversized declared binary metadata were refused. Static inspection confirms 1-MiB streaming SHA blocks, a 512-MiB limit only for precise existing build vllm .so paths and a 64-MiB bound for ordinary source metadata. Locked source loaders compile the pinned Python bytes and refuse old Python/pyc fallback. This prepares source selection; it does not prove binary ABI or GPU availability.

Commands and full replays are saved as observer-counterexamples-02.py, observer-counterexamples-03.py and nested-binary-spec-01.py in this directory. They ran on the server using CUDA_VISIBLE_DEVICES empty, PYTHONDONTWRITEBYTECODE=1, project .venv/bin/python -I -S and a backend import guard. Backend imports and GPU/binary execution calls were zero. Original source SHA and the budget ledger were unchanged by the independent reviews.

Final reviewed observer SHA:
- p4_gpu_step_observation.py: 2dbd58eb095413520c5818cf5bb982ae6ca584933fc9b6db567eef41b29ca690
- p4_native_window_journal.py: 3a9ded39846cccf88ee9aceed2e16b86d8a7bbe953adfea3ba1a302b2c5aa8c3
- qualify_p4_native_gpu.py: 370c40dd5b4bc0bcf91a5ac9402d65ee80ea5426f34762632ebda9e11abd5491

The next allowed GPU stage still needs explicit scope/budget and begins with original off/shadow native primitive qualification. Measured method benefit, a complete causal GPU cost recorder and effect evaluation are outstanding.
