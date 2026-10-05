# Derived immutable snapshot reuse

This change only avoids repeated `dataclasses.replace` and the accompanying
`SystemSnapshot.__post_init__` for one unchanged original observation during an
active single-file retry. The one-entry key contains the original snapshot's
object identity, the complete live-step scalar tuple and the runtime signature
prefix. Its retained values are frozen observations and scalar tuples; it holds
no runner, capture, ready descriptor, Future, job, slot or other owner.

Every call continues through the original `policy.issue_preview`, freshness and
maximum-wait checks, the second real live-state check and the existing deferral
recording path. No decision or permission is cached. Unknown or expired state,
different snapshot/state/runtime signature, a progress override, shadow mode,
live end before recording and bridge fault invalidate the derived value. Original
observation timestamps are preserved. Reactor retry-key checks are unchanged by
this bridge change; any separate reactor change must be reviewed independently.

The added CPU tests compare the actual frozen v3 `preview_issue` function with
the new function on the same value classes, including state transitions and
failure cases. A source-extracted original reactor replay executes 155 fixed
deferrals and verifies identical decisions, slot reserve/release counts and old
timestamps while `dataclasses.replace` calls fall from 156 to 2. The prior
16-test retry replay remains unmodified. CPU timing is a development diagnostic,
not a GPU result or a claim that total-request performance has improved.

Run the new test with `python -B -I -S test_enriched_snapshot_reuse.py`.
`SERVER11_AUTHOR_SOURCE_ROOT` selects the unchanged author/common package root;
`SERVER11_V3_CANDIDATE_ROOT` optionally locates the preserved v3 candidate.
