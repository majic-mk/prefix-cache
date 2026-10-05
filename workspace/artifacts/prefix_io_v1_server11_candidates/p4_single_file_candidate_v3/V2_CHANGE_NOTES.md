# Empty-window observation correction

V1 is preserved. Of all copied implementation sources, only reactor.py changes:
the original `_prefix_p4_order` and `_prefix_p4_read_batch_ids` return early for
an empty eligible window, bounded by the existing 32-parent/64-work limits.
Oversized and nonempty windows retain the original collection and decision path.
No queue, polling/sleep, cache lifetime, cost, budget, SLO, event or model hook
changes. Mandatory/continuation/terminal paths are unchanged.

The original `_has_work` includes retained shared staging. Consequently a
preload can keep the owner pumping. Previously three advisory entries built a
full publication before finding no eligible work. An independent source-extracted
CPU replay of 10,000 such iterations counts 30,000 collections in V1 and zero in
V2, with unchanged retained cache and `_has_work=True`. Its bridge observes one
epoch and no choices: those counters did not measure collection frequency.
The replay uses a counted empty publication, so its elapsed time is not the real
collection cost and is not GPU performance evidence.

Completion ETA still finishes through the unchanged `_prefix_p4_parent_terminal`
at original successful/failed completion. Nonempty H2D continuation still takes
its original `_prefix_stage_decide` collection before submission. Omitting empty
advice therefore does not create a completion or release witness. No ETA or
release qualification is added.

CPU validation: 8 new source/empty/nonempty/mandatory-window/off/oversized/terminal
regressions plus the copied 143 tests passed (151 total). The new test compares
V1/V2 ASTs to require exactly these two changed methods. For remote naming, set
`SERVER11_PREVIOUS_CANDIDATE_DIR` to the unchanged V1 candidate root before running
`python -B -I -S test_empty_p4_window.py`. Original integration/GPU qualification
must be run by the parent after freezing actual deployed source paths. No GPU was
used to develop or test this correction; no performance improvement is claimed.
