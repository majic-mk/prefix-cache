# P4 runtime v4 / actual C4 calibration v6

This runtime uses the actual candidate-v4 common reactor and new six-window
native calibration `server11-native-cost-six-window-06`. Its source, measurements
and guard references are under `server11-native-cost-v6-20261003`; v5 cost values
and its source-lock SHA do not qualify v4. The copied harness test retains the
historical v5 launcher only as an unchanged-original-function AST reference.

Before freeze, root must independently finish and validate all six v6 windows,
the original guard shutdown/session drain, before/after source checks, raw
serialization, original fit/heldout calculation and actual C4 source binding.
No new cost numbers are present in this directory. The receipt factory replays
all that evidence and derives the original calibration-only upper bound and the
A-only engineering budget. Formal service SLOs remain unknown.

Freeze is append-only and requires the exact newly verified v6 source-lock SHA:

```text
python -B -I -S control_p4_single_file.py freeze --calibration-lock-sha256 <actual-v6-lock-sha256>
```

The binding puts the actual C4 reactor in `runtime_common_refs`, removes that
duplicate from overlay refs, and retains all other candidate/runtime source
references. The C4 receipt requires every runtime C4 source byte to match the v6
calibration closure. `prepare` strictly replays the receipt before writing any
arm config/scope and records its actual dynamic bounds. Missing/failed v6
evidence blocks preparation. The original 300+20-second arm limits and original
guard/cumulative budget are unchanged.

Arm labels are `off-04`, `shadow-04`, `on-04`, in that conditional order. The
independent runtime validator uses the same strict receipt-derived bound and
requires the runtime reactor to be a common calibration source. It preserves
the distinction between functional qualification and failure of the selected
step cost-migration gate in on mode. No GPU benefit is inferred from this
preparation or CPU fixtures.

The actual running owner is additionally checked before measured requests:
`type(reactor).__module__` must be the loaded C4 `py_kvcache.reactor`; the original
`_run` method is verified by the existing source/code-object verifier and its
code filename must be the C4 file. Every actually loaded `py_kvcache` and
`prefix_io_control` module must reside under C4 and match its locked bytes. The
raw `native_source_binding` records those references, parent cap eight and exact
requested bridge identity. The independent validator ties this binding to the
receipt's common C4 source and actual runtime identity evidence. Merely having
both a legacy and a C4 source in the inherited lock is insufficient. Binding
runs before capture attachment, so on mode may still have the initial shadow
flag; existing measured before/after policy snapshots verify the final mode.
