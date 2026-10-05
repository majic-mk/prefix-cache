# Canonical parent importer repair

Actual off04 stopped with unknown path hook, phase0, exit1, session drained.
Root reproduced it on CPU with normal -B site startup: two standard hooks plus
the installed editable vLLM namespace hook. The earlier -I -S fixture startup
skipped .pth initialization. The author's canonical third_party directory was
already backed by a genuine standard cached FileFinder; global hook count was
an unrelated condition.

This new helper preserves the frozen V2 files. Its only implementation change
validates that alreadyexisting canonical parent cache: exact FileFinder class,
plain path, exact list/tuple/string loader fields and standard loader identities,
no instance override. Missing/None/unknown or changed cache refuses. The original
secondary PathFinder query remains, and the same cached object must survive it.
Global hooks for other directories are not read, called, removed or modified.
No .pth/package/system/backend/model/kernel change is made.

Original BoundAuthorFinder, source pins, cached capability function firstcall,
exception scalar/type/traceback proof, sole loaded author parent path, physical
absence checks, required import rejection and restore API are unchanged. The
capability adapter still returns False only after the exact original missing
error and bounded absence proof. Unmatched errors propagate as the same object.
No meta finder returns None or allows an unlocked fallback.

Local Python3.12 -B -I -S:14/14 PASS0.923s, zero skips. The existing cache contract
now covers missing/None/unknown/override/scalar drift and instance replacement.
One new contract compiles the actual installed editable finder source4717B,
SHA11446ee6db26a2901e8df4e15096cc393a2e819426cdd397efe803aa6d2f2753,
without calling install, then proves the third hook is not invoked by original
or secondary queries using the same cache. No framework/compiler/GPU/RPC runs.

Root must additionally replay actual CPU normal-site startup without -S, match
the real child mode, and run the server14 source contracts before assembling the
new normal V4/names05/source lock. The old failed04 attempt is closed; shadow04
is blocked. This source candidate grants no GPU authority or normal-output,
collector, cost, production or effect qualification. Commands and exact source
refs are in OPTIONAL_CAPABILITY_CPU_PLAN.json. API names remain
verify_optional_absence and install_capability_probe with detach/evidence.
