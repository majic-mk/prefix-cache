# Existing CUDA13 SDK layout

This independent candidate fixes only the process-local layout required by
the existing FlashInfer compiler/version/link selection. It does not grant a
GPU run, change old frozen normal sources, skip architecture checks, disable
JIT or alter the model/cache executor.

The original audited server inventory contains1934 regular bin/include/nvvm
files (217111529bytes), plus byte-bound actual driver and libcudart13 refs.
The original CPU compiler proof already passed nvcc13.0.88 SM120f object
compilation and host linking, without loading the resulting shared object.
The helper consumes those original schemas, hashes all actual files and
rejects drift, extra/symlink files, incorrect counters or failed proof.

`load_audited_assets()` and the CLI perform read-only preflight: no mkdir,
framework import, compiler command or GPU call. `prepare_overlay()` creates
only an exclusive new `<approved run>/details/runtime-cache/cuda13-sdk`.
It links existing bin/include/nvvm, creates a real lib64, aliases libcudart.so
and libcudart.so.13 to the existing runtime, and links lib64/stubs/libcuda.so
to the actual existing driver binary for the original linker command only.
The stubs directory never enters LD_LIBRARY_PATH. Assets are checked again
after creation; failed creation leaves partial new evidence and blocks startup.

The returned environment is a copy for the future guarded child only. Roots
and compiler are fixed to the overlay; PATH only prepends its bin and runtime
library search only prepends its lib64. Foreign root/compiler overrides,
FlashInfer extra flags/launchers, architecture overrides and disabled JIT are
rejected. The helper never changes os.environ or system/driver files.

Local Windows tests:12 defined,10 passed,2 explicit Linux symlink/containment
skips. Actual Windows symlink creation failed WinError1314; no local success
is claimed. The first AST harness had two Windows separator assertions that
were corrected to match original os.path.join. Receipt retains these failures.
Root must run all12 on Linux with no skips, and real asset preflight, before
preparing a separate minimal normal clone with this SDK hook and new names02.
All GPU/model/JIT/effect qualification remains false until separately tested.

Server CPU commands and exact original inventory/proof refs are in
CUDA13_SDK_CPU_PLAN.json. The old off failure cannot be retried under its used
name or followed by shadow. Future GPU execution requires a new source lock,
scope and explicit human authorization; this candidate is CPU-only.
