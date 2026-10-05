# CPU KV byte evidence candidate

`g3_kv_byte_evidence.py` checks a bounded synthetic transcript against the current frozen G2 layout source. It verifies complete canonical layer-major pages, job/key/block association, declared completion order, byte equality and SHA-256 across producer, D2H, SSD write/read and consumer restore captures. The pinned staging allocator requires aligned logical storage blocks, so this limited subset has zero outer SSD padding; unsupported padding or partial files are rejected.

This module has no real capture or installer. Its stage labels and completion markers are CPU fixture assertions. The immutable result always reports GPU, production, effect and real-byte qualification as false and lists the missing runtime provenance. Caller origin strings and qualified receipts do not grant authority. No framework import, GPU operation, sync, extra get_finished, policy change or original source edit occurs.

Run `test_g3_kv_byte_evidence.py --source-root ROOT --lock LOCK` under the server stdlib interpreter with CUDA_VISIBLE_DEVICES empty. Local mirror defaults are supported for CPU-only tests. All files must restore and capture before any consumer compute. The final local run passed 32 distinct tests; the first attempt's two restricted-AST harness errors and both actual logs are retained. See G3_KV_BYTE_CPU_PLAN.json for exact scope and command.

The prior five-file candidate is retained byte-for-byte in historical-byte-before-global-before-compute. Its actual two-file ordering counterexample is saved in CPU_BYTE_GLOBAL_COUNTEREXAMPLE_OLD.json. The repair adds only the missing global before-compute condition and one regression test; no native authority or supported layout is expanded.
