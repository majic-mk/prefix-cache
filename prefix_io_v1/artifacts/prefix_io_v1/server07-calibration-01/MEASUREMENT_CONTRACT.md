# Native AIO P1 calibration contract
Server: connect.westd:38819, GPU GPU-f8744916-1693-fa6a-93b6-7f503c03459c. Existing 8-hour cumulative ledger; no download or system changes.
Pinned author commits unchanged. This is acquisition and subsequent original planner validation, not a research-policy benchmark.

Three native cost observations: f = cold recomputation, g_mem = exact staging restore, g_ssd = SSD restore. Axis N = reusable prefix tokens. Each identical request contains N+1 explicit token IDs and generates one token. First-token latency comes from author RequestStateStats.first_token_latency (seconds), not import/model loading/whole GPU job duration. N=16,64,128,256,512,1024; repeat 0 retained as warmup and repeats 1..5 as measurements. The extra one-token query is charged equally to all paths; this is a conservative full-request cost proxy, not isolated kernel cost. Compare holdout admission before trusting it beyond this diagnostic.

Frozen GPU KV 256 MiB, pinned staging 128 MiB, storage block 16 tokens = 917504 aligned bytes, iodepth 4, LRU, shared preload/lookahead 1, async store and original copy merging. Calibration context 1040 accommodates terminal prefix N=1024 plus query; any exported planner domain is restricted to <=1024 (no above-knot extrapolation). Exact GPU Prefix stays enabled. Explicit GPU-only resets are calibration interventions and are logged; they are not a formal traffic pattern.

Local relative alias Qwen/Qwen2.5-7B-Instruct points to the previously hash-verified offline model. The same model identity is used by all stages; FileMapper path validation and Prefix hashing are unchanged.

Cost acquisition uses original load_planner=off and absent thresholds; no fabricated curves are supplied. Planned validation must use original load_planner=on with actually measured curves. Original admission is never overridden.

Native worker RPC submits queued stores using _submit_store_jobs and waits via original Future/worker.wait. It never consumes get_finished or calls complete_store; ordinary subsequent engine steps propagate completions to scheduler fences. Each trace is flushed after accepted I/O completes. All source/trace IDs, actual success, H2D, logical bytes and successful read completion bytes are checked. Unique prompts prevent accidental shared Prefix across measured trials. SSD stage restarts the engine (empty GPU/staging) while retaining published files; direct I/O remains original.

First connector diagnostic initially failed the collector's external/internal ID join. Raw traces show successful staging load; the driver was corrected to the exact author InputProcessor external-ID + eight-hex-suffix mapping. The failed launch is retained, not relabeled as passed.

Median samples are exported only after path, output token and identity checks. All outliers/failures remain in raw reports. Interpolation uses original PCHIP; golden checks only validate numeric consistency, not empirical accuracy. If SSD is slower/no profitable crossing exists, retain decline evidence; never invent a zero threshold to force admission.
