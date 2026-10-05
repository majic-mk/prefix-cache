# Prefix I/O V1 dependency layout

Canonical frozen sources:
- upstream/py-kvcache @ 3abba7a502d553f6e7e2e58b92086487e3395d7e
- upstream/vllm-author @ 817a7e3124f817cd6e549581d3e5483207a753a4 (author with_profiling)
- upstream/kvcache-experiments @ 0e023a84a21246b9bbc06266fa8070397eccbdc9
- upstream/simple-profiler @ ec0d563bf68856df83c5824ac579700ec076b9e2

work/py-kvcache is the common CPU repair worktree. Both snapshots and worktree are excluded only in this checkout's .git/info/exclude. Patches and exact locks are part of project delivery; dependency code keeps its original license.

vllm, vllm-source, vllm-locked and vllm-archive are preserved transport/debug attempts, not build inputs. The canonical vllm-author Git checkout includes both audited author commits; shallow history was transported intact with its shallow metadata. The failed shallow bundle clone is not used.

CPU tests use PYTHONPATH, not a globally installed modified py-kvcache. simple-profiler is installed editable only in .venv-prefix. No vLLM build or GPU runtime installation was attempted in this 2 GiB CPU container.
