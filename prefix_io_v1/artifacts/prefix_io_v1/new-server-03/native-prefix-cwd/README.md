# Native smoke run-directory isolation

This P1 experiment-only change validates/resolves local model and manifest paths and the new output directory before recording working_directory in frozen-config.json. It then changes only the current process cwd to that absolute output directory before importing vLLM or constructing LLM. Author profiler relative files results_engine_core_0.json, merge.json and merge.json.registry therefore remain within the unique run/details directory. Author model/cache engine code is unchanged.

CPU verification: 24 existing provenance/environment unit tests passed in 0.10s. Source compile and ordering checks passed; reverse patch application check passed. No GPU/model operation or model download was performed.

Commands:

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/prefix_io_v1_native_prefix --junitxml=artifacts/prefix_io_v1/new-server-03/native-prefix-cwd/cpu-tests.xml
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python artifacts/prefix_io_v1/new-server-03/native-prefix-cwd/verify_source.py
```

Patch boundary: 0002-native-prefix-run-directory.patch is an incremental patch after new-server-03/native-prefix-adaptation/0001-provider-provenance.patch (including its runtime-cache settings). It contains only the cwd/frozen-directory change, so apply each exactly once in that order. Do not regenerate the older provider patch from this newer full file or include these lines twice. Source/patch SHA-256 are in verification.json. The original native-prefix-preparation new-file patch precedes both.
