import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

root = Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
script = root / "experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py"
evidence = root / "artifacts/prefix_io_v1/new-server-02/native-prefix-preparation"
evidence.mkdir(parents=True, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
source = script.read_text()
compile(source, str(script), "exec")
ns = runpy.run_path(str(script), run_name="cpu_source_review")
assert "torch" not in sys.modules and "vllm" not in sys.modules
author = ns["AUTHOR_ROOT"]
tree = ast.parse((author / "vllm/entrypoints/llm.py").read_text())
llm_class = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "LLM")
init = next(x for x in llm_class.body if isinstance(x, ast.FunctionDef) and x.name == "__init__")
llm_args = {x.arg for x in init.args.args + init.args.kwonlyargs}
engine_tree = ast.parse((author / "vllm/engine/arg_utils.py").read_text())
engine_class = next(x for x in engine_tree.body if isinstance(x, ast.ClassDef) and x.name == "EngineArgs")
engine_fields = {x.target.id for x in engine_class.body if isinstance(x, ast.AnnAssign)
                 and isinstance(x.target, ast.Name)}
assert set(ns["ENGINE"]) <= llm_args | engine_fields
sampling_tree = ast.parse((author / "vllm/sampling_params.py").read_text())
sampling_class = next(x for x in sampling_tree.body if isinstance(x, ast.ClassDef) and x.name == "SamplingParams")
sampling_fields = {x.target.id for x in sampling_class.body if isinstance(x, ast.AnnAssign)
                   and isinstance(x.target, ast.Name)}
assert set(ns["SAMPLING"]) <= sampling_fields
assert len(ns["PROMPT_TOKEN_IDS"]) == 128 and ns["EXPECTED_HOT_TOKENS"] == 112
assert ns["ENGINE"]["kv_cache_memory_bytes"] == 64 * 1024 ** 2
assert "reset_prefix_cache" not in source
env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
commands = [
    [sys.executable, str(script), "--help"],
    [sys.executable, str(script), "--model-dir", str(root / "MODEL_NOT_DOWNLOADED"),
     "--model-plan", str(root / "REAL_PLAN_NOT_AVAILABLE.json"),
     "--output-dir", str(root / "experiments/prefix_io_v1/runs/UNEXECUTED-native-prefix-review")],
]
checks = []
for idx, command in enumerate(commands):
    result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=15)
    (evidence / f"cpu-check-{idx}.log").write_text(result.stdout)
    checks.append({"command": command, "exit": result.returncode,
                   "log": f"cpu-check-{idx}.log"})
assert checks[0]["exit"] == 0
assert checks[1]["exit"] != 0
assert "one complete GPU UUID must be bound" in (evidence / "cpu-check-1.log").read_text()
assert not (root / "experiments/prefix_io_v1/runs/UNEXECUTED-native-prefix-review").exists()
script_sha = hashlib.sha256(script.read_bytes()).hexdigest()
plan = {
    "schema_version": 1,
    "status": "PREPARED_NOT_EXECUTED",
    "scope": "P1 native GPU-only exact Prefix synthetic-token path smoke",
    "model_id": ns["MODEL_ID"],
    "model_revision": None,
    "model_revision_status": "UNOBTAINED_OFFICIAL_NETWORK_UNAVAILABLE",
    "model_download_plan": None,
    "actual_model_manifest": None,
    "gpu_execution": "UNEXECUTED",
    "author_commit": ns["AUTHOR_COMMIT"],
    "script_sha256": script_sha,
    "engine": ns["ENGINE"],
    "sampling": ns["SAMPLING"],
    "prompt_token_ids": ns["PROMPT_TOKEN_IDS"],
    "expected_cached_tokens": [0, 112],
    "expected_output_check": "exact equality of the two 16-token output ID lists",
    "official_model_files_required_before_run": True,
    "no_download_in_smoke": True,
    "requires_budget_runner_offline_environment": True,
    "not_qualified": ["SSD", "staging", "production KV byte identity",
                      "actual KV tensor allocation", "end-to-end cache", "performance", "full P1"],
}
(evidence / "frozen-pending-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
verification = {"scope": "CPU compile, pure-stdlib import, source API names, CLI and fail-closed guard",
                "status": "PASSED_CPU_PREPARATION_ONLY",
                "gpu_operations": 0, "model_downloads": 0,
                "torch_or_vllm_imported": False,
                "source_api_engine_keys": sorted(ns["ENGINE"]),
                "source_api_sampling_keys": sorted(ns["SAMPLING"]),
                "checks": checks, "script_sha256": script_sha}
(evidence / "cpu-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
patch = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", str(script.relative_to(root))],
                       cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
assert patch.returncode == 1 and not patch.stderr
patch_path = evidence / "0001-native-gpu-prefix-smoke.patch"
patch_path.write_text(patch.stdout)
verification["patch_sha256"] = hashlib.sha256(patch_path.read_bytes()).hexdigest()
(evidence / "cpu-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
print(json.dumps(verification, indent=2))
