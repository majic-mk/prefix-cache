"""CPU-only verification of the provider adaptation and actual saved provenance."""
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[4]
evidence = Path(__file__).resolve().parent
script = root / "experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py"
before = evidence / "native_gpu_prefix_smoke.before.py"
tests = root / "tests/prefix_io_v1_native_prefix/test_model_provenance.py"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
source = script.read_text()
compile(source, str(script), "exec")
ns = runpy.run_path(str(script), run_name="cpu_verification")
old = runpy.run_path(str(before), run_name="cpu_before_verification")
for key in ("ENGINE", "SAMPLING", "PROMPT_TOKEN_IDS", "EXPECTED_HOT_TOKENS",
            "AUTHOR_COMMIT", "MODEL_ID"):
    assert ns[key] == old[key], key
author = ns["AUTHOR_ROOT"]
tree = ast.parse((author / "vllm/entrypoints/llm.py").read_text())
llm = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "LLM")
init = next(x for x in llm.body if isinstance(x, ast.FunctionDef) and x.name == "__init__")
allowed = {x.arg for x in init.args.args + init.args.kwonlyargs}
eng = ast.parse((author / "vllm/engine/arg_utils.py").read_text())
engcls = next(x for x in eng.body if isinstance(x, ast.ClassDef) and x.name == "EngineArgs")
allowed |= {x.target.id for x in engcls.body if isinstance(x, ast.AnnAssign)
            and isinstance(x.target, ast.Name)}
assert set(ns["ENGINE"]) <= allowed
manifest = root / "artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json"
plan = json.loads(manifest.read_text())
provider = ns["validate_provider"](plan)
assert provider["provider"] == "modelscope"
assert not {"torch", "vllm", "modelscope"} & set(sys.modules)
env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
commands = [
    [sys.executable, str(script), "--help"],
    [sys.executable, str(script), "--model-dir", str(root / "MODEL_NOT_DOWNLOADED"),
     "--model-plan", str(manifest),
     "--output-dir", str(root / "experiments/prefix_io_v1/runs/UNEXECUTED-provider-cpu-check")],
]
cli = []
for i, command in enumerate(commands):
    result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=15)
    (evidence / f"cli-{i}.log").write_text(result.stdout)
    cli.append({"command": command, "exit": result.returncode, "log": f"cli-{i}.log"})
assert cli[0]["exit"] == 0 and cli[1]["exit"] == 1
assert "one complete GPU UUID" in (evidence / "cli-1.log").read_text()
assert not (root / "experiments/prefix_io_v1/runs/UNEXECUTED-provider-cpu-check").exists()
xml = ET.parse(evidence / "cpu-tests.xml").getroot()
suites = list(xml.iter("testsuite"))
assert sum(int(s.attrib["tests"]) for s in suites) == 24
assert not sum(int(s.attrib[k]) for s in suites for k in ("errors", "failures"))
rel_script = str(script.relative_to(root))
rel_tests = str(tests.relative_to(root))
patch = "".join(difflib.unified_diff(before.read_text().splitlines(True), source.splitlines(True),
                                    fromfile="a/" + rel_script, tofile="b/" + rel_script))
patch += "".join(difflib.unified_diff([], tests.read_text().splitlines(True),
                                     fromfile="/dev/null", tofile="b/" + rel_tests))
patch_path = evidence / "0001-provider-provenance.patch"
patch_path.write_text(patch)
check = subprocess.run(["git", "apply", "--reverse", "--check", str(patch_path)],
                       cwd=root, capture_output=True, text=True, timeout=15)
assert check.returncode == 0, check.stderr
report = {
    "scope": "CPU source/CLI/synthetic unit tests plus actual saved provenance hash verification",
    "status": "PASSED_CPU_ONLY", "gpu_operations": 0, "model_downloads": 0,
    "model_weights_validated": False, "gpu_modules_imported": False,
    "unit_tests": {"passed": 24, "synthetic_fixture_only": True},
    "source_API_fields": "verified by pinned author source AST",
    "engine_sampling_prompt_and_expected_cache_counts": "unchanged",
    "actual_provider": provider["provider"], "revision_namespace": provider["revision_namespace"],
    "revision": plan["revision"], "official_provenance_roles": [x["role"] for x in provider["provenance"]["evidence"]],
    "actual_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "script_before_sha256": hashlib.sha256(before.read_bytes()).hexdigest(),
    "script_after_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    "tests_sha256": hashlib.sha256(tests.read_bytes()).hexdigest(),
    "patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
    "reverse_patch_check": "passed", "cli": cli,
}
(evidence / "cpu-verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
