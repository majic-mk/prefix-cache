"""CPU candidates only: temporary synthetic evidence is never a GPU result."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("calibration_preparation_test",
    ROOT / "experiments/prefix_io_v1/scripts/prepare_native_calibration_plan.py")
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    real = plan.ROOT
    # Only pinned original Python source is copied, not a repo, model or weights.
    for relative in (plan.PARETO, plan.BREAK_EVEN):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((real / relative).read_bytes())
    cfg = {"num_hidden_layers": 28, "num_key_value_heads": 4, "hidden_size": 3584,
           "num_attention_heads": 28, "torch_dtype": "bfloat16"}
    model_dir = tmp_path / "model"
    dump(model_dir / "config.json", cfg)
    manifest_path = tmp_path / "official-plan.json"
    identity = {"model_id": plan.MODEL_ID, "revision": "a" * 40,
                "source": "https://modelscope.cn", "provider": "modelscope",
                "revision_namespace": "modelscope_git_commit"}
    dump(manifest_path, {**identity, "files": [{"path": "config.json",
              "hash_algorithm": "sha256", "hash": plan.sha(model_dir / "config.json")}]})
    identity.update(plan_path=str(manifest_path), plan_sha256=plan.sha(manifest_path),
                    manifest_sha256=plan.sha(manifest_path))
    engine = {"dtype": "bfloat16", "kv_cache_dtype": "auto", "quantization": None,
              "kv_cache_memory_bytes": 67108864, "block_size": 16, "max_model_len": 256}
    storage = {"tensor_count": 28, "dtype": "torch.bfloat16", "device_type": "cuda",
               "num_blocks": 73, "actual_kv_tensor_allocation_bytes": 66977792,
               "declared_kv_tensor_bytes": 66977792}
    prompt = list(range(128))
    cold = {"prompt_token_ids": prompt, "output_token_ids": [1] * 16, "num_cached_tokens": 0}
    repeat = dict(cold, num_cached_tokens=112)
    frozen = {"author_commit": plan.AUTHOR_COMMIT, "model": identity, "model_dir": str(model_dir),
              "engine": engine, "expected_cached_tokens": [0, 112],
              "prompt_token_ids": prompt, "sampling": {"max_tokens": 16},
              "environment": {"CUDA_VISIBLE_DEVICES": "GPU-11111111-2222-3333-4444-555555555555"}}
    result = {"status": "PASSED_GPU_PREFIX_PATH_ONLY", "native_engine_shutdown": "completed",
              "effective_model_dtype": "torch.bfloat16", "actual_kv_tensor_dtype": "torch.bfloat16",
              "effective_model_quantization": None, "effective_kv_cache_dtype": "auto",
              "effective_kv_cache_memory_bytes": 67108864, "results": [cold, repeat],
              "kv_tensor_storage_metadata": storage, "actual_kv_tensor_allocation_bytes": 66977792}
    directory = tmp_path / "smoke"
    dump(directory / "frozen-config.json", frozen)
    dump(directory / "smoke-result.json", result)
    uuid = frozen["environment"]["CUDA_VISIBLE_DEVICES"]
    dump(directory.parent / "result.json", {"exit": 0, "child_exit": 0, "timed_out": False,
        "session_drained": True, "error": None, "interrupted_signal": None,
        "gpu_job_attempted": True, "session_members_after_cleanup": [], "gpu_uuid": uuid,
        "evidence": str(directory.parent), "command": ["python", "smoke.py", "--model-dir", str(model_dir),
             "--model-plan", str(manifest_path), "--output-dir", str(directory)]})
    dump(tmp_path / "experiments/prefix_io_v1/locks/environment.json",
         {"gpu_uuid": uuid, "native_GPU_Prefix_evidence": str(directory / "smoke-result.json")})
    dump(tmp_path / "experiments/prefix_io_v1/locks/dependency-lock.json",
         {"repositories": {n: {"commit": c} for n, c in
           {"vllm-author": plan.AUTHOR_COMMIT, "py-kvcache": plan.BREAK_EVEN_COMMIT,
            "kvcache-experiments": plan.PARETO_COMMIT}.items()},
          "model_lock": {**identity, "id": plan.MODEL_ID, "local_path": str(model_dir),
              "manifest": str(manifest_path), "verified_local_weights": True}})
    monkeypatch.setattr(plan, "ROOT", tmp_path)
    return directory, frozen, result, tmp_path


@pytest.mark.parametrize("sizes,repeats", [([17, 81], 1), ([192, 32, 128], 4), ([7], 10)])
def test_original_job_plan_varies_with_inputs_and_stays_non_executable(fixture, sizes, repeats):
    directory, _, _, _ = fixture
    output = plan.prepare(directory, sizes, repeats)
    assert [x["doc_size"] for x in output["candidate_jobs"]] == sorted(sizes)
    assert all(x["n_requests"] == repeats and x["prefix_reuse_pct"] == 0
               and x["server_config"] == "baseline" for x in output["candidate_jobs"])
    assert output["planned_measured_requests"] == len(sizes) * repeats
    assert output["calibration_measurements"] == {"f": None, "g_mem": None, "g_ssd": None}
    assert not output["driver_ready"] and not output["actual_token_fit_verified"]
    assert not output["configuration_frozen_for_execution"]
    assert output["execution"] == "UNEXECUTED" and "curves" not in output
    assert "nominal" in output["doc_size_semantics"]
    assert output["identity"]["kv_bytes_per_token"] == 57344


@pytest.mark.parametrize("sizes,repeats", [([], 3), ([0], 3), ([256], 3), ([32, 32], 3),
                                         ([32], 0), ([32], 11), (list(range(1, 10)), 2)])
def test_unbounded_or_invalid_candidates_rejected(fixture, sizes, repeats):
    with pytest.raises(ValueError):
        plan.prepare(fixture[0], sizes, repeats)


@pytest.mark.parametrize("field,value", [
    ("status", "FAILED"), ("native_engine_shutdown", "failed"),
    ("actual_kv_tensor_dtype", "torch.float16"), ("effective_kv_cache_dtype", "fp8"),
    ("actual_kv_tensor_allocation_bytes", 1), ("effective_model_quantization", "fp8"),
])
def test_unqualified_prior_result_rejected(fixture, field, value):
    directory, _, result, _ = fixture
    result[field] = value
    dump(directory / "smoke-result.json", result)
    with pytest.raises(ValueError):
        plan.prepare(directory, [64], 3)


def test_cold_and_repeat_output_mismatch_rejected(fixture):
    directory, _, result, _ = fixture
    result["results"][1]["output_token_ids"] = [2] * 16
    dump(directory / "smoke-result.json", result)
    with pytest.raises(ValueError, match="equality"):
        plan.prepare(directory, [64], 3)


def test_manifest_mutation_rejected(fixture):
    directory, frozen, _, _ = fixture
    path = Path(frozen["model"]["plan_path"])
    content = json.loads(path.read_text())
    content["revision"] = "b" * 40
    dump(path, content)
    with pytest.raises(ValueError, match="manifest hash"):
        plan.prepare(directory, [64], 3)


@pytest.mark.parametrize("name", [plan.PARETO, plan.BREAK_EVEN])
def test_untrusted_author_source_rejected_before_execution(fixture, name):
    path = fixture[3] / name
    marker = fixture[3] / "MUST_NOT_EXIST"
    path.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n")
    expected = plan.PARETO_SHA if name == plan.PARETO else plan.BREAK_EVEN_SHA
    with pytest.raises(ValueError, match="source SHA"):
        plan.author_module(name, expected, {"Job", "build_job_plan"} if name == plan.PARETO else None)
    assert not marker.exists()


@pytest.mark.parametrize("version", [None, 2])
def test_scalar_file_cannot_masquerade_as_v2(fixture, version):
    path = fixture[3] / "old.json"
    dump(path, {"schema_version": version, "break_even_ssd_tokens": 123,
                "break_even_mem_tokens": 0})
    with pytest.raises(ValueError, match="v1/scalar-only"):
        plan.prepare(fixture[0], [64], 3, path)


def test_synthetic_v2_structure_never_claims_calibration_even_without_golden(fixture):
    directory, frozen, _, root = fixture
    # Arbitrary test fixtures validate parsing only, never emitted as calibration.
    raw = {"model_name": frozen["model_dir"], "kv_dtype": "auto", "kv_bytes_per_token": 57344,
           "curves": {name: {"floor": 0, "knots": {"16": 0.001, "32": 0.002}}
                      for name in ("f", "g_mem", "g_ssd")}}
    path = root / "synthetic-v2.json"
    dump(path, raw)
    review = plan.prepare(directory, [64], 3, path)["optional_curve_review"]
    assert review["status"] == "STRUCTURE_ONLY" and review["golden_points"] == 0
    assert not review["usable_for_planner"] and not review["measurement_provenance_verified"]
    raw["kv_dtype"] = "fp8"
    dump(path, raw)
    with pytest.raises(ValueError, match="dtype or KV geometry"):
        plan.prepare(directory, [64], 3, path)


def test_no_research_or_gpu_imports():
    assert not {"torch", "vllm", "py_kvcache", "flashinfer", "matplotlib", "numpy"} & set(sys.modules)


@pytest.mark.parametrize("field,value", [("exit", 1), ("child_exit", 1), ("timed_out", True),
                                         ("session_drained", False), ("gpu_uuid", "GPU-wrong"),
                                         ("session_members_after_cleanup", [42])])
def test_failed_or_other_gpu_budget_run_rejected(fixture, field, value):
    p = fixture[0].parent / "result.json"
    d = json.loads(p.read_text()); d[field] = value; dump(p, d)
    with pytest.raises(ValueError):
        plan.prepare(fixture[0], [64], 3)


def test_current_lock_model_revision_mismatch_rejected(fixture):
    p = fixture[3] / "experiments/prefix_io_v1/locks/dependency-lock.json"
    d = json.loads(p.read_text()); d["model_lock"]["revision"] = "b" * 40; dump(p, d)
    with pytest.raises(ValueError, match="current model lock"):
        plan.prepare(fixture[0], [64], 3)


def test_runner_model_path_mismatch_rejected(fixture):
    p = fixture[0].parent / "result.json"
    d = json.loads(p.read_text())
    d["command"][d["command"].index("--model-dir") + 1] = str(fixture[3])
    dump(p, d)
    with pytest.raises(ValueError, match="command path mismatch"):
        plan.prepare(fixture[0], [64], 3)
