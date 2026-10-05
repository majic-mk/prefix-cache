"""CPU-only synthetic provenance fixtures; no real model/GPU qualification."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py"
spec = importlib.util.spec_from_file_location("native_prefix_cpu_subject", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
REVISION = "1" * 40


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    directory = tmp_path / "model"
    directory.mkdir()
    config = {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"],
              "num_hidden_layers": 28, "hidden_size": 3584,
              "num_attention_heads": 28, "num_key_value_heads": 4,
              "torch_dtype": "bfloat16", "vocab_size": 152064,
              "use_sliding_window": False}
    contents = {
        "config.json": json.dumps(config).encode(),
        "model.safetensors.index.json": json.dumps({
            "weight_map": {"synthetic.weight": "model-00001-of-00001.safetensors"}}).encode(),
        "model-00001-of-00001.safetensors": b"SYNTHETIC CPU fixture, never loaded by a model",
        "tokenizer_config.json": b'{"test_fixture_only":true}',
    }
    records = []
    for name, data in contents.items():
        (directory / name).write_bytes(data)
        records.append({"path": name, "bytes": len(data),
                        "hash_algorithm": "sha256", "hash": hashlib.sha256(data).hexdigest()})
    evidence = []
    urls = {
        "official_release": "https://qwenlm.github.io/blog/qwen2.5/",
        "provider_model": "https://modelscope.cn/api/v1/models/" + smoke.MODEL_ID,
        "pinned_files": "https://modelscope.cn/api/v1/models/" + smoke.MODEL_ID
                        + "/repo/files?Revision=" + REVISION + "&Recursive=true",
    }
    for role, filename in (("config", "config.json"), ("safetensors_index", "model.safetensors.index.json"),
                           ("tokenizer_config", "tokenizer_config.json")):
        urls[role] = ("https://modelscope.cn/api/v1/models/" + smoke.MODEL_ID
                      + "/repo?Revision=" + REVISION + "&FilePath=" + filename)
    for role, url in urls.items():
        data = ("SYNTHETIC provenance fixture: " + role).encode()
        name = role + ".fixture"
        (tmp_path / name).write_bytes(data)
        evidence.append({"role": role, "url": url, "path": name,
                         "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    plan = {"model_id": smoke.MODEL_ID, "revision": REVISION,
            "source": "https://modelscope.cn", "provider": "modelscope",
            "trust_remote_code": False, "files": records,
            "provenance": {"publisher": "Qwen",
                           "official_channel_url": "https://modelscope.cn/organization/qwen",
                           "publisher_github": "https://github.com/QwenLM",
                           "revision_namespace": "modelscope_git_commit", "evidence": evidence}}
    path = tmp_path / "synthetic-manifest.json"
    return directory, path, plan


def validate(fixture):
    directory, path, plan = fixture
    path.write_text(json.dumps(plan))
    return smoke.validate_local_model(directory, path)[1]


def test_modelscope_keeps_provider_namespace_and_manifest_hash(fixture):
    result = validate(fixture)
    assert result["provider"] == "modelscope"
    assert result["source"] == "https://modelscope.cn"
    assert result["revision_namespace"] == "modelscope_git_commit"
    assert result["revision"] == REVISION
    assert result["manifest_sha256"] == hashlib.sha256(fixture[1].read_bytes()).hexdigest()
    assert len(result["provenance"]["evidence"]) == 6


def test_existing_huggingface_plan_remains_accepted(fixture):
    plan = fixture[2]
    plan.update(source="https://huggingface.co")
    plan.pop("provider")
    plan.pop("provenance")
    result = validate(fixture)
    assert result["provider"] == "huggingface"
    assert result["revision_namespace"] == "huggingface_git_commit"
    assert result["provenance"] is None


@pytest.mark.parametrize("source", ["http://modelscope.cn", "https://modelscope.cn.evil",
                                   "https://example.org", "https://modelscope.cn/"])
def test_rejects_unapproved_source(fixture, source):
    fixture[2]["source"] = source
    with pytest.raises(RuntimeError, match="source"):
        validate(fixture)


@pytest.mark.parametrize("field,value", [("provider", "huggingface"),
                                         ("revision", "master"),
                                         ("revision", None),
                                         ("trust_remote_code", True)])
def test_rejects_identity_changes(fixture, field, value):
    fixture[2][field] = value
    with pytest.raises(RuntimeError):
        validate(fixture)


def test_modelscope_requires_explicit_provider(fixture):
    fixture[2].pop("provider")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        validate(fixture)


def test_modelscope_requires_publisher_provenance(fixture):
    fixture[2].pop("provenance")
    with pytest.raises(RuntimeError, match="provenance required"):
        validate(fixture)


def test_modelscope_rejects_huggingface_revision_namespace(fixture):
    fixture[2]["provenance"]["revision_namespace"] = "huggingface_git_commit"
    with pytest.raises(RuntimeError, match="provenance required"):
        validate(fixture)


@pytest.mark.parametrize("change", ["missing", "tampered_bytes", "wrong_url_revision",
                                    "outside_project", "duplicate_role", "publisher_url"])
def test_rejects_bad_provenance(fixture, tmp_path, change):
    evidence = fixture[2]["provenance"]["evidence"]
    if change == "missing":
        evidence.pop()
    elif change == "tampered_bytes":
        (tmp_path / evidence[0]["path"]).write_bytes(b"modified synthetic evidence")
    elif change == "wrong_url_revision":
        evidence[2]["url"] = evidence[2]["url"].replace(REVISION, "2" * 40)
    elif change == "outside_project":
        evidence[0]["path"] = str(tmp_path / evidence[0]["path"])
    elif change == "duplicate_role":
        evidence.append(copy.deepcopy(evidence[0]))
    elif change == "publisher_url":
        evidence[0]["url"] = "https://example.org/release"
    with pytest.raises(RuntimeError):
        validate(fixture)


def test_modelscope_rejects_git_blob_hash(fixture):
    fixture[2]["files"][0]["hash_algorithm"] = "git_blob_sha1"
    with pytest.raises(RuntimeError, match="content SHA-256"):
        validate(fixture)


def test_modified_weight_bytes_rejected(fixture):
    (fixture[0] / "model-00001-of-00001.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="official plan"):
        validate(fixture)


def test_extra_safetensors_rejected(fixture):
    (fixture[0] / "unverified.safetensors").write_bytes(b"extra")
    with pytest.raises(RuntimeError, match="verified official index"):
        validate(fixture)


def test_runtime_environment_stays_native_and_disables_telemetry(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    for key in smoke.RUNTIME_PATH_ENV_KEYS:
        monkeypatch.setenv(key, "/outside-project")
    for key in ("VLLM_USE_MODELSCOPE", "VLLM_NO_USAGE_STATS", "VLLM_DO_NOT_TRACK"):
        monkeypatch.setenv(key, "true")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    smoke.configure_runtime_environment()
    assert os.environ["VLLM_USE_MODELSCOPE"] == "0"
    assert os.environ["VLLM_NO_USAGE_STATS"] == os.environ["VLLM_DO_NOT_TRACK"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == os.environ["TRANSFORMERS_OFFLINE"] == "1"
    for key in smoke.RUNTIME_PATH_ENV_KEYS:
        path = Path(os.environ[key])
        assert path.is_relative_to(tmp_path) and path.is_dir()
    assert Path(os.environ["TMPDIR"]).name == ".p1tmp"


def test_research_and_gpu_modules_not_loaded():
    assert not {"torch", "vllm", "modelscope", "kvcache", "py_kvcache"} & set(sys.modules)
