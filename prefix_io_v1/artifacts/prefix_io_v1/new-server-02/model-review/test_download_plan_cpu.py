"""Synthetic CPU inputs only: none of these revisions/hashes identifies the real model."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "experiments/prefix_io_v1/scripts/prepare_qwen_download.py"
spec = importlib.util.spec_from_file_location("prepare_qwen_download", SCRIPT)
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)
PIN = "1" * 40


def inputs():
    config = json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}).encode()
    index = json.dumps({"weight_map": {
        "layer.a": "model-00001-of-00002.safetensors",
        "layer.b": "model-00002-of-00002.safetensors",
    }}).encode()
    small = {
        "config.json": config, "tokenizer.json": b"{}", "tokenizer_config.json": b"{}",
        "model.safetensors.index.json": index,
    }
    siblings = [{"rfilename": name, "size": len(data), "blobId": plan.git_blob_sha1(data)}
                for name, data in small.items()]
    siblings += [
        {"rfilename": "model-00001-of-00002.safetensors", "size": 1000,
         "lfs": {"sha256": "a"*64, "size": 1000}},
        {"rfilename": "model-00002-of-00002.safetensors", "size": 2000,
         "lfs": {"sha256": "b"*64, "size": 2000}},
    ]
    meta = {"id": plan.MODEL_ID, "sha": PIN, "private": False, "gated": False,
            "siblings": siblings}
    permissions = {"allow_model_downloads": True, "max_model_download_gib": 20}
    return meta, config, index, permissions


def prepare(data=None, pin=PIN):
    meta, config, index, permissions = data or inputs()
    return plan.prepare(meta, pin, config, index, permissions)


def test_exact_frozen_list_remains_unexecuted_and_unreserved():
    meta, config, index, permissions = inputs()
    result = prepare((meta, config, index, permissions))
    assert result["revision"] == PIN
    assert len(result["files"]) == 6
    assert result["total_bytes"] == sum(item["size"] for item in meta["siblings"])
    assert all("/resolve/" + PIN + "/" in item["url"] for item in result["files"])
    assert result["download_execution"] == "UNEXECUTED"
    assert result["budget_reservation"] == "NOT_RESERVED"
    assert result["ledger_status"] == "NOT_VERIFIED"
    assert result["download_ready"] is False and result["trust_remote_code"] is False


@pytest.mark.parametrize("revision", ["main", None, "", "1"*39])
def test_unfrozen_revision_rejected(revision):
    with pytest.raises(ValueError, match="frozen"):
        prepare(pin=revision)


@pytest.mark.parametrize("mutation", ["revision", "model", "size", "lfs", "lfs_size",
                                     "missing_config", "missing_shard", "duplicate"])
def test_incomplete_or_inconsistent_metadata_is_rejected(mutation):
    meta, config, index, permissions = inputs()
    if mutation == "revision":
        meta["sha"] = "2"*40
    elif mutation == "model":
        meta["id"] = "different/model"
    elif mutation == "size":
        del meta["siblings"][-1]["size"]
    elif mutation == "lfs":
        del meta["siblings"][-1]["lfs"]["sha256"]
    elif mutation == "lfs_size":
        meta["siblings"][-1]["lfs"]["size"] = 100
    elif mutation == "missing_config":
        meta["siblings"] = [s for s in meta["siblings"] if s["rfilename"] != "config.json"]
    elif mutation == "missing_shard":
        meta["siblings"].pop()
    else:
        meta["siblings"].append(copy.deepcopy(meta["siblings"][-1]))
    with pytest.raises(ValueError):
        prepare((meta, config, index, permissions))


@pytest.mark.parametrize("value", [None, False, 0, -1, "NaN", "Infinity", "bad"])
def test_budget_must_be_explicit_finite_positive(value):
    meta, config, index, permissions = inputs()
    permissions["max_model_download_gib"] = value
    with pytest.raises(ValueError):
        prepare((meta, config, index, permissions))


def test_total_larger_than_20gib_fails_before_any_download():
    meta, config, index, permissions = inputs()
    meta["siblings"][-1]["size"] = meta["siblings"][-1]["lfs"]["size"] = 20 * plan.GIB
    with pytest.raises(ValueError, match="exceed"):
        prepare((meta, config, index, permissions))


def test_disabled_permission_fails_closed():
    meta, config, index, permissions = inputs()
    permissions["allow_model_downloads"] = False
    with pytest.raises(ValueError, match="authorized"):
        prepare((meta, config, index, permissions))


@pytest.mark.parametrize("kind", ["config_hash", "index_hash", "missing_index"])
def test_pinned_small_files_are_verified(kind):
    meta, config, index, permissions = inputs()
    if kind == "config_hash":
        config = config.replace(b"qwen2", b"qwen3")
    elif kind == "index_hash":
        index = index.replace(b"layer.a", b"layer.c")
    else:
        index = None
    with pytest.raises(ValueError):
        prepare((meta, config, index, permissions))


def test_remote_code_mapping_is_rejected_even_with_matching_file_hash():
    meta, config, index, permissions = inputs()
    obj = json.loads(config)
    obj["auto_map"] = {"AutoModel": "remote.CustomModel"}
    config = json.dumps(obj).encode()
    item = meta["siblings"][0]
    item["size"] = len(config)
    item["blobId"] = plan.git_blob_sha1(config)
    with pytest.raises(ValueError, match="remote-code"):
        prepare((meta, config, index, permissions))
