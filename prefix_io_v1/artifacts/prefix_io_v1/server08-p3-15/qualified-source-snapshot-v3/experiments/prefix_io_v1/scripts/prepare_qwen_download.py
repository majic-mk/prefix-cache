#!/usr/bin/env python3
"""Prepare a fixed Qwen download plan from already obtained official HF metadata.

Plan only: no network, downloads, GPU, model import, reservations, or ledger edits.
An external budget runner must atomically verify permissions and reserve remaining
download bytes before executing this plan. This script cannot authorize execution.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
GIB = 1024 ** 3
ALLOWED_AUX = (
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt",
    "added_tokens.json", "special_tokens_map.json", "chat_template.jinja",
)
REQUIRED_AUX = ("config.json", "tokenizer.json", "tokenizer_config.json")


def load_json(path):
    data = Path(path).read_bytes()
    if len(data) > 2 * 1024 ** 2:
        raise ValueError("metadata/config/index input exceeds 2 MiB")
    return json.loads(data), data


def git_blob_sha1(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def budget_bytes(permissions):
    if permissions.get("allow_model_downloads") is not True:
        raise ValueError("model downloads are not authorized")
    value = permissions.get("max_model_download_gib")
    if isinstance(value, bool) or value is None:
        raise ValueError("finite explicit model download budget required")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid model budget") from exc
    if not amount.is_finite() or amount <= 0:
        raise ValueError("finite positive model budget required")
    byte_limit = amount * GIB
    if byte_limit != byte_limit.to_integral_value():
        raise ValueError("model budget must express a whole number of bytes")
    # This fixed task may not expand its original 20 GiB ceiling.
    return min(int(byte_limit), 20 * GIB)


def file_record(name, sibling, revision, require_lfs=False):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
        raise ValueError("repository file name is not a safe top-level file")
    size = sibling.get("size")
    if type(size) is not int or size <= 0:
        raise ValueError(f"{name}: missing or invalid exact size")
    lfs = sibling.get("lfs")
    if lfs is not None:
        sha = lfs.get("sha256") if isinstance(lfs, dict) else None
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError(f"{name}: missing LFS SHA-256")
        if lfs.get("size") != size:
            raise ValueError(f"{name}: inconsistent LFS size")
        algorithm = "sha256"
    else:
        if require_lfs:
            raise ValueError(f"{name}: safetensors require official LFS metadata")
        sha = sibling.get("blobId")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(f"{name}: missing Git blob SHA-1")
        algorithm = "git_blob_sha1"
    return {
        "path": name, "bytes": size, "hash_algorithm": algorithm, "hash": sha,
        "url": f"https://huggingface.co/{MODEL_ID}/resolve/{revision}/{name}",
    }


def verify_small_input(record, data):
    if record["bytes"] != len(data):
        raise ValueError(f"{record['path']}: actual metadata file size differs")
    actual = (hashlib.sha256(data).hexdigest() if record["hash_algorithm"] == "sha256"
              else git_blob_sha1(data))
    if actual != record["hash"]:
        raise ValueError(f"{record['path']}: content does not match pinned metadata hash")


def prepare(metadata, revision, config_bytes, index_bytes, permissions):
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("a frozen 40-hex revision is required; main is not allowed")
    if metadata.get("sha") != revision:
        raise ValueError("official metadata revision differs from requested pin")
    if metadata.get("id", metadata.get("modelId")) != MODEL_ID:
        raise ValueError("metadata is not for the fixed approved model")
    if metadata.get("private") is not False or metadata.get("gated") not in (False, None):
        raise ValueError("metadata does not prove public, ungated access")
    if not isinstance(metadata.get("siblings"), list):
        raise ValueError("official blobs=true sibling metadata is required")
    siblings = {}
    for item in metadata["siblings"]:
        if not isinstance(item, dict) or not isinstance(item.get("rfilename"), str):
            raise ValueError("malformed sibling metadata")
        name = item["rfilename"]
        if name in siblings:
            raise ValueError("ambiguous duplicate sibling")
        siblings[name] = item
    if any(name not in siblings for name in REQUIRED_AUX):
        raise ValueError("required config/tokenizer files are missing")
    selected = [name for name in ALLOWED_AUX if name in siblings]
    config_record = file_record("config.json", siblings["config.json"], revision)
    verify_small_input(config_record, config_bytes)
    config = json.loads(config_bytes)
    if config.get("model_type") != "qwen2":
        raise ValueError("unexpected model_type for fixed Qwen2.5 model")
    if config.get("architectures") != ["Qwen2ForCausalLM"] or config.get("auto_map"):
        raise ValueError("unexpected architecture or remote-code mapping")
    if "model.safetensors.index.json" in siblings:
        if index_bytes is None:
            raise ValueError("pinned safetensors index content is required")
        name = "model.safetensors.index.json"
        verify_small_input(file_record(name, siblings[name], revision), index_bytes)
        index = json.loads(index_bytes)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("missing safetensors weight_map")
        if any(not isinstance(value, str) for value in weight_map.values()):
            raise ValueError("invalid safetensors shard name")
        weights = sorted(set(weight_map.values()))
        if any(not re.fullmatch(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors", name)
               for name in weights):
            raise ValueError("unexpected/unsafe safetensors shard name")
        selected.append("model.safetensors.index.json")
    elif "model.safetensors" in siblings and index_bytes is None:
        weights = ["model.safetensors"]
    else:
        raise ValueError("no verified safetensors layout")
    if any(name not in siblings for name in weights):
        raise ValueError("index references a shard absent from pinned metadata")
    records = [file_record(name, siblings[name], revision) for name in selected]
    records.extend(file_record(name, siblings[name], revision, require_lfs=True) for name in weights)
    records.sort(key=lambda item: item["path"])
    total = sum(item["bytes"] for item in records)
    limit = budget_bytes(permissions)
    if total > limit:
        raise ValueError(f"required files exceed authorized bytes: {total} > {limit}")
    return {
        "schema_version": 1, "model_id": MODEL_ID, "revision": revision,
        "trust_remote_code": False, "source": "https://huggingface.co",
        "files": records, "total_bytes": total,
        "total_gib": str(Decimal(total) / GIB), "authorized_ceiling_bytes": limit,
        "metadata_api_url": f"https://huggingface.co/api/models/{MODEL_ID}/revision/{revision}?blobs=true",
        "download_execution": "UNEXECUTED", "download_ready": False,
        "budget_reservation": "NOT_RESERVED", "ledger_status": "NOT_VERIFIED",
        "blockers": [
            "External runner must lock/check the shared ledger and current permissions.",
            "Remaining required bytes must be reserved before any weight request.",
            "Existing and resumed files must retain pinned size/hash verification.",
            "Unknown concurrent or failed request accounting must fail closed.",
            "Actual bytes written and outstanding reservations must be recorded durably.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path,
                        help="Official HF blobs=true model metadata, already saved")
    parser.add_argument("--revision", required=True, help="Explicit frozen 40-hex commit")
    parser.add_argument("--config", required=True, type=Path,
                        help="config.json fetched from this same revision")
    parser.add_argument("--index", type=Path,
                        help="model.safetensors.index.json fetched from this same revision")
    parser.add_argument("--permissions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New plan JSON; no overwrite")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists")
    from prefix_io_control.config import read_yaml, validate_permissions
    metadata, metadata_bytes = load_json(args.metadata)
    _, config_bytes = load_json(args.config)
    index_bytes = load_json(args.index)[1] if args.index else None
    permissions = validate_permissions(read_yaml(args.permissions))
    result = prepare(metadata, args.revision, config_bytes, index_bytes, permissions)
    result["input_sha256"] = {
        "official_metadata": hashlib.sha256(metadata_bytes).hexdigest(),
        "config": hashlib.sha256(config_bytes).hexdigest(),
        "index": hashlib.sha256(index_bytes).hexdigest() if index_bytes else None,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "revision": args.revision,
                      "total_bytes": result["total_bytes"], "download_execution": "UNEXECUTED"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
