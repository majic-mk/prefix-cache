#!/usr/bin/env python3
"""P1 native GPU Prefix path smoke; run only through run_gpu_stage.py.

This controlled synthetic-token diagnostic has no py-kvcache/SSD connector,
no performance claim, and no end-to-end or production-KV byte qualification.
Only stdlib imports occur before local/offline validation. No network or download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[3]
AUTHOR_ROOT = ROOT / "third_party/work/vllm-author-build"
AUTHOR_COMMIT = "817a7e3124f817cd6e549581d3e5483207a753a4"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PROMPT_TOKEN_IDS = list(range(1000, 1128))
ENGINE = {
    "dtype": "bfloat16",
    "kv_cache_dtype": "bfloat16",
    "quantization": None,
    "load_format": "safetensors",
    "trust_remote_code": False,
    "skip_tokenizer_init": True,
    "tensor_parallel_size": 1,
    "distributed_executor_backend": "uni",
    "max_model_len": 256,
    "max_num_seqs": 1,
    "max_num_batched_tokens": 256,
    "block_size": 16,
    "kv_cache_memory_bytes": 67108864,
    "gpu_memory_utilization": 0.70,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": False,
    "cpu_offload_gb": 0,
    "offload_group_size": 0,
    "kv_offloading_size": None,
    "kv_transfer_config": None,
    "attention_config": {"backend": "TRITON_ATTN"},
    "enforce_eager": True,
    "compilation_config": 0,
    "generation_config": "vllm",
    "disable_log_stats": True,
    "seed": 0,
}
SAMPLING = {
    "temperature": 0.0,
    "seed": 0,
    "max_tokens": 16,
    "min_tokens": 16,
    "ignore_eos": True,
    "detokenize": False,
}
EXPECTED_HOT_TOKENS = ((len(PROMPT_TOKEN_IDS) - 1) // ENGINE["block_size"]
                       * ENGINE["block_size"])


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_new_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def under_project(path):
    resolved = Path(path).resolve(strict=True)
    require(resolved.is_relative_to(ROOT), "input must resolve inside this project")
    return resolved


def digest_file(path, algorithm):
    size = path.stat().st_size
    digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
    if algorithm == "git_blob_sha1":
        digest.update(b"blob " + str(size).encode("ascii") + b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return size, digest.hexdigest()


def validate_provider(plan):
    source = plan.get("source")
    providers = {"https://huggingface.co": "huggingface",
                 "https://modelscope.cn": "modelscope"}
    require(source in providers and plan.get("trust_remote_code") is False,
            "unexpected model plan source")
    provider = plan.get("provider", "huggingface" if source == "https://huggingface.co" else None)
    require(provider == providers[source], "provider/source identity mismatch")
    namespace = provider + "_git_commit"
    provenance = plan.get("provenance")
    if provider == "modelscope":
        require(isinstance(provenance, dict)
                and provenance.get("publisher") == "Qwen"
                and provenance.get("official_channel_url") == "https://modelscope.cn/organization/qwen"
                and provenance.get("publisher_github") == "https://github.com/QwenLM"
                and provenance.get("revision_namespace") == namespace,
                "official ModelScope publisher provenance required")
        required_roles = {"official_release", "provider_model", "pinned_files",
                          "config", "safetensors_index", "tokenizer_config"}
        seen = set()
        for item in provenance.get("evidence", []):
            role = item["role"]
            require(role in required_roles and role not in seen, "unexpected provenance role")
            seen.add(role)
            reference = Path(item["path"])
            require(not reference.is_absolute(), "provenance path must be project-relative")
            path = under_project(ROOT / reference)
            size, sha = digest_file(path, "sha256")
            require(size == item["bytes"] and sha == item["sha256"],
                    "saved official provenance evidence hash/size mismatch")
            parsed = urlsplit(item["url"])
            if role == "official_release":
                require(item["url"] == "https://qwenlm.github.io/blog/qwen2.5/",
                        "official Qwen release URL required")
            else:
                base = "/api/v1/models/" + MODEL_ID
                suffix = "" if role == "provider_model" else "/repo/files" if role == "pinned_files" else "/repo"
                require(parsed.scheme == "https" and parsed.netloc == "modelscope.cn"
                        and parsed.path == base + suffix, "unexpected provider evidence URL")
                if role != "provider_model":
                    query = parse_qs(parsed.query)
                    require(query.get("Revision") == [plan["revision"]],
                            "provenance URL revision differs from ModelScope commit")
                    if role not in ("pinned_files",):
                        filename = {"config": "config.json",
                                    "safetensors_index": "model.safetensors.index.json",
                                    "tokenizer_config": "tokenizer_config.json"}[role]
                        require(query.get("FilePath") == [filename], "provenance file URL mismatch")
        require(seen == required_roles, "incomplete official ModelScope provenance")
    return {"source": source, "provider": provider, "revision_namespace": namespace,
            "provenance": provenance}


RUNTIME_CACHE_SUBDIRS = {
    "VLLM_CACHE_ROOT": "vllm",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "XDG_CACHE_HOME": "xdg",
    "HF_HOME": "huggingface",
}
RUNTIME_PATH_ENV_KEYS = (*RUNTIME_CACHE_SUBDIRS, "TMPDIR")


def configure_runtime_environment():
    # Local files use the same native HF-format loader regardless of provenance.
    # These affect only this process and its workers, never the server environment.
    os.environ.update(VLLM_USE_MODELSCOPE="0", VLLM_NO_USAGE_STATS="1", VLLM_DO_NOT_TRACK="1")
    # Required by the author's public callable RPC serializer. Trusted local
    # offline workers only; this script starts no HTTP/API service.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    base = ROOT / "experiments/prefix_io_v1"
    paths = {key: base / "runtime-cache" / name for key, name in RUNTIME_CACHE_SUBDIRS.items()}
    paths["TMPDIR"] = base / "runtime-tmp"
    for key, path in paths.items():
        path = path.resolve()
        require(path.is_relative_to(ROOT), "runtime cache/temp path escapes project")
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)


def validate_local_model(model_dir, plan_path):
    """Validate actual local bytes against the separately pinned official plan."""
    model_dir = under_project(model_dir)
    plan_path = under_project(plan_path)
    require(model_dir.is_dir(), "local model directory required")
    raw_plan = plan_path.read_bytes()
    plan = json.loads(raw_plan)
    require(plan["model_id"] == MODEL_ID, "wrong fixed model identity")
    revision = plan["revision"]
    require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
            "real frozen 40-hex model revision required; main/null rejected")
    provider = validate_provider(plan)
    verified = {}
    for record in plan["files"]:
        name = record["path"]
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                and name not in (".", "..") and name not in verified,
                "unsafe or duplicate model file name")
        path = under_project(model_dir / name)
        require(path.is_file(), f"model file missing: {name}")
        algorithm = record["hash_algorithm"]
        require(algorithm in ("sha256", "git_blob_sha1"), "unknown model hash")
        require(provider["provider"] != "modelscope" or algorithm == "sha256",
                "ModelScope files require content SHA-256")
        actual_size, actual_hash = digest_file(path, algorithm)
        require(actual_size == record["bytes"] and actual_hash == record["hash"],
                f"local model file does not match official plan: {name}")
        verified[name] = {"bytes": actual_size, "hash_algorithm": algorithm,
                          "hash": actual_hash}
    require("config.json" in verified and "model.safetensors.index.json" in verified,
            "this smoke requires the complete official sharded safetensors layout")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    require(bool(shards) and all(name in verified and name.endswith(".safetensors")
                                for name in shards), "unverified weight shard")
    require({path.name for path in model_dir.glob("*.safetensors")} == shards,
            "local safetensors files differ from the verified official index")
    config = json.loads((model_dir / "config.json").read_text())
    expected = {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"],
                "num_hidden_layers": 28, "hidden_size": 3584,
                "num_attention_heads": 28, "num_key_value_heads": 4}
    require(all(config.get(key) == value for key, value in expected.items()),
            "unexpected fixed Qwen2.5-7B architecture")
    require(not config.get("auto_map") and not config.get("quantization_config")
            and not config.get("use_sliding_window"), "unsupported model variant")
    require(config.get("dtype", config.get("torch_dtype")) == "bfloat16",
            "official BF16 weights required")
    require(max(PROMPT_TOKEN_IDS) < config["vocab_size"], "invalid prompt token IDs")
    return model_dir, {**provider, "model_id": MODEL_ID, "revision": revision,
                       "manifest_sha256": hashlib.sha256(raw_plan).hexdigest(),
                       "plan_path": str(plan_path),
                       "plan_sha256": hashlib.sha256(raw_plan).hexdigest(),
                       "verified_local_files": verified}


def assert_no_external_cache():
    require(not any(name == "kvcache" or name.startswith("kvcache.")
                    or name == "py_kvcache" or name.startswith("py_kvcache.")
                    for name in sys.modules), "unexpected py-kvcache import")


def summarize_output(outputs):
    require(len(outputs) == 1, "expected one RequestOutput")
    output = outputs[0]
    require(output.finished and len(output.outputs) == 1, "request not finished")
    require(list(output.prompt_token_ids) == PROMPT_TOKEN_IDS, "prompt IDs changed")
    require(type(output.num_cached_tokens) is int, "native cached-token evidence absent")
    completion = output.outputs[0]
    tokens = list(completion.token_ids)
    require(len(tokens) == SAMPLING["max_tokens"], "unexpected output token count")
    return {"request_id": output.request_id,
            "prompt_token_ids": list(output.prompt_token_ids),
            "output_token_ids": tokens,
            "num_cached_tokens": output.num_cached_tokens,
            "finish_reason": completion.finish_reason}



def worker_kv_storage_metadata(worker, expected_layers=28, budget_bytes=67108864):
    """Read bounded tensor metadata through the author's existing worker RPC."""
    import torch

    def checked(condition, message):
        if not condition:
            raise RuntimeError(message)

    checked(type(expected_layers) is int and expected_layers == 28,
            "only the fixed 28-layer Qwen architecture is qualified")
    checked(type(budget_bytes) is int and budget_bytes == 67108864,
            "unexpected frozen KV tensor budget")
    runner = worker.model_runner
    caches = runner.kv_caches
    config = runner.kv_cache_config
    checked(isinstance(caches, list) and len(caches) == expected_layers,
            "uninitialized or unexpected native KV tensor list")
    groups = config.kv_cache_groups
    declared = config.kv_cache_tensors
    checked(len(groups) == 1 and len(groups[0].layer_names) == expected_layers
            and len(set(groups[0].layer_names)) == expected_layers,
            "unexpected native KV layer groups")
    checked(0 < len(declared) <= expected_layers and type(config.num_blocks) is int
            and config.num_blocks > 0, "uninitialized native KV tensor configuration")
    sizes = [item.size for item in declared]
    checked(all(type(size) is int and size > 0 for size in sizes),
            "invalid declared KV backing size")
    declared_bytes = sum(sizes)
    checked(declared_bytes <= budget_bytes, "declared KV tensor sizes exceed budget")
    unique = {}
    shapes = []
    for tensor in caches:
        checked(isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda"
                and tensor.dtype == torch.bfloat16,
                "native KV cache must contain CUDA BF16 tensors")
        shape = tuple(tensor.shape)
        checked(0 < len(shape) <= 8 and all(type(dim) is int and dim > 0 for dim in shape),
                "invalid native KV tensor shape")
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        pointer = storage.data_ptr()
        checked(type(size) is int and 0 < size <= budget_bytes
                and type(pointer) is int and pointer > 0,
                "empty or invalid native KV backing storage")
        # Storage pointers, unlike view pointers, identify shared backing memory.
        # Pointers remain local to this RPC and are not returned as evidence.
        key = (tensor.device.type, tensor.device.index, pointer)
        checked(key not in unique or unique[key] == size,
                "inconsistent size for a shared native KV storage")
        unique[key] = size
        shapes.append(list(shape))
    actual_bytes = sum(unique.values())
    checked(0 < actual_bytes <= budget_bytes and actual_bytes == declared_bytes,
            "actual KV tensor backing differs from the initialized native configuration")
    return {"tensor_count": len(caches), "unique_storage_count": len(unique),
            "tensor_shapes": shapes, "dtype": "torch.bfloat16", "device_type": "cuda",
            "num_blocks": config.num_blocks,
            "declared_kv_tensor_bytes": declared_bytes,
            "actual_kv_tensor_allocation_bytes": actual_bytes,
            "measurement": "sum_unique_untyped_storage_nbytes_metadata_only",
            "cuda_allocator_reserved_bytes": None,
            "physical_release_witness": None}


def main():
    configure_runtime_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-plan", required=True, type=Path,
                        help="Real official fixed-revision Hugging Face or ModelScope manifest")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New subdirectory under experiments/prefix_io_v1/runs")
    args = parser.parse_args()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        require(os.environ.get(name) == "1", f"{name}=1 required from budget runner")
    require(re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                         os.environ.get("CUDA_VISIBLE_DEVICES", "")),
            "one complete GPU UUID must be bound by the authorized budget runner")
    output_dir = args.output_dir.resolve()
    require(output_dir.is_relative_to(ROOT / "experiments/prefix_io_v1/runs"),
            "output must remain inside the project run directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "FAILED", "scope": "native GPU Prefix synthetic-token smoke",
              "gpu_model_initialization_attempted": False,
              "ssd_or_staging_qualified": False, "end_to_end_cache_qualified": False,
              "production_kv_byte_identity_qualified": False,
              "performance_claim": False, "results": []}
    llm = None
    try:
        model_dir, model = validate_local_model(args.model_dir, args.model_plan)
        actual_commit = subprocess.check_output(
            ["git", "-C", str(AUTHOR_ROOT), "rev-parse", "HEAD"], text=True).strip()
        require(actual_commit == AUTHOR_COMMIT, "author source commit mismatch")
        frozen = {"schema_version": 1, "model": model, "model_dir": str(model_dir),
                  "author_commit": actual_commit, "engine": ENGINE, "sampling": SAMPLING,
                  "prompt_token_ids": PROMPT_TOKEN_IDS,
                  "expected_cached_tokens": [0, EXPECTED_HOT_TOKENS],
                  "environment": {key: os.environ[key] for key in
                                  ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                                   "CUDA_VISIBLE_DEVICES", "VLLM_USE_MODELSCOPE",
                                   "VLLM_NO_USAGE_STATS", "VLLM_DO_NOT_TRACK",
                                   "VLLM_ALLOW_INSECURE_SERIALIZATION", *RUNTIME_PATH_ENV_KEYS)},
                  "scope": report["scope"], "working_directory": str(output_dir)}
        write_new_json(output_dir / "frozen-config.json", frozen)
        # Author profiler paths are relative to cwd; preserve each run separately.
        os.chdir(output_dir)
        assert_no_external_cache()
        import vllm
        from vllm import LLM, SamplingParams

        require(Path(vllm.__file__).resolve().is_relative_to(AUTHOR_ROOT),
                "vLLM import is outside the pinned author build worktree")
        report["vllm_module"] = str(Path(vllm.__file__).resolve())
        report["gpu_model_initialization_attempted"] = True
        llm = LLM(model=str(model_dir), **ENGINE)
        config = llm.llm_engine.vllm_config
        require(config.kv_transfer_config is None and config.cache_config.kv_offloading_size is None,
                "external KV connector/offloading must remain disabled")
        require(config.cache_config.enable_prefix_caching
                and config.cache_config.block_size == ENGINE["block_size"]
                and config.cache_config.cache_dtype == "bfloat16",
                "effective native cache settings differ from frozen smoke")
        require(config.cache_config.kv_cache_memory_bytes == ENGINE["kv_cache_memory_bytes"],
                "effective KV pool budget differs from frozen 64 MiB")
        report["effective_kv_cache_memory_bytes"] = config.cache_config.kv_cache_memory_bytes
        storage_reports = llm.collective_rpc(worker_kv_storage_metadata, timeout=10.0)
        require(isinstance(storage_reports, list) and len(storage_reports) == 1,
                "exactly one native worker KV metadata result required")
        storage_report = storage_reports[0]
        report["kv_tensor_storage_metadata"] = storage_report
        report["actual_kv_tensor_allocation_bytes"] = storage_report["actual_kv_tensor_allocation_bytes"]
        write_new_json(output_dir / "kv-tensor-storage-metadata.json", storage_report)
        for name in ("cold", "repeat"):
            outputs = llm.generate([{"prompt_token_ids": PROMPT_TOKEN_IDS.copy()}],
                                   SamplingParams(**SAMPLING), use_tqdm=False)
            item = summarize_output(outputs)
            item["phase"] = name
            report["results"].append(item)
            write_new_json(output_dir / f"{name}.json", item)
        cold, repeat = report["results"]
        require(cold["num_cached_tokens"] == 0, "cold request unexpectedly cached")
        require(repeat["num_cached_tokens"] == EXPECTED_HOT_TOKENS,
                "repeat did not prove the expected native exact Prefix hit")
        require(cold["output_token_ids"] == repeat["output_token_ids"],
                "cold/repeat output token IDs differ; no tolerance fallback")
        assert_no_external_cache()
        report["status"] = "PASSED_GPU_PREFIX_PATH_ONLY"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
    finally:
        if llm is not None:
            try:
                # Pinned native EngineCoreClient API; no replacement lifecycle.
                llm.llm_engine.engine_core.shutdown(timeout=15.0)
                report["native_engine_shutdown"] = "completed"
            except Exception as exc:
                report["status"] = "FAILED"
                report["shutdown_error"] = f"{type(exc).__name__}: {exc}"
        write_new_json(output_dir / "smoke-result.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASSED_GPU_PREFIX_PATH_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
