"""Replay only this round's input and shared frontend-patch checks; no GPU imports."""
from pathlib import Path
import ast
import hashlib
import json


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    inputs = json.loads((here / "FROZEN_REQUEST_INPUTS_02.json").read_bytes())
    source = inputs["source"]
    raw = Path(source["path"]).read_bytes()
    assert len(raw) == source["bytes"]
    assert hashlib.sha256(raw).hexdigest() == source["sha256"]
    originals = {str(row["request_id"]): row for row in json.loads(raw)["requests"]}
    records = inputs["records"]
    assert len(records) == 12
    assert inputs["frontend_max_concurrency"] == 3
    assert inputs["engine_max_num_seqs"] == 1
    assert inputs["frozen_common"]["preload_lookahead_requests"] == 2
    for index, row in enumerate(records):
        assert row["prompt_token_ids"] == originals[str(row["source_request_id"])]["prompt_token_ids"]
        assert len(row["prompt_token_ids"]) == 768
        assert row["scheduled_ns"] == index * 250_000_000
        assert row["max_tokens"] == 128
    assert records[-1]["prompt_token_ids"] == records[0]["prompt_token_ids"]
    blocks = {tuple(row["prompt_token_ids"][:n]) for row in records[:11] for n in range(16, 769, 16)}
    assert len(blocks) == 528
    assert 528 * 917504 > inputs["frozen_common"]["kv_cache_memory_bytes"] + inputs["frozen_common"]["staging_bytes"]
    assert 768 + 128 <= inputs["frozen_common"]["max_model_len"]
    runner = root / "artifacts/prefix_io_v1_server13_candidates/public_development_gpu_20261005/u_collection_runtime/strong_trace_runner_v7.py"
    original = runner.read_text(encoding="utf-8")
    before = '    require(config_engine["max_num_seqs"] >= concurrency, "common engine supports frozen concurrency")'
    after = '    integer(config_engine["max_num_seqs"], "original engine running sequence bound", 1, 8)\n    integer(concurrency, "frontend accepted/waiting request bound", 1, 8)'
    assert original.count(before) == 1
    patched = original.replace(before, after)
    compile(patched, str(runner), "exec", dont_inherit=True)
    original_ast, patched_ast = ast.parse(original), ast.parse(patched)
    for name in ("drive_original_engine", "main"):
        old = next(node for node in original_ast.body if isinstance(node, ast.FunctionDef) and node.name == name)
        new = next(node for node in patched_ast.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
    print(json.dumps({"status": "PASS_CPU_INPUT_AND_PATCH_CHECKS_ONLY", "requests": 12,
        "prompt_tokens_each": 768, "output_tokens_each": 128, "prefix_blocks": 528,
        "prefix_bytes": 484442112, "engine_driver_and_main_AST_unchanged": True,
        "GPU_runs": 0, "patch_applied_to_server": False,
        "limited_cost_consumer_implemented": False, "GPU_comparison_ready": False}))


if __name__ == "__main__":
    main()
