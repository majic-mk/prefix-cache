"""Read-only CPU model asset verification; no tensor/framework/model imports.

Reuse exact ASTs of five locked existing offline verifier functions. Execute
neither smoke.main nor runtime-cache configuration. Stream content hashes only.
Emit JSONL progress and one final receipt to stdout; write no remote files.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import parse_qs, urlsplit

PROJECT = Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
PLAN_REL = "artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json"
MODEL_REL = "models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444"
OFFLINE_REL = "experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py"
PLAN_EXPECTED = (7310, "9b1c79986aa1fd62eb9064ecb8b4e2a03fc235a34ac2d69c33a4fcb531a91017")
OFFLINE_EXPECTED = (21990, "7d31cce851ca2334fedbe6b66c4e2d9bfff19e8768bafaa52f68e414518d77e2")
FUNCTIONS = ("require", "under_project", "digest_file", "validate_provider", "validate_local_model")
FORBIDDEN_MODULES = ("torch", "vllm", "transformers", "safetensors", "numpy", "pynvml")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def emit(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)


def pinned_bytes(path, expected):
    require(path.is_file() and not path.is_symlink(), "pinned input absent or symlink: " + str(path))
    raw = path.read_bytes()
    require((len(raw), hashlib.sha256(raw).hexdigest()) == expected,
            "fixed source/plan bytes and SHA do not match: " + str(path))
    return raw


def stable_stat(path):
    st = path.stat()
    return dict(device=st.st_dev, inode=st.st_ino, bytes=st.st_size,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


def execute(project):
    project = project.resolve(strict=True)
    plan_path, model_dir, source = project/PLAN_REL, project/MODEL_REL, project/OFFLINE_REL
    plan_bytes = pinned_bytes(plan_path, PLAN_EXPECTED)
    source_bytes = pinned_bytes(source, OFFLINE_EXPECTED)
    plan = json.loads(plan_bytes)
    require(plan["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
            and plan["revision"] == "16c174980d8a1492910551634b4969e69cdc2444"
            and plan["provider"] == "modelscope" and plan["source"] == "https://modelscope.cn"
            and plan["trust_remote_code"] is False, "fixed official model/provider identity differs")
    files = plan["files"]
    require(type(files) is list and len(files) == 11, "exact eleven official files required")
    require(not model_dir.is_symlink() and model_dir.is_dir()
            and model_dir.resolve(strict=True).is_relative_to(project), "local model directory escapes project")
    evidence = {r["role"]: r for r in plan["provenance"]["evidence"]}
    pinned = evidence["pinned_files"]
    pinned_path = project/pinned["path"]
    pinned_raw = pinned_bytes(pinned_path, (pinned["bytes"], pinned["sha256"]))
    provider_listing = json.loads(pinned_raw)
    require(provider_listing["Code"] == 200 and provider_listing["Success"] is True,
            "pinned official provider listing is not a successful response")
    listing_revision = provider_listing["Data"]["LatestCommitter"]["Id"]
    # This saved ModelScope response leaves the body commit Id empty. Preserve
    # that limitation; the original validator binds the frozen URL Revision
    # and saved response SHA, rather than inventing a missing body identifier.
    require(listing_revision in ("", plan["revision"]), "provider body declares a conflicting revision")
    listing_url = urlsplit(pinned["url"])
    require(listing_url.scheme == "https" and listing_url.netloc == "modelscope.cn"
            and listing_url.path == "/api/v1/models/Qwen/Qwen2.5-7B-Instruct/repo/files"
            and parse_qs(listing_url.query).get("Revision") == [plan["revision"]],
            "saved pinned-listing URL does not bind the fixed revision")
    rows = provider_listing["Data"]["Files"]
    official = {r["Path"]: r for r in rows}
    require(len(official) == len(rows), "duplicate official provider file path")
    names = set()
    snapshots = {}
    official_matches = []
    for record in files:
        name = record["path"]
        require(type(name) is str and re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                and name not in (".", "..") and name not in names, "unsafe/duplicate model file")
        names.add(name)
        require(record["hash_algorithm"] == "sha256" and re.fullmatch(r"[a-f0-9]{64}",record["hash"]),
                "official content SHA-256 unavailable; refuse to synthesize expected hash")
        row = official[name]
        require(row["Type"] == "blob" and type(row["Size"]) is int and row["Size"] == record["bytes"]
                and row["Sha256"] == record["hash"], "expected plan hash/bytes differ from saved official listing")
        url = urlsplit(record["url"])
        query = parse_qs(url.query)
        require(url.scheme == "https" and url.netloc == "modelscope.cn"
                and url.path == "/api/v1/models/Qwen/Qwen2.5-7B-Instruct/repo"
                and query.get("Revision") == [plan["revision"]] and query.get("FilePath") == [name],
                "fixed model file URL/revision differs")
        path = model_dir/name
        require(path.is_file() and not path.is_symlink() and path.resolve(strict=True).parent == model_dir,
                "model file missing, symlink or unexpected location: " + name)
        snapshots[name] = stable_stat(path)
        require(snapshots[name]["bytes"] == record["bytes"], "model file size mismatch: " + name)
        official_matches.append(dict(path=name,bytes=record["bytes"],sha256=record["hash"]))
    require(sum(r["bytes"] for r in files) == plan["total_bytes"], "official plan total size mismatch")
    weight_names = {r["path"] for r in files if r["path"].endswith(".safetensors")}
    require(len(weight_names) == 4, "exact four official weight shards required")
    emit(dict(event="PREFLIGHT_PASS",official_expected_sha256_available=True,
        expected_hashes_matched_to_saved_official_listing=True,file_count=11,
        model_bytes=plan["total_bytes"],weight_bytes=sum(r["bytes"] for r in files if r["path"] in weight_names),
        no_model_or_framework_import=True))
    tree = ast.parse(source_bytes.decode("utf-8"),filename=str(source))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    require(len(functions) == len(FUNCTIONS) and {n.name for n in functions} == set(FUNCTIONS),
            "locked five offline functions absent")
    ast_refs = {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in functions}
    namespace = dict(ROOT=project,MODEL_ID=plan["model_id"],PROMPT_TOKEN_IDS=list(range(1000,1128)),
        hashlib=hashlib,json=json,Path=Path,re=re,parse_qs=parse_qs,urlsplit=urlsplit)
    module = ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[]))
    exec(compile(module,str(source)+"::readonly-offline-function-clone","exec"),namespace)
    digest = namespace["digest_file"]
    hashes = []
    def audited_digest(path, algorithm):
        path = Path(path)
        before = stable_stat(path)
        started = time.monotonic()
        size, value = digest(path,algorithm)
        after = stable_stat(path)
        require(before == after and size == before["bytes"], "file changed while hashing: " + str(path))
        category = "model_payload" if path.parent == model_dir else "saved_provider_provenance"
        item = dict(path=str(path.relative_to(project)),category=category,bytes=size,
            algorithm=algorithm,sha256=value,stat_before=before,stat_after=after,
            wall_seconds=time.monotonic()-started)
        hashes.append(item)
        emit(dict(event="FILE_STREAM_HASH_COMPLETE",**item))
        return size,value
    namespace["digest_file"] = audited_digest
    try:
        _, verified = namespace["validate_local_model"](model_dir,plan_path)
    except Exception as exc:
        emit(dict(event="MODEL_PAYLOAD_CPU_READONLY_FAILED",error=type(exc).__name__+": "+str(exc),
            completed_stream_hashes=hashes,actual_GPU_runs=0,no_download=True,no_model_loaded=True,
            remote_application_writes=0))
        raise
    pinned_bytes(plan_path,PLAN_EXPECTED)
    pinned_bytes(source,OFFLINE_EXPECTED)
    for record in files:
        require(stable_stat(model_dir/record["path"]) == snapshots[record["path"]],
                "model file metadata changed during full asset verification: " + record["path"])
    # Recheck small provenance sources after payload reads. Do not reread weights.
    for item in plan["provenance"]["evidence"]:
        pinned_bytes(project/item["path"],(item["bytes"],item["sha256"]))
    require(not any(name in sys.modules or any(n.startswith(name+".") for n in sys.modules)
                    for name in FORBIDDEN_MODULES), "forbidden backend/framework import detected")
    payload_hashes = [h for h in hashes if h["category"] == "model_payload"]
    require(len(payload_hashes) == 11 and len(verified["verified_local_files"]) == 11,
            "complete original offline verifier did not cover all eleven files")
    index = json.loads((model_dir/"model.safetensors.index.json").read_text())
    return dict(event="MODEL_PAYLOAD_CPU_READONLY_PASS",status="PASS_OFFICIAL_ASSET_BYTES_SHA256_ONLY",
        completed_at_utc=datetime.now(timezone.utc).isoformat(),project=str(project),model_dir=str(model_dir),
        model_id=plan["model_id"],revision=plan["revision"],provider=plan["provider"],
        plan_ref=dict(path=PLAN_REL,bytes=PLAN_EXPECTED[0],sha256=PLAN_EXPECTED[1]),
        offline_source_ref=dict(path=OFFLINE_REL,bytes=OFFLINE_EXPECTED[0],sha256=OFFLINE_EXPECTED[1]),
        official_listing_ref=dict(path=pinned["path"],bytes=pinned["bytes"],sha256=pinned["sha256"]),
        provider_listing_body_commit_id=listing_revision,
        provider_revision_binding="fixed saved API URL Revision and independently pinned response SHA; no body commit ID present" if not listing_revision else "saved URL and response body commit ID",
        reused_exact_function_ast_refs=ast_refs,official_expected_hashes=official_matches,
        official_expected_sha256_available=True,expected_hashes_matched_to_saved_official_listing=True,
        verified_local_files=verified["verified_local_files"],stream_hash_records=hashes,
        model_file_count=11,weight_shard_count=4,model_bytes=sum(h["bytes"] for h in payload_hashes),
        weight_bytes=sum(h["bytes"] for h in payload_hashes if h["path"].endswith(".safetensors")),
        weight_map_entries=len(index["weight_map"]),index_tensor_payload_bytes=index["metadata"]["total_size"],
        all_model_metadata_stable=True,all_provider_evidence_bytes_rechecked=True,
        model_loaded=False,tensors_loaded=False,tokenizer_loaded=False,cache_created=False,
        framework_imports=0,actual_GPU_runs=0,gpu_verified=False,production_qualified=False,
        effect_verified=False,P4_complete=False,downloads=0,remote_application_writes=0,
        original_source_modified=False,permissions_modified=False,budget_ledger_modified=False,
        original_offline_model_validation_passed=True,
        limitations=["Offline bytes match the separately frozen plan and saved official provider response; no fresh network request was made.",
            "No tensor load, normal model execution, kernel/runtime compatibility, real GPU event timer or method effect was tested."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",type=Path,default=PROJECT)
    args = parser.parse_args()
    started = time.monotonic()
    result = execute(args.project)
    result["wall_seconds"] = time.monotonic()-started
    emit(result)
