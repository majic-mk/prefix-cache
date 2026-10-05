"""Append-new local G2 source lock and raw-byte upload parts; no authority.

This uses only stdlib and frozen local sources/prior actual model byte receipts.
It neither connects a server nor imports/initializes a framework or a GPU.
Required paths are read from the final runner's exact verify_source_lock AST.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

BASELINE = "artifacts/prefix_io_v1_server09_g1_20261001/gpu-source-lock.json"
BASELINE_REF = (444671, "5b438ebc1ec014c22d1d501915600b4cd9bc777ad9f05da1f7ac9b1a18cc1181")
PAYLOAD_RECEIPT = "artifacts/prefix_io_v1_server09_candidates/model_payload_readonly/SERVER09_MODEL_PAYLOAD_READONLY_RESULT_V2.json"
PAYLOAD_RECEIPT_REF = (97302, "6eed7b029b9d8c6f0e01e3d1ea2e5e9ea7aaee004d0266fab50c0a2a727c1cbb")
DEST = "artifacts/prefix_io_v1/server09-g2-normal-worker-20261001"
LOCAL = "artifacts/prefix_io_v1_server09_candidates/g2_normal_worker"
MIRROR = "artifacts/prefix_io_v1_server09_candidates/g2_source_readonly"
SIZE_LIMIT = 60000


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    def unique(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    raw = path.read_bytes()
    return raw, json.loads(raw.decode("utf-8"), object_pairs_hook=unique)


def valid_ref(row):
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "exact ref shape")
    path = row["path"]
    require(type(path) is str and path and not path.startswith("/") and "\\" not in path
            and all(piece not in ("", ".", "..") for piece in path.split("/")), "relative POSIX ref")
    require(type(row["bytes"]) is int and row["bytes"] >= 0 and
            type(row["sha256"]) is str and re.fullmatch(r"[a-f0-9]{64}", row["sha256"]), "exact ref size/SHA")
    return row


def local_ref(root, relative):
    path = root / relative
    require(path.is_file() and not path.is_symlink(), "local frozen source absent/symlink")
    require(path.resolve().is_relative_to(root), "local source outside workspace")
    raw = path.read_bytes()
    return dict(path=relative, bytes=len(raw), sha256=sha(raw))


def pinned_input(root, relative, expected):
    row = local_ref(root, relative)
    require((row["bytes"], row["sha256"]) == expected, "frozen input bytes/SHA drift: " + relative)
    return row


def static_value(node, names):
    if isinstance(node, ast.Constant):
        require(type(node.value) in (str, int, bool, type(None)), "only scalar literal")
        return node.value
    if isinstance(node, ast.Name):
        require(node.id in names, "unknown literal source constant")
        return names[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = static_value(node.left, names), static_value(node.right, names)
        require(type(left) is str and type(right) is str, "only source-path string addition")
        return left + right
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [static_value(value, names) for value in node.elts]
        return list(values) if isinstance(node, ast.List) else tuple(values) if isinstance(node, ast.Tuple) else set(values)
    raise ValueError("nonliteral required-path/source expression")


def runner_required_paths(raw):
    tree = ast.parse(raw.decode("utf-8"))
    names = {}
    needed = {"DEST", "SCRIPT", "WORKER", "SCALAR", "FRAME", "AUTHOR", "RUNNER", "CONTEXT", "BASELINE_LOCK",
              "BASELINE_SHA", "GUARD", "PREPARE", "QUALIFY", "SMOKE", "PERMISSIONS", "MODEL_PLAN", "MODEL",
              "RUNTIME_CACHE_SOURCE_REFS"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in needed:
                names[key] = static_value(node.value, names)
    require(set(names) == needed and names["DEST"] == DEST and names["BASELINE_SHA"] == BASELINE_REF[1],
            "exact final normal constants required")
    verify = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_source_lock"]
    require(len(verify) == 1, "one actual verify_source_lock function")
    required = [node.value for node in ast.walk(verify[0]) if isinstance(node, ast.Assign) and
                len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "required"]
    require(len(required) == 1, "one actual required source set")
    paths = static_value(required[0], dict(names, baseline_relative=names["BASELINE_LOCK"]))
    require(type(paths) is set and all(type(path) is str for path in paths), "actual required source path set")
    return names, paths, sha(ast.dump(verify[0], include_attributes=False).encode())


def build(root, args):
    local = root / LOCAL
    output = local / "gpu-source-lock-candidate.json"
    parts_folder = local / "source-lock-upload-parts"
    audit_path = local / "CPU_SOURCE_LOCK_PREPARATION_RESULT.json"
    require(not output.exists() and not parts_folder.exists() and not audit_path.exists(), "append-new local outputs only")
    original_inputs = {}
    baseline_row = pinned_input(root, BASELINE, BASELINE_REF)
    original_inputs[BASELINE] = baseline_row
    _, baseline = read_json(root / BASELINE)
    require(type(baseline) is dict and type(baseline.get("files")) is list and len(baseline["files"]) == 2052,
            "exact frozen G1 2052 refs")
    rows = [deepcopy(valid_ref(row)) for row in baseline["files"]]
    index = {row["path"]: row for row in rows}
    require(len(index) == 2052, "no duplicate old source refs")
    added, repeated, provenance = [], [], []

    def add(row, role, input_ref):
        row = deepcopy(valid_ref(row))
        provenance.append(dict(ref=row, role=role, input_ref=input_ref))
        if row["path"] in index:
            require(index[row["path"]] == row, "conflicting old/new source ref: " + row["path"])
            repeated.append(row)
        else:
            rows.append(row)
            index[row["path"]] = row
            added.append(row)

    final_shas = {"run_g2_normal_model_lifecycle.py": args.run_sha,
                  "g2_worker_observation.py": args.worker_sha, "G2_NORMAL_MODEL_CPU_PLAN.json": args.plan_sha}
    for name, expected_sha in final_shas.items():
        row = local_ref(root, LOCAL + "/" + name)
        require(row["sha256"] == expected_sha and re.fullmatch(r"[a-f0-9]{64}", expected_sha), "final approved normal source changed")
        original_inputs[row["path"]] = row
        add(dict(row, path=DEST + "/" + name), "final normal implementation/plan", row)
    normal_raw = (local / "run_g2_normal_model_lifecycle.py").read_bytes()
    names, required, verify_ast_sha = runner_required_paths(normal_raw)
    add(dict(path=names["BASELINE_LOCK"], bytes=baseline_row["bytes"], sha256=baseline_row["sha256"]),
        "preserved old G1 lock itself; not G2 authority", baseline_row)
    for local_relative, remote_relative, role, expected in (
        ("artifacts/prefix_io_v1_server09_candidates/runtime_connector/p4_runtime_scalar_connector.py", names["SCALAR"], "frozen scalar connector",
            (14795, "347fb842989e8e14c73968528396721b546a11d7312206aae2efbb3739b0f3cb")),
        ("artifacts/prefix_io_v1_server09_candidates/collector/p4_full_step_frame_adapter.py", names["FRAME"], "frozen frame adapter",
            (16067, "bea850fd60831010c2ac0bf88c1f5f121f3ffca50d63ab56759319ed6e512582"))):
        row = pinned_input(root, local_relative, expected)
        original_inputs[local_relative] = row
        add(dict(row, path=remote_relative), role, row)

    context_meta = MIRROR + "/TORCH_CONTEXT_SOURCE_REF.json"
    context_input = local_ref(root, context_meta); original_inputs[context_meta] = context_input
    _, context = read_json(root / context_meta)
    context = {key: context[key] for key in ("path", "bytes", "sha256")}
    require(context["path"] == names["CONTEXT"], "actual Torch context path differs")
    context_actual = local_ref(root, MIRROR + "/" + context["path"])
    require((context_actual["bytes"], context_actual["sha256"]) == (context["bytes"], context["sha256"]), "actual Torch context mirror drift")
    original_inputs[context_actual["path"]] = context_actual
    add(context, "actual server Torch decorator source mirrored without import", context_input)

    cache_meta = MIRROR + "/RUNTIME_CACHE_ENV_SOURCE_REFS.json"
    cache_input = local_ref(root, cache_meta); original_inputs[cache_meta] = cache_input
    _, cache_refs = read_json(root / cache_meta)
    expected_cache = [dict(path=path, bytes=size, sha256=value) for path, size, value in names["RUNTIME_CACHE_SOURCE_REFS"]]
    require(type(cache_refs) is list and len(cache_refs) == 3 and cache_refs == expected_cache, "three actual cache source refs match final runner")
    for row in cache_refs:
        actual = local_ref(root, MIRROR + "/" + row["path"])
        require((actual["bytes"], actual["sha256"]) == (row["bytes"], row["sha256"]), "actual cache source mirror drift")
        original_inputs[actual["path"]] = actual
        add(row, "actual server supported process-cache source mirrored without import", cache_input)

    receipt_input = pinned_input(root, PAYLOAD_RECEIPT, PAYLOAD_RECEIPT_REF)
    original_inputs[PAYLOAD_RECEIPT] = receipt_input
    _, actual_model = read_json(root / PAYLOAD_RECEIPT)
    require(actual_model["remote_exit"] == 0 and actual_model["remote_stderr"] == "", "prior actual CPU asset read must have succeeded")
    result = actual_model["result"]
    require(result["status"] == "PASS_OFFICIAL_ASSET_BYTES_SHA256_ONLY" and result["framework_imports"] == 0 and
            result["model_loaded"] is False and result["actual_GPU_runs"] == 0 and result["remote_application_writes"] == 0 and
            result["all_model_metadata_stable"] is True and result["all_provider_evidence_bytes_rechecked"] is True and
            result["official_expected_sha256_available"] is True and result["expected_hashes_matched_to_saved_official_listing"] is True,
            "prior actual complete official asset CPU byte receipt, never runtime qualification")
    add(result["plan_ref"], "official fixed model plan from actual CPU asset receipt", receipt_input)
    streams = result["stream_hash_records"]
    require(type(streams) is list and len(streams) == 17 and Counter(row["category"] for row in streams) ==
            Counter(model_payload=11, saved_provider_provenance=6), "11 actual model plus six provenance streams")
    for stream in streams:
        require(stream["algorithm"] == "sha256" and stream["stat_before"] == stream["stat_after"] and
                stream["stat_before"]["bytes"] == stream["bytes"], "prior actual stable stream SHA evidence")
        add({key: stream[key] for key in ("path", "bytes", "sha256")}, "actual CPU stream: " + stream["category"], receipt_input)

    selected = (
        ("g2_native_drain", "server09-g2-native-drain-20261001",
            (("g2_native_drain_provider.py", 21997, "261f8cd2438a4941d41077b20c95bfab4ca4418858bbab09aa7a3ef459353a44"),
             ("G2_NATIVE_DRAIN_CPU_PLAN.json", 14096, "04d97daf3b3d72102947f711f2aa8ee331f797deccfed853e3acee42bd9b2fe6"))),
        ("g2_boundary", "server09-g2-boundary-20261001",
            (("g2_full_step_boundary_contract.py", 15897, "756b5825f851d372985fc2d02a5ec870d079961092ce41005b8a05a24f4a6f13"),
             ("G2_CPU_BOUNDARY_PLAN.json", 12607, "9b7f18a75a4a26a8a019a55e3799c0d077f6b8128493b9d8c10dd77cf2f50bad"))),
        ("context_v3", "server09-context-v3-20261001",
            (("p4_complete_trace_context_v3.py", 12962, "07dd3af5932395dcc07b325b52be53ed624f4b4af8cfc4bf47dd0b7c58bbb477"),
             ("P4_COMPLETE_TRACE_CONTEXT_V3_PLAN.json", 14930, "1a2571b6a823c30bae553b66bbbf294993403df7ab02ae025e88b05c00fff8e9"))),
        ("paired_v3", "server09-paired-v3-20261001",
            (("p4_paired_context_v3.py", 12836, "00c069beda7bbe154ea006c622fc7301092c5f872efcc34acdc2e06c2da4d0f8"),
             ("P4_PAIRED_CONTEXT_V3_PLAN.json", 6671, "2529462b8e2c1da91b3c5af2a1bb5a5cd8eb1c4f2105f2a08536203d047b04e8"))))
    for source_folder, remote_folder, file_refs in selected:
        for name, size, expected_sha in file_refs:
            row = pinned_input(root, "artifacts/prefix_io_v1_server09_candidates/" + source_folder + "/" + name, (size, expected_sha))
            original_inputs[row["path"]] = row
            add(dict(row, path="artifacts/prefix_io_v1/" + remote_folder + "/" + name), "frozen independent CPU-only supporting implementation/plan", row)

    require(required <= set(index), "actual final verify_source_lock required paths missing")
    require(index[names["GUARD"]] == dict(path=names["GUARD"], bytes=13013, sha256="3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a"), "old current budget guard differs")
    require(index[names["PREPARE"]] == dict(path=names["PREPARE"], bytes=17280, sha256="3059e689f1dcbedc450c867d84e6ba1d6393c0400d710bbbd5dde5495c871fe3"), "old original permission parser differs")
    require(index[names["MODEL_PLAN"]] == dict(path=names["MODEL_PLAN"], bytes=7310, sha256="9b1c79986aa1fd62eb9064ecb8b4e2a03fc235a34ac2d69c33a4fcb531a91017"), "old official model plan differs")
    require(rows[:2052] == baseline["files"] and len(rows) == len(index), "all old refs preserved exactly without duplicates")
    model_paths = {row["path"] for row in streams if row["category"] == "model_payload"}
    require(len(model_paths) == 11 and all(path.startswith(names["MODEL"] + "/") for path in model_paths), "actual fixed model paths")
    for row in rows:
        limit = 4 * 1024**3 if row["path"].startswith(names["MODEL"] + "/") else 512 * 1024**2
        require(row["bytes"] <= limit, "actual final checker source byte bound")
    candidate = dict(schema_version=1, status="CPU_ONLY_G2_SOURCE_CANDIDATE",
        allow_gpu_initialization=False, allow_gpu_runs=False, GPU_collector_verified=False,
        production_qualified=False, effect_verified=False,
        baseline_source_lock=dict(path=names["BASELINE_LOCK"], bytes=baseline_row["bytes"], sha256=baseline_row["sha256"]),
        source_binding=dict(old_frozen_refs_preserved=2052, added_refs=len(added), total_refs=len(rows),
            original_G1_lock_modified=False, complete_current_remote_hash_recheck_performed=False,
            model_payload_refs_from_prior_actual_CPU_full_stream_sha=True,
            required_paths_from_final_verify_source_lock_AST=True, actual_verify_source_lock_ast_sha256=verify_ast_sha),
        authority_boundary="This source inventory is CPU preparation only. It does not create a human record, G2 scope, GPU permission or budget reservation. Separate future explicit authority and actual full server byte recheck remain required.",
        files=rows)
    raw = (json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    require(len(raw) <= 4 * 1024**2, "actual final bounded JSON checker")
    lock_ref = dict(path=DEST + "/gpu-source-lock-candidate.json", bytes=len(raw), sha256=sha(raw))
    payloads, parts = [], []
    for offset in range(0, len(raw), SIZE_LIMIT):
        payload = raw[offset:offset + SIZE_LIMIT]
        name = "part-%04d.bin" % (len(parts) + 1)
        payloads.append((name, payload))
        parts.append(dict(path=name, offset=offset, bytes=len(payload), sha256=sha(payload)))
    require(b"".join(data for _, data in payloads) == raw, "raw part reconstruction differs")
    require(all(0 < row["bytes"] <= SIZE_LIMIT and row["offset"] == n * SIZE_LIMIT for n, row in enumerate(parts)), "contiguous bounded raw parts")
    # Recheck every small frozen input after composing the inventory. No model
    # bytes/remote source are reread here; previous actual stream proof is bound.
    for relative, expected in original_inputs.items():
        require(local_ref(root, relative) == expected, "local input changed during preparation")
    require(sha((root / BASELINE).read_bytes()) == BASELINE_REF[1], "old lock was modified")
    forbidden = ("torch", "vllm", "py_kvcache", "flashinfer", "triton", "transformers", "numpy", "pynvml")
    require(not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in forbidden),
            "no framework/backend import in source inventory preparation")
    with output.open("xb") as handle:
        handle.write(raw)
    parts_folder.mkdir()
    for name, data in payloads:
        with (parts_folder / name).open("xb") as handle:
            handle.write(data)
    manifest = dict(schema_version=1, status="CPU_ONLY_G2_SOURCE_CANDIDATE",
        whole_file=lock_ref, parts_directory=DEST + "/source-lock-upload-parts", encoding="raw byte slices, no text/newline conversion",
        maximum_part_bytes=SIZE_LIMIT, part_count=len(parts), parts=parts,
        reconstruction="Read each part in manifest order, verify exact bytes/SHA and contiguous offset, concatenate original bytes, verify whole bytes/SHA before append-new publishing.",
        allow_gpu_initialization=False, allow_gpu_runs=False, remote_upload_performed=False,
        actual_GPU_runs=0, original_G1_lock_modified=False)
    manifest_path = parts_folder / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write((json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    disk_rebuilt = b"".join((parts_folder / row["path"]).read_bytes() for row in parts)
    require(len(disk_rebuilt) == lock_ref["bytes"] and sha(disk_rebuilt) == lock_ref["sha256"] and
            disk_rebuilt == output.read_bytes(), "actual local part files reconstruct original candidate bytes")
    audit = dict(schema_version=1, status="PASS_LOCAL_CPU_SOURCE_CANDIDATE_PREPARATION_ONLY",
        lock_ref=lock_ref, manifest_ref=local_ref(root, manifest_path.relative_to(root).as_posix()),
        baseline_input_ref=baseline_row, final_normal_input_refs=[original_inputs[LOCAL + "/" + name] for name in final_shas],
        exact_old_2052_rows_and_order_preserved=True, old_baseline_bytes_SHA_rechecked=True,
        required_paths=sorted(required), required_paths_count=len(required), required_paths_present=True,
        actual_verify_source_lock_ast_sha256=verify_ast_sha, new_ref_count=len(added), new_refs=added,
        already_present_equal_refs=repeated, ref_provenance=provenance,
        actual_model_payload_receipt_ref=receipt_input, actual_model_stream_ref_count=17,
        local_part_count=len(parts), local_part_bytes_limit=SIZE_LIMIT, actual_disk_part_reconstruction_exact=True,
        current_remote_all_source_bytes_verified=False, framework_imports=0, actual_GPU_runs=0,
        remote_writes=0, human_record_created=False, GPU_authorization_created=False,
        permissions_modified=False, budget_ledger_modified=False,
        gpu_verified=False, production_qualified=False, effect_verified=False)
    with audit_path.open("xb") as handle:
        handle.write((json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps(dict(lock_ref=lock_ref, old_refs=2052, added_refs=len(added), total_refs=len(rows),
        part_count=len(parts), part_max_bytes=max(row["bytes"] for row in parts),
        manifest_ref=local_ref(root, manifest_path.relative_to(root).as_posix()),
        audit_ref=local_ref(root, audit_path.relative_to(root).as_posix()), new_ref_paths=[row["path"] for row in added]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-sha", required=True)
    parser.add_argument("--worker-sha", required=True)
    parser.add_argument("--plan-sha", required=True)
    parsed = parser.parse_args()
    require(all(re.fullmatch(r"[a-f0-9]{64}", value) for value in (parsed.run_sha, parsed.worker_sha, parsed.plan_sha)), "final approved SHA arguments")
    build(parsed.workspace.resolve(strict=True), parsed)
