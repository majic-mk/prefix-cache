"""Append-new ninja G2 source candidate and raw slices, with no authority.

Uses only frozen local bytes and the original actual CPU ninja audit. Original
4035 ordered refs are retained. No remote connection, backend import, binary
execution, GPU scope, permission change or ledger write occurs here.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

REMOTE_ROOT = "/root/autodl-tmp/prefix-io-v1-handoff/project"
LOCAL = "artifacts/prefix_io_v1_server09_candidates/g2_normal_worker_cuda13_ninja"
DEST = "artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-ninja-20261001"
G1_LOCAL = "artifacts/prefix_io_v1_server09_g1_20261001/gpu-source-lock.json"
G1_REMOTE = "artifacts/prefix_io_v1/server09-g1-20261001/gpu-source-lock.json"
G1_REF = (444671, "5b438ebc1ec014c22d1d501915600b4cd9bc777ad9f05da1f7ac9b1a18cc1181")
G2_LOCAL = "artifacts/prefix_io_v1_server09_candidates/g2_normal_worker/gpu-source-lock-candidate.json"
G2_REF = (452304, "5e7cc04fe759411800b895169d669f75cade6b2f0adda34d3992ecf2a7dbd760")
PRIOR_LOCAL = "artifacts/prefix_io_v1_server09_candidates/g2_normal_worker_cuda13/gpu-source-lock-candidate.json"
PRIOR_REMOTE = "artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-20261001/gpu-source-lock-candidate.json"
PRIOR_REF = (895738, "f3d21df27310ce2c5199ab69f2a62f80b2111390f3ad160aba6ea843dd06f1d9")
NINJA_AUDIT_LOCAL = "artifacts/prefix_io_v1_server09_g2_cuda13_20261001/ACTUAL_CPU_BUILD_LAUNCHER_AUDIT.json"
NINJA_AUDIT_REF = (15438, "edcbefe014178bec50a66b13f3b8358f6cc1550f6159cd971936c197b2645f36")
NINJA = dict(path=".venv/bin/ninja", bytes=370448,
             sha256="08639e194fffa7f08b259fc4abfa4803aff66b64de52549cee42ec527d55cea6")
PART_LIMIT = 60000


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    raw = path.read_bytes()
    return raw, json.loads(raw.decode("utf-8"), object_pairs_hook=unique)


def valid_ref(row):
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "exact source ref")
    path = row["path"]
    require(type(path) is str and path and not path.startswith("/") and "\\" not in path and ":" not in path and
            all(piece not in ("", ".", "..") for piece in path.split("/")), "project-relative POSIX source path")
    require(type(row["bytes"]) is int and row["bytes"] >= 0 and type(row["sha256"]) is str and
            re.fullmatch(r"[a-f0-9]{64}", row["sha256"]), "exact source bytes/hash")
    return row


def local_ref(root, relative):
    path = root / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), "regular frozen local source")
    raw = path.read_bytes()
    return valid_ref(dict(path=relative, bytes=len(raw), sha256=sha(raw)))


def pinned(root, relative, expected):
    row = local_ref(root, relative)
    require((row["bytes"], row["sha256"]) == expected, "frozen input drift: " + relative)
    return row


def literal(node, names):
    if isinstance(node, ast.Constant):
        require(type(node.value) in (str, int, bool, type(None)), "bounded scalar source literal")
        return node.value
    if isinstance(node, ast.Name):
        require(node.id in names, "unresolved source constant")
        return names[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = literal(node.left, names), literal(node.right, names)
        require(type(left) is str and type(right) is str, "only source string addition")
        return left + right
    if isinstance(node, (ast.Tuple, ast.Set, ast.List)):
        values = [literal(value, names) for value in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else set(values) if isinstance(node, ast.Set) else values
    if isinstance(node, ast.Dict):
        require(all(key is not None for key in node.keys), "no expanded literal dict")
        keys = [literal(key, names) for key in node.keys]
        require(len(keys) == len(set(keys)), "no duplicate literal dict key")
        return dict(zip(keys, [literal(value, names) for value in node.values]))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        require(not node.args and all(keyword.arg is not None for keyword in node.keywords), "source dict keywords only")
        return {keyword.arg: literal(keyword.value, names) for keyword in node.keywords}
    if isinstance(node, ast.Subscript):
        owner, key = literal(node.value, names), literal(node.slice, names)
        require(type(owner) in (dict, tuple, list) and type(key) in (str, int), "bounded literal lookup")
        return owner[key]
    if isinstance(node, ast.GeneratorExp):
        require(len(node.generators) == 1 and not node.generators[0].ifs and
                node.generators[0].is_async == 0, "single source-ref literal projection")
        generator = node.generators[0]; values = literal(generator.iter, names)
        require(type(values) in (tuple, list), "bounded source-ref sequence")
        target = generator.target
        require(isinstance(target, (ast.Tuple, ast.List)) and all(isinstance(item, ast.Name) for item in target.elts),
                "simple source-ref tuple target")
        output = []
        for value in values:
            require(type(value) in (tuple, list) and len(value) == len(target.elts), "source-ref tuple arity")
            nested = dict(names); nested.update((key.id, member) for key, member in zip(target.elts, value))
            output.append(literal(node.elt, nested))
        return output
    raise ValueError("nonliteral source expression")


def runner_required(raw):
    tree = ast.parse(raw.decode("utf-8")); names = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try: names[node.targets[0].id] = literal(node.value, names)
            except (ValueError, KeyError, IndexError, TypeError): pass
    require(names.get("DEST") == DEST and names.get("SCRIPT") == DEST + "/run_g2_normal_model_lifecycle.py" and
            names.get("WORKER") == DEST + "/g2_worker_observation.py" and names.get("BASELINE_LOCK") == G1_REMOTE and
            names.get("BASELINE_SHA") == G1_REF[1] and type(names.get("SDK_SOURCE_REFS")) is tuple and
            len(names["SDK_SOURCE_REFS"]) == 8,
            "distinct fixed destination and unchanged old G1 baseline")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_source_lock"]
    require(len(functions) == 1, "one actual source checker")
    function = functions[0]
    assignments = [node for node in ast.walk(function) if isinstance(node, ast.Assign) and len(node.targets) == 1 and
                   isinstance(node.targets[0], ast.Name) and node.targets[0].id == "required"]
    require(len(assignments) == 1, "one actual required set")
    environment = dict(names, baseline_relative=names["BASELINE_LOCK"])
    required = literal(assignments[0].value, environment)
    require(type(required) is set, "actual required set type")
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "required"]
    for call in calls:
        require(len(call.args) == 1 and not call.keywords and call.func.attr in ("update", "add"), "bounded required-set append only")
        value = literal(call.args[0], environment)
        if call.func.attr == "add": required.add(value)
        else:
            require(type(value) in (list, tuple, set), "required update projected literal source list")
            required.update(value)
    require(all(type(path) is str for path in required) and NINJA["path"] in required, "actual checker requires pinned ninja")
    ninja_literals = [value for value in names.values() if value == NINJA or value == tuple(NINJA.values())]
    require(ninja_literals, "actual source constant exact ninja path/bytes/hash")
    return names, required, sha(ast.dump(function, include_attributes=False).encode())


def load_inputs(root):
    inputs = {}
    for local, expected in ((G1_LOCAL, G1_REF), (G2_LOCAL, G2_REF), (PRIOR_LOCAL, PRIOR_REF), (NINJA_AUDIT_LOCAL, NINJA_AUDIT_REF)):
        inputs[local] = pinned(root, local, expected)
    _, g1 = read_json(root / G1_LOCAL); _, g2 = read_json(root / G2_LOCAL); _, prior = read_json(root / PRIOR_LOCAL)
    require(type(g1.get("files")) is list and len(g1["files"]) == 2052 and
            type(g2.get("files")) is list and len(g2["files"]) == 2088 and
            type(prior.get("files")) is list and len(prior["files"]) == 4035 and
            prior["files"][:2088] == g2["files"] and prior["files"][:2052] == g1["files"], "all old row values/order preserved")
    require(prior.get("status") == "CPU_ONLY_G2_SOURCE_CANDIDATE" and prior.get("allow_gpu_runs") is False and
            prior.get("allow_gpu_initialization") is False, "prior candidate does not grant authority")
    for row in prior["files"]: valid_ref(row)
    require(len({row["path"] for row in prior["files"]}) == 4035 and NINJA["path"] not in {row["path"] for row in prior["files"]},
            "all prior source refs unique; ninja is new")
    _, audit = read_json(root / NINJA_AUDIT_LOCAL)
    require(audit["result"]["exit"] == 0 and audit["result"]["stderr"] == "", "actual original CPU audit succeeded")
    actual = json.loads(audit["result"]["stdout"])
    require(actual.get("status") == "READ_ONLY_CPU_BUILD_LAUNCHER_AUDIT" and
            all(type(actual.get(key)) is int and actual[key] == 0 for key in ("new_GPU_jobs", "framework_imports", "compiler_executions")) and
            actual["tools"]["ninja"]["selected"] is None, "actual CPU-only launcher lookup audit")
    matches = [row for row in actual["candidate_ninja_files"] if row.get("path") == REMOTE_ROOT + "/" + NINJA["path"]]
    require(len(matches) == 1, "one actual candidate ninja file")
    ninja = matches[0]
    require(all(ninja.get(key) is True for key in ("exists", "regular_file", "executable")) and
            ninja.get("is_symlink") is False and ninja.get("resolved") == ninja["path"] and
            ninja.get("bytes") == NINJA["bytes"] and type(ninja.get("bytes")) is int and ninja.get("sha256") == NINJA["sha256"],
            "original actual canonical executable ninja bytes/hash")
    return inputs, g1, g2, prior


def build(root, args):
    inputs, g1, g2, prior = load_inputs(root)
    if args.validate_base_assets_only:
        print(json.dumps(dict(status="PASS_LOCAL_FROZEN_OLD4035_AND_ACTUAL_NINJA_AUDIT_ONLY", old_G1_refs=2052,
            old_G2_refs=2088, old_CUDA13_refs=4035, new_ninja_ref=NINJA, final_lock_created=False,
            current_remote_bytes_rechecked=False, GPU_runs=0), indent=2)); return
    finals = {"run_g2_normal_model_lifecycle.py": args.run_sha, "g2_worker_observation.py": args.worker_sha,
              "G2_NORMAL_MODEL_CPU_PLAN.json": args.plan_sha}
    require(all(type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) for value in finals.values()), "approved final source SHAs required")
    output = root / LOCAL / "gpu-source-lock-candidate.json"; folder = root / LOCAL / "source-lock-upload-parts"
    receipt = root / LOCAL / "CPU_SOURCE_LOCK_PREPARATION_RESULT.json"
    require(not output.exists() and not folder.exists() and not receipt.exists(), "append-new output files only")
    rows = deepcopy(prior["files"]); index = {row["path"]: row for row in rows}; added, provenance = [], []

    def add(row, evidence, role):
        row = deepcopy(valid_ref(row))
        require(row["path"] not in index, "only five distinct new paths, never replace old ref")
        rows.append(row); index[row["path"]] = row; added.append(row)
        provenance.append(dict(ref=row, input_evidence_ref=evidence, role=role))

    for name, expected in finals.items():
        local = LOCAL + "/" + name; row = local_ref(root, local)
        require(row["sha256"] == expected, "independently frozen final source drift")
        inputs[local] = row; add(dict(row, path=DEST + "/" + name), row, "distinct final ninja normal source/plan")
    names, required, ast_sha = runner_required((root / LOCAL / "run_g2_normal_model_lifecycle.py").read_bytes())
    add(NINJA, inputs[NINJA_AUDIT_LOCAL], "prior actual CPU-read existing executable; current server recheck pending")
    add(dict(path=PRIOR_REMOTE, bytes=PRIOR_REF[0], sha256=PRIOR_REF[1]), inputs[PRIOR_LOCAL], "unchanged whole CUDA13 candidate lock itself; not authority")
    require(len(added) == 5 and len(rows) == len(index) == 4040 and rows[:4035] == prior["files"] and
            rows[:2088] == g2["files"] and rows[:2052] == g1["files"], "exact 4035+5 row inventory and all legacy ordering")
    require(required <= set(index), "actual final source checker required paths missing")
    for path, size, digest in names["SDK_SOURCE_REFS"]:
        require(index.get(path) == dict(path=path, bytes=size, sha256=digest), "original unchanged eight SDK source rows")
    for row in rows:
        limit = 4 * 1024**3 if row["path"].startswith(names["MODEL"] + "/") else 512 * 1024**2
        require(row["bytes"] <= limit, "actual checker source byte bounds")
    candidate = dict(schema_version=1, status="CPU_ONLY_G2_SOURCE_CANDIDATE",
        allow_gpu_initialization=False, allow_gpu_runs=False, GPU_collector_verified=False, production_qualified=False, effect_verified=False,
        baseline_source_lock=dict(path=names["BASELINE_LOCK"], bytes=G1_REF[0], sha256=G1_REF[1]),
        prior_CUDA13_source_candidate=dict(path=PRIOR_REMOTE, bytes=PRIOR_REF[0], sha256=PRIOR_REF[1]),
        source_binding=dict(old_G1_refs_preserved=2052, old_G2_refs_preserved=2088, old_CUDA13_refs_preserved=4035,
            old_rows_preserved_in_original_order=True, original_locks_modified=False, added_refs=5, total_refs=4040,
            complete_current_remote_hash_recheck_performed=False, ninja_ref_from_original_actual_CPU_launcher_audit=True,
            required_paths_from_final_verify_source_lock_AST=True, actual_verify_source_lock_ast_sha256=ast_sha),
        authority_boundary="CPU source inventory preparation only. No new human authorization, active scope, GPU permission, job retry, reservation or budget expansion is created. The unchanged single original ledger and future separate explicit bounded permission remain required.",
        files=rows)
    raw = (json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    require(len(raw) <= 4 * 1024**2 and len(rows) <= 8192, "actual checker JSON/file count bounds")
    lock_ref = dict(path=DEST + "/gpu-source-lock-candidate.json", bytes=len(raw), sha256=sha(raw)); parts = []
    for offset in range(0, len(raw), PART_LIMIT):
        data = raw[offset:offset + PART_LIMIT]; name = "part-%04d.bin" % (len(parts) + 1)
        parts.append((dict(path=name, offset=offset, bytes=len(data), sha256=sha(data)), data))
    require(b"".join(data for _, data in parts) == raw and all(0 < row["bytes"] <= PART_LIMIT and row["offset"] == number * PART_LIMIT
            for number, (row, _) in enumerate(parts)), "contiguous exact bounded raw slices")
    for local, expected in inputs.items(): require(local_ref(root, local) == expected, "local frozen input changed during preparation")
    forbidden = ("torch", "vllm", "py_kvcache", "flashinfer", "triton", "transformers", "numpy", "pynvml")
    require(not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in forbidden), "no framework/backend import")
    with output.open("xb") as handle: handle.write(raw)
    folder.mkdir()
    for row, data in parts:
        with (folder / row["path"]).open("xb") as handle: handle.write(data)
    manifest = dict(schema_version=1, status="CPU_ONLY_G2_SOURCE_CANDIDATE", whole_file=lock_ref,
        parts_directory=DEST + "/source-lock-upload-parts", encoding="raw byte slices; no newline/text conversion",
        maximum_part_bytes=PART_LIMIT, part_count=len(parts), parts=[row for row, _ in parts],
        reconstruction="Verify each manifest-ordered part bytes/SHA and contiguous offset; concatenate raw bytes and verify whole bytes/SHA before append-new publication.",
        allow_gpu_initialization=False, allow_gpu_runs=False, actual_GPU_runs=0, remote_upload_performed=False, original_locks_modified=False)
    manifest_path = folder / "manifest.json"
    with manifest_path.open("xb") as handle: handle.write((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    require(b"".join((folder / row["path"]).read_bytes() for row, _ in parts) == output.read_bytes() == raw, "actual disk parts reconstruct whole bytes")
    audit = dict(schema_version=1, status="PASS_LOCAL_CPU_SOURCE_CANDIDATE_PREPARATION_ONLY", lock_ref=lock_ref,
        manifest_ref=local_ref(root, manifest_path.relative_to(root).as_posix()),
        builder_ref=local_ref(root, Path(__file__).resolve().relative_to(root).as_posix()), original_frozen_input_refs=list(inputs.values()),
        exact_old_2052_G1_rows_and_order_preserved=True, exact_old_2088_G2_rows_and_order_preserved=True,
        exact_old_4035_CUDA13_rows_and_order_preserved=True, old_lock_bytes_SHA_rechecked=True,
        required_paths=sorted(required), required_paths_count=len(required), actual_verify_source_lock_ast_sha256=ast_sha,
        new_ref_count=5, total_refs=4040, new_refs=added, ref_provenance=provenance,
        local_part_count=len(parts), local_part_bytes_limit=PART_LIMIT, actual_disk_part_reconstruction_exact=True,
        current_remote_all_source_bytes_verified=False, framework_imports=0, actual_GPU_runs=0, remote_writes=0,
        human_record_created=False, GPU_authorization_created=False, permissions_modified=False, budget_ledger_modified=False,
        gpu_verified=False, production_qualified=False, effect_verified=False)
    with receipt.open("xb") as handle: handle.write((json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps(dict(lock_ref=lock_ref, old_G1_refs=2052, old_G2_refs=2088, old_CUDA13_refs=4035, added_refs=5, total_refs=4040,
        part_count=len(parts), part_max_bytes=max(row["bytes"] for row, _ in parts),
        manifest_ref=local_ref(root, manifest_path.relative_to(root).as_posix()),
        audit_ref=local_ref(root, receipt.relative_to(root).as_posix()), new_refs=added), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-sha"); parser.add_argument("--worker-sha"); parser.add_argument("--plan-sha")
    parser.add_argument("--validate-base-assets-only", action="store_true")
    parsed = parser.parse_args(); build(parsed.workspace.resolve(strict=True), parsed)
