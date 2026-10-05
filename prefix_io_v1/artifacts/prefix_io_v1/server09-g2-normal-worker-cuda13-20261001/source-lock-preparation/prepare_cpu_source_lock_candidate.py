"""Append-new CUDA13 G2 source candidate and raw upload slices; stdlib only.

Original G1 and G2 rows/ordering are preserved. Actual server inventories are
consumed as prior byte evidence, never represented as a current server read.
This script creates no human authorization, active scope, or reservation.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

REMOTE_ROOT = "/root/autodl-tmp/prefix-io-v1-handoff/project"
LOCAL = "artifacts/prefix_io_v1_server09_candidates/g2_normal_worker_cuda13"
DEST = "artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-20261001"
G1_LOCAL = "artifacts/prefix_io_v1_server09_g1_20261001/gpu-source-lock.json"
G1_REMOTE = "artifacts/prefix_io_v1/server09-g1-20261001/gpu-source-lock.json"
G1_REF = (444671, "5b438ebc1ec014c22d1d501915600b4cd9bc777ad9f05da1f7ac9b1a18cc1181")
G2_LOCAL = "artifacts/prefix_io_v1_server09_candidates/g2_normal_worker/gpu-source-lock-candidate.json"
G2_REMOTE = "artifacts/prefix_io_v1/server09-g2-normal-worker-20261001/gpu-source-lock-candidate.json"
G2_REF = (452304, "5e7cc04fe759411800b895169d669f75cade6b2f0adda34d3992ecf2a7dbd760")
SDK_LOCAL = "artifacts/prefix_io_v1_server09_candidates/g2_cuda13_toolchain"
SDK_REMOTE = "artifacts/prefix_io_v1/server09-cuda13-toolchain-v2-20261001"
AUDIT_LOCAL = "artifacts/prefix_io_v1_server09_g2_20261001"
CPU_REMOTE = "artifacts/prefix_io_v1/server09-cuda13-cpu-20261001"
SDK_FILES = (
    (SDK_LOCAL + "/cuda13_sdk_overlay.py", SDK_REMOTE + "/cuda13_sdk_overlay.py", 16929,
     "ed846361788e9bdde853e6e0aa16e361329e3f2dffbfbe2c4f396acf9e9d1ecf"),
    (SDK_LOCAL + "/CUDA13_SDK_CPU_PLAN.json", SDK_REMOTE + "/CUDA13_SDK_CPU_PLAN.json", 10025,
     "8044db267791d5f217e3dd9c478375787aa73a74d53d19117f6bcebbc1f0a659"),
    (AUDIT_LOCAL + "/cuda13-cpu/CUDA13_SOURCE_INVENTORY.json", CPU_REMOTE + "/CUDA13_SOURCE_INVENTORY.json", 692329,
     "2ed079701c53d3e37005cf33432937e6c442f06cad16c5aded06efffcc7c36e0"),
    (AUDIT_LOCAL + "/cuda13-cpu/CPU_COMPILE_LINK_RESULT.json", CPU_REMOTE + "/CPU_COMPILE_LINK_RESULT.json", 5589,
     "3c3f9b092aea3ca89e07f31530a2ae3b38ab65cf0a8868821031f1d7f8d14e3f"),
)
SELECTION_FILES = (
    ("flashinfer/compilation_context.py", 3909, "e55c7f58a83810590e18686954527cc62735a6e91c3fa67286a5e7f00afc9be5"),
    ("flashinfer/jit/cpp_ext.py", 12445, "9d20f28baed969411220456b140a2786a081a2b7195d2433a70daec3c8ae7403"),
    ("flashinfer/jit/utils.py", 2644, "a32b738a91bafbe392c69e10fd11533ff98adbecbba9d769a19f2c5e920f5f31"),
    ("torch/version.py", 317, "323d35171ef1184f1d7db3bbd1f3d3e227e0e826be8fd52200778346e17c873f"),
)
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
    relative = row["path"]
    require(type(relative) is str and relative and not relative.startswith("/") and
            "\\" not in relative and ":" not in relative and
            all(part not in ("", ".", "..") for part in relative.split("/")), "relative POSIX source path")
    require(type(row["bytes"]) is int and row["bytes"] >= 0 and
            type(row["sha256"]) is str and re.fullmatch(r"[a-f0-9]{64}", row["sha256"]), "exact source bytes/SHA")
    return row


def local_ref(root, relative):
    path = root / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), "frozen local regular source")
    raw = path.read_bytes()
    return valid_ref(dict(path=relative, bytes=len(raw), sha256=sha(raw)))


def pinned(root, relative, expected):
    row = local_ref(root, relative)
    require((row["bytes"], row["sha256"]) == expected, "frozen input drift: " + relative)
    return row


def project_ref(absolute):
    require(type(absolute) is dict and set(absolute) == {"path", "bytes", "sha256"}, "exact original absolute file ref")
    path = absolute["path"]
    require(type(path) is str and path.startswith(REMOTE_ROOT + "/"), "ordinary source must be within actual project ROOT")
    return valid_ref(dict(absolute, path=path[len(REMOTE_ROOT) + 1:]))


def literal(node, names):
    if isinstance(node, ast.Constant):
        require(type(node.value) in (str, int, bool, type(None)), "bounded source literal")
        return node.value
    if isinstance(node, ast.Name):
        require(node.id in names, "unknown source constant: " + node.id)
        return names[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = literal(node.left, names), literal(node.right, names)
        require(type(left) is str and type(right) is str, "source path string addition only")
        return left + right
    if isinstance(node, (ast.Tuple, ast.Set, ast.List)):
        values = [literal(value, names) for value in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else set(values) if isinstance(node, ast.Set) else values
    raise ValueError("nonliteral source expression")


def runner_required(raw):
    tree = ast.parse(raw.decode("utf-8"))
    needed = {"DEST", "SCRIPT", "WORKER", "AUTHOR", "RUNNER", "CONTEXT", "BASELINE_LOCK", "BASELINE_SHA",
              "GUARD", "PREPARE", "QUALIFY", "SMOKE", "MODEL_PLAN", "MODEL", "PERMISSIONS", "SCALAR", "FRAME",
              "SDK_DEST", "SDK_MODULE", "SDK_PLAN", "SDK_INVENTORY", "SDK_PROOF", "SDK_SOURCE_REFS"}
    names = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in needed:
                names[node.targets[0].id] = literal(node.value, names)
    require(set(names) == needed and names["DEST"] == DEST and names["BASELINE_LOCK"] == G1_REMOTE and
            names["BASELINE_SHA"] == G1_REF[1], "new destination and unchanged old baseline constants")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_source_lock"]
    require(len(functions) == 1, "one actual source checker")
    assigns = [node for node in ast.walk(functions[0]) if isinstance(node, ast.Assign) and len(node.targets) == 1
               and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "required"]
    require(len(assigns) == 1, "one actual required source set")
    required = literal(assigns[0].value, dict(names, baseline_relative=G1_REMOTE))
    require(type(required) is set, "actual required paths set")
    updates = [node for node in ast.walk(functions[0]) if isinstance(node, ast.Call) and
               isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and
               node.func.value.id == "required" and node.func.attr == "update"]
    expected = ast.parse("required.update(path for path, _, _ in SDK_SOURCE_REFS)").body[0].value
    require(len(updates) == 1 and ast.dump(updates[0], include_attributes=False) ==
            ast.dump(expected, include_attributes=False), "exact SDK required-path generator only")
    require(type(names["SDK_SOURCE_REFS"]) is tuple and len(names["SDK_SOURCE_REFS"]) == 8,
            "eight frozen SDK and version-selection source refs")
    required.update(path for path, _, _ in names["SDK_SOURCE_REFS"])
    return names, required, sha(ast.dump(functions[0], include_attributes=False).encode())


def load_inputs(root):
    inputs = {}
    for relative, expected in ((G1_LOCAL, G1_REF), (G2_LOCAL, G2_REF)):
        inputs[relative] = pinned(root, relative, expected)
    _, g1 = read_json(root / G1_LOCAL)
    _, g2 = read_json(root / G2_LOCAL)
    require(type(g1.get("files")) is list and len(g1["files"]) == 2052 and
            type(g2.get("files")) is list and len(g2["files"]) == 2088 and
            g2["files"][:2052] == g1["files"], "all exact G1/G2 rows and original order")
    require(g2.get("status") == "CPU_ONLY_G2_SOURCE_CANDIDATE" and
            g2.get("allow_gpu_initialization") is False and g2.get("allow_gpu_runs") is False,
            "old G2 source candidate is not authority")
    rows = [deepcopy(valid_ref(row)) for row in g2["files"]]
    require(len({row["path"] for row in rows}) == 2088, "no prior source duplicate")
    sdk = []
    for local, remote, size, digest in SDK_FILES:
        actual = pinned(root, local, (size, digest)); inputs[local] = actual
        sdk.append((dict(actual, path=remote), actual, "frozen SDK implementation/plan or original CPU inventory/proof"))
    for relative, size, digest in SELECTION_FILES:
        remote = ".venv/lib/python3.12/site-packages/" + relative
        local = AUDIT_LOCAL + "/toolchain-audit-sources-exact/" + remote
        actual = pinned(root, local, (size, digest)); inputs[local] = actual
        sdk.append((dict(actual, path=remote), actual, "exact actual server compiler/architecture selection source mirror"))
    _, inventory = read_json(root / SDK_FILES[2][0])
    _, proof = read_json(root / SDK_FILES[3][0])
    require(inventory.get("status") == "CPU_ONLY_EXISTING_CUDA13_SOURCE_INVENTORY" and
            inventory.get("GPU_operations") == 0 and type(inventory.get("GPU_operations")) is int and
            inventory.get("framework_imports") == 0 and type(inventory.get("framework_imports")) is int and
            inventory.get("system_or_driver_modified") is False and inventory.get("downloads") is False,
            "actual original CPU-only toolkit inventory")
    toolkit = REMOTE_ROOT + "/.venv/lib/python3.12/site-packages/nvidia/cu13"
    require(inventory.get("toolkit_root") == toolkit and type(inventory.get("directories")) is dict and
            set(inventory["directories"]) == {"bin", "include", "nvvm"}, "exact existing toolkit tree roots")
    assets = []
    expected_counts = {"bin": 10, "include": 1922, "nvvm": 2}
    for key in ("bin", "include", "nvvm"):
        tree = inventory["directories"][key]
        require(type(tree) is dict and set(tree) == {"root", "files", "file_count", "total_bytes", "inventory_sha256"}
                and tree["root"] == toolkit + "/" + key and type(tree["files"]) is list and
                type(tree["file_count"]) is int and tree["file_count"] == len(tree["files"]) == expected_counts[key],
                "original exact toolkit tree schema/count")
        digest = sha(json.dumps(tree["files"], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii"))
        require(digest == tree["inventory_sha256"], "original inventory original-order digest")
        total = 0
        seen = set()
        for file in tree["files"]:
            require(type(file) is dict and set(file) == {"path", "bytes", "sha256", "relative"}, "original actual tree file schema")
            relative = file["relative"]
            valid_ref(dict(path=relative, bytes=file["bytes"], sha256=file["sha256"]))
            require(file["path"] == tree["root"] + "/" + relative and relative not in seen,
                    "unique original absolute/relative source binding")
            seen.add(relative); total += file["bytes"]
            assets.append(project_ref({k: file[k] for k in ("path", "bytes", "sha256")}))
        require(type(tree["total_bytes"]) is int and tree["total_bytes"] == total, "tree original byte accounting")
    require(len(assets) == 1934 and sum(row["bytes"] for row in assets) == 217111529, "all actual 1934 toolkit assets")
    cudart = project_ref(inventory["cudart"])
    require(cudart == dict(path=".venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13", bytes=704288,
            sha256="96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0"), "actual installed cudart13 ref")
    driver = inventory["driver"]
    require(driver == dict(path="/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05", bytes=91501576,
            sha256="76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6"), "actual external driver pinned indirectly only")
    require(proof.get("status") == "PASS_CPU_CUDA13_SM120F_COMPILE_AND_HOST_LINK_ONLY" and
            proof.get("manifest_ref") == dict(path=REMOTE_ROOT + "/" + SDK_FILES[2][1], bytes=SDK_FILES[2][2], sha256=SDK_FILES[2][3]) and
            proof.get("source_inventory_counts") == expected_counts and proof.get("source_inventory_bytes") == 217111529 and
            proof.get("driver_ref") == driver and proof.get("cudart_ref") == inventory["cudart"] and
            all(type(proof.get(k)) is int and proof[k] == 0 for k in ("GPU_operations", "GPU_kernels_executed", "framework_imports")) and
            all(proof.get(k) is False for k in ("shared_object_loaded", "system_or_driver_modified", "downloads", "stubs_on_runtime_library_path")),
            "original compiler proof is CPU-only exact inventory evidence")
    return inputs, g1, g2, sdk, assets, cudart, driver


def build(root, args):
    inputs, g1, g2, sdk, assets, cudart, driver = load_inputs(root)
    if args.validate_base_assets_only:
        print(json.dumps(dict(status="PASS_LOCAL_PINNED_OLD_LOCKS_AND_CUDA13_INVENTORY_ONLY", old_G1_refs=2052,
            old_G2_refs=2088, toolkit_asset_refs=len(assets), toolkit_asset_bytes=217111529, GPU_runs=0,
            current_remote_bytes_rechecked=False, final_lock_created=False), indent=2)); return
    finals = {"run_g2_normal_model_lifecycle.py": args.run_sha, "g2_worker_observation.py": args.worker_sha,
              "G2_NORMAL_MODEL_CPU_PLAN.json": args.plan_sha}
    require(all(type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) for value in finals.values()),
            "final independently approved source SHA arguments required")
    output = root / LOCAL / "gpu-source-lock-candidate.json"
    folder = root / LOCAL / "source-lock-upload-parts"
    audit_path = root / LOCAL / "CPU_SOURCE_LOCK_PREPARATION_RESULT.json"
    require(not output.exists() and not folder.exists() and not audit_path.exists(), "append-new outputs only")
    rows = deepcopy(g2["files"]); index = {row["path"]: row for row in rows}
    added, repeated, provenance = [], [], []

    def add(row, evidence, role):
        row = deepcopy(valid_ref(row))
        if row["path"] in index:
            require(index[row["path"]] == row, "old/new source path conflict: " + row["path"]); repeated.append(row)
        else:
            rows.append(row); index[row["path"]] = row; added.append(row)
        provenance.append(dict(ref=row, source_evidence_ref=evidence, role=role))

    for name, approved_sha in finals.items():
        local = LOCAL + "/" + name; actual = local_ref(root, local)
        require(actual["sha256"] == approved_sha, "independently frozen normal source drift")
        inputs[local] = actual
        add(dict(actual, path=DEST + "/" + name), actual, "final distinct CUDA13 normal runner/worker/plan")
    raw_runner = (root / LOCAL / "run_g2_normal_model_lifecycle.py").read_bytes()
    names, required, checker_ast_sha = runner_required(raw_runner)
    add(dict(path=G2_REMOTE, bytes=G2_REF[0], sha256=G2_REF[1]), inputs[G2_LOCAL], "unchanged prior G2 source candidate itself; not authority")
    for row, evidence, role in sdk: add(row, evidence, role)
    expected_sdk = tuple((row["path"], row["bytes"], row["sha256"]) for row, _, _ in sdk)
    require(names["SDK_SOURCE_REFS"] == expected_sdk, "actual runner's exact eight SDK source rows")
    inventory_input = inputs[SDK_FILES[2][0]]
    for row in assets: add(row, inventory_input, "prior actual CPU stream in full existing toolkit inventory; current server recheck pending")
    add(cudart, inventory_input, "prior actual existing cudart13 bytes; no new library install")
    require(rows[:2088] == g2["files"] and rows[:2052] == g1["files"] and len(index) == len(rows), "all old ordered rows untouched")
    require(required <= set(index), "actual final source checker required paths absent")
    require(index[G1_REMOTE] == dict(path=G1_REMOTE, bytes=G1_REF[0], sha256=G1_REF[1]), "unchanged G1 baseline lock itself")
    for row in rows:
        limit = 4 * 1024**3 if row["path"].startswith(names["MODEL"] + "/") else 512 * 1024**2
        require(row["bytes"] <= limit, "actual runner file bounds")
    candidate = dict(schema_version=1, status="CPU_ONLY_G2_SOURCE_CANDIDATE",
        allow_gpu_initialization=False, allow_gpu_runs=False, GPU_collector_verified=False,
        production_qualified=False, effect_verified=False,
        baseline_source_lock=dict(path=G1_REMOTE, bytes=G1_REF[0], sha256=G1_REF[1]),
        prior_G2_source_candidate=dict(path=G2_REMOTE, bytes=G2_REF[0], sha256=G2_REF[1]),
        source_binding=dict(old_G1_refs_preserved=2052, old_G2_refs_preserved=2088,
            original_G1_lock_modified=False, original_G2_lock_modified=False, added_refs=len(added), total_refs=len(rows),
            complete_current_remote_hash_recheck_performed=False, prior_actual_toolkit_inventory_refs=1934,
            external_driver_ref_bound_via_original_inventory_and_helper=True,
            required_paths_from_final_verify_source_lock_AST=True, actual_verify_source_lock_ast_sha256=checker_ast_sha),
        authority_boundary="CPU inventory preparation only. No new human record, active scope, GPU permission, reservation, retry or experiment is authorized. Original system driver is pinned inside the byte-locked original manifest and verified by the locked SDK helper, not added as a project-relative row.",
        files=rows)
    raw = (json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    require(len(raw) <= 4 * 1024**2 and len(rows) <= 8192, "actual runner JSON/ref count bounds")
    lock_ref = dict(path=DEST + "/gpu-source-lock-candidate.json", bytes=len(raw), sha256=sha(raw))
    parts = []
    for offset in range(0, len(raw), PART_LIMIT):
        data = raw[offset:offset + PART_LIMIT]; name = "part-%04d.bin" % (len(parts) + 1)
        parts.append((dict(path=name, offset=offset, bytes=len(data), sha256=sha(data)), data))
    require(b"".join(data for _, data in parts) == raw, "exact raw reconstruction before publish")
    for relative, expected in inputs.items(): require(local_ref(root, relative) == expected, "local source input changed during preparation")
    forbidden = ("torch", "vllm", "py_kvcache", "flashinfer", "triton", "transformers", "numpy", "pynvml")
    require(not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in forbidden), "no backend import")
    with output.open("xb") as handle: handle.write(raw)
    folder.mkdir()
    for row, data in parts:
        with (folder / row["path"]).open("xb") as handle: handle.write(data)
    manifest = dict(schema_version=1, status="CPU_ONLY_G2_SOURCE_CANDIDATE", whole_file=lock_ref,
        parts_directory=DEST + "/source-lock-upload-parts", encoding="raw byte slices; no text or newline conversion",
        maximum_part_bytes=PART_LIMIT, part_count=len(parts), parts=[row for row, _ in parts],
        reconstruction="Verify each manifest-ordered part's exact bytes/SHA and contiguous offset, concatenate raw bytes, verify whole bytes/SHA before append-new publication.",
        allow_gpu_initialization=False, allow_gpu_runs=False, actual_GPU_runs=0, remote_upload_performed=False,
        original_G1_lock_modified=False, original_G2_lock_modified=False)
    manifest_path = folder / "manifest.json"
    with manifest_path.open("xb") as handle: handle.write((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
    rebuilt = b"".join((folder / row["path"]).read_bytes() for row, _ in parts)
    require(rebuilt == output.read_bytes() == raw and sha(rebuilt) == lock_ref["sha256"], "actual disk raw parts reconstruct lock")
    audit = dict(schema_version=1, status="PASS_LOCAL_CPU_SOURCE_CANDIDATE_PREPARATION_ONLY", lock_ref=lock_ref,
        manifest_ref=local_ref(root, manifest_path.relative_to(root).as_posix()), builder_ref=local_ref(root, Path(__file__).resolve().relative_to(root).as_posix()),
        baseline_G1_input_ref=inputs[G1_LOCAL], prior_G2_input_ref=inputs[G2_LOCAL],
        final_normal_input_refs=[inputs[LOCAL + "/" + name] for name in finals], original_frozen_input_refs=list(inputs.values()),
        exact_old_2052_G1_rows_and_order_preserved=True, exact_old_2088_G2_rows_and_order_preserved=True,
        old_lock_bytes_SHA_rechecked=True, required_paths=sorted(required), required_paths_count=len(required),
        actual_verify_source_lock_ast_sha256=checker_ast_sha, new_ref_count=len(added), new_refs=added,
        already_present_equal_refs=repeated, ref_provenance=provenance, actual_toolkit_inventory_file_count=1934,
        actual_toolkit_inventory_tree_bytes=217111529, actual_toolkit_inventory_ref=inventory_input,
        external_driver_ref_not_in_ordinary_project_rows=driver, external_driver_verified_by_future_locked_helper=True,
        local_part_count=len(parts), local_part_bytes_limit=PART_LIMIT, actual_disk_part_reconstruction_exact=True,
        current_remote_all_source_bytes_verified=False, framework_imports=0, actual_GPU_runs=0, remote_writes=0,
        human_record_created=False, GPU_authorization_created=False, permissions_modified=False,
        budget_ledger_modified=False, gpu_verified=False, production_qualified=False, effect_verified=False)
    with audit_path.open("xb") as handle: handle.write((json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps(dict(lock_ref=lock_ref, old_G1_refs=2052, old_G2_refs=2088, added_refs=len(added), total_refs=len(rows),
        part_count=len(parts), part_max_bytes=max(row["bytes"] for row, _ in parts),
        manifest_ref=local_ref(root, manifest_path.relative_to(root).as_posix()),
        audit_ref=local_ref(root, audit_path.relative_to(root).as_posix()),
        new_ref_roles={"distinct_normal_source_and_plan": 3, "prior_G2_lock_itself": 1,
                       "SDK_and_selection_source_rows": 8, "toolkit_tree_assets": 1934, "cudart13": 1}), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-sha")
    parser.add_argument("--worker-sha")
    parser.add_argument("--plan-sha")
    parser.add_argument("--validate-base-assets-only", action="store_true")
    parsed = parser.parse_args()
    build(parsed.workspace.resolve(strict=True), parsed)
