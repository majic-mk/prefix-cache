"""Verify this small CPU delivery archive and independent reanalysis parity."""
import hashlib
import json
from pathlib import Path
import tarfile

BASE = Path(__file__).resolve().parent
REPLAY = BASE / "server_replay"
def require(ok, message):
    if not ok:
        raise RuntimeError(message)
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
receipt = read(REPLAY / "ARCHIVE_RECEIPT.json")
manifest_raw = (REPLAY / "DELIVERY_MANIFEST.json").read_bytes()
manifest = json.loads(manifest_raw)
archive = REPLAY / "FEASIBILITY_EVIDENCE.tar.gz"
require(archive.stat().st_size == receipt["archive_bytes"], "archive size")
require(sha(archive.read_bytes()) == receipt["archive_sha256"], "archive SHA")
require(sha(manifest_raw) == receipt["manifest_sha256"], "manifest SHA")
expected = {r["path"]: r for r in manifest["files"]}
require(len(expected) == manifest["file_count"] == receipt["data_file_count"], "unique manifest paths")
seen = set()
with tarfile.open(archive, "r|gz") as tar:
    for member in tar:
        require(member.isfile() and not member.issym() and not member.islnk(), "regular member only")
        require(member.name not in seen, "duplicate archive member")
        seen.add(member.name)
        raw = tar.extractfile(member).read()
        if member.name == "DELIVERY_MANIFEST.json":
            require(raw == manifest_raw, "embedded manifest")
            continue
        row = expected[member.name]
        require(len(raw) == row["bytes"] and sha(raw) == row["sha256"], "member: " + member.name)
        copy = REPLAY / member.name
        if copy.is_file():
            require(copy.read_bytes() == raw, "download matches archived member: " + member.name)
require(seen == set(expected) | {"DELIVERY_MANIFEST.json"}, "closed archive file set")
runtime = read(REPLAY / "CPU_RUNTIME_REANALYSIS.json")
local_runtime = read(BASE / "runtime_review/ANALYSIS.json")
for key in ("arms", "retry", "delta_decomposition", "v6_pairs", "profiles",
            "profile_comparison", "checks", "check_count"):
    require(runtime[key] == local_runtime[key], "independent runtime parity: " + key)
opportunity = read(REPLAY / "CPU_STRONG_U_REANALYSIS.json")
local_opportunity = read(BASE / "opportunity_review/STRONG_U_REANALYSIS_FINAL.json")
require(opportunity["strong_U_runs"] == local_opportunity["strong_U_runs"], "independent U parity")
before = read(REPLAY / "CPU_CONTRACT_BEFORE_V2.json")
after = read(REPLAY / "CPU_CONTRACT_AFTER.json")
require(before["gpu_ledger"] == after["gpu_ledger"], "GPU ledger unchanged")
require(before["reference_checks"] == after["reference_checks"], "nine frozen references unchanged")
decision = read(REPLAY / "FEASIBILITY_DECISION.json")
require(decision["new_GPU_seconds"] == 0 and not decision["decision"]["resume_paid_GPU_now"],
        "closed no-GPU decision")
state = read(REPLAY / "DELIVERY_FINAL_STATE.json")
require(state["GPU_seconds"] == 0 and state["gpu_device_nodes"] == [] and
        state["active_reservation"] is None and state["tracked_git_status"] == "",
        "final no-card state")
result = dict(status="PASS_ALL_DELIVERY_MEMBERS_AND_INDEPENDENT_REANALYSIS",
    archive_sha256=receipt["archive_sha256"], manifest_sha256=receipt["manifest_sha256"],
    data_files_sha_verified=len(expected), embedded_manifest_verified=True, extracted=False,
    independent_runtime_fields_equal=True, independent_strong_U_equal=True,
    original_GPU_ledger_unchanged=True, new_GPU_seconds=0)
with (REPLAY / "LOCAL_DELIVERY_VERIFICATION.json").open("x", encoding="utf-8") as f:
    json.dump(result, f, indent=2);f.write("\n")
print(json.dumps(result))
