"""Independent targeted replay of original retained failures, CPU stdlib only."""
from pathlib import Path
import sys, importlib.abc, json, hashlib
from dataclasses import replace
root = Path.cwd()
sys.path.insert(0, str(root / "third_party/work/prefix-io-p4-02-cpu/src"))
attempts = []
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "vllm", "py_kvcache", "numpy", "cupy", "cuda"):
            attempts.append(fullname)
            raise RuntimeError("CPU independent review forbids backend import")
sys.meta_path.insert(0, Guard())
from prefix_io_control.p4_production_table_contract import EvidenceRef, TableContext, TableContractError
from prefix_io_control.p4_verified_cost_loader import PreparedCostTable, load_semantically_verified_table
from prefix_io_control.p4_cost_table import CostTable

out = root / "artifacts/prefix_io_v1/server08-p4-02-cpu/eta/review-cost/final-counterexample-replay-03"
out.mkdir(exist_ok=False)
sources = [
    "third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py",
    "third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_verified_cost_loader.py",
]
expected = ["860876ae2d6c6a0bceee52b07c1681f5c0a11b71822f58b775e5e638406f8b3f",
            "23fddf7c159ff2bdcb98c000ad18f90c804183f0a5af9bf08be24ac3b54d368d"]
source_refs = [dict(path=p, sha256=hashlib.sha256((root/p).read_bytes()).hexdigest()) for p in sources]
assert [r["sha256"] for r in source_refs] == expected
fixture_parent = root / "artifacts/prefix_io_v1/server08-p4-02-cpu/eta/review-cost/counterexamples-01"
ledger = root / "experiments/prefix_io_v1/gpu-budget-ledger.json"
ledger_before = ledger.read_bytes()
assert json.loads(ledger_before)["active_reservation"] is None
results = []
def ref(directory, path):
    data = (directory/path).read_bytes()
    return EvidenceRef(path, len(data), hashlib.sha256(data).hexdigest())
def load(directory):
    candidate = json.loads((directory/"candidate.json").read_text())
    qualification = json.loads((directory/"qualification.json").read_text())
    return load_semantically_verified_table(
        directory, "candidate.json", expected_context=TableContext.from_mapping(candidate["context"]),
        qualification_ref=ref(directory, "qualification.json"),
        expected_verifier_ref=EvidenceRef.from_mapping(qualification["verifier_ref"]),
        plan_path="plan.json", expected_plan_ref=ref(directory, "plan.json"),
    )

prepared_native_label = None
for name in ("workload-hash-overlap", "cross-cell-split-overlap", "invalid-active-batch",
             "forged-native-origin", "abi-source-drift", "gpu-identity-drift"):
    directory = fixture_parent/name
    old_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
    try:
        table = load(directory)
        cell = table.verification.cells[0].cost
        row = dict(case=name, accepted=True, semantic_status=table.verification.status,
                   production_qualified=table.production_qualified, gpu_verified=table.gpu_verified,
                   production_lookup=table.lookup(cell.load_signature, cell.existing_io,
                                                 cell.stage, cell.physical_bytes),
                   total_ns=cell.total_ns)
        assert name == "forged-native-origin" and table.production_qualified is False
        assert row["production_lookup"] is None and table.gpu_verified is False
        prepared_native_label = table
    except TableContractError as exc:
        row = dict(case=name, accepted=False, error=str(exc))
        assert name != "forged-native-origin"
    assert old_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
    row["original_fixture_bytes_unchanged"] = True
    results.append(row)

class FakeLookup:
    def lookup(self, *args, **kwargs):
        return dict(production_qualified=True, total_ns=1, source="CPU_FABRICATED")
wrapper = []
for name, verification, table in (
    ("original_fake_wrapper", None, FakeLookup()),
    ("valid_verification_fake_lookup", prepared_native_label.verification, FakeLookup()),
    ("valid_verification_wrong_source",
     prepared_native_label.verification, CostTable(prepared_native_label.cpu_mock_table.cells, scope="mock_only", source_sha256="f"*64)),
    ("valid_verification_empty_cells",
     prepared_native_label.verification, CostTable((), scope="mock_only", source_sha256=prepared_native_label.cpu_mock_table.source_sha256)),
):
    try:
        PreparedCostTable(verification, table)
    except (TypeError, ValueError) as exc:
        wrapper.append(dict(case=name, rejected=True, error=str(exc)))
    else:
        raise AssertionError("public fake/mismatched wrapper accepted: " + name)

receipt = dict(status="PASS_INDEPENDENT_FINAL_COUNTEREXAMPLES_AND_WRAPPER", original_cases=results,
               wrapper_negative_cases=wrapper, source_refs=source_refs,
               forbidden_backend_import_attempts=attempts, gpu_workloads_run=0,
               production_capability_obtained=False,
               ledger_unchanged=ledger.read_bytes()==ledger_before,
               ledger_sha256=hashlib.sha256(ledger_before).hexdigest(),
               source_bytes_unchanged=all(hashlib.sha256((root/r["path"]).read_bytes()).hexdigest()==r["sha256"] for r in source_refs),
               limitations=[
                   "Consistent native_gpu_recording text verifies CPU semantics only, never actual GPU origin.",
                   "Margin validation supplies empirical residual envelope; not independent effect evaluation or confidence interval.",
                   "Existing driver still lacks complete native execution-window recorder; raw-to-analysis builder cannot supply missing observations.",
               ])
assert attempts == [] and receipt["ledger_unchanged"] and receipt["source_bytes_unchanged"]
(out/"result.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
