"""Strict CPU preflight for P4 production-table candidates.

A bound candidate is NOT a qualified CostTable. Byte identity and a reported
GPU status do not establish real measurements. The independent paired GPU
verifier and native interference-quota application are still required. This
module never imports a backend or calls/executes a referenced verifier.
"""
from dataclasses import dataclass, fields, asdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from .dispatch_budget import Amount, STAGES, integer
from .p4_cost_table import CostCell, BASES

MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_EVIDENCE_BYTES = 32 * 1024 * 1024
MAX_CELLS = 128
REQUIRED_MEASUREMENT_ROLES = frozenset((
    "baseline_wrapper", "action_wrapper", "baseline_observations",
    "action_observations", "pair_analysis",
))
UNCERTAINTY_METHODS = frozenset(("paired_interval_margin", "paired_residual_margin"))


class TableContractError(ValueError):
    pass


def _require(test, message):
    if not test:
        raise TableContractError(message)


def _keys(obj, names, label):
    _require(type(obj) is dict and set(obj) == set(names), label + ": exact schema keys required")


def _text(value, label):
    _require(type(value) is str and bool(value.strip()) and len(value) <= 128,
             label + ": bounded nonempty text required")


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, label + ": integer required")


def _sha(value, label):
    _require(type(value) is str and len(value) == 64 and
             all(c in "0123456789abcdef" for c in value), label + ": SHA-256 required")


def _path(root, value):
    _require(type(value) is str and 0 < len(value) <= 512 and "\\" not in value
             and "\x00" not in value, "bounded relative POSIX path required")
    pure = PurePosixPath(value)
    _require(len(pure.parts) > 0 and not pure.is_absolute() and ":" not in pure.parts[0] and
             all(p not in ("", ".", "..") for p in value.split("/")),
             "absolute/traversal/ambiguous path rejected")
    current = root
    for part in pure.parts:
        current = current / part
        _require(not current.is_symlink(), "symlink evidence is forbidden")
    try:
        resolved = current.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TableContractError("missing or invalid evidence path") from exc
    _require(root in resolved.parents and resolved.is_file(), "evidence must be a project file")
    return resolved


def _hash_file(path):
    size = path.stat().st_size
    _require(size <= MAX_EVIDENCE_BYTES, "metadata evidence exceeds bounded CPU read")
    digest = sha256()
    actual_size = 0
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(65536), b""):
            actual_size += len(data)
            _require(actual_size <= MAX_EVIDENCE_BYTES, "metadata evidence grew beyond bounded CPU read")
            digest.update(data)
    _require(actual_size == size, "evidence changed during bounded read")
    return actual_size, digest.hexdigest()


@dataclass(frozen=True)
class EvidenceRef:
    path: str
    bytes: int
    sha256: str

    def __post_init__(self):
        _require(type(self.path) is str, "evidence path must be text")
        _integer(self.bytes, "evidence bytes")
        _sha(self.sha256, "evidence digest")

    @classmethod
    def from_mapping(cls, obj):
        _keys(obj, ("path", "bytes", "sha256"), "evidence reference")
        return cls(**obj)

    def verify(self, root):
        path = _path(root, self.path)
        size, digest = _hash_file(path)
        _require((size, digest) == (self.bytes, self.sha256), "changed evidence size/SHA")
        return path


@dataclass(frozen=True)
class TableContext:
    run_id: str
    model_sha256: str
    gpu_uuid: str
    native_source_sha256: str
    vllm_source_sha256: str
    kv_layout_sha256: str
    kernel_mode: str
    torch_version: str
    cuda_version: str
    driver_version: str
    transfer_quantum_bytes: int
    gpu_pool_bytes: int
    stage_capacity_bytes: int
    internal_step_budget_ns: int
    cost_basis: str
    uncertainty_method: str

    def __post_init__(self):
        for name in ("run_id", "gpu_uuid", "kernel_mode", "torch_version",
                     "cuda_version", "driver_version"):
            _text(getattr(self, name), name)
        _require(self.gpu_uuid.startswith("GPU-") and len(self.gpu_uuid) > 4,
                 "explicit physical GPU identity required")
        for name in ("model_sha256", "native_source_sha256", "vllm_source_sha256",
                     "kv_layout_sha256"):
            _sha(getattr(self, name), name)
        for name in ("transfer_quantum_bytes", "gpu_pool_bytes", "stage_capacity_bytes",
                     "internal_step_budget_ns"):
            _integer(getattr(self, name), name, 1)
        _require(type(self.cost_basis) is str and self.cost_basis in BASES, "one fixed cost-accounting basis required")
        _require(type(self.uncertainty_method) is str and self.uncertainty_method in UNCERTAINTY_METHODS, "explicit uncertainty method required")

    @classmethod
    def from_mapping(cls, obj):
        _keys(obj, tuple(f.name for f in fields(cls)), "table context")
        return cls(**obj)


@dataclass(frozen=True)
class BoundCell:
    cell_id: str
    cost: CostCell
    measurement_refs: tuple
    paired_runs_reported: int
    windows_reported: int


@dataclass(frozen=True)
class BlockedProductionCandidate:
    context: TableContext
    cells: tuple[BoundCell, ...]
    candidate_ref: EvidenceRef
    qualification_ref: EvidenceRef
    expected_verifier_ref: EvidenceRef
    bound_evidence_count: int
    metadata_binding_complete: bool = True

    @property
    def production_qualified(self):
        return False

    @property
    def gpu_verified(self):
        return False

    @property
    def status(self):
        return "BLOCKED_PRODUCTION_GPU_QUALIFICATION"

    @property
    def missing_requirements(self):
        return ("independent_real_paired_gpu_verifier",
                "native_interference_quota_application_and_gpu_qualification")


def _duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key rejected")
        result[key] = value
    return result


def _read_json(path, expected_ref=None):
    _require(path.stat().st_size <= MAX_JSON_BYTES, "bounded JSON metadata required")
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    _require(len(raw) <= MAX_JSON_BYTES, "JSON metadata grew beyond bounded CPU read")
    size, digest = len(raw), sha256(raw).hexdigest()
    if expected_ref is not None:
        _require((size, digest) == (expected_ref.bytes, expected_ref.sha256),
                 "JSON changed between reference verification and parse")
    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_duplicate_keys,
                            parse_constant=lambda value: (_ for _ in ()).throw(
                                TableContractError("nonfinite JSON number rejected")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TableContractError("strict UTF-8 JSON required") from exc
    return parsed, size, digest


def _cell(obj, context, root):
    _keys(obj, ("cell_id", "load", "existing_io", "stage", "physical_bytes",
                "baseline_ns", "incremental_or_joint_ns", "uncertainty_ns",
                "paired_runs_reported", "windows_reported", "measurement_refs"), "cost cell")
    _text(obj["cell_id"], "cell id")
    _keys(obj["load"], ("active_decode", "batch", "prefill_tokens", "context_length"), "load state")
    for name, value in obj["load"].items():
        _integer(value, name, 1 if name in ("batch", "context_length") else 0)
    existing = obj["existing_io"]
    _require(type(existing) is list and len(existing) == 4, "exact four-stage existing I/O required")
    amounts = []
    for value in existing:
        _keys(value, ("ops", "bytes"), "existing I/O")
        _integer(value["ops"], "existing I/O operations")
        _integer(value["bytes"], "existing I/O bytes")
        amounts.append(Amount(value["ops"], value["bytes"]))
    _require(obj["stage"] in STAGES, "supported physical stage required")
    _integer(obj["physical_bytes"], "physical bytes", 1)
    _require(obj["physical_bytes"] % context.transfer_quantum_bytes == 0,
             "physical bytes must match frozen legal transfer quantum")
    for name in ("baseline_ns", "incremental_or_joint_ns", "uncertainty_ns"):
        _integer(obj[name], name, 1 if name == "baseline_ns" else 0)
    _integer(obj["paired_runs_reported"], "reported paired runs", 1)
    _integer(obj["windows_reported"], "reported windows", 1)
    measurements = obj["measurement_refs"]
    _require(type(measurements) is dict and set(measurements) == REQUIRED_MEASUREMENT_ROLES,
             "complete separate baseline/action observations and pair-analysis bindings required")
    refs = []
    for role in sorted(measurements):
        ref = EvidenceRef.from_mapping(measurements[role])
        path = ref.verify(root)
        _require(path.suffix in (".json", ".jsonl"), "measurement metadata JSON/JSONL required")
        refs.append((role, ref))
    _require(measurements["baseline_wrapper"] != measurements["action_wrapper"] and
             measurements["baseline_observations"] != measurements["action_observations"],
             "a paired control/action cannot be the same evidence artifact")
    load = obj["load"]
    exact_signature = (context.model_sha256, context.gpu_uuid, context.kv_layout_sha256,
        context.kernel_mode, load["active_decode"], load["batch"], load["prefill_tokens"],
        load["context_length"], context.transfer_quantum_bytes)
    cost = CostCell(exact_signature, tuple(amounts), obj["stage"], obj["physical_bytes"],
        context.cost_basis, obj["baseline_ns"], obj["incremental_or_joint_ns"], obj["uncertainty_ns"])
    return BoundCell(obj["cell_id"], cost, tuple(refs),
                     obj["paired_runs_reported"], obj["windows_reported"])


def load_production_candidate(root, candidate_path, *, expected_context,
                              qualification_ref, expected_verifier_ref):
    """Strict artifact binding only; even valid-looking fake reports stay blocked.

    expected_context/verifier identity come from the independently frozen run
    plan, not from candidate declarations. No referenced script is executed.
    Receipt booleans, reported PASS, hashes and counts cannot open this gate.
    """
    root = Path(root).resolve(strict=True)
    _require(type(expected_context) is TableContext, "independently frozen exact context required")
    _require(type(qualification_ref) is EvidenceRef and type(expected_verifier_ref) is EvidenceRef,
             "explicit independently pinned qualification/verifier refs required")
    path = _path(root, candidate_path)
    _require(path.suffix == ".json", "candidate JSON required")
    data, size, digest = _read_json(path)
    _keys(data, ("schema_version", "scope", "context", "cells"), "production candidate")
    _require(type(data["schema_version"]) is int and data["schema_version"] == 1,
             "candidate schema version must be exact integer 1")
    _require(data["scope"] == "production_candidate", "mock/P3 conditional scope cannot qualify")
    context = TableContext.from_mapping(data["context"])
    _require(context == expected_context, "model/hardware/layout/kernel/source/context mismatch")
    raw_cells = data["cells"]
    _require(type(raw_cells) is list and 1 <= len(raw_cells) <= MAX_CELLS, "bounded nonempty cells required")
    cells = tuple(_cell(obj, context, root) for obj in raw_cells)
    _require(len({c.cell_id for c in cells}) == len(cells), "duplicate cell id")
    _require(len({c.cost.key for c in cells}) == len(cells), "ambiguous exact cost-state/action cell")
    candidate_ref = EvidenceRef(candidate_path, size, digest)
    verifier_path = expected_verifier_ref.verify(root)
    _require(verifier_path.suffix == ".py", "independent verifier must be a pinned Python source artifact")
    qualification_path = qualification_ref.verify(root)
    _require(qualification_path.suffix == ".json", "qualification binding report JSON required")
    report, _, _ = _read_json(qualification_path, qualification_ref)
    _keys(report, ("schema_version", "scope", "status", "candidate_ref", "context",
                   "verifier_ref", "measurement_refs"), "qualification binding report")
    _require(type(report["schema_version"]) is int and report["schema_version"] == 1,
             "qualification schema version must be exact integer 1")
    _require(report["scope"] == "production_qualification_candidate" and
             report["status"] in ("PENDING_INDEPENDENT_VERIFICATION", "PASS_REPORTED"),
             "P3/CPU/report-self-qualification is not an independent P4 verdict")
    _require(EvidenceRef.from_mapping(report["candidate_ref"]) == candidate_ref,
             "qualification report binds a different candidate")
    _require(TableContext.from_mapping(report["context"]) == context, "qualification context mismatch")
    _require(EvidenceRef.from_mapping(report["verifier_ref"]) == expected_verifier_ref,
             "untrusted/changed independent verifier chain")
    refs = report["measurement_refs"]
    _require(type(refs) is list and len(refs) <= MAX_CELLS * 5, "bounded qualification evidence refs")
    declared = tuple(EvidenceRef.from_mapping(ref) for ref in refs)
    _require(len(set(declared)) == len(declared), "duplicate qualification evidence")
    actual = frozenset(ref for cell in cells for _, ref in cell.measurement_refs)
    _require(frozenset(declared) == actual, "incomplete or extra measurement chain")
    for ref in declared:
        ref.verify(root)
    # The returned cell values and source digest must describe the same bytes,
    # even if a candidate changes while its referenced evidence is being read.
    candidate_ref.verify(root)
    qualification_ref.verify(root)
    expected_verifier_ref.verify(root)
    return BlockedProductionCandidate(context, cells, candidate_ref, qualification_ref,
                                      expected_verifier_ref, len(actual) + 3)
