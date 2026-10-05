"""Exact finite interference-cost interface, with a closed production gate.

P3 conditional/native-owned windows are not a production interference table.
CPU mock cells exercise the arithmetic only. No constructor flag can turn them
into GPU-qualified evidence; the P4 paired GPU qualification remains required.
"""
from dataclasses import dataclass
from hashlib import sha256
import inspect
import json
from pathlib import Path
import uuid
from .dispatch_budget import Amount, STAGES, ZERO, integer, vector

BASES = frozenset(("existing_io_plus_delta", "no_io_plus_joint"))


@dataclass(frozen=True)
class DevelopmentCostInput:
    """One observed A/B cell; no production, holdout, or tail qualification."""
    gpu_uuid: str
    common_runtime_domain_sha256: str
    source_lock_sha256: str
    model_sha256: str
    kv_layout_sha256: str
    kernel_mode: str
    native_source_sha256: str
    collector_source_sha256: str
    cuda_event_source_sha256: str
    runtime_refs: tuple
    cells: tuple
    budget_ns: int
    a_ns: int
    b_ns: int


def _development_ref(root, refs, row):
    """Check the actual independently frozen bytes, without imports or GPU."""
    if type(row) is not dict or set(row) != {"path", "bytes", "sha256"}:
        raise ValueError("strict development source reference required")
    name = row["path"]
    if (type(name) is not str or not name or "\\" in name or ":" in name or
            any(part in ("", ".", "..") for part in name.split("/")) or
            refs.get(name) != row):
        raise ValueError("actual development reference absent from current lock")
    integer(row["bytes"], "actual development bytes", 1)
    digest = row["sha256"]
    if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("actual development reference SHA-256 required")
    path = Path(root).resolve(strict=True)
    for part in name.split("/"):
        path /= part
        if path.is_symlink():
            raise ValueError("development evidence symlink refused")
    if not path.is_file() or path.stat().st_size != row["bytes"]:
        raise ValueError("actual development file missing or size changed")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError("actual development source byte drift")
    return raw


def load_gpu_development_table(root, refs, pair_binding_ref, *, driver,
                               expected_gpu_uuid, expected_common_runtime_domain_sha256):
    """Reuse a source-pinned pure raw-pair validator, then create one cell.

    The validator must replay completed original guards, raw outputs/CUDA frames
    and native I/O. A document flag is never sufficient. Its module is a leaf
    in the same current closure and its result must equal the pinned material.
    No production issuer, SLO, statistical upper bound or CPU mock is used.
    """
    if (type(expected_gpu_uuid) is not str or not expected_gpu_uuid.startswith("GPU-") or
            expected_gpu_uuid != "GPU-" + str(uuid.UUID(expected_gpu_uuid.removeprefix("GPU-"))) or
            type(expected_common_runtime_domain_sha256) is not str or len(expected_common_runtime_domain_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in expected_common_runtime_domain_sha256)):
        raise ValueError("actual canonical GPU UUID and frozen common domain required")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate development input JSON key")
            result[key] = value
        return result
    raw = _development_ref(root, refs, pair_binding_ref)
    document = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite development input")))
    if type(document) is not dict or set(document) != {"schema", "material", "evidence_refs"} or document["schema"] != "bounded_I_actual_AB_cost_input_v1":
        raise ValueError("one actual bounded I A/B cost input required")
    validate = getattr(driver, "validate_actual_development_pair", None)
    if not callable(validate):
        raise ValueError("actual completed raw A/B validator unavailable")
    source = inspect.getsourcefile(validate)
    if source is None:
        raise ValueError("actual raw-pair validator source unavailable")
    relative = Path(source).resolve(strict=True).relative_to(Path(root).resolve(strict=True)).as_posix()
    validator_ref = refs.get(relative)
    _development_ref(root, refs, validator_ref)
    evidence = document["evidence_refs"]
    if type(evidence) is not list or not evidence or len(evidence) > 64:
        raise ValueError("actual bounded A/B evidence references required")
    for row in evidence:
        _development_ref(root, refs, row)
    if len({row["path"] for row in evidence}) != len(evidence):
        raise ValueError("duplicate A/B evidence reference")
    material = validate(root, refs, pair_binding_ref,
        expected_gpu_uuid=expected_gpu_uuid,
        expected_common_runtime_domain_sha256=expected_common_runtime_domain_sha256)
    if material != document["material"]:
        raise ValueError("actual A/B replay differs from pinned cost input")
    required = {"gpu_uuid", "common_runtime_domain_sha256", "source_lock_ref", "model_manifest_ref",
                "kv_layout_ref", "native_source_ref", "collector_source_ref", "cuda_event_source_ref",
                "kernel_mode", "context_length", "a_ns", "b_ns", "budget_ns", "stage", "physical_bytes",
                "operations", "existing_io", "batch", "active_decode", "prefill_tokens"}
    if type(material) is not dict or set(material) != required:
        raise ValueError("exact replayed development material required")
    if (material["gpu_uuid"] != expected_gpu_uuid or
            material["common_runtime_domain_sha256"] != expected_common_runtime_domain_sha256 or
            material["kernel_mode"] != "eager" or material["stage"] != "ssd_read" or
            material["physical_bytes"] != 917504 or type(material["physical_bytes"]) is not int or
            material["operations"] != 1 or type(material["operations"]) is not int or
            [material["batch"], material["active_decode"], material["prefill_tokens"]] != [1, 1, 0] or
            any(type(material[k]) is not int for k in ("batch", "active_decode", "prefill_tokens")) or
            material["existing_io"] != [{"ops": 0, "bytes": 0}] * 4 or
            any(type(v) is not int for row in material["existing_io"] for v in row.values())):
        raise ValueError("current GPU/common domain single decode SSD unit and ZERO I/O required")
    for name in ("context_length", "a_ns", "b_ns", "budget_ns"):
        integer(material[name], "actual positive " + name, 1)
    if material["budget_ns"] != material["a_ns"]:
        raise ValueError("only explicit observed A engineering threshold; no SLO/upper claim")
    source_rows = [material[name] for name in ("source_lock_ref", "model_manifest_ref", "kv_layout_ref",
                   "native_source_ref", "collector_source_ref", "cuda_event_source_ref")]
    for row in source_rows:
        _development_ref(root, refs, row)
    key = (material["gpu_uuid"], material["common_runtime_domain_sha256"], material["model_manifest_ref"]["sha256"],
           material["kv_layout_ref"]["sha256"], 1, 1, 0, material["context_length"], 917504)
    cell = CostCell(key, ZERO, "ssd_read", 917504, "existing_io_plus_delta", material["a_ns"],
                    max(0, material["b_ns"] - material["a_ns"]), 0)
    rows = {row["path"]: row for row in source_rows + evidence + [pair_binding_ref, validator_ref]}
    for row in rows.values():
        _development_ref(root, refs, row)
    if _development_ref(root, refs, pair_binding_ref) != raw:
        raise ValueError("actual cost input changed during replay")
    identity = DevelopmentCostInput(material["gpu_uuid"], material["common_runtime_domain_sha256"],
        material["source_lock_ref"]["sha256"], material["model_manifest_ref"]["sha256"], material["kv_layout_ref"]["sha256"],
        "eager", material["native_source_ref"]["sha256"], material["collector_source_ref"]["sha256"],
        material["cuda_event_source_ref"]["sha256"], tuple((r["path"], r["bytes"], r["sha256"]) for r in rows.values()),
        (key,), material["budget_ns"], material["a_ns"], material["b_ns"])
    table = CostTable((cell,), scope="conditional", source_sha256=pair_binding_ref["sha256"])
    object.__setattr__(table, "scope", "gpu_development")
    object.__setattr__(table, "_development_input", identity)
    return table


def signature(values):
    if type(values) is not tuple or not 1 <= len(values) <= 16:
        raise ValueError("finite exact model/load/hardware signature required")
    for value in values:
        if type(value) not in (str, int) or (type(value) is str and
                (not value or len(value) > 128)):
            raise ValueError("signature contains bounded scalar identities only")


@dataclass(frozen=True)
class CostCell:
    load_signature: tuple
    existing_io: tuple[Amount, ...]
    stage: str
    physical_bytes: int
    basis: str
    baseline_ns: int
    incremental_or_joint_ns: int
    uncertainty_ns: int

    def __post_init__(self):
        signature(self.load_signature)
        vector(self.existing_io)
        if self.stage not in STAGES or self.basis not in BASES:
            raise ValueError("explicit physical stage and unique cost basis required")
        integer(self.physical_bytes, "physical bytes", 1)
        for name in ("baseline_ns", "incremental_or_joint_ns", "uncertainty_ns"):
            integer(getattr(self, name), name)

    @property
    def key(self):
        return (self.load_signature, self.existing_io, self.stage, self.physical_bytes)

    @property
    def total_ns(self):
        # existing I/O is either already in T0, or already in the joint cost.
        # There is intentionally no extra existing-I/O additive term.
        return self.baseline_ns + self.incremental_or_joint_ns + self.uncertainty_ns


@dataclass(frozen=True)
class CostEstimate:
    total_ns: int
    basis: str
    baseline_ns: int
    incremental_or_joint_ns: int
    uncertainty_ns: int
    source_sha256: str
    mock_only: bool = True
    production_qualified: bool = False


class CostTable:
    """Immutable cells; unavailable means None rather than zero.

    Current P4 CPU delivery deliberately accepts only mock_only/conditional
    scopes. A future production loader needs actual paired GPU validation and a
    frozen compatible source/geometry identity. It must not be enabled by a
    synthetic boolean or a P3 conditional receipt.
    """
    __slots__ = ("cells", "scope", "source_sha256", "_gpu_proof", "_development_input")

    @property
    def development_input(self):
        value = self._development_input
        if self.scope != "gpu_development" or type(value) is not DevelopmentCostInput:
            return None
        return value

    @property
    def production_qualified(self):
        if self.scope != "gpu_verified_exact_cells":
            return False
        from .gpu_cell_issuer import _table_proof_valid
        return _table_proof_valid(self._gpu_proof, self.cells, self.source_sha256)

    def __setattr__(self, name, value):
        raise AttributeError("cost tables are immutable")

    def __init__(self, cells=(), *, scope="conditional", source_sha256):
        if scope not in ("mock_only", "conditional"):
            raise ValueError("P4 production GPU qualification is unavailable")
        if type(source_sha256) is not str or len(source_sha256) != 64 or any(
                c not in "0123456789abcdef" for c in source_sha256):
            raise ValueError("actual immutable source SHA-256 required")
        if type(cells) is not tuple or len(cells) > 128 or any(type(c) is not CostCell for c in cells):
            raise ValueError("bounded immutable cost cells required")
        if len({c.key for c in cells}) != len(cells):
            raise ValueError("ambiguous matching cells")
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "_gpu_proof", None)
        object.__setattr__(self, "_development_input", None)

    def lookup(self, load_signature, existing_io, stage, physical_bytes, *, execution="production"):
        signature(load_signature)
        vector(existing_io)
        if stage not in STAGES:
            raise ValueError("unknown physical stage")
        integer(physical_bytes, "physical bytes", 1)
        if execution not in ("production", "cpu_mock", "gpu_development"):
            raise ValueError("explicit production/CPU mock lookup scope required")
        if execution == "production":
            if not self.production_qualified:
                return None
        elif execution == "gpu_development":
            if self.development_input is None:
                return None
        elif self.scope != "mock_only":
            return None
        key = (load_signature, existing_io, stage, physical_bytes)
        matches = [c for c in self.cells if c.key == key]
        if len(matches) != 1:
            return None
        c = matches[0]
        return CostEstimate(c.total_ns, c.basis, c.baseline_ns, c.incremental_or_joint_ns,
                            c.uncertainty_ns, self.source_sha256,
                            mock_only=execution == "cpu_mock",
                            production_qualified=execution == "production" and self.production_qualified)


def load_conditional_table(path):
    """Retain provenance of a real P3 conditional JSON, never authorize control."""
    data = Path(path).read_bytes()
    parsed = json.loads(data)
    if type(parsed) is not dict:
        raise ValueError("conditional table must be a JSON object")
    # A self-declared qualification cannot bypass the missing P4 GPU validator.
    return CostTable((), scope="conditional", source_sha256=sha256(data).hexdigest())


def _from_verified_gpu_capability(proof):
    """Private entry; a JSON flag or CPU candidate cannot provide this capability."""
    from .gpu_cell_issuer import _table_material
    cells, source_sha256, identity = _table_material(proof)
    if type(cells) is not tuple or not 1 <= len(cells) <= 8:
        raise ValueError("finite verified GPU cells required")
    table = CostTable(cells, scope="conditional", source_sha256=source_sha256)
    object.__setattr__(table, "scope", "gpu_verified_exact_cells")
    object.__setattr__(table, "_gpu_proof", proof)
    if not table.production_qualified:
        raise ValueError("raw GPU issuer capability rejected")
    return table
