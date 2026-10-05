"""Exact finite interference-cost interface, with a closed production gate.

P3 conditional/native-owned windows are not a production interference table.
CPU mock cells exercise the arithmetic only. No constructor flag can turn them
into GPU-qualified evidence; the P4 paired GPU qualification remains required.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from .dispatch_budget import Amount, STAGES, integer, vector

BASES = frozenset(("existing_io_plus_delta", "no_io_plus_joint"))


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
    __slots__ = ("cells", "scope", "source_sha256", "_gpu_proof")

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

    def lookup(self, load_signature, existing_io, stage, physical_bytes, *, execution="production"):
        signature(load_signature)
        vector(existing_io)
        if stage not in STAGES:
            raise ValueError("unknown physical stage")
        integer(physical_bytes, "physical bytes", 1)
        if execution not in ("production", "cpu_mock"):
            raise ValueError("explicit production/CPU mock lookup scope required")
        if execution == "production":
            if not self.production_qualified:
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
