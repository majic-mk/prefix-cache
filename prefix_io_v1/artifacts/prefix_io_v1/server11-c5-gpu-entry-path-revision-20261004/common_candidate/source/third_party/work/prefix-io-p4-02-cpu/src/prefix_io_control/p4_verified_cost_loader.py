"""Prepared exact cost loader: semantic CPU readiness never opens production."""
from dataclasses import dataclass
from .p4_cost_table import CostTable
from .p4_production_table_contract import load_production_candidate
from .p4_paired_measurement_verifier import (
    SemanticPairedVerification, load_verification_plan, verify_paired_measurements,
)


@dataclass(frozen=True)
class PreparedCostTable:
    verification: SemanticPairedVerification
    cpu_mock_table: CostTable

    def __post_init__(self):
        if type(self.verification) is not SemanticPairedVerification or type(self.cpu_mock_table) is not CostTable:
            raise TypeError("exact semantic verification and original CostTable required")
        if (self.verification.status != "PASS_CPU_SEMANTICS_NOT_GPU_QUALIFICATION" or
                self.cpu_mock_table.scope != "mock_only" or
                self.cpu_mock_table.source_sha256 != self.verification.candidate_ref.sha256 or
                self.cpu_mock_table.cells != tuple(c.cost for c in self.verification.cells)):
            raise ValueError("prepared mock cells/source must equal semantic recomputation")

    @property
    def source_ref(self):
        return self.verification.candidate_ref

    @property
    def cell_count(self):
        return len(self.verification.cells)

    @property
    def gpu_verified(self):
        return False

    @property
    def production_qualified(self):
        return False

    @property
    def status(self):
        return "PREPARED_CPU_SEMANTICS_GPU_BLOCKED"

    def lookup(self, load_signature, existing_io, stage, physical_bytes, *, execution="production"):
        # Enforce this wrapper contract even if the underlying interface changes.
        if execution == "production":
            return None
        return self.cpu_mock_table.lookup(load_signature, existing_io, stage,
                                          physical_bytes, execution=execution)


def load_semantically_verified_table(root, candidate_path, *, expected_context,
        qualification_ref, expected_verifier_ref, plan_path, expected_plan_ref):
    plan = load_verification_plan(root, plan_path, expected_plan_ref=expected_plan_ref)
    candidate = load_production_candidate(root, candidate_path,
        expected_context=expected_context, qualification_ref=qualification_ref,
        expected_verifier_ref=expected_verifier_ref)
    verification = verify_paired_measurements(root, candidate, expected_plan=plan)
    table = CostTable(tuple(cell.cost for cell in verification.cells), scope="mock_only",
                      source_sha256=verification.candidate_ref.sha256)
    return PreparedCostTable(verification, table)
