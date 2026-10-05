"""Strict startup binding of prepared measurements; it grants no GPU capability."""
from dataclasses import dataclass
from pathlib import PurePosixPath
from .p4_production_table_contract import TableContext, EvidenceRef, _keys, _require

def relative(value):
    _require(type(value) is str and 0 < len(value) <= 512 and "\\" not in value and "\x00" not in value,
             "bounded project-relative metadata path required")
    p=PurePosixPath(value)
    _require(not p.is_absolute() and ":" not in p.parts[0] and
             all(x not in ("",".","..") for x in value.split("/")),
             "relative metadata path cannot escape project")
    return value

@dataclass(frozen=True)
class PreparedCostRequest:
    candidate_path: str
    expected_context: TableContext
    qualification_ref: EvidenceRef
    expected_verifier_ref: EvidenceRef
    plan_path: str
    expected_plan_ref: EvidenceRef

    def load(self,root):
        from .p4_verified_cost_loader import load_semantically_verified_table
        return load_semantically_verified_table(root,self.candidate_path,
            expected_context=self.expected_context,qualification_ref=self.qualification_ref,
            expected_verifier_ref=self.expected_verifier_ref,plan_path=self.plan_path,
            expected_plan_ref=self.expected_plan_ref)

def parse_prepared_cost_request(raw):
    _keys(raw,("kind","candidate_path","expected_context","qualification_ref",
        "expected_verifier_ref","plan_path","expected_plan_ref"),"prepared cost startup request")
    _require(raw["kind"] == "paired_semantic_candidate","only strict paired semantic candidate input supported")
    candidate=relative(raw["candidate_path"]);plan=relative(raw["plan_path"])
    refs=[EvidenceRef.from_mapping(raw[k]) for k in
          ("qualification_ref","expected_verifier_ref","expected_plan_ref")]
    for ref in refs:relative(ref.path)
    _require(refs[2].path == plan,"independently frozen plan path must match")
    return PreparedCostRequest(candidate,TableContext.from_mapping(raw["expected_context"]),*refs[:2],plan,refs[2])
