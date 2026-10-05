"""Audit local handoff inputs and freeze non-executable bounded P1 templates.

No server connection, model load, GPU query, rental or remote-state mutation.
Never overwrite existing output artifacts or reinterpret old QA as v2 evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from probekv.decoupled_v2_plan import (audit_censuses, bounded_action_specification,
                                     cpu_gpu_preflight, stable_digest, verify_observed_exclusion_ids)
from probekv.io import atomic_write_json, git_commit, sha256_file, write_jsonl


# These are actual group identities in the local prior census and the later QA
# report, not invented new candidate IDs. Their new-v2 eligibility is excluded.
OBSERVED_SUPPORT_GROUPS = (
    "2WikiMultiHopQA:2c4d2edd4f5e68f97c99d235d365ca5c4b0332d79d44b6dc18d4444d11ad7df6",
    "2WikiMultiHopQA:3de89412564f872ebadbd9379c223efcf45e542047f72367aaee46c10a86c5cb",
)
CENSUS_BINDINGS = (
    ("support-census-2wiki-632b877-v1.json", "2621bae0bbf2e0cd76386cca8854c9be76c0bbe699c26ce0651a1a89edd3a00c"),
    ("support-census-hotpot-632b877-v1.json", "f84c5ac0f09aae8162bfb024d36d0230c603fd272c59112d064095595a55f26f"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    outputs = ("input_qualification.jsonl", "data_qualification_summary.json",
        "cohort_manifest.json", "planned_actions.csv", "source_construction_plan.json", "gpu_preflight.json")
    if any((args.output / name).exists() for name in outputs):
        raise FileExistsError("P0 artifacts are immutable; select a fresh output directory")
    assets, censuses = [], []
    for name, expected in CENSUS_BINDINGS:
        path = root / name
        record = dict(path=name, expected_sha256=expected, actual_sha256=None,
                      status="MISSING", evidence_scope="historical_census_only")
        if path.is_file():
            record["actual_sha256"] = sha256_file(path)
            if record["actual_sha256"] != expected:
                raise ValueError("historical census file digest mismatch: " + name)
            record["status"] = "LOCAL_HASH_VERIFIED_NOT_RAW_REQUALIFIED"
            censuses.append(json.loads(path.read_text(encoding="utf-8")))
        assets.append(record)
    if len(censuses) == len(CENSUS_BINDINGS):
        verify_observed_exclusion_ids(censuses, OBSERVED_SUPPORT_GROUPS)
    qa_report = root / "docs/SUPPORT_SOURCE_QA_20260918.md"
    support_report = root / "docs/SUPPORT_SPAN_CENSUS_20260918.md"
    # The detailed raw pilot is not present locally; retain the provenance limit.
    for path in (qa_report, support_report):
        assets.append(dict(path=path.relative_to(root).as_posix(),
            actual_sha256=sha256_file(path) if path.is_file() else None,
            status="HISTORICAL_DOCUMENT_ONLY" if path.is_file() else "MISSING"))
    dependencies = [
        "artifacts/prepared_stream/2wiki/cases.jsonl",
        "artifacts/prepared_stream/hotpotqa/cases.jsonl",
        "artifacts/schema10-profile-25f0cc1/mistral/development_partition.jsonl",
        "data/official/2wiki-train-complete.json",
        "data/official/hotpotqa-official.json",
        "artifacts/support-freeze-1e03b2e-v1/pilot.json",
    ]
    for name in dependencies:
        path = root / name
        assets.append(dict(path=name, actual_sha256=sha256_file(path) if path.is_file() else None,
            status="PRESENT_REQUALIFICATION_REQUIRED" if path.is_file() else "NOT_LOCAL",
            no_remote_fetch_performed=True))
    rows, summary = audit_censuses(censuses, previously_observed_groups=OBSERVED_SUPPORT_GROUPS)
    summary["assets"] = assets
    summary["exclusion_basis"] = dict(group_ids=list(OBSERVED_SUPPORT_GROUPS),
        reports=["docs/SUPPORT_SPAN_CENSUS_20260918.md", "docs/SUPPORT_SOURCE_QA_20260918.md"],
        raw_prior_pilot_locally_verified=False,
        conservative_exclusion_only=True)
    summary["issues"] = [
        {"id": "DATA-01", "issue": "The two rich C4 census groups already have prior QA outcomes",
         "handling": "Exclude from fresh v2 cohort; retain as regression/exploration only"},
        {"id": "DATA-02", "issue": "Prior rich groups contain 112/200 tokens, not the draft's legacy 512",
         "handling": "Do not pad/truncate/relabel; require explicit geometry resolution before freezing a new cohort"},
        {"id": "DATA-03", "issue": "Raw official inputs, prepared cases, old partition, tokenizer and full prior pilots not locally rebound",
         "handling": "Census hash verification alone cannot establish original prompts, chronology or complete lineage isolation"},
        {"id": "DATA-04", "issue": "Historical census Markdown transcribes the second rich group with one extra e after 872",
         "handling": "Use exact group ID from the hash-verified census, reject unknown exclusion IDs; do not modify old report"},
        {"id": "PLAN-01", "issue": "Independent S0 birth prompt/position/construction recipe is unspecified",
         "handling": "Leave S0 unbound; do not silently pick target, earliest or isolated text-only Source"},
        {"id": "PLAN-02", "issue": "Mixed birth parents/recipe and construction/check time limits are not frozen",
         "handling": "Zero currently executable builds; exact birth-level ledger required before GPU"},
    ]
    contract = root / "ProbeKV_Codex_First_Handoff/04_实验契约草案.json"
    contract_sha = sha256_file(contract)
    manifest, actions, builds = bounded_action_specification(contract_sha)
    own_sources = [root / "src/probekv/decoupled_v2_plan.py", Path(__file__).resolve()]
    local_digest = stable_digest({str(p.relative_to(root)): sha256_file(p) for p in own_sources})
    manifest["runtime_commit_at_p0"] = git_commit(root)
    manifest["planner_local_source_digest"] = local_digest
    manifest["data_qualification_summary_sha256"] = stable_digest(summary)
    preflight = cpu_gpu_preflight(contract_sha256=contract_sha, runtime_commit=git_commit(root),
                                 code_worktree_digest=local_digest)
    preflight["worktree_digest_scope"] = "planner/generator files only; not whole runtime qualification"
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "input_qualification.jsonl", rows)
    atomic_write_json(args.output / "data_qualification_summary.json", summary)
    atomic_write_json(args.output / "cohort_manifest.json", manifest)
    atomic_write_json(args.output / "source_construction_plan.json", builds)
    atomic_write_json(args.output / "gpu_preflight.json", preflight)
    with (args.output / "planned_actions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(actions[0]))
        writer.writeheader()
        writer.writerows(actions)
    print(json.dumps(dict(status="BLOCKED", audited_census_rows=len(rows),
        template_actions=len(actions), bound_actions=0, gpu_actions_executed=0,
        output=str(args.output.resolve()))))


if __name__ == "__main__":
    main()
