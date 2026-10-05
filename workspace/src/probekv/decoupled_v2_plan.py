"""CPU-only, fail-closed P0 input audit and bounded experiment specifications.

This module cannot authorize GPU execution. A frozen action *template* is not a
qualified cohort: it deliberately leaves request/Artifact identities unset when
the original data, construction recipes or current authorization are absent.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


STRATEGY_ID = "decoupled_qd_source_origin_v2"
SEED = 20260918
P1E_TARGET_CAP = 30
PER_GROUP_CAP = 10
P1M_TARGET_CAP = 12
P1E_ACTION_CAP = 180
P1M_ACTION_CAP = 36


def stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def deterministic_target_cap(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Cap by content group before outcomes; do not manufacture missing units.

    This is only an ordering/capping utility. It does not make a row eligible,
    split related components, or grant permission to execute it.
    """
    keys = [(str(r["group_id"]), str(r["origin_example_id"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate target/group in action proposal")
    ordered = sorted(rows, key=lambda r: (stable_digest([SEED, r["group_id"],
                        r["origin_example_id"]]), r["group_id"], r["origin_example_id"]))
    counts: Counter = Counter()
    result = []
    for row in ordered:
        if counts[row["group_id"]] >= PER_GROUP_CAP:
            continue
        result.append(dict(row))
        counts[row["group_id"]] += 1
        if len(result) == P1E_TARGET_CAP:
            break
    return result


def dependency_components(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Join shared content, origins and explicitly supplied lineage parents.

    The caller still has to prove lineage completeness. These components are a
    conservative audit of the available records, not a new frozen partition.
    """
    parents = {str(r["group_id"]): str(r["group_id"]) for r in rows}

    def find(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    owner: dict[tuple[str, str], str] = {}
    for row in rows:
        group = str(row["group_id"])
        keys = [("content", str(row.get("content_key", group)))]
        ids = list(row["source_origin_ids"])
        ids += [str(t["origin_example_id"]) for t in row["targets"]]
        ids += list(row.get("lineage_parent_origin_ids", ()))
        keys += [("origin", str(origin)) for origin in ids]
        for key in keys:
            if key in owner:
                a, b = sorted((find(group), find(owner[key])))
                parents[b] = a
            else:
                owner[key] = group
    return {group: find(group) for group in sorted(parents)}


def verify_observed_exclusion_ids(censuses: Sequence[Mapping[str, Any]],
                                  observed_groups: Iterable[str]) -> None:
    """Catch ID transcription errors instead of silently allowing seen data."""
    available = {r["group_id"] for c in censuses for r in c["rows"]}
    if not set(observed_groups).issubset(available):
        raise ValueError("prior observed group identity not found in hash-verified census")


def audit_censuses(censuses: Sequence[Mapping[str, Any]], *,
                   previously_observed_groups: Iterable[str]) -> tuple[list[dict], dict]:
    """Audit existing census records without reading answers or model outputs.

    A census explicitly lacks original prompt geometry and lineage proof. Even
    an unobserved, 512-token row therefore remains BLOCKED, never qualified.
    """
    observed = set(previously_observed_groups)
    rows: list[dict] = []
    for census in censuses:
        if census.get("kind") != "unused_calibration_support_census_v1":
            raise ValueError("not a recognized supporting-span census")
        if (census.get("locked_test_accessed") is not False or
                census.get("outcome_based_sampling") is not False):
            raise ValueError("unsafe or unknown census data isolation")
        for original in census["rows"]:
            row = dict(original, dataset=census["dataset"])
            sources = row["source_origin_ids"]
            targets = [t["origin_example_id"] for t in row["targets"]]
            blockers = ["ORIGINAL_RAW_CASES_PARTITION_AND_TOKENIZER_NOT_REVALIDATED",
                        "FULL_PROMPT_AND_ABSOLUTE_POSITIONS_NOT_REVALIDATED",
                        "COMPLETE_LINEAGE_AND_PRIOR_EXPOSURE_NOT_VERIFIED",
                        "HISTORICAL_SOURCE_BUILD_EVIDENCE_NOT_BOUND"]
            if len(sources) != 4 or len(set(sources)) != 4:
                blockers.append("FOUR_DISTINCT_HISTORICAL_ORIGINS_REQUIRED")
            if set(sources).intersection(targets) or len(set(targets)) != len(targets):
                blockers.append("HISTORY_CONSUMER_IDENTITY_OVERLAP")
            if row["group_id"] in observed:
                blockers.append("PREVIOUSLY_OBSERVED_GROUP_REGRESSION_ONLY")
            if row["token_count"] != 512:
                blockers.append("LEGACY_512_TOKEN_GEOMETRY_UNSATISFIED")
            if not targets:
                blockers.append("NO_FUTURE_CONSUMER_IN_CENSUS")
            if any(t.get("support_stratum") == "audit_rejected" for t in row["targets"]):
                blockers.append("AMBIGUOUS_SUPPORT_LABEL_PRESERVED")
            row.update(status="BLOCKED", new_v2_qualified=False,
                       source_ids_are_origin_ids_not_artifact_ids=True,
                       input_sha256=dict(census["input_sha256"]),
                       known_previously_observed=row["group_id"] in observed,
                       blockers=blockers)
            rows.append(row)
    components = dependency_components(rows)
    observed_components = {components[g] for g in observed if g in components}
    for row in rows:
        row["known_dependency_component"] = components[row["group_id"]]
        row["full_lineage_verified"] = False
        if (components[row["group_id"]] in observed_components and
                row["group_id"] not in observed):
            row["blockers"].append("KNOWN_ORIGIN_DEPENDENCY_ON_OBSERVED_GROUP")
    summary = dict(kind="decoupled_v2_local_data_qualification", status="BLOCKED",
        census_rows=len(rows), groups=len(components),
        known_dependency_components=len(set(components.values())),
        previously_observed_group_count=sum(r["known_previously_observed"] for r in rows),
        legacy_512_token_rows=sum(r["token_count"] == 512 for r in rows),
        full_support_target_pairs=sum(t["support_stratum"] == "full_support_sentence"
            for r in rows for t in r["targets"]),
        full_support_pairs_outside_two_known_observed_groups=sum(t["support_stratum"] == "full_support_sentence"
            for r in rows if not r["known_previously_observed"] for t in r["targets"]),
        freshness_of_remaining_pairs_established=False,
        qualified_target_count=0, qualified_source_artifact_count=0,
        component_counts_are_lower_bound_not_split_proof=True,
        source_construction_executed=0, real_model_executed=False,
        gpu_execution_allowed=False, locked_test_accessed=False, paper_evidence=False)
    return rows, summary


def bounded_action_specification(contract_sha256: str) -> tuple[dict, list[dict], dict]:
    """Freeze capped slots, not fabricated target or Source identities.

    Source creation is a separate ledger. Currently permitted source builds are
    zero: neither the S0 recipe nor the mixed parents/time budgets are frozen.
    """
    if len(contract_sha256) != 64 or any(c not in "0123456789abcdef" for c in contract_sha256):
        raise ValueError("actual contract file SHA256 required")
    actions = []
    for phase, count, arms in (("P1-E", P1E_TARGET_CAP,
            ("dense", "historical_1", "historical_2", "historical_3", "historical_4", "S0")),
            ("P1-M", P1M_TARGET_CAP, ("dense", "E", "M1"))):
        for slot in range(1, count + 1):
            for arm in arms:
                actions.append(dict(phase=phase, target_slot=slot, arm=arm,
                    request_id=None, content_group_id=None, source_artifact_id=None,
                    source_artifact_sha256=None, birth_request_id=None,
                    input_sha256=None, execution_manifest_sha256=None,
                    status="UNBOUND_NOT_EXECUTABLE", execute=False,
                    repair_ratio=0.15 if arm != "dense" else None,
                    repair="legacy_normalized_kv" if arm != "dense" else None,
                    first_selective_reuse_layer=9 if arm != "dense" else None,
                    source_origin_policy="EXACT_ONLY" if phase == "P1-E" else "PAIRED_E_M1_ISOLATED"))
    builds = dict(kind="separate_source_construction_ledger", status="BLOCKED",
        exact_group_count=None, paired_birth_count=None,
        currently_permitted_total_source_builds=0, planned_actual_source_builds=[],
        conditional_bounds={
            "P1-E": {"historical_builds_per_unique_group": 4,
                "independent_S0_builds_per_unique_group": 1,
                "unique_group_ceiling": P1E_TARGET_CAP,
                "derived_child_build_ceiling": 5 * P1E_TARGET_CAP,
                "S0_construction_recipe": None},
            "P1-M": {"paired_E_M1_builds_per_unique_birth": 2,
                "unique_birth_ceiling": P1M_TARGET_CAP,
                "derived_child_build_ceiling": 2 * P1M_TARGET_CAP,
                "upstream_exact_parent_build_manifest": None,
                "additional_parent_builds_permitted": 0},
        }, per_action_time_budget_seconds=None, total_time_budget_seconds=None,
        self_alignment_check_actions=None, additional_dense_reference_actions=None,
        isolated_propagation_case_cap=6, model_G2_quality_actions=0,
        blocker="Bind exact births, parents, S0 recipe, numerical checks and time/cost caps before any build",
        diagnostic_objects_can_populate_online_pool=False)
    manifest = dict(kind="decoupled_v2_bounded_action_template", strategy_id=STRATEGY_ID,
        contract_sha256=contract_sha256, seed=SEED, template_frozen=True,
        actual_cohort_frozen=False, action_bindings_complete=False, status="BLOCKED",
        target_cap=P1E_TARGET_CAP, per_group_cap=PER_GROUP_CAP,
        mixed_target_cap=P1M_TARGET_CAP, actual_target_count=0,
        actual_qualified_groups=[], cohort_sha256=None,
        main_action_caps={"P1-E": P1E_ACTION_CAP, "P1-M": P1M_ACTION_CAP},
        currently_permitted_gpu_actions=0, gpu_execution_allowed=False,
        diagnostic_only=True, paper_evidence=False, locked_test_accessed=False,
        chronology="corpus_pseudotime_requires_revalidation",
        old_partitions_modified=False, stage_dependencies={"P1-E": ["P0-E"], "P1-M": ["P0-M"]},
        stops={"P1E_complementary_groups": 2, "macro_headroom_f1": 0.01,
               "alternative_quality_coverage_space": 0.05, "mixed_vs_E_mean_f1_margin": 0.01,
               "rescue_batches_max": 1, "rescue_automatic_execution": False},
        blockers=["QUALIFIED_ORIGINAL_COHORT_MISSING", "S0_CONSTRUCTION_RECIPE_UNSPECIFIED",
                  "MIXED_BIRTH_AND_PARENT_RECIPE_UNBOUND", "SOURCE_BUILD_AND_CHECK_BUDGET_UNBOUND",
                  "CURRENT_GPU_AUTHORIZATION_MISSING", "P0_REAL_ENGINE_E_M_NUMERICS_NOT_EVALUATED"])
    manifest["action_template_sha256"] = stable_digest(actions)
    manifest["construction_template_sha256"] = stable_digest(builds)
    return manifest, actions, builds


def cpu_gpu_preflight(*, contract_sha256: str, runtime_commit: str,
                      code_worktree_digest: str) -> dict:
    """No remote probe, historical permission inheritance, or default prices."""
    return dict(kind="decoupled_v2_gpu_preflight", status="BLOCKED",
        contract_sha256=contract_sha256, runtime_commit=runtime_commit,
        local_code_worktree_digest=code_worktree_digest,
        active_instance_id=None, gpu_uuid=None, approval_reference=None,
        maximum_cost=None, maximum_gpu_hours=None, current_unit_price=None,
        remote_code_sha_verified=False, model_revision_verified=False,
        tokenizer_verified=False, patch_verified=False,
        native_exact_and_mixed_correctness="NOT_EVALUATED",
        qualified_cohort_available=False, action_bindings_complete=False,
        gpu_execution_allowed=False, automatic_rental_allowed=False,
        remote_connected=False, gpu_actions_executed=0,
        historical_budget_is_permission=False,
        missing=["current authorized instance identity and time window",
                 "current hourly price and explicit total time/cost cap",
                 "matching clean runtime/patch/model/tokenizer/extension qualification",
                 "original data, provenance-isolated cohort and exact action bindings",
                 "bounded Source/S0/E/M1 construction and numerical checks"],
        formal_profile_bundle_frozen=False, gpu_runtime_qualified=False,
        h1_h2_execution_allowed=False, paper_evidence=False, locked_test_accessed=False)
