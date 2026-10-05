import copy
import unittest

from probekv.decoupled_v2_plan import (audit_censuses, bounded_action_specification,
    cpu_gpu_preflight, dependency_components, deterministic_target_cap, verify_observed_exclusion_ids)


class DecoupledP0PlanTests(unittest.TestCase):
    def census(self):
        return dict(kind="unused_calibration_support_census_v1", dataset="fixture",
            locked_test_accessed=False, outcome_based_sampling=False,
            input_sha256={}, rows=[dict(group_id="group", case_id="case", token_count=512,
                source_origin_ids=["s1", "s2", "s3", "s4"],
                targets=[dict(origin_example_id="t1", support_stratum="full_support_sentence")])])

    def test_census_not_raw_qualification(self):
        rows, summary = audit_censuses([self.census()], previously_observed_groups=())
        self.assertEqual(summary["qualified_target_count"], 0)
        self.assertFalse(rows[0]["new_v2_qualified"])
        self.assertIn("FULL_PROMPT_AND_ABSOLUTE_POSITIONS_NOT_REVALIDATED", rows[0]["blockers"])

    def test_previously_seen_and_short_geometry_do_not_become_new_cohort(self):
        c = self.census()
        c["rows"][0]["token_count"] = 200
        rows, summary = audit_censuses([c], previously_observed_groups=["group"])
        self.assertEqual(summary["full_support_pairs_outside_two_known_observed_groups"], 0)
        self.assertIn("PREVIOUSLY_OBSERVED_GROUP_REGRESSION_ONLY", rows[0]["blockers"])
        self.assertIn("LEGACY_512_TOKEN_GEOMETRY_UNSATISFIED", rows[0]["blockers"])

    def test_identity_and_ambiguous_support_are_preserved(self):
        c = self.census()
        c["rows"][0]["targets"] = [dict(origin_example_id="s1", support_stratum="audit_rejected")]
        rows, _ = audit_censuses([c], previously_observed_groups=())
        self.assertIn("HISTORY_CONSUMER_IDENTITY_OVERLAP", rows[0]["blockers"])
        self.assertIn("AMBIGUOUS_SUPPORT_LABEL_PRESERVED", rows[0]["blockers"])
        self.assertEqual(rows[0]["targets"][0]["support_stratum"], "audit_rejected")

    def test_unsafe_or_unknown_isolation_rejected(self):
        for field in ("locked_test_accessed", "outcome_based_sampling"):
            c = self.census()
            c.pop(field)
            with self.assertRaises(ValueError):
                audit_censuses([c], previously_observed_groups=())

    def test_exclusion_typo_fails_instead_of_leaking_seen_group(self):
        verify_observed_exclusion_ids([self.census()], ["group"])
        with self.assertRaises(ValueError):
            verify_observed_exclusion_ids([self.census()], ["groupp"])

    def test_dependency_component_not_group_label_is_independence_unit(self):
        a = self.census()["rows"][0]
        b = dict(group_id="b", source_origin_ids=["bs"],
                 targets=[dict(origin_example_id="bt")], lineage_parent_origin_ids=["s1"])
        c = dict(group_id="c", source_origin_ids=["cs"], targets=[dict(origin_example_id="ct")],
                 lineage_parent_origin_ids=["bt"])
        components = dependency_components([a, b, c])
        self.assertEqual(len(set(components.values())), 1)

    def test_observed_dependency_component_also_blocked(self):
        c = self.census()
        extra = copy.deepcopy(c["rows"][0])
        extra["group_id"] = "other"
        extra["targets"] = [dict(origin_example_id="t2", support_stratum="full_support_sentence")]
        c["rows"].append(extra)
        rows, _ = audit_censuses([c], previously_observed_groups=["group"])
        self.assertIn("KNOWN_ORIGIN_DEPENDENCY_ON_OBSERVED_GROUP", rows[1]["blockers"])

    def test_deterministic_caps_without_outcome_selection(self):
        rows = [dict(group_id=str(g), origin_example_id=f"{g}:{t}") for g in range(5) for t in range(20)]
        chosen = deterministic_target_cap(rows)
        self.assertEqual(len(chosen), 30)
        self.assertEqual(chosen, deterministic_target_cap(rows[::-1]))
        self.assertTrue(all(sum(r["group_id"] == str(g) for r in chosen) <= 10 for g in range(5)))
        self.assertEqual(len(deterministic_target_cap(rows[:3])), 3)
        with self.assertRaises(ValueError):
            deterministic_target_cap(rows + rows[:1])

    def test_bounded_unbound_actions_never_claim_execution(self):
        manifest, actions, builds = bounded_action_specification("a" * 64)
        self.assertEqual(sum(a["phase"] == "P1-E" for a in actions), 180)
        self.assertEqual(sum(a["phase"] == "P1-M" for a in actions), 36)
        self.assertTrue(all(a["request_id"] is None and not a["execute"] for a in actions))
        self.assertFalse(manifest["actual_cohort_frozen"])
        self.assertIsNone(manifest["cohort_sha256"])
        self.assertEqual(builds["currently_permitted_total_source_builds"], 0)
        self.assertEqual(builds["model_G2_quality_actions"], 0)
        self.assertIsNone(builds["conditional_bounds"]["P1-E"]["S0_construction_recipe"])

    def test_contract_digest_must_be_actual_sha_shape(self):
        for invalid in ("", "pending", "z" * 64):
            with self.assertRaises(ValueError):
                bounded_action_specification(invalid)

    def test_cpu_preflight_cannot_inherit_historical_gpu_permission(self):
        report = cpu_gpu_preflight(contract_sha256="a"*64, runtime_commit="commit",
                                  code_worktree_digest="b"*64)
        self.assertFalse(report["gpu_execution_allowed"])
        self.assertFalse(report["remote_connected"])
        self.assertIsNone(report["current_unit_price"])
        self.assertIsNone(report["active_instance_id"])
        self.assertEqual(report["native_exact_and_mixed_correctness"], "NOT_EVALUATED")


if __name__ == "__main__":
    unittest.main()
