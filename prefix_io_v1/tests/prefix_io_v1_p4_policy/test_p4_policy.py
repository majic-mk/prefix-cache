"""CPU-only contract tests, not GPU or counterfactual performance evidence."""
from dataclasses import replace, FrozenInstanceError
import tempfile
import unittest
from pathlib import Path

from prefix_io_control.dependencies import Parent, Resource, ResourceId
from prefix_io_control.dispatch_budget import Amount, ZERO
from prefix_io_control.dispatch_shadow import ShadowState
from prefix_io_control.simple_stage_policy import SimpleStageConfig, make_dispatch_controller
from prefix_io_control.p4_types import (
    P4Config, SystemSnapshot, WorkDescriptor, ReleaseWitness, WaitingTarget,
)
from prefix_io_control.p4_policy import make_p4_policy
from prefix_io_control.p4_cost_table import CostCell, CostTable, load_conditional_table

NOW = 100
CAPS = frozenset(("native_ready_work", "owner_release_protocol", "gpu_owner_generation",
                  "gpu_active_refs", "gpu_protectors", "cpu_owner_generation",
                  "cpu_active_refs", "cpu_protectors", "restore_parent_completion"))


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.p1 = Parent("r", 1, 12, 0, 0, False, False, 3, 140)
        self.p2 = Parent("r", 2, 5, 0, 0, False, False, 2, 150)
        self.id1 = ResourceId("r", "pool", 0, 11, 4)
        self.id2 = ResourceId("r", "pool", 0, 12, 7)
        self.w1 = WorkDescriptor("r", 2, 1, 0, "ssd_write", 16, 1, 99, False)
        self.w2 = WorkDescriptor("r", 2, 2, 0, "ssd_write", 8, 0, 99, False)
        self.works = (self.w2, self.w1)
        self.target = WaitingTarget("wait", 0, 99, need_gpu_bytes=16)
        self.snapshot = SystemSnapshot("r", 2, 99, CAPS, (self.p1, self.p2),
            ShadowState("r", 99, ZERO, 32, 2, True), (self.target,),
            ((self.id1, 4), (self.id2, 7)), None, None, ("model", "gpu", 2, 16400, 16))
        self.r1 = Resource(self.id1, 16, 0, frozenset((1,)), "observed_blocking")
        self.witness = ReleaseWitness("wait", self.r1, (self.p1,), "gpu",
            (self.w1.work_id,), "original-native-parent-source-fence", 140)
        self.policy = make_p4_policy("r", P4Config("dependency_only", 10, 100))

    def choose(self, *, snapshot=None, works=None, witnesses=None, policy=None, now=NOW, epoch=2):
        return (policy or self.policy).choose(
            self.snapshot if snapshot is None else snapshot,
            self.works if works is None else works,
            (self.witness,) if witnesses is None else witnesses,
            now_ns=now, expected_epoch=epoch)

    def assert_native(self, result):
        self.assertNotEqual(result.action, "selected")
        self.assertEqual(result.selected_work_ids, tuple(w.work_id for w in self.works))
        self.assertTrue(result.tail_unchanged)
        self.assertFalse(result.gpu_qualified)

    def test_off_factory_has_no_policy(self):
        self.assertIsNone(make_p4_policy("r", P4Config("off", 10, 100)))

    def test_values_are_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            self.snapshot.snapshot_epoch = 3

    def test_scalar_only_work_identity(self):
        with self.assertRaises(ValueError):
            replace(self.w1, child_id=object())
        with self.assertRaises(ValueError):
            replace(self.w1, child_id=(1, 2))

    def test_no_resource_native_reference(self):
        with self.assertRaises(TypeError):
            replace(self.witness, resource=object())

    def test_known_release_reorders_only_existing_work(self):
        result = self.choose()
        self.assertEqual(result.action, "selected")
        self.assertEqual(result.selected_work_ids, (self.w1.work_id, self.w2.work_id))
        self.assertEqual(result.gpu_release_credit_bytes, 16)

    def test_shadow_proposes_but_actual_remains_native(self):
        shadow = make_p4_policy("r", P4Config("shadow", 10, 100))
        result = self.choose(policy=shadow)
        self.assertEqual(result.action, "observed")
        self.assertEqual(result.selected_work_ids, (self.w2.work_id, self.w1.work_id))
        self.assertEqual(result.proposed_work_ids, (self.w1.work_id, self.w2.work_id))

    def test_stale_time_and_future_time_fallback(self):
        self.assert_native(self.choose(now=110))
        self.assert_native(self.choose(now=98))

    def test_stale_run_fallback(self):
        s = replace(self.snapshot, run_id="other", native_state=None, parents=(), generations=())
        self.assert_native(self.choose(snapshot=s))

    def test_epoch_mismatch_fallback(self):
        self.assert_native(self.choose(epoch=3))

    def test_missing_gpu_capability_has_no_credit(self):
        s = replace(self.snapshot, capabilities=frozenset(("native_ready_work",)))
        r = self.choose(snapshot=s)
        self.assert_native(r)
        self.assertIsNone(r.gpu_release_credit_bytes)

    def test_unknown_active_references_not_zero(self):
        w = replace(self.witness, resource=replace(self.r1, active_refs=None))
        self.assert_native(self.choose(witnesses=(w,)))

    def test_active_reference_blocks_release(self):
        w = replace(self.witness, resource=replace(self.r1, active_refs=1))
        self.assert_native(self.choose(witnesses=(w,)))

    def test_generation_reuse_blocks_stale_witness(self):
        s = replace(self.snapshot, generations=((self.id1, 5), (self.id2, 7)))
        self.assert_native(self.choose(snapshot=s))

    def test_unknown_generation_blocks_release(self):
        self.assert_native(self.choose(snapshot=replace(self.snapshot, generations=())))

    def test_foreign_run_physical_identity_rejected(self):
        foreign = replace(self.id1, run_id="foreign")
        with self.assertRaises(ValueError):
            replace(self.snapshot, generations=((foreign, 4),))
        with self.assertRaises(ValueError):
            replace(self.w1, resource_identity=foreign, generation=4)
        with self.assertRaises(ValueError):
            replace(self.witness, parents=(replace(self.p1, run_id="foreign"),))

    def test_stale_work_generation_blocks_reordering(self):
        w = replace(self.w1, generation=4, resource_identity=self.id1)
        s = replace(self.snapshot, generations=((self.id1, 5), (self.id2, 7)))
        result = self.choose(snapshot=s, works=(self.w2, w))
        self.assertNotEqual(result.action, "selected")

    def test_multi_protector_requires_all_parents_and_work(self):
        r = replace(self.r1, protecting_jobs=frozenset((1, 2)))
        w = replace(self.witness, resource=r, parents=(self.p1, self.p2),
                    release_scope="multi_protector", estimated_unblock_ns=150)
        self.assert_native(self.choose(witnesses=(w,)))
        w = replace(w, work_ids=(self.w1.work_id, self.w2.work_id))
        result = self.choose(witnesses=(w,))
        self.assertEqual(result.action, "selected")
        self.assertEqual(result.gpu_release_credit_bytes, 16)

    def test_multi_protector_not_or_semantics(self):
        r = replace(self.r1, protecting_jobs=frozenset((1, 2)))
        w = replace(self.witness, resource=r, release_scope="multi_protector")
        self.assert_native(self.choose(witnesses=(w,)))

    def test_parent_scope_not_child_completion(self):
        self.assert_native(self.choose(witnesses=(replace(self.witness, release_scope="per_file"),)))

    def test_parent_error_never_release(self):
        bad = replace(self.p1, failed=True, inflight_files=1)
        s = replace(self.snapshot, parents=(bad, self.p2))
        w = replace(self.witness, parents=(bad,))
        self.assert_native(self.choose(snapshot=s, witnesses=(w,)))

    def test_partial_future_not_parent_completion(self):
        bad = replace(self.p1, future_done=True, done_files=1)
        s = replace(self.snapshot, parents=(bad, self.p2))
        w = replace(self.witness, parents=(bad,))
        self.assert_native(self.choose(snapshot=s, witnesses=(w,)))

    def test_many_files_do_not_mean_deep_dependency(self):
        p = replace(self.p1, total_files=1000)
        s = replace(self.snapshot, parents=(p, self.p2))
        w = replace(self.witness, parents=(p,))
        self.assertEqual(self.choose(snapshot=s, witnesses=(w,)).action, "selected")

    def test_depth_bound_and_unknown_depth(self):
        for depth in (4, None):
            p = replace(self.p1, remaining_stages=depth)
            s = replace(self.snapshot, parents=(p, self.p2))
            w = replace(self.witness, parents=(p,))
            self.assert_native(self.choose(snapshot=s, witnesses=(w,)))

    def test_unknown_estimate_not_zero(self):
        self.assert_native(self.choose(witnesses=(replace(self.witness, estimated_unblock_ns=None),)))

    def test_legal_parent_time_includes_all_submitted_work(self):
        self.assert_native(self.choose(witnesses=(replace(self.witness, estimated_unblock_ns=130),)))

    def test_submitted_work_not_reordered(self):
        result = self.choose(works=(self.w2, replace(self.w1, submitted=True)))
        self.assertNotEqual(result.action, "selected")
        self.assertIn("submitted", result.reason)

    def test_native_continuation_mandatory_shutdown_progress(self):
        for progress in ("continuation", "mandatory", "mandatory_support", "shutdown", "age"):
            result = self.choose(works=(self.w2, replace(self.w1, progress=progress)))
            self.assertNotEqual(result.action, "selected")
            self.assertEqual(result.reason, "native_progress_before_policy")

    def test_candidate_victim_cannot_claim_observed_release(self):
        w = replace(self.witness, resource=replace(self.r1, evidence="candidate"))
        self.assert_native(self.choose(witnesses=(w,)))

    def test_clean_reclaim_precedes_writeback(self):
        target = replace(self.target, need_gpu_bytes=0, need_cpu_bytes=16)
        s = replace(self.snapshot, targets=(target,), cpu_clean_reclaimable_bytes=16)
        w = replace(self.witness, resource_kind="cpu")
        result = self.choose(snapshot=s, witnesses=(w,))
        self.assert_native(result)
        self.assertEqual(result.reason, "native_clean_cpu_reclaim_first")

    def test_immediate_gpu_reuse_precedes_writeback(self):
        r = self.choose(snapshot=replace(self.snapshot, gpu_immediately_reusable_bytes=16))
        self.assert_native(r)
        self.assertEqual(r.reason, "native_immediate_gpu_reuse_first")

    def test_duplicate_resources_do_not_double_credit(self):
        s = replace(self.snapshot, targets=(replace(self.target, need_gpu_bytes=32),))
        self.assert_native(self.choose(snapshot=s, witnesses=(self.witness, self.witness)))

    def test_bounded_multi_resource_union_satisfies_deficit(self):
        r2 = Resource(self.id2, 16, 0, frozenset((2,)), "observed_blocking")
        w2 = replace(self.witness, resource=r2, parents=(self.p2,), work_ids=(self.w2.work_id,),
                     estimated_unblock_ns=150)
        s = replace(self.snapshot, targets=(replace(self.target, need_gpu_bytes=32),))
        r = self.choose(snapshot=s, witnesses=(self.witness, w2))
        self.assertEqual(r.action, "selected")
        self.assertEqual(r.gpu_release_credit_bytes, 32)
        self.assertEqual(len(r.closure_resource_ids), 2)

    def test_error_overlap_selects_bytes_in_dependency_only(self):
        r2 = Resource(self.id2, 16, 0, frozenset((2,)), "observed_blocking")
        w1 = replace(self.witness, estimated_unblock_ns=150, error_ns=10, interference_ns=9)
        w2 = replace(w1, resource=r2, parents=(self.p2,), work_ids=(self.w2.work_id,),
                     estimated_unblock_ns=155, interference_ns=2)
        r = self.choose(witnesses=(w1, w2))
        self.assertEqual(r.proposed_work_ids[0], self.w2.work_id)

    def test_dependency_only_does_not_use_interference_signal(self):
        r2 = Resource(self.id2,16,0,frozenset((2,)),"observed_blocking")
        small = replace(self.w1,nbytes=4)
        large = replace(self.w2,nbytes=32)
        w1 = replace(self.witness,work_ids=(small.work_id,),estimated_unblock_ns=150,
                     error_ns=10,interference_ns=1000)
        w2 = replace(w1,resource=r2,parents=(self.p2,),work_ids=(large.work_id,),
                     estimated_unblock_ns=155,interference_ns=0)
        r = self.choose(works=(large,small),witnesses=(w1,w2))
        self.assertEqual(r.selected_work_ids[0],small.work_id)

    def test_unblock_time_precedes_interference_outside_error(self):
        r2 = Resource(self.id2, 16, 0, frozenset((2,)), "observed_blocking")
        w2 = replace(self.witness, resource=r2, parents=(self.p2,),
                     work_ids=(self.w2.work_id,), estimated_unblock_ns=190, interference_ns=0)
        w1 = replace(self.witness, interference_ns=100)
        r = self.choose(witnesses=(w1, w2))
        self.assertEqual(r.proposed_work_ids[0], self.w1.work_id)

    def test_age_keeps_native_target_order_for_normal_waiters(self):
        other = WaitingTarget("other", 1, 99, need_gpu_bytes=16)
        s = replace(self.snapshot, targets=(other, self.target))
        self.assertEqual(self.choose(snapshot=s).target_id, "wait")

    def test_aged_target_prevents_indefinite_yielding(self):
        other = WaitingTarget("other", 1, 0, need_gpu_bytes=16)
        s = replace(self.snapshot, targets=(self.target, other))
        w = replace(self.witness, target_id="other")
        self.assertEqual(self.choose(snapshot=s, witnesses=(w,)).target_id, "other")

    def test_64_work_window_does_not_own_or_drop_native_tail(self):
        works = tuple(replace(self.w1, child_id=i) for i in range(65))
        r = self.choose(works=works)
        self.assertEqual(r.action, "native_fallback")
        self.assertEqual(r.observed_work_count, 64)
        self.assertTrue(r.window_truncated)
        self.assertTrue(r.tail_unchanged)

    def test_32_parent_window_and_8_closure_limit(self):
        with self.assertRaises(ValueError):
            replace(self.snapshot, parents=tuple(replace(self.p1, job_id=i) for i in range(33)))
        self.assert_native(self.choose(witnesses=(self.witness,) * 9))

    def test_interference_joint_require_real_production_qualification(self):
        for mode in ("interference", "joint"):
            p = make_p4_policy("r", P4Config(mode, 10, 100, 20))
            r = self.choose(policy=p)
            self.assert_native(r)
            self.assertEqual(r.reason, "production_interference_gpu_gate_blocked")

    def test_finite_legal_batches_do_not_split_native_descriptors(self):
        p = make_p4_policy("r", P4Config("dependency_only", 10, 100))
        works = tuple(replace(self.w1, child_id=i, nbytes=8, minimum_unit_bytes=8)
                      for i in range(8))
        result = p.choose_batch(self.snapshot, works, now_ns=100, expected_epoch=2)
        self.assertEqual(result.storage_units, 8)
        self.assertEqual(result.physical_bytes, 64)
        goal = p.choose_batch(self.snapshot, works, now_ns=100, expected_epoch=2,
                             target_work_ids=(works[0].work_id,works[1].work_id))
        self.assertEqual(goal.storage_units, 2)
        self.assertEqual(len(goal.work_ids), 2)
        unsplittable = (replace(self.w1, nbytes=24, minimum_unit_bytes=8),)
        self.assertEqual(p.choose_batch(self.snapshot, unsplittable, now_ns=100,
                                       expected_epoch=2).action, "native_fallback")

    def test_batch_mixed_unit_geometry_cannot_change_candidate_meaning(self):
        first = replace(self.w1,nbytes=8,minimum_unit_bytes=8)
        second = replace(self.w2,nbytes=16,minimum_unit_bytes=16)
        result = self.policy.choose_batch(self.snapshot,(first,second),now_ns=100,expected_epoch=2)
        self.assertEqual(result.action,"native_fallback")
        self.assertEqual(result.reason,"mixed_frozen_storage_unit_geometry")

    def test_batch_target_bool_alias_and_duplicate_ids_rejected(self):
        first = replace(self.w1,nbytes=8,minimum_unit_bytes=8)
        with self.assertRaises(ValueError):
            self.policy.choose_batch(self.snapshot,(first,),now_ns=100,expected_epoch=2,
                                     target_work_ids=((True,0,"ssd_write"),))
        with self.assertRaises(ValueError):
            self.policy.choose_batch(self.snapshot,(first,),now_ns=100,expected_epoch=2,
                                     target_work_ids=(first.work_id,first.work_id))

    def test_batch_never_delays_native_progress_or_rebuilds_fused_work(self):
        p = make_p4_policy("r", P4Config("joint", 10, 100, 20))
        result = p.choose_batch(self.snapshot, (replace(self.w1, progress="continuation"),),
                                now_ns=100, expected_epoch=2)
        self.assertEqual(result.action, "native_fallback")
        self.assertEqual(result.reason, "native_progress_batch_before_policy")

    def test_issue_preview_independently_checks_native_selectability(self):
        for work in (replace(self.w1,accepted=False),replace(self.w1,created_ns=101),
                     replace(self.w1,submitted=True)):
            r = self.policy.issue_preview(work,self.snapshot,now_ns=100,expected_epoch=2)
            self.assertEqual(r.action,"native_fallback")
        work = replace(self.w1,resource_identity=self.id1,generation=4)
        s = replace(self.snapshot,generations=((self.id1,5),(self.id2,7)))
        self.assertEqual(self.policy.issue_preview(work,s,now_ns=100,
                                                  expected_epoch=2).action,"native_fallback")

    def test_combined_interference_without_joint_witness_is_unknown(self):
        r2 = Resource(self.id2,16,0,frozenset((2,)),"observed_blocking")
        w2 = replace(self.witness,resource=r2,parents=(self.p2,),work_ids=(self.w2.work_id,),
                     estimated_unblock_ns=150,interference_ns=17)
        c1,_ = self.policy._witness(replace(self.witness,interference_ns=11),
                                   self.snapshot,self.works,self.target,100)
        c2,_ = self.policy._witness(w2,self.snapshot,self.works,self.target,100)
        self.assertIsNone(self.policy._combine((c1,c2),self.works).interference_ns)

    def test_restore_closure_cannot_complete_an_unrelated_write_parent(self):
        target = replace(self.target,need_gpu_bytes=0,restore_parent_id=2)
        s = replace(self.snapshot,targets=(target,))
        w = replace(self.witness,resource_kind="restore",completed_restore_parent_id=2)
        self.assert_native(self.choose(snapshot=s,witnesses=(w,)))

    def test_restore_closure_requires_correct_parent_and_read_h2d_direction(self):
        target = replace(self.target,need_gpu_bytes=0,restore_parent_id=1)
        s = replace(self.snapshot,targets=(target,))
        work = replace(self.w1,stage="h2d")
        w = replace(self.witness,resource_kind="restore",completed_restore_parent_id=1,
                    work_ids=(work.work_id,))
        result = self.choose(snapshot=s,works=(self.w2,work),witnesses=(w,))
        self.assertEqual(result.action,"selected")
        wrong = replace(work,stage="d2h")
        w = replace(w,work_ids=(wrong.work_id,))
        result = self.choose(snapshot=s,works=(self.w2,wrong),witnesses=(w,))
        self.assertNotEqual(result.action,"selected")

    def test_same_shared_controller_consumes_one_per_success(self):
        q = 8
        caps = tuple(Amount(2, q * 2) for _ in range(4))
        c = SimpleStageConfig("fixed", 1_000_000, q, caps, caps,
            Amount(2, q * 2), Amount(2, q * 2), q * 2, q * 2, 0, 32, 100, 1000)
        controller = make_dispatch_controller("r", c)
        controller.bind()
        p = make_p4_policy("r", P4Config("dependency_only", 100, 1000))
        s = replace(self.snapshot, monotonic_ns=99)
        for _ in range(2):
            self.assertEqual(p.issue_preview(self.w2, s, now_ns=100, expected_epoch=2).action, "issue")
            decision = controller.decide("ssd_write", 8, s.native_state, now_ns=100,
                                         work_id=self.w2.work_id)
            controller.accepted(decision.attempt)
        self.assertEqual(controller.used[1], Amount(2, 16))
        self.assertEqual(controller.epoch_refreshes, 1)
        self.assertEqual(controller.decide("ssd_write", 8, s.native_state, now_ns=100,
                                          work_id=self.w2.work_id).action, "defer")
        self.assertFalse(hasattr(p, "used"))
        self.assertFalse(hasattr(p, "pending"))


class CostTests(unittest.TestCase):
    def setUp(self):
        self.sig = ("model-sha", "gpu-uuid", "kernel", 2, 16400, 16)
        self.existing = (Amount(1, 16), Amount(1, 8), Amount(1, 4), Amount(1, 4))
        self.cell = CostCell(self.sig, self.existing, "ssd_read", 16,
                             "existing_io_plus_delta", 10, 3, 2)
        self.table = CostTable((self.cell,), scope="mock_only", source_sha256="a"*64)

    def test_existing_io_delta_is_added_once(self):
        r = self.table.lookup(self.sig, self.existing, "ssd_read", 16, execution="cpu_mock")
        self.assertEqual(r.total_ns, 15)
        self.assertTrue(r.mock_only)
        self.assertFalse(r.production_qualified)

    def test_no_io_joint_is_not_plus_existing_cost_again(self):
        c = replace(self.cell, basis="no_io_plus_joint", baseline_ns=7, incremental_or_joint_ns=6)
        t = CostTable((c,), scope="mock_only", source_sha256="b"*64)
        self.assertEqual(t.lookup(self.sig, self.existing, "ssd_read", 16,
                                 execution="cpu_mock").total_ns, 15)

    def test_exact_hardware_load_io_stage_bytes_matching(self):
        for sig, io, stage, nbytes in (
            (self.sig[:-1] + (8,), self.existing, "ssd_read", 16),
            (self.sig, ZERO, "ssd_read", 16), (self.sig, self.existing, "ssd_write", 16),
            (self.sig, self.existing, "ssd_read", 8)):
            self.assertIsNone(self.table.lookup(sig, io, stage, nbytes, execution="cpu_mock"))

    def test_production_cannot_use_mock_even_when_exact(self):
        self.assertIsNone(self.table.lookup(self.sig, self.existing, "ssd_read", 16))
        with self.assertRaises(AttributeError):
            self.table.production_qualified = True
        with self.assertRaises(ValueError):
            CostTable((self.cell,), scope="production_qualified", source_sha256="a"*64)

    def test_conditional_table_never_authorizes_any_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conditional.json"
            path.write_text('{"status":"PASS","gpu_verified":true,"production_qualified":true}')
            table = load_conditional_table(path)
            self.assertFalse(table.production_qualified)
            self.assertIsNone(table.lookup(self.sig, self.existing, "ssd_read", 16,
                                          execution="cpu_mock"))

    def test_bad_units_negative_boolean_and_duplicate_cell_rejected(self):
        for kwargs in ({"physical_bytes":True}, {"baseline_ns":-1},
                       {"basis":"mixed-double-counted"}, {"uncertainty_ns":True}):
            with self.assertRaises(ValueError):
                replace(self.cell, **kwargs)
        with self.assertRaises(ValueError):
            CostTable((self.cell,self.cell), scope="mock_only", source_sha256="a"*64)

    def test_cpu_mock_preview_gate_and_progress_override(self):
        p = make_p4_policy("r", P4Config("interference", 10, 100, 14), table=self.table)
        work = WorkDescriptor("r", 2, 1, 0, "ssd_read", 16, 0, 99, False)
        snap = SystemSnapshot("r", 2, 99, frozenset(("native_ready_work",)),
            native_state=ShadowState("r",99,self.existing,16,1,True),load_signature=self.sig)
        self.assertEqual(p.issue_preview(work,snap,now_ns=100,expected_epoch=2).action,
                         "native_fallback")
        mock = p.issue_preview(work,snap,now_ns=100,expected_epoch=2,execution="cpu_mock")
        self.assertEqual(mock.action,"defer")
        self.assertEqual(mock.predicted_total_ns,15)
        for reason in ("mandatory", "continuation", "shutdown"):
            r = p.issue_preview(work,snap,now_ns=100,expected_epoch=2,progress=reason,
                                execution="cpu_mock")
            self.assertEqual(r.action,"issue")
            self.assertTrue(r.progress_override)
            self.assertFalse(r.production_qualified)

    def test_finite_mock_batches_obey_exact_joint_cost_and_target(self):
        works = tuple(WorkDescriptor("r",2,1,i,"ssd_read",8,i,99,False,
                                     minimum_unit_bytes=8) for i in range(8))
        cells = tuple(replace(self.cell, physical_bytes=n*8,
                              incremental_or_joint_ns=n*2) for n in (1,2,4,8))
        table = CostTable(cells, scope="mock_only", source_sha256="c"*64)
        p = make_p4_policy("r",P4Config("joint",10,100,20),table=table)
        snap = SystemSnapshot("r",2,99,frozenset(("native_ready_work",)),
            native_state=ShadowState("r",99,self.existing,128,1,True),load_signature=self.sig)
        r = p.choose_batch(snap,works,now_ns=100,expected_epoch=2,execution="cpu_mock")
        self.assertEqual(r.storage_units,4)
        self.assertEqual(r.predicted_total_ns,20)
        self.assertTrue(r.mock_only)
        self.assertFalse(r.production_qualified)
        goal = p.choose_batch(snap,works,now_ns=100,expected_epoch=2,execution="cpu_mock",
                              target_work_ids=(works[0].work_id,))
        self.assertEqual(goal.storage_units,1)
        self.assertEqual(p.choose_batch(snap,works,now_ns=100,
                                       expected_epoch=2).action,"native_fallback")

    def test_unknown_and_stale_baseline_fallback_not_zero(self):
        p = make_p4_policy("r", P4Config("joint", 10, 100, 20), table=self.table)
        work = WorkDescriptor("r",2,1,0,"ssd_read",16,0,99,False)
        snap = SystemSnapshot("r",2,99,frozenset(("native_ready_work",)),
            native_state=ShadowState("r",80,self.existing,16,1,True),load_signature=self.sig)
        self.assertEqual(p.issue_preview(work,snap,now_ns=100,expected_epoch=2,
                                        execution="cpu_mock").action,"native_fallback")


if __name__ == "__main__":
    unittest.main()
