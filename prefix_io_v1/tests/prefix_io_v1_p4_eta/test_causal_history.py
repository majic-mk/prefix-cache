"""Causal closure arithmetic and negative cases; CPU only, no GPU evidence."""
from dataclasses import FrozenInstanceError, replace
import unittest
from prefix_io_control.p4_eta import (
    ClosureGeometry, NativeClosureHistory, CausalEstimate, CompletedClosureMeasurement)

G = ClosureGeometry((("ssd_read", 4096, 4096, 1),))
C = ("actual-native-context-unqualified", 4096, 4096)
class ETATests(unittest.TestCase):
    def history(self, **kw):
        return NativeClosureHistory("r", max_age_ns=kw.get("max_age_ns", 1000000),
                                    minimum_samples=kw.get("minimum_samples", 4))
    def observe(self, h, pid, t, g=G, c=C):
        return h.observe(run_id="r", parent_id=pid, geometry=g, context=c, now_ns=t)
    def finish(self, h, pid, t, **kw):
        return h.finish(run_id="r", parent_id=pid, now_ns=t,
            successful=kw.get("successful",True), drain_known=kw.get("drain_known",True))
    def warm(self,h,n=4):
        for pid in range(1,n+1):
            self.observe(h,pid,pid*100)
            self.finish(h,pid,pid*100+10+pid)
    def estimate(self,h,pid,t,g=G,c=C,execution="shadow_diagnostic"):
        return h.estimate(run_id="r",parent_id=pid,geometry=g,context=c,now_ns=t,execution=execution)

    def test_measurement_is_ready_to_whole_parent_drain(self):
        h=self.history();self.observe(h,1,100);s=self.finish(h,1,127)
        self.assertEqual((s.first_observed_ns,s.completed_ns,s.elapsed_ns),(100,127,27))
        self.assertIsNone(s.prior_forecast_duration_ns)
        self.assertFalse(s.gpu_qualified)
    def test_geometry_and_measurement_values_frozen(self):
        h=self.history();self.observe(h,1,100);s=self.finish(h,1,120)
        with self.assertRaises(FrozenInstanceError):G.rows=()
        with self.assertRaises(FrozenInstanceError):s.elapsed_ns=1
    def test_no_prediction_before_minimum_completed_samples(self):
        h=self.history();self.warm(h,3);self.observe(h,4,400)
        self.assertIsNone(self.estimate(h,4,400))
        self.assertEqual(h.snapshot()["successful_measurements"],3)
    def test_completed_history_has_empirical_error_and_provenance(self):
        h=self.history();self.warm(h);self.observe(h,5,500)
        e=self.estimate(h,5,500)
        self.assertEqual((e.duration_ns,e.empirical_error_ns,e.sample_count),(13,2,4))
        self.assertEqual((e.earliest_completed_ns,e.latest_completed_ns),(111,414))
        self.assertFalse(e.production_qualified)
    def test_current_sample_cannot_train_its_own_forecast(self):
        h=self.history();self.warm(h);self.observe(h,5,500);s=self.finish(h,5,540)
        self.assertEqual((s.prior_forecast_duration_ns,s.prior_forecast_sample_count),(13,4))
        self.assertEqual((s.causal_absolute_residual_ns,s.sequence),(27,5))
    def test_prior_residual_is_counted_only_after_completion(self):
        h=self.history();self.warm(h);self.observe(h,5,500)
        before=self.estimate(h,5,500);self.assertEqual(before.empirical_error_ns,2)
        self.finish(h,5,540);self.observe(h,6,600)
        after=self.estimate(h,6,600);self.assertEqual(after.empirical_error_ns,27)
    def test_duplicate_observation_does_not_reset_age(self):
        h=self.history();self.observe(h,1,100)
        self.assertFalse(self.observe(h,1,119));s=self.finish(h,1,120)
        self.assertEqual(s.elapsed_ns,20)
    def test_changed_context_does_not_reset_first_observation(self):
        h=self.history();self.observe(h,1,100)
        self.assertFalse(self.observe(h,1,110,c=("other",4096)))
        s=self.finish(h,1,130);self.assertEqual(s.context,C);self.assertEqual(s.elapsed_ns,30)
    def test_retired_or_reused_parent_cannot_create_new_sample(self):
        h=self.history();self.observe(h,1,100);self.finish(h,1,120)
        self.assertFalse(self.observe(h,1,130))
        self.assertIsNone(self.finish(h,1,150));self.assertEqual(h.completed,1)
    def test_out_of_order_first_observation_is_conservatively_omitted(self):
        h=self.history();self.observe(h,4,100)
        self.assertFalse(self.observe(h,3,101));self.assertEqual(h.omitted,1)
    def test_failed_parent_is_not_success_timing(self):
        h=self.history();self.observe(h,1,100)
        self.assertIsNone(self.finish(h,1,110,successful=False))
        self.assertEqual((h.completed,h.discarded),(0,1))
    def test_unknown_physical_drain_is_not_success_timing(self):
        h=self.history();self.observe(h,1,100)
        self.assertIsNone(self.finish(h,1,110,drain_known=False))
        self.assertEqual(h.snapshot()["retained_cell_samples"],0)
    def test_zero_interval_is_unknown_not_measured_zero(self):
        h=self.history();self.observe(h,1,100);self.assertIsNone(self.finish(h,1,100))
        self.assertEqual(h.discarded,1)
    def test_clock_reversal_cannot_backdate_completion(self):
        h=self.history();self.observe(h,1,100)
        with self.assertRaisesRegex(ValueError,"backwards"):self.finish(h,1,99)
        self.assertEqual(h.snapshot()["pending_count"],1);self.assertEqual(h.completed,0)
    def test_same_timestamp_completion_is_not_strictly_past_evidence(self):
        h=self.history();self.warm(h);self.observe(h,5,414)
        self.assertIsNone(self.estimate(h,5,414))
    def test_future_history_cannot_leak_into_earlier_forecast(self):
        h=self.history();self.warm(h)
        with self.assertRaisesRegex(ValueError,"backwards"):self.observe(h,5,413)
        self.assertEqual(h.snapshot()["pending_count"],0)
    def test_foreign_run_rejected(self):
        h=self.history()
        with self.assertRaisesRegex(ValueError,"run differs"):
            h.observe(run_id="other",parent_id=1,geometry=G,context=C,now_ns=100)
        with self.assertRaisesRegex(ValueError,"run differs"):
            h.finish(run_id="other",parent_id=1,now_ns=100,successful=True,drain_known=True)
    def test_geometry_mismatch_has_no_compatible_forecast(self):
        h=self.history();self.warm(h);self.observe(h,5,500)
        g=ClosureGeometry((("h2d",4096,4096,1),))
        self.assertIsNone(self.estimate(h,5,500,g=g))
    def test_context_mismatch_has_no_compatible_forecast(self):
        h=self.history();self.warm(h);self.observe(h,5,500)
        self.assertIsNone(self.estimate(h,5,500,c=("different",)))
    def test_no_parent_pending_has_no_forecast(self):
        h=self.history();self.warm(h);self.assertIsNone(self.estimate(h,5,500))
    def test_overdue_point_is_unknown_and_not_immediate_release(self):
        h=self.history();self.warm(h);self.observe(h,5,500)
        self.assertIsNone(self.estimate(h,5,513))
    def test_stale_completed_samples_are_not_current_evidence(self):
        h=self.history(max_age_ns=20);self.warm(h);self.observe(h,5,500)
        self.assertIsNone(self.estimate(h,5,500))
    def test_cpu_history_cannot_qualify_production_even_when_complete(self):
        h=self.history();self.warm(h);self.observe(h,5,500)
        self.assertIsNone(self.estimate(h,5,500,execution="production"))
        self.assertFalse(h.production_qualified)
        with self.assertRaises(AttributeError):h.production_qualified=True
    def test_pending_window_omits_observation_never_work(self):
        h=self.history()
        for pid in range(1,33):self.assertTrue(self.observe(h,pid,pid))
        self.assertFalse(self.observe(h,33,33))
        self.assertEqual(h.snapshot()["pending_count"],32)
        self.assertIsNone(self.finish(h,33,34))
        self.assertEqual(h.snapshot()["pending_count"],32)
    def test_cells_and_samples_are_bounded(self):
        h=self.history();pid=0;t=0
        for cell in range(40):
            for _ in range(20):
                pid+=1;t+=10;self.observe(h,pid,t,c=("cell",cell));self.finish(h,pid,t+1)
        s=h.snapshot()
        self.assertEqual((s["cell_count"],s["retained_cell_samples"]),(32,512))
        self.assertEqual(len(s["recent_measurements"]),64)
    def test_abort_discards_only_scalar_pending_metadata(self):
        h=self.history();self.observe(h,1,100);h.abort_all()
        self.assertEqual((h.snapshot()["pending_count"],h.discarded),(0,1))
        self.assertFalse(h.snapshot()["held_job_or_resource_owners"])
        self.assertEqual(h.snapshot()["new_work_queues"],0)
    def test_bool_times_and_flags_rejected(self):
        h=self.history()
        with self.assertRaises((ValueError,TypeError)):self.observe(h,1,True)
        self.observe(h,1,100)
        with self.assertRaises(ValueError):
            h.finish(run_id="r",parent_id=1,now_ns=110,successful=1,drain_known=True)
    def test_context_rejects_mutable_or_owner_values(self):
        for c in ([],(object(),),({},),("",),tuple(range(17))):
            with self.assertRaises((ValueError,TypeError)):
                self.observe(self.history(),1,100,c=c)
    def test_invalid_geometry_never_redefines_native_quantum(self):
        bad=((),(("d2h",4096,4096,1),),(("ssd_read",4100,4096,1),),
             (("ssd_read",4096,4096,65),),(("ssd_read",4096,4096,1),)*2)
        for rows in bad:
            with self.assertRaises((ValueError,TypeError)):ClosureGeometry(rows)
    def test_unknown_execution_scope_rejected(self):
        h=self.history();self.observe(h,1,100)
        with self.assertRaises(ValueError):self.estimate(h,1,100,execution="allow_gpu")
    def test_duplicate_completion_does_not_double_count(self):
        h=self.history();self.observe(h,1,100);self.finish(h,1,120)
        self.assertIsNone(self.finish(h,1,121));self.assertEqual(h.completed,1)
