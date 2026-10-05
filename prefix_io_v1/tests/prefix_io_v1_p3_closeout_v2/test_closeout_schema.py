"""Pure stdlib schema tests: metadata projections only; no native/model/GPU import."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/prefix_io_v1/scripts/analyze_p3_closeout_p316_v2.py"
spec = importlib.util.spec_from_file_location("p316_closeout_v2_unit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

RECORDED = json.loads("{\"mixed01\":{\"raw\":{\"native_cpu_probe_requested\":true,\"native_metadata_cpu_intervals\":{\"warmup_and_start_boundary\":{\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":8,\"thread_cpu_ns\":56082},\"_prefix_stage_settle\":{\"calls\":16,\"thread_cpu_ns\":51473},\"_prefix_stage\":{\"calls\":24,\"thread_cpu_ns\":297447},\"_prefix_stage_copy\":{\"calls\":8,\"thread_cpu_ns\":459428},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":185722},\"observer_sink\":{\"calls\":333,\"thread_cpu_ns\":1838817}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":92360},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":66265}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}}},\"phase_totals\":{\"warmup\":{\"calls\":390,\"thread_cpu_ns\":2888969},\"cohort\":{\"calls\":2,\"thread_cpu_ns\":158625},\"tail\":{\"calls\":0,\"thread_cpu_ns\":0}},\"clock\":\"thread_time_ns\",\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false},\"cohort_and_tail\":{\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":2067,\"thread_cpu_ns\":8768902},\"_prefix_stage_settle\":{\"calls\":12201,\"thread_cpu_ns\":26689120},\"_prefix_stage\":{\"calls\":21380,\"thread_cpu_ns\":215231035},\"_prefix_stage_copy\":{\"calls\":3022,\"thread_cpu_ns\":122521544},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":166655,\"thread_cpu_ns\":1164003301}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":274464},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":99528}}},\"phase_totals\":{\"warmup\":{\"calls\":0,\"thread_cpu_ns\":0},\"cohort\":{\"calls\":205325,\"thread_cpu_ns\":1537213902},\"tail\":{\"calls\":2,\"thread_cpu_ns\":373992}},\"clock\":\"thread_time_ns\",\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false},\"finish\":{\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}}},\"phase_totals\":{\"warmup\":{\"calls\":0,\"thread_cpu_ns\":0},\"cohort\":{\"calls\":0,\"thread_cpu_ns\":0},\"tail\":{\"calls\":0,\"thread_cpu_ns\":0}},\"clock\":\"thread_time_ns\",\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false},\"scope\":\"outermost metadata/control/observer hook CPU including timing instrumentation; explicit RPC phase boundaries; no whole-model CPU claim\"},\"native_kv\":{\"native_metadata_cpu\":{\"schema_version\":1,\"clock\":\"thread_time_ns\",\"valid\":true,\"error\":null,\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":94938},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":32583}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}}},\"phase_totals\":{\"warmup\":{\"calls\":2,\"thread_cpu_ns\":127521},\"cohort\":{\"calls\":0,\"thread_cpu_ns\":0},\"tail\":{\"calls\":0,\"thread_cpu_ns\":0}},\"publication_calls\":1,\"publication_thread_cpu_ns\":289126,\"scope\":\"bounded pure metadata/control/observer hooks; includes timer overhead\",\"excludes\":[\"native GPU/IO launch and wait\",\"model execution CPU\",\"frontend/core whole-process CPU\",\"unwrapped cache bookkeeping\"],\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false}},\"cohort_probe_start\":{\"native_metadata_cpu\":{\"schema_version\":1,\"clock\":\"thread_time_ns\",\"valid\":true,\"error\":null,\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":8,\"thread_cpu_ns\":56082},\"_prefix_stage_settle\":{\"calls\":16,\"thread_cpu_ns\":51473},\"_prefix_stage\":{\"calls\":24,\"thread_cpu_ns\":297447},\"_prefix_stage_copy\":{\"calls\":8,\"thread_cpu_ns\":459428},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":2,\"thread_cpu_ns\":280660},\"observer_sink\":{\"calls\":334,\"thread_cpu_ns\":1871400}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":92360},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":66265}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}}},\"phase_totals\":{\"warmup\":{\"calls\":392,\"thread_cpu_ns\":3016490},\"cohort\":{\"calls\":2,\"thread_cpu_ns\":158625},\"tail\":{\"calls\":0,\"thread_cpu_ns\":0}},\"publication_calls\":3,\"publication_thread_cpu_ns\":500855,\"scope\":\"bounded pure metadata/control/observer hooks; includes timer overhead\",\"excludes\":[\"native GPU/IO launch and wait\",\"model execution CPU\",\"frontend/core whole-process CPU\",\"unwrapped cache bookkeeping\"],\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false}},\"probe\":{\"native_metadata_cpu\":{\"schema_version\":1,\"clock\":\"thread_time_ns\",\"valid\":true,\"error\":null,\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":8,\"thread_cpu_ns\":56082},\"_prefix_stage_settle\":{\"calls\":16,\"thread_cpu_ns\":51473},\"_prefix_stage\":{\"calls\":24,\"thread_cpu_ns\":297447},\"_prefix_stage_copy\":{\"calls\":8,\"thread_cpu_ns\":459428},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":2,\"thread_cpu_ns\":280660},\"observer_sink\":{\"calls\":334,\"thread_cpu_ns\":1871400}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":2067,\"thread_cpu_ns\":8768902},\"_prefix_stage_settle\":{\"calls\":12201,\"thread_cpu_ns\":26689120},\"_prefix_stage\":{\"calls\":21380,\"thread_cpu_ns\":215231035},\"_prefix_stage_copy\":{\"calls\":3022,\"thread_cpu_ns\":122521544},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":92360},\"observer_sink\":{\"calls\":166656,\"thread_cpu_ns\":1164069566}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":274464},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":99528}}},\"phase_totals\":{\"warmup\":{\"calls\":392,\"thread_cpu_ns\":3016490},\"cohort\":{\"calls\":205327,\"thread_cpu_ns\":1537372527},\"tail\":{\"calls\":2,\"thread_cpu_ns\":373992}},\"publication_calls\":4,\"publication_thread_cpu_ns\":696571,\"scope\":\"bounded pure metadata/control/observer hooks; includes timer overhead\",\"excludes\":[\"native GPU/IO launch and wait\",\"model execution CPU\",\"frontend/core whole-process CPU\",\"unwrapped cache bookkeeping\"],\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false}},\"final_probe\":{\"native_metadata_cpu\":{\"schema_version\":1,\"clock\":\"thread_time_ns\",\"valid\":true,\"error\":null,\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":8,\"thread_cpu_ns\":56082},\"_prefix_stage_settle\":{\"calls\":16,\"thread_cpu_ns\":51473},\"_prefix_stage\":{\"calls\":24,\"thread_cpu_ns\":297447},\"_prefix_stage_copy\":{\"calls\":8,\"thread_cpu_ns\":459428},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":2,\"thread_cpu_ns\":280660},\"observer_sink\":{\"calls\":334,\"thread_cpu_ns\":1871400}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":2067,\"thread_cpu_ns\":8768902},\"_prefix_stage_settle\":{\"calls\":12201,\"thread_cpu_ns\":26689120},\"_prefix_stage\":{\"calls\":21380,\"thread_cpu_ns\":215231035},\"_prefix_stage_copy\":{\"calls\":3022,\"thread_cpu_ns\":122521544},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":92360},\"observer_sink\":{\"calls\":166656,\"thread_cpu_ns\":1164069566}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":274464},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":99528}}},\"phase_totals\":{\"warmup\":{\"calls\":392,\"thread_cpu_ns\":3016490},\"cohort\":{\"calls\":205327,\"thread_cpu_ns\":1537372527},\"tail\":{\"calls\":2,\"thread_cpu_ns\":373992}},\"publication_calls\":5,\"publication_thread_cpu_ns\":813050,\"scope\":\"bounded pure metadata/control/observer hooks; includes timer overhead\",\"excludes\":[\"native GPU/IO launch and wait\",\"model execution CPU\",\"frontend/core whole-process CPU\",\"unwrapped cache bookkeeping\"],\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false}}},\"analysis\":{\"native_metadata_cpu\":{\"intervals\":{\"warmup_and_start_boundary\":{\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":8,\"thread_cpu_ns\":56082},\"_prefix_stage_settle\":{\"calls\":16,\"thread_cpu_ns\":51473},\"_prefix_stage\":{\"calls\":24,\"thread_cpu_ns\":297447},\"_prefix_stage_copy\":{\"calls\":8,\"thread_cpu_ns\":459428},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":185722},\"observer_sink\":{\"calls\":333,\"thread_cpu_ns\":1838817}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":92360},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":66265}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}}},\"phase_totals\":{\"warmup\":{\"calls\":390,\"thread_cpu_ns\":2888969},\"cohort\":{\"calls\":2,\"thread_cpu_ns\":158625},\"tail\":{\"calls\":0,\"thread_cpu_ns\":0}},\"clock\":\"thread_time_ns\",\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false},\"cohort_and_tail\":{\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":2067,\"thread_cpu_ns\":8768902},\"_prefix_stage_settle\":{\"calls\":12201,\"thread_cpu_ns\":26689120},\"_prefix_stage\":{\"calls\":21380,\"thread_cpu_ns\":215231035},\"_prefix_stage_copy\":{\"calls\":3022,\"thread_cpu_ns\":122521544},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":166655,\"thread_cpu_ns\":1164003301}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":274464},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":99528}}},\"phase_totals\":{\"warmup\":{\"calls\":0,\"thread_cpu_ns\":0},\"cohort\":{\"calls\":205325,\"thread_cpu_ns\":1537213902},\"tail\":{\"calls\":2,\"thread_cpu_ns\":373992}},\"clock\":\"thread_time_ns\",\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false},\"finish\":{\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}}},\"phase_totals\":{\"warmup\":{\"calls\":0,\"thread_cpu_ns\":0},\"cohort\":{\"calls\":0,\"thread_cpu_ns\":0},\"tail\":{\"calls\":0,\"thread_cpu_ns\":0}},\"clock\":\"thread_time_ns\",\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false}},\"cohort_and_tail_thread_cpu_seconds\":1.537587894,\"final_cumulative\":{\"schema_version\":1,\"clock\":\"thread_time_ns\",\"valid\":true,\"error\":null,\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":8,\"thread_cpu_ns\":56082},\"_prefix_stage_settle\":{\"calls\":16,\"thread_cpu_ns\":51473},\"_prefix_stage\":{\"calls\":24,\"thread_cpu_ns\":297447},\"_prefix_stage_copy\":{\"calls\":8,\"thread_cpu_ns\":459428},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":2,\"thread_cpu_ns\":280660},\"observer_sink\":{\"calls\":334,\"thread_cpu_ns\":1871400}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":2067,\"thread_cpu_ns\":8768902},\"_prefix_stage_settle\":{\"calls\":12201,\"thread_cpu_ns\":26689120},\"_prefix_stage\":{\"calls\":21380,\"thread_cpu_ns\":215231035},\"_prefix_stage_copy\":{\"calls\":3022,\"thread_cpu_ns\":122521544},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":92360},\"observer_sink\":{\"calls\":166656,\"thread_cpu_ns\":1164069566}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":274464},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":99528}}},\"phase_totals\":{\"warmup\":{\"calls\":392,\"thread_cpu_ns\":3016490},\"cohort\":{\"calls\":205327,\"thread_cpu_ns\":1537372527},\"tail\":{\"calls\":2,\"thread_cpu_ns\":373992}},\"publication_calls\":5,\"publication_thread_cpu_ns\":813050,\"scope\":\"bounded pure metadata/control/observer hooks; includes timer overhead\",\"excludes\":[\"native GPU/IO launch and wait\",\"model execution CPU\",\"frontend/core whole-process CPU\",\"unwrapped cache bookkeeping\"],\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false},\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false,\"scope\":\"pure metadata/control/observer hook thread CPU; timer overhead included; native/backend/model CPU excluded\"}},\"provenance\":{\"label\":\"server08-p3-16-mixed-01-off\",\"result\":{\"path\":\"experiments/prefix_io_v1/runs/server08-p3-16-mixed-01-off/details/result.json\",\"sha256\":\"b9dea3b9726b57a8f5d9ea28de9124af69d15567c446f6b94387b0e724cb8dff\"},\"analysis\":{\"path\":\"artifacts/prefix_io_v1/server08-p3-16/mixed-01-off-analysis.json\",\"sha256\":\"2f445de051e81dc219bd5e3ec8e576519ba5945704170e2b2799b4bbb33f93e7\"}}},\"observer02\":{\"raw\":{\"native_cpu_probe_requested\":true,\"native_metadata_cpu_intervals\":{\"warmup_and_start_boundary\":{\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":15,\"thread_cpu_ns\":82748},\"_prefix_stage_settle\":{\"calls\":1671,\"thread_cpu_ns\":3047576},\"_prefix_stage\":{\"calls\":2702,\"thread_cpu_ns\":21824067},\"_prefix_stage_copy\":{\"calls\":640,\"thread_cpu_ns\":17501057},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":2,\"thread_cpu_ns\":237544},\"observer_sink\":{\"calls\":3135,\"thread_cpu_ns\":18663820}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":61717},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":54963}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}}},\"phase_totals\":{\"warmup\":{\"calls\":8165,\"thread_cpu_ns\":61356812},\"cohort\":{\"calls\":2,\"thread_cpu_ns\":116680},\"tail\":{\"calls\":0,\"thread_cpu_ns\":0}},\"clock\":\"thread_time_ns\",\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false},\"cohort_and_tail\":{\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":23,\"thread_cpu_ns\":2279125}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":250334},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":83731}}},\"phase_totals\":{\"warmup\":{\"calls\":0,\"thread_cpu_ns\":0},\"cohort\":{\"calls\":23,\"thread_cpu_ns\":2279125},\"tail\":{\"calls\":2,\"thread_cpu_ns\":334065}},\"clock\":\"thread_time_ns\",\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false},\"finish\":{\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}}},\"phase_totals\":{\"warmup\":{\"calls\":0,\"thread_cpu_ns\":0},\"cohort\":{\"calls\":0,\"thread_cpu_ns\":0},\"tail\":{\"calls\":0,\"thread_cpu_ns\":0}},\"clock\":\"thread_time_ns\",\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false},\"scope\":\"outermost metadata/control/observer hook CPU including timing instrumentation; explicit RPC phase boundaries; no whole-model CPU claim\"},\"native_kv\":{\"native_metadata_cpu\":{\"schema_version\":1,\"clock\":\"thread_time_ns\",\"valid\":true,\"error\":null,\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":238458},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":82572}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}}},\"phase_totals\":{\"warmup\":{\"calls\":2,\"thread_cpu_ns\":321030},\"cohort\":{\"calls\":0,\"thread_cpu_ns\":0},\"tail\":{\"calls\":0,\"thread_cpu_ns\":0}},\"publication_calls\":1,\"publication_thread_cpu_ns\":274662,\"scope\":\"bounded pure metadata/control/observer hooks; includes timer overhead\",\"excludes\":[\"native GPU/IO launch and wait\",\"model execution CPU\",\"frontend/core whole-process CPU\",\"unwrapped cache bookkeeping\"],\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false}},\"cohort_probe_start\":{\"native_metadata_cpu\":{\"schema_version\":1,\"clock\":\"thread_time_ns\",\"valid\":true,\"error\":null,\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":15,\"thread_cpu_ns\":82748},\"_prefix_stage_settle\":{\"calls\":1671,\"thread_cpu_ns\":3047576},\"_prefix_stage\":{\"calls\":2702,\"thread_cpu_ns\":21824067},\"_prefix_stage_copy\":{\"calls\":640,\"thread_cpu_ns\":17501057},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":3,\"thread_cpu_ns\":476002},\"observer_sink\":{\"calls\":3136,\"thread_cpu_ns\":18746392}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":61717},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":54963}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}}},\"phase_totals\":{\"warmup\":{\"calls\":8167,\"thread_cpu_ns\":61677842},\"cohort\":{\"calls\":2,\"thread_cpu_ns\":116680},\"tail\":{\"calls\":0,\"thread_cpu_ns\":0}},\"publication_calls\":4,\"publication_thread_cpu_ns\":577427,\"scope\":\"bounded pure metadata/control/observer hooks; includes timer overhead\",\"excludes\":[\"native GPU/IO launch and wait\",\"model execution CPU\",\"frontend/core whole-process CPU\",\"unwrapped cache bookkeeping\"],\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false}},\"probe\":{\"native_metadata_cpu\":{\"schema_version\":1,\"clock\":\"thread_time_ns\",\"valid\":true,\"error\":null,\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":15,\"thread_cpu_ns\":82748},\"_prefix_stage_settle\":{\"calls\":1671,\"thread_cpu_ns\":3047576},\"_prefix_stage\":{\"calls\":2702,\"thread_cpu_ns\":21824067},\"_prefix_stage_copy\":{\"calls\":640,\"thread_cpu_ns\":17501057},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":3,\"thread_cpu_ns\":476002},\"observer_sink\":{\"calls\":3136,\"thread_cpu_ns\":18746392}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":61717},\"observer_sink\":{\"calls\":24,\"thread_cpu_ns\":2334088}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":250334},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":83731}}},\"phase_totals\":{\"warmup\":{\"calls\":8167,\"thread_cpu_ns\":61677842},\"cohort\":{\"calls\":25,\"thread_cpu_ns\":2395805},\"tail\":{\"calls\":2,\"thread_cpu_ns\":334065}},\"publication_calls\":5,\"publication_thread_cpu_ns\":698862,\"scope\":\"bounded pure metadata/control/observer hooks; includes timer overhead\",\"excludes\":[\"native GPU/IO launch and wait\",\"model execution CPU\",\"frontend/core whole-process CPU\",\"unwrapped cache bookkeeping\"],\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false}},\"final_probe\":{\"native_metadata_cpu\":{\"schema_version\":1,\"clock\":\"thread_time_ns\",\"valid\":true,\"error\":null,\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":15,\"thread_cpu_ns\":82748},\"_prefix_stage_settle\":{\"calls\":1671,\"thread_cpu_ns\":3047576},\"_prefix_stage\":{\"calls\":2702,\"thread_cpu_ns\":21824067},\"_prefix_stage_copy\":{\"calls\":640,\"thread_cpu_ns\":17501057},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":3,\"thread_cpu_ns\":476002},\"observer_sink\":{\"calls\":3136,\"thread_cpu_ns\":18746392}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":61717},\"observer_sink\":{\"calls\":24,\"thread_cpu_ns\":2334088}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":250334},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":83731}}},\"phase_totals\":{\"warmup\":{\"calls\":8167,\"thread_cpu_ns\":61677842},\"cohort\":{\"calls\":25,\"thread_cpu_ns\":2395805},\"tail\":{\"calls\":2,\"thread_cpu_ns\":334065}},\"publication_calls\":6,\"publication_thread_cpu_ns\":812567,\"scope\":\"bounded pure metadata/control/observer hooks; includes timer overhead\",\"excludes\":[\"native GPU/IO launch and wait\",\"model execution CPU\",\"frontend/core whole-process CPU\",\"unwrapped cache bookkeeping\"],\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false}}},\"analysis\":{\"native_metadata_cpu\":{\"intervals\":{\"warmup_and_start_boundary\":{\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":15,\"thread_cpu_ns\":82748},\"_prefix_stage_settle\":{\"calls\":1671,\"thread_cpu_ns\":3047576},\"_prefix_stage\":{\"calls\":2702,\"thread_cpu_ns\":21824067},\"_prefix_stage_copy\":{\"calls\":640,\"thread_cpu_ns\":17501057},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":2,\"thread_cpu_ns\":237544},\"observer_sink\":{\"calls\":3135,\"thread_cpu_ns\":18663820}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":61717},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":54963}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}}},\"phase_totals\":{\"warmup\":{\"calls\":8165,\"thread_cpu_ns\":61356812},\"cohort\":{\"calls\":2,\"thread_cpu_ns\":116680},\"tail\":{\"calls\":0,\"thread_cpu_ns\":0}},\"clock\":\"thread_time_ns\",\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false},\"cohort_and_tail\":{\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":23,\"thread_cpu_ns\":2279125}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":250334},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":83731}}},\"phase_totals\":{\"warmup\":{\"calls\":0,\"thread_cpu_ns\":0},\"cohort\":{\"calls\":23,\"thread_cpu_ns\":2279125},\"tail\":{\"calls\":2,\"thread_cpu_ns\":334065}},\"clock\":\"thread_time_ns\",\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false},\"finish\":{\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"observer_sink\":{\"calls\":0,\"thread_cpu_ns\":0}}},\"phase_totals\":{\"warmup\":{\"calls\":0,\"thread_cpu_ns\":0},\"cohort\":{\"calls\":0,\"thread_cpu_ns\":0},\"tail\":{\"calls\":0,\"thread_cpu_ns\":0}},\"clock\":\"thread_time_ns\",\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false}},\"cohort_and_tail_thread_cpu_seconds\":0.00261319,\"final_cumulative\":{\"schema_version\":1,\"clock\":\"thread_time_ns\",\"valid\":true,\"error\":null,\"counting\":\"outermost_only; nested inclusive hook costs are never added twice\",\"phase_regions\":{\"warmup\":{\"_prefix_stage_decide\":{\"calls\":15,\"thread_cpu_ns\":82748},\"_prefix_stage_settle\":{\"calls\":1671,\"thread_cpu_ns\":3047576},\"_prefix_stage\":{\"calls\":2702,\"thread_cpu_ns\":21824067},\"_prefix_stage_copy\":{\"calls\":640,\"thread_cpu_ns\":17501057},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":3,\"thread_cpu_ns\":476002},\"observer_sink\":{\"calls\":3136,\"thread_cpu_ns\":18746392}},\"cohort\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":61717},\"observer_sink\":{\"calls\":24,\"thread_cpu_ns\":2334088}},\"tail\":{\"_prefix_stage_decide\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_settle\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_stage_copy\":{\"calls\":0,\"thread_cpu_ns\":0},\"_prefix_capacity_snapshot\":{\"calls\":0,\"thread_cpu_ns\":0},\"_capture_owner_snapshot\":{\"calls\":1,\"thread_cpu_ns\":250334},\"observer_sink\":{\"calls\":1,\"thread_cpu_ns\":83731}}},\"phase_totals\":{\"warmup\":{\"calls\":8167,\"thread_cpu_ns\":61677842},\"cohort\":{\"calls\":25,\"thread_cpu_ns\":2395805},\"tail\":{\"calls\":2,\"thread_cpu_ns\":334065}},\"publication_calls\":6,\"publication_thread_cpu_ns\":812567,\"scope\":\"bounded pure metadata/control/observer hooks; includes timer overhead\",\"excludes\":[\"native GPU/IO launch and wait\",\"model execution CPU\",\"frontend/core whole-process CPU\",\"unwrapped cache bookkeeping\"],\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false},\"algorithm_cpu_claim\":false,\"full_production_cpu_breakdown\":false,\"scope\":\"pure metadata/control/observer hook thread CPU; timer overhead included; native/backend/model CPU excluded\"}},\"provenance\":{\"label\":\"server08-p3-16-observer-02-on\",\"result\":{\"path\":\"experiments/prefix_io_v1/runs/server08-p3-16-observer-02-on/details/result.json\",\"sha256\":\"d4a19c9f2ec1d0967d94a041a9f950c2a70c738aede3ed6188ab07e2bbc28267\"},\"analysis\":{\"path\":\"artifacts/prefix_io_v1/server08-p3-16/observer-02-on-analysis.json\",\"sha256\":\"c37736f1faac1ccc57ea0133aa9617c60536755b0f5ed44b7b14690e4fcb2556\"}}}}")
NATIVE_HASHES = json.loads("{\"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/__init__.py\":\"27a469e4bd7ea7ae62ca91486307a2a3b4bb2906b140917ca63b1e7b5de08813\",\"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/_pchip.py\":\"d3f186b52ae296181ae3e547cb8516ba1538a20062bb76152c6042b13ff5d397\",\"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/break_even.py\":\"2419c9470df0e43558ccc729aa3927ebd4d1381e00276d4caaa57b3cf19c42a6\",\"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/cost_model.py\":\"e85922054fe96dce5c0221f1cc17cc3f1b2d7addd7eae452cabad305fc412397\",\"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/file_mapper.py\":\"165239153b9bf1b475992358683a8d43849d74f0e841326704e1c419abda72dc\",\"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/fs_config.py\":\"7a2429d15f6d6009da5f7965ba6b15279a9ef25b348cadd29927f61b9c488698\",\"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/liburing_file.py\":\"6a8995ca6e5f49ae470e8c49dbf3d98caf2d09dd091cb6f08f9e892c051414f5\",\"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/load_planner.py\":\"92581e373d7e5c0903520417fb0ce80141f2f65506bf16c7ecbde94ab1ca5fe2\",\"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/profiling.py\":\"b6025d8afb9256be2a2d35df01e4bf96f3926a2948ecc88a32c2c63336b333d3\",\"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py\":\"5c5129bd75313d86a1012794cfda8096115818e4e806775427ee1f7bed9c6ca2\",\"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/staging.py\":\"18d2b634b894fec32d7abbf767132afbd4fc5ec66fef738ffef724f3e95e615b\",\"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/staging_cache.py\":\"f505da0a942dfded2019fd5b292a986ddb7a29e18bb47c568f9a614eb2a0d928\",\"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/transfer.py\":\"1dcc5f4370e3db694d34924038a8fdd5a2eb12cf7f04ba81072dc969f275c1ba\",\"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/vllm.py\":\"e88732f3165f8421263f6c3bc78448e225cccb8ee3a7cdef690d78dee0a08f3d\",\"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/linux_aio.py\":\"0a987479722c7d6b520b99c9b283febb55c5145fc17c5623b8b1151c4f45a6d7\",\"third_party/work/py-kvcache-p3-16-cpu/tests/__init__.py\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\",\"third_party/work/py-kvcache-p3-16-cpu/tests/test_break_even.py\":\"c94b2dc35ff334fc4d9d2db00131b0758adde2f24730a0a679b5dddb0f9ddae4\",\"third_party/work/py-kvcache-p3-16-cpu/tests/test_cost_model.py\":\"87b8c81d9922529d5e8fe009c15e8be59e7b25f26240e7c32733bedfdda7cf8b\",\"third_party/work/py-kvcache-p3-16-cpu/tests/test_e2e_kvcache.py\":\"ca398167b82c20267c3836088eb50d5e05ce38999b73414184d92537b412b1d8\",\"third_party/work/py-kvcache-p3-16-cpu/tests/test_fs_layout.py\":\"10dd8ead816b59733e40f87238d958d57643332fa065431229596efd6ecf9466\",\"third_party/work/py-kvcache-p3-16-cpu/tests/test_load_planner.py\":\"78390041c1d5d2de4f7a8da6d067bf6fca580c0d39e7f0c6a990a0efe6e30c19\",\"third_party/work/py-kvcache-p3-16-cpu/tests/test_planner_manager.py\":\"f1cbcdf6a10d40f3b8e91b20f192288b2069852df54f1ba90450dcfb0517116e\",\"third_party/work/py-kvcache-p3-16-cpu/tests/test_py_kvcache_async_open.py\":\"d9caf9364f9c2e5425ffcdbcf21ed6bd9910e0056938b8c9e87d6c27483144c3\",\"third_party/work/py-kvcache-p3-16-cpu/tests/test_py_kvcache_copy_batch.py\":\"5c78c5458152f06580e06bcc067b564a52cfb95c2c77eeb9dd5e4e15285fa408\",\"third_party/work/py-kvcache-p3-16-cpu/tests/test_py_kvcache_lookup_cache.py\":\"e05c4c4c194b2633bcf12da429a66183c91dba4c0f6aa8bf1da2f3dad16e3b20\",\"third_party/work/py-kvcache-p3-16-cpu/tests/test_py_kvcache_ntensor.py\":\"f04017c754cd7079a54e4a71c1e6085edb6db7b2d7bc28f7f7cd173b3143d10c\",\"third_party/work/py-kvcache-p3-16-cpu/tests/test_py_kvcache_preload_handler.py\":\"3cdc25ae53ded51007a1518db4f753976b7576aceea9de89bc14f0d4afc38927\",\"third_party/work/py-kvcache-p3-16-cpu/tests/test_py_kvcache_preload_reactor.py\":\"fac1051cbf47371c29849d1d3e5fcc937531a188783afd2534348bdbaed031e6\",\"third_party/work/py-kvcache-p3-16-cpu/tests/test_py_kvcache_shared_preload.py\":\"a24f5059b767325e94e5bcf0558239be930687790f38c8a99c50319b012aa045\",\"third_party/work/py-kvcache-p3-16-cpu/tests/test_staging_cache.py\":\"a2574dda4fab661092d05ba9332f93277240a29f8fd21c94fe083f509a264dd4\",\"third_party/work/py-kvcache-p3-16-cpu/tests/test_vllm_compat.py\":\"9a73c5cca1977b8f8f5509fc564fafe469eca7e21bc739eb00158ac01a77e742\"}")
FROZEN_V1_SHA = "648265d5ca9a94188d175b3ea51ca65c0e333a0567d6141203db4bdf277513d7"


def cpu_pair(label="mixed01"):
    item = copy.deepcopy(RECORDED[label])
    return item["raw"], item["analysis"]


def phase_totals(point):
    for phase, rows in point["phase_regions"].items():
        point["phase_totals"][phase] = {key: sum(r[key] for r in rows.values())
                                       for key in ("calls", "thread_cpu_ns")}


def domain(name="cap1024-l2", gib=1):
    if name == "cap960-l1":
        return dict(capacity_domain_id=name, staging_bytes=1006632960, staging_mem_gib=.9375,
                    preload_lookahead_requests=1, io_depth=8, slot_count=1097,
                    cache_preload_ceiling_slots=1089)
    return dict(capacity_domain_id=name, staging_bytes=1073741824, staging_mem_gib=gib,
                preload_lookahead_requests=2, io_depth=8, slot_count=1170,
                cache_preload_ceiling_slots=1162)


class CPUProjectionTests(unittest.TestCase):
    def test_real_mixed_cpu_canonical_passes(self):
        raw, analysis = cpu_pair()
        self.assertEqual(audit.cpu_measurement(raw, analysis), 1.537587894)
        self.assertEqual(RECORDED["mixed01"]["provenance"]["label"], "server08-p3-16-mixed-01-off")

    def test_real_quiet_cpu_passes_with_zero_finish_and_nonzero_publication(self):
        raw, analysis = cpu_pair("observer02")
        self.assertEqual(audit.cpu_measurement(raw, analysis), .00261319)
        self.assertEqual(raw["native_metadata_cpu_intervals"]["finish"]["phase_totals"]["tail"]["thread_cpu_ns"], 0)
        self.assertGreater(raw["final_probe"]["native_metadata_cpu"]["publication_calls"],
                           raw["probe"]["native_metadata_cpu"]["publication_calls"])

    def test_raw_interval_envelope_missing_or_extra_rejected(self):
        for change in ("missing", "extra"):
            with self.subTest(change=change):
                raw, analysis = cpu_pair()
                if change == "missing":
                    del raw["native_metadata_cpu_intervals"]["scope"]
                else:
                    raw["native_metadata_cpu_intervals"]["unknown"] = {}
                with self.assertRaises(audit.AuditError):
                    audit.cpu_measurement(raw, analysis)

    def test_raw_and_analysis_scope_changes_rejected(self):
        for where in ("raw", "analysis", "point"):
            with self.subTest(where=where):
                raw, analysis = cpu_pair()
                if where == "raw":
                    raw["native_metadata_cpu_intervals"]["scope"] = "whole model CPU"
                elif where == "analysis":
                    analysis["native_metadata_cpu"]["scope"] = "whole model CPU"
                else:
                    raw["probe"]["native_metadata_cpu"]["scope"] = "whole model CPU"
                with self.assertRaises(audit.AuditError):
                    audit.cpu_measurement(raw, analysis)

    def test_analysis_cannot_include_raw_wrapper_scope(self):
        raw, analysis = cpu_pair()
        analysis["native_metadata_cpu"]["intervals"]["scope"] = raw["native_metadata_cpu_intervals"]["scope"]
        with self.assertRaises(audit.AuditError):
            audit.cpu_measurement(raw, analysis)

    def test_unknown_or_invalid_snapshot_is_not_zero(self):
        for value in (None, False):
            with self.subTest(value=value):
                raw, analysis = cpu_pair()
                raw["probe"]["native_metadata_cpu"]["valid"] = value
                with self.assertRaises(audit.AuditError):
                    audit.cpu_measurement(raw, analysis)

    def test_bool_counter_rejected_even_when_equal_to_zero(self):
        raw, analysis = cpu_pair()
        raw["native_kv"]["native_metadata_cpu"]["phase_regions"]["cohort"]["_prefix_stage"]["calls"] = False
        with self.assertRaises(audit.AuditError):
            audit.cpu_measurement(raw, analysis)

    def test_per_region_cancelling_tamper_rejected_even_with_unchanged_totals(self):
        raw, analysis = cpu_pair()
        rows = raw["native_metadata_cpu_intervals"]["cohort_and_tail"]["phase_regions"]["cohort"]
        rows["_prefix_stage"]["thread_cpu_ns"] += 1
        rows["observer_sink"]["thread_cpu_ns"] -= 1
        analysis["native_metadata_cpu"]["intervals"] = {
            k: copy.deepcopy(v) for k, v in raw["native_metadata_cpu_intervals"].items() if k != "scope"}
        with self.assertRaises(audit.AuditError):
            audit.cpu_measurement(raw, analysis)

    def test_region_counter_backwards_rejected_despite_self_consistent_totals(self):
        raw, analysis = cpu_pair()
        p = raw["cohort_probe_start"]["native_metadata_cpu"]
        p["phase_regions"]["warmup"]["_capture_owner_snapshot"]["calls"] = 0
        phase_totals(p)
        with self.assertRaises(audit.AuditError):
            audit.cpu_measurement(raw, analysis)

    def test_wrong_cpu_seconds_or_nonfinite_rejected(self):
        for value in (1.537587893 + .1, float("nan")):
            with self.subTest(value=value):
                raw, analysis = cpu_pair()
                analysis["native_metadata_cpu"]["cohort_and_tail_thread_cpu_seconds"] = value
                with self.assertRaises(audit.AuditError):
                    audit.cpu_measurement(raw, analysis)

    def test_publication_not_refunded_or_double_counted(self):
        raw, analysis = cpu_pair()
        final = raw["final_probe"]["native_metadata_cpu"]
        final["publication_thread_cpu_ns"] = raw["probe"]["native_metadata_cpu"]["publication_thread_cpu_ns"] - 1
        analysis["native_metadata_cpu"]["final_cumulative"] = copy.deepcopy(final)
        with self.assertRaises(audit.AuditError):
            audit.cpu_measurement(raw, analysis)
        raw, analysis = cpu_pair()
        analysis["native_metadata_cpu"]["cohort_and_tail_thread_cpu_seconds"] += (
            raw["probe"]["native_metadata_cpu"]["publication_thread_cpu_ns"] -
            raw["cohort_probe_start"]["native_metadata_cpu"]["publication_thread_cpu_ns"]) / 1e9
        with self.assertRaises(audit.AuditError):
            audit.cpu_measurement(raw, analysis)

    def test_claim_or_clock_or_missing_region_rejected(self):
        for change in ("claim", "clock", "region", "final"):
            with self.subTest(change=change):
                raw, analysis = cpu_pair()
                if change == "claim":
                    analysis["native_metadata_cpu"]["algorithm_cpu_claim"] = True
                elif change == "clock":
                    raw["probe"]["native_metadata_cpu"]["clock"] = "process_time_ns"
                elif change == "region":
                    del raw["probe"]["native_metadata_cpu"]["phase_regions"]["cohort"]["observer_sink"]
                else:
                    analysis["native_metadata_cpu"]["final_cumulative"]["publication_calls"] += 1
                with self.assertRaises(audit.AuditError):
                    audit.cpu_measurement(raw, analysis)


class CapacityDomainTests(unittest.TestCase):
    def test_registered_gib_int_and_float_equivalent_only_for_gib(self):
        self.assertTrue(audit.same_capacity_domain(domain(gib=1), domain(gib=1.0)))
        self.assertEqual(audit.normalized_capacity_domain(domain())["staging_mem_gib"], 1.0)
        self.assertEqual(audit.normalized_capacity_domain(domain("cap960-l1"))["staging_bytes"], 1006632960)

    def test_bool_nonfinite_and_unknown_gib_rejected(self):
        for value in (True, None, float("nan"), float("inf"), "1", 1.00000000000001):
            with self.subTest(value=value), self.assertRaises(audit.AuditError):
                audit.normalized_capacity_domain(domain(gib=value))

    def test_physical_integer_types_and_values_are_never_coerced(self):
        for key, value in (("staging_bytes", 1073741824.0), ("io_depth", 8.0),
                           ("slot_count", 1170.0), ("preload_lookahead_requests", True),
                           ("cache_preload_ceiling_slots", 1161), ("staging_bytes", 1073741823)):
            with self.subTest(key=key, value=value):
                item = domain(); item[key] = value
                with self.assertRaises(audit.AuditError):
                    audit.normalized_capacity_domain(item)

    def test_domain_formula_depth_and_horizon_are_frozen(self):
        for key, value in (("capacity_domain_id", "cap1024-l1"), ("io_depth", 1),
                           ("preload_lookahead_requests", 1), ("slot_count", 1171)):
            with self.subTest(key=key):
                item = domain(); item[key] = value
                with self.assertRaises(audit.AuditError):
                    audit.normalized_capacity_domain(item)
        self.assertFalse(audit.same_capacity_domain(domain(), domain("cap960-l1")))

    def test_exact_domain_keys_required(self):
        for change in ("missing", "extra"):
            with self.subTest(change=change):
                item = domain()
                if change == "missing":
                    del item["slot_count"]
                else:
                    item["new_budget"] = 0
                with self.assertRaises(audit.AuditError):
                    audit.normalized_capacity_domain(item)


class AnchorFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.expected = {k: v for k, v in NATIVE_HASHES.items() if "/py_kvcache/" in k}
        self.expected.update({"src/prefix_io_control/unit_%d.py" % n: "a"*64 for n in range(21)})
        self.expected.update({"third_party/work/vllm-author-build/vllm/unit_%d.py" % n: "b"*64 for n in range(1844)})
        self.expected.update({k: "c"*64 for k in audit.CALIBRATION_OTHER_SOURCES})
        self.locked = dict(self.expected)
        self.locked.update({k: v for k, v in NATIVE_HASHES.items() if "/tests/" in k})
        self.locked.update({"docs/fixture_%d.txt" % n: "d"*64 for n in range(699)})
        self.assertEqual(len(self.expected), 1884)
        self.assertEqual(len(self.locked), 2599)
        self.lock_ref = self.write("artifacts/prefix_io_v1/server08-p3-16/execution-lock-08.json", self.locked)
        self.pin = mock.patch.object(audit, "P316_EXECUTION_LOCK_SHA", self.lock_ref["sha256"])
        self.pin.start(); self.addCleanup(self.pin.stop)
        self.source = dict(schema_version=1, status="PASS_FROZEN_INPUT_BYTE_IDENTITIES_AT_STORAGE_BLOCK",
                           all_hashes_match=True, locked_input_count=2599, execution_lock=self.lock_ref,
                           input_files={str(self.root/k) if i % 2 else k: v
                                        for i, (k, v) in enumerate(self.locked.items())})
        self.manifest = {"native_source_hashes": copy.deepcopy(NATIVE_HASHES)}
        self.publish_source(self.source)

    def write(self, relative, value):
        p = self.root / relative; p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("x", encoding="utf-8") as f:
            json.dump(value, f, sort_keys=True, allow_nan=False)
        return {"path": relative, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    def publish_source(self, source):
        suffix = len(list(self.root.glob("source*.json")))
        source_ref = self.write("source%d.json" % suffix, source)
        receipt_ref = self.write("receipt%d.json" % suffix, {
            "checks": {"source_lock": {"evidence_complete": True, "producer_refs": [source_ref]}}})
        self.manifest["delivery_receipts"] = [{"name": "source_lock", "receipt": receipt_ref}]

    def anchor(self):
        return audit.calibration_source_anchor(audit.Evidence(self.root), self.manifest)

    def test_complete1884_sources_are_anchored_to_actual_frozen_lock_fileref(self):
        expected, proof = self.anchor()
        self.assertEqual(expected, self.expected)
        self.assertEqual(proof["anchored_runtime_source_count"], 1884)
        audit.anchored_calibration_sources(expected, self.expected)

    def test_missing_runtime_key_extra_path_and_bad_digest_rejected(self):
        expected, _ = self.anchor()
        for change in ("missing", "extra", "digest"):
            with self.subTest(change=change):
                observed = dict(expected)
                if change == "missing":
                    observed.pop(next(iter(observed)))
                elif change == "extra":
                    observed["src/prefix_io_control/extra.py"] = "a"*64
                else:
                    observed[next(iter(observed))] = "not-a-sha"
                with self.assertRaises(audit.AuditError):
                    audit.anchored_calibration_sources(expected, observed)

    def test_four_mutually_same_wrong_source_maps_still_rejected_by_lock(self):
        expected, _ = self.anchor()
        wrong = dict(expected); wrong["third_party/work/vllm-author-build/vllm/unit_0.py"] = "f"*64
        peers = [copy.deepcopy(wrong) for _ in range(4)]
        self.assertTrue(all(peer == peers[0] for peer in peers))
        for peer in peers:
            with self.assertRaises(audit.AuditError):
                audit.anchored_calibration_sources(expected, peer)

    def test_verified_input_map_change_rejected_even_if_new_receipt_sha_valid(self):
        source = copy.deepcopy(self.source)
        source["input_files"][next(iter(source["input_files"]))] = "f"*64
        self.publish_source(source)
        with self.assertRaises(audit.AuditError):
            self.anchor()

    def test_missing_fileref_unpinned_lock_and_incomplete_proof_rejected(self):
        original = copy.deepcopy(self.source)
        source = copy.deepcopy(original); source["execution_lock"] = {"path": self.lock_ref["path"]}
        self.publish_source(source)
        with self.assertRaises(audit.AuditError):
            self.anchor()
        source = copy.deepcopy(original); source["execution_lock"] = dict(self.lock_ref, sha256="f"*64)
        self.publish_source(source)
        with self.assertRaises(audit.AuditError):
            self.anchor()
        self.manifest["delivery_receipts"] = []
        with self.assertRaises(audit.AuditError):
            self.anchor()

    def test_normalized_lock_alias_and_observed_path_alias_rejected(self):
        source = copy.deepcopy(self.source)
        key = next(iter(self.locked))
        source["input_files"][str(self.root/key)] = self.locked[key]
        source["input_files"][key] = self.locked[key]
        self.publish_source(source)
        with self.assertRaises(audit.AuditError):
            self.anchor()
        for path in ("/absolute.py", "src/../src/a.py", "src//a.py", "src\\a.py"):
            with self.subTest(path=path), self.assertRaises(audit.AuditError):
                audit.anchored_calibration_sources({path: "a"*64}, {path: "a"*64})

    def test_failed_or_unknown_source_byte_verification_is_not_completion(self):
        for value in (False, None):
            with self.subTest(value=value):
                source = copy.deepcopy(self.source); source["all_hashes_match"] = value
                self.publish_source(source)
                with self.assertRaises(audit.AuditError):
                    self.anchor()

    def test_original_lock_pin_cannot_follow_new_malicious_map_and_fileref(self):
        changed = dict(self.locked)
        changed["third_party/work/vllm-author-build/vllm/unit_0.py"] = "f"*64
        new_ref = self.write("other-lock.json", changed)
        source = copy.deepcopy(self.source)
        source["execution_lock"] = new_ref; source["input_files"] = changed
        self.publish_source(source)
        with self.assertRaises(audit.AuditError):
            self.anchor()

    def test_full31_partition_keeps_tests_and_requires_exact_runtime15(self):
        expected, _ = self.anchor()
        scope = audit.calibration_native_source_scope(
            NATIVE_HASHES, expected, "third_party/work/py-kvcache-p3-16-cpu")
        self.assertEqual((scope["runtime_source_count"], scope["test_source_count"]), (15, 16))
        self.assertTrue(scope["full_test_hashes_remain_required_by_CPU_and_patch"])
        full = dict(NATIVE_HASHES)
        full.pop(next(k for k in full if "/tests/" in k))
        with self.assertRaises(audit.AuditError):
            audit.calibration_native_source_scope(full, expected, "third_party/work/py-kvcache-p3-16-cpu")
        observed = dict(expected)
        observed.pop(next(k for k in observed if "/py_kvcache/" in k))
        with self.assertRaises(audit.AuditError):
            audit.calibration_native_source_scope(NATIVE_HASHES, observed, "third_party/work/py-kvcache-p3-16-cpu")

    def test_missing_runs_never_select_winner_or_promote_p3_or_p4(self):
        review = self.write("review.json", {"mandatory_p3_gates": [{}]*11})
        prereg = self.write("prereg.json", {
            "core_order": ["U", "F16", "P4", "P4", "F16", "U"],
            "supplement_order": ["F8", "P512", "P512", "F8"],
            "capacity_order": ["cap960-l1", "cap1024-l2"],
            "formal_performance_claim": False, "p4_enabled": False, "configs": [], "plans": []})
        manifest = dict(self.manifest, schema_version=1, gpu_uuid="GPU-fixture",
                        requirements_review=review, preregistration=prereg,
                        core_runs=[], finite_runs=[{"arm": "F8"}], capacity_gates=[], capacity_runs=[],
                        calibration_runs=[], hot_runs=[], observation_runs=[], cpu_receipts=[],
                        native_qualifications=[], lineage_receipts=[])
        result = audit.analyze(self.root, manifest)
        self.assertEqual(result["status"], "P3_INCOMPLETE")
        self.assertIsNone(result["descriptive_arm_medians"])
        self.assertFalse(result["p3_phase_closed"])
        self.assertFalse(result["p4_enabled"])
        self.assertFalse(result["p4_next_allowed"]["requirements_evidence_complete"])
        unmet = {x["name"] for x in result["unmet_gates"]}
        self.assertTrue({"core_models", "finite_models", "finite_capacity_and_preload_disposition",
                         "gpu_hot_selected_baseline_negative_control"} <= unmet)


class UnchangedGateTests(unittest.TestCase):
    def test_original_frozen_reader_and_physical_gates_unchanged(self):
        old = ROOT / "experiments/prefix_io_v1/scripts/analyze_p3_closeout_p316.py"
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(), FROZEN_V1_SHA)
        before = {n.name: n for n in ast.parse(old.read_text()).body if isinstance(n, ast.FunctionDef)}
        after = {n.name: n for n in ast.parse(SCRIPT.read_text()).body if isinstance(n, ast.FunctionDef)}
        for name in ("gpu_wrapper", "actual_controls", "pressure_proof", "cpu_receipts",
                     "patch_roundtrip", "arm_summary", "audit_checks", "native_qualifications"):
            with self.subTest(name=name):
                self.assertEqual(ast.dump(before[name], include_attributes=False),
                                 ast.dump(after[name], include_attributes=False))


if __name__ == "__main__":
    unittest.main()
