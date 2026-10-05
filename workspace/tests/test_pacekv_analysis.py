"""Synthetic records ONLY test rejection/aggregation, never GPU correctness."""
import json
from pathlib import Path
import tempfile
import unittest

from pacekv.analysis import analyze
from pacekv.audit import file_sha, write_new_json
from pacekv.plan import pilot_plan


class AnalysisTests(unittest.TestCase):
    def fixture(self, root):
        import torch
        p = pilot_plan()
        for name in ("runtime.json", "summary.json", "launch.json"):
            write_new_json(root/name, dict(code_digest="unit_test_not_gpu", unit_test_fixture=True))
        for n in p["prompt_tokens"]:
            for name in ("input-%s.json" % n, "qualification-cost-%s.json" % n):
                write_new_json(root/name, dict(unit_test_fixture=True))
            (root/("roundtrip-%s.pt" % n)).write_bytes(b"unit test placeholder; not actual tensors")
            for path in ("cpu_pinned", "ssd_staged_page_cache_uncontrolled"):
                write_new_json(root/("correctness-%s-%s.json" % (n, path)), dict(
                    source_before="a", host_digest="a", destination_digest="a", source_after="a",
                    token_ids=list(range(32)), token_ids_reference=list(range(32)),
                    logits_relative_l2_by_position=[0.0]*32))
                torch.save(dict(reference=[torch.ones(1, 4)]*32,
                                candidate=[torch.ones(1, 4)]*32),
                           root/("logits-%s-%s.pt" % (n, path)))
            for r in range(p["warmup_rounds"] + p["repetitions"]):
                for arm in p["arms"]:
                    transfers, totals = [], dict(h2d=0, d2h=0)
                    for direction in ("h2d", "d2h"):
                        if arm in (direction, "bidirectional"):
                            totals[direction] = p["transfer_bytes_per_direction_per_round"]
                            transfers.append(dict(direction=direction, bytes=totals[direction],
                                                  issued_ns=0, completion_observed_ns=1, cuda_copy_ms=.1))
                    write_new_json(root/("decode-%s-%02d-%s.json" % (n, r, arm)), dict(
                        token_visible_ns=list(range(64)), token_ids=list(range(64)),
                        round_index=r, arm=arm, prompt_tokens=n, ids_equal_reference=True,
                        dma_ownership=dict(outstanding=0), decode_ns=64, total_with_drain_ns=65,
                        transfer_events=transfers, transfer_bytes=totals, simulated_unit_fixture=True))
        self.manifest(root)

    def manifest(self, root):
        payload = dict(code_digest="unit_test_not_gpu", plan_sha256=pilot_plan()["plan_sha256"],
                       files={p.name: file_sha(p) for p in root.iterdir() if p.name != "evidence_manifest.json"})
        (root/"evidence_manifest.json").write_text(json.dumps(payload))

    def test_digest_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            (root/"runtime.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "digest"):analyze(root)

    def test_summary_passed_cannot_replace_raw_token_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            target=root/"decode-512-02-h2d.json"
            row=json.loads(target.read_text());row["token_ids"][0]=999
            target.write_text(json.dumps(row));self.manifest(root)
            with self.assertRaisesRegex(ValueError, "tokens disagree"):analyze(root)

    def test_omitted_arm_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            # Remove only a freshly created disposable test fixture.
            (root/"decode-512-02-h2d.json").unlink();self.manifest(root)
            with self.assertRaisesRegex(ValueError, "missing required"):analyze(root)

    def test_incomplete_copy_budget_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            target=root/"decode-512-02-h2d.json"
            row=json.loads(target.read_text());row["transfer_events"]=[]
            target.write_text(json.dumps(row));self.manifest(root)
            with self.assertRaisesRegex(ValueError, "I/O exposure"):analyze(root)


if __name__ == "__main__":unittest.main()
