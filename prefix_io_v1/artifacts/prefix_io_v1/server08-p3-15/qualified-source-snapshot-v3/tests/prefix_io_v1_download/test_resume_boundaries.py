"""CPU-only explicit Range continuation; fake streams, temporary files/ledgers only."""
from __future__ import annotations
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_download_boundaries import dl, Response, rig, forbid_real_network, REV, WEIGHT
from test_redirect_routes import Body, cdn_url

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "resume_under_test", ROOT / "experiments/prefix_io_v1/scripts/resume_pinned_model.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


class ResumeRig:
    def __init__(self, rig, monkeypatch):
        self.r = rig
        monkeypatch.setattr(rs, "dl", dl)
        self.rec = next(row for row in rig.plan["files"] if row["path"] == WEIGHT)
        rig.plan["source"] = "https://modelscope.cn"
        for rec in rig.plan["files"]:
            rec["url"] = (f"https://modelscope.cn/api/v1/models/{dl.MODEL_ID}/repo"
                          f"?Revision={REV}&FilePath={rec['path']}")
        rig.save_plan()
        self.manifest_sha = hashlib.sha256(rig.manifest.read_bytes()).hexdigest()
        self.offset = 37
        self.original = rig.payloads[WEIGHT][:self.offset]
        rig.target.mkdir(parents=True)
        self.source = rig.target / (WEIGHT + ".failed.partial")
        self.source.write_bytes(self.original)
        self.prefix_sha = hashlib.sha256(self.original).hexdigest()
        self.event = {
            "kind": "model_download", "id": "failed-event", "label": "failed",
            "file": WEIGHT, "url": self.rec["url"], "reserved_bytes": self.rec["bytes"] + 1,
            "runner_pid": 1, "started_unix": 1, "manifest_sha256": self.manifest_sha,
            "actual_payload_bytes": self.offset, "charged_bytes": self.rec["bytes"] + 1,
            "ended_unix": 2, "error": "InterruptedError: signal 15", "published": False}
        self.failed_path = (rig.root / "artifacts/prefix_io_v1/new-server-03/downloads"
                            "/failed/result.json")
        self.failed_path.parent.mkdir(parents=True)
        self.failed = {"label": "failed", "status": "FAILED", "source": rig.plan["source"],
                       "revision": REV, "manifest_sha256": self.manifest_sha,
                       "target": str(rig.target), "files": [self.event]}
        self.save_failure()
        self.initial = copy.deepcopy(rig.initial)
        self.initial.update(model_download_bytes=self.rec["bytes"] + 1,
                            model_payload_received_bytes=self.offset,
                            download_events=[copy.deepcopy(self.event)])
        rig.save_budget(self.initial)
        self.requests = []
        self.responses = []
        self.factory = self.default_response
        self.on_open = None
        monkeypatch.setattr(dl, "make_opener", lambda: self)
    @property
    def remainder(self):
        return self.rec["bytes"] - self.offset
    def save_failure(self):
        self.failed_path.write_text(json.dumps(self.failed))
    def default_response(self, req):
        response = Response(self.r.payloads[WEIGHT][self.offset:], req.full_url)
        response.status = 206
        response.headers["Content-Range"] = (
            f"bytes {self.offset}-{self.rec['bytes']-1}/{self.rec['bytes']}")
        return response
    def open(self, req, timeout):
        budget = self.r.budget()
        assert budget["active_reservation"]["reserved_bytes"] == self.remainder + 1
        assert budget["active_reservation"]["kind"] == "model_resume"
        assert req.get_header("Range") == f"bytes={self.offset}-"
        assert req._prefix_download_expected["hash"] == self.rec["hash"]
        assert req._prefix_download_expected["revision"] == REV
        assert self.source.read_bytes() == self.original
        self.requests.append(req)
        if self.on_open:
            self.on_open()
        response = self.factory(req)
        self.responses.append(response)
        return response
    def run(self, **overrides):
        args = dict(root=self.r.root, plan_path=self.r.manifest, target=self.r.target,
                    label="resume", failed_label="failed", filename=WEIGHT,
                    partial_path=self.source, offset=self.offset,
                    partial_sha256=self.prefix_sha)
        args.update(overrides)
        return rs.resume(**args)
    def assert_failed_charge(self, received):
        budget = self.r.budget()
        assert budget["model_download_bytes"] == self.initial["model_download_bytes"] + self.remainder + 1
        assert budget["model_payload_received_bytes"] == self.offset + received
        assert budget["active_reservation"] is None
        assert budget["download_events"][0] == self.event
        assert budget["download_events"][-1]["published"] is False
        assert self.source.read_bytes() == self.original
        assert not (self.r.target / WEIGHT).exists()


@pytest.fixture
def rr(rig, monkeypatch):
    return ResumeRig(rig, monkeypatch)


def test_success_preserves_source_and_prior_charge_and_verifies_entire_file(rr, monkeypatch):
    history = []
    real_fsync = dl.os.fsync
    monkeypatch.setattr(dl.os, "fsync", lambda fd: (history.append(fd), real_fsync(fd))[1])
    rr.on_open = lambda: pytest.fail("reservation was not durably written") if len(history) < 5 else None
    result = rr.run()
    assert result["status"] == "VERIFIED_RESUMED_FILE_ONLY"
    assert (rr.r.target / WEIGHT).read_bytes() == rr.r.payloads[WEIGHT]
    assert rr.source.read_bytes() == rr.original
    assert not (rr.r.target / (WEIGHT + ".resume.partial")).exists()
    budget = rr.r.budget()
    assert budget["download_events"][0] == rr.event
    assert budget["model_download_bytes"] == rr.initial["model_download_bytes"] + rr.remainder
    assert budget["model_payload_received_bytes"] == rr.rec["bytes"]
    assert budget["events"] == rr.initial["events"]
    assert budget["gpu_wall_seconds"] == rr.initial["gpu_wall_seconds"]
    assert budget["active_reservation"] is None


@pytest.mark.parametrize("kind", ["model_download", "gpu"])
def test_any_active_reservation_rejected_before_copy_or_network(rr, kind):
    budget = rr.r.budget()
    budget["active_reservation"] = {"kind": kind, "reserved_bytes": 1}
    rr.r.save_budget(budget)
    before = rr.r.ledger.read_bytes()
    with pytest.raises(RuntimeError, match="unfinished reservation"):
        rr.run()
    assert not rr.requests and rr.r.ledger.read_bytes() == before
    assert not (rr.r.target / (WEIGHT + ".resume.partial")).exists()


def test_shared_lock_blocks_resume(rr):
    before = rr.r.ledger.read_bytes()
    with rr.r.ledger.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            rr.run()
    assert not rr.requests and rr.r.ledger.read_bytes() == before


@pytest.mark.parametrize("available", ["exact", "one_short"])
def test_remaining_budget_includes_overrun_probe_and_old_failure(rr, available):
    budget = rr.r.budget()
    budget["model_download_bytes"] = 20 * dl.GIB - (rr.remainder + 1) + (available == "one_short")
    rr.r.save_budget(budget)
    if available == "exact":
        assert rr.run()["status"] == "VERIFIED_RESUMED_FILE_ONLY"
    else:
        before = rr.r.ledger.read_bytes()
        with pytest.raises(RuntimeError, match="budget"):
            rr.run()
        assert not rr.requests and rr.r.ledger.read_bytes() == before


@pytest.mark.parametrize("case", ["manifest", "target", "status", "ledger_event", "wrong_url",
                                 "not_conservative", "offset_beyond_received", "wrong_file"])
def test_source_failure_is_bound_to_exact_manifest_partial_and_settled_event(rr, case):
    if case == "manifest":
        rr.failed["manifest_sha256"] = "f" * 64
    elif case == "target":
        rr.failed["target"] = str(rr.r.root / "elsewhere")
    elif case == "status":
        rr.failed["status"] = "RUNNING"
    elif case == "ledger_event":
        rr.event["id"] = "absent"
    elif case == "wrong_url":
        rr.event["url"] += "&other=1"
    elif case == "not_conservative":
        rr.event["charged_bytes"] -= 1
    elif case == "offset_beyond_received":
        rr.event["actual_payload_bytes"] = rr.offset - 1
    else:
        rr.event["file"] = "model-00002-of-00002.safetensors"
    rr.save_failure()
    if case in {"wrong_url", "not_conservative", "offset_beyond_received"}:
        budget = rr.r.budget()
        budget["download_events"] = [copy.deepcopy(rr.event)]
        rr.r.save_budget(budget)
    before = rr.r.ledger.read_bytes()
    with pytest.raises(RuntimeError):
        rr.run()
    assert not rr.requests and rr.r.ledger.read_bytes() == before


@pytest.mark.parametrize("case", ["zero", "end", "size", "hash", "foreign_path", "symlink"])
def test_explicit_partial_and_known_offset_are_required(rr, case):
    args = {}
    if case == "zero": args["offset"] = 0
    if case == "end": args["offset"] = rr.rec["bytes"]
    if case == "size": args["offset"] = rr.offset - 1
    if case == "hash": args["partial_sha256"] = "0" * 64
    if case == "foreign_path": args["partial_path"] = rr.r.root / "other.partial"
    if case == "symlink":
        rr.source.unlink()
        elsewhere = rr.r.root / "original"
        elsewhere.write_bytes(rr.original)
        rr.source.symlink_to(elsewhere)
    before = rr.r.ledger.read_bytes()
    with pytest.raises(RuntimeError):
        rr.run(**args)
    assert not rr.requests and rr.r.ledger.read_bytes() == before


@pytest.mark.parametrize("place", ["destination", "new_partial", "output"])
def test_no_overwrite_of_existing_files_or_label(rr, place):
    if place == "destination": target = rr.r.target / WEIGHT
    elif place == "new_partial": target = rr.r.target / (WEIGHT + ".resume.partial")
    else:
        target = rr.failed_path.parent.parent / "resume"
        target.mkdir()
    if place != "output": target.write_bytes(b"preserve")
    with pytest.raises((RuntimeError, FileExistsError)):
        rr.run()
    assert not rr.requests
    if place != "output": assert target.read_bytes() == b"preserve"


@pytest.mark.parametrize("when", ["before_copy", "before_request"])
def test_disk_checked_for_extra_full_copy_and_remaining_body(rr, monkeypatch, when):
    free = iter([0] if when == "before_copy" else [100 * dl.GIB, 0])
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda path: SimpleNamespace(free=next(free)))
    before = rr.r.ledger.read_bytes()
    with pytest.raises(RuntimeError, match="disk"):
        rr.run()
    assert not rr.requests and rr.r.ledger.read_bytes() == before
    assert rr.source.read_bytes() == rr.original


@pytest.mark.parametrize("case", ["http200", "missing_range", "wrong_start", "wrong_end",
                                 "wrong_total", "star_total", "missing_length", "wrong_length",
                                 "compressed"])
def test_range_headers_rejected_without_reading_body_and_conservatively_charged(rr, case):
    def response(req):
        out = rr.default_response(req)
        if case == "http200": out.status = 200
        elif case == "missing_range": del out.headers["Content-Range"]
        elif case == "wrong_start": out.headers["Content-Range"] = f"bytes 0-{rr.rec['bytes']-1}/{rr.rec['bytes']}"
        elif case == "wrong_end": out.headers["Content-Range"] = f"bytes {rr.offset}-{rr.rec['bytes']}/{rr.rec['bytes']}"
        elif case == "wrong_total": out.headers["Content-Range"] = f"bytes {rr.offset}-{rr.rec['bytes']-1}/{rr.rec['bytes']+1}"
        elif case == "star_total": out.headers["Content-Range"] = f"bytes {rr.offset}-{rr.rec['bytes']-1}/*"
        elif case == "missing_length": del out.headers["Content-Length"]
        elif case == "wrong_length": out.headers["Content-Length"] = str(rr.remainder + 1)
        else: out.headers["Content-Encoding"] = "gzip"
        return out
    rr.factory = response
    with pytest.raises(RuntimeError):
        rr.run()
    assert rr.responses[0].read_sizes == [] and rr.responses[0].closed
    rr.assert_failed_charge(0)


@pytest.mark.parametrize("case", ["short", "overlong", "wrong_hash", "interrupted"])
def test_stream_is_bounded_failed_and_preserves_original(rr, case):
    def response(req):
        out = rr.default_response(req)
        payload = rr.r.payloads[WEIGHT][rr.offset:]
        if case == "short": payload = payload[:-1]
        if case == "overlong": payload += b"EXCESS BYTES MUST NEVER ALL BE READ"
        if case == "wrong_hash": payload = b"x" * len(payload)
        out.stream = __import__("io").BytesIO(payload)
        if case == "interrupted":
            original_read = out.read
            def interrupted(amount):
                if out.returned:
                    raise InterruptedError("signal 15")
                return original_read(min(amount, 8))
            out.read = interrupted
        return out
    rr.factory = response
    with pytest.raises(RuntimeError):
        rr.run()
    actual = rr.responses[0].returned
    assert actual <= rr.remainder + 1
    if case == "overlong": assert actual == rr.remainder + 1
    if case == "interrupted": assert actual == 8
    rr.assert_failed_charge(actual)
    partial = rr.r.target / (WEIGHT + ".resume.partial")
    assert partial.read_bytes().startswith(rr.original)


def test_settlement_persistence_failure_retains_reservation(rr, monkeypatch):
    real_atomic = dl.atomic_json
    def fail_settlement(path, obj):
        if path == rr.r.ledger and obj.get("active_reservation") is None:
            raise OSError("injected settlement persistence failure")
        return real_atomic(path, obj)
    monkeypatch.setattr(dl, "atomic_json", fail_settlement)
    with pytest.raises(OSError, match="settlement"):
        rr.run()
    assert rr.r.budget()["active_reservation"]["kind"] == "model_resume"
    monkeypatch.setattr(dl, "atomic_json", real_atomic)
    with pytest.raises(RuntimeError, match="unfinished reservation"):
        rr.run(label="another")
    assert len(rr.requests) == 1


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_existing_exact_redirect_preserves_range_and_pinned_binding(code):
    handler = dl.VerifiedRedirect()
    rec = {"hash_algorithm": "sha256", "hash": "b" * 64, "path": WEIGHT, "revision": REV}
    import urllib.request
    request = urllib.request.Request(
        f"https://modelscope.cn/api/v1/models/{dl.MODEL_ID}/repo?Revision={REV}&FilePath={WEIGHT}",
        headers={"Range": "bytes=37-", "Accept-Encoding": "identity"})
    request.timeout = 60
    request._prefix_download_expected = rec
    body = Body()
    captures = []
    class Parent:
        def open(self, redirected, timeout):
            assert body.closed
            assert redirected.get_header("Range") == "bytes=37-"
            assert redirected.get_header("Accept-encoding") == "identity"
            assert redirected._prefix_download_expected == rec
            assert redirected._prefix_download_redirect_hops == 1
            captures.append(redirected)
            return "captured-without-network"
    handler.add_parent(Parent())
    assert getattr(handler, "http_error_" + str(code))(
        request, body, code, "Found", {"Location": cdn_url(rec)}) == "captured-without-network"
    assert len(captures) == 1
