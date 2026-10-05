"""CPU regression tests. All tokens below are deliberately synthetic."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("redact_delivery", Path(__file__).with_name("redact_delivery.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

COOKIE = 'anonymous_cookie=synthetic_cookie_token_123456; Path=/; Note="quoted\\\\text"'
AUTH = "synthetic_auth_token_987654321"
def url(token=AUTH):
    return "https://example.invalid/file?" + "auth_" + "key=" + token + "&revision=abc"
RAW = {"headers": [["Set-Cookie", COOKIE]], "location": url()}
PUBLIC = {"headers": [["Set-Cookie", "REDACTED"]], "location": url("REDACTED")}

def redactor():
    return module.redactor_from_pair(RAW, PUBLIC)

@pytest.mark.parametrize("levels", range(7))
def test_nested_json_string_escaping(levels):
    text = json.dumps(RAW)
    for _ in range(levels):
        text = json.dumps({"stdout": text})
    result, counts = redactor().redact(text)
    assert AUTH not in result and "synthetic_cookie_token_123456" not in result
    assert sum(counts.values()) >= 2
    json.loads(result)

def test_generic_auth_key_and_idempotence():
    original = url("new_unseen_temporary_token_112233")
    result, counts = redactor().redact(original)
    assert result == url("REDACTED")
    assert counts == {"additional_auth_key": 1}
    assert redactor().redact(result) == (result, {})

def test_pair_fail_closed():
    with pytest.raises(ValueError, match="unexpected"):
        module.redactor_from_pair(RAW, {"headers": [["Set-Cookie", COOKIE]], "location": "different"})
    with pytest.raises(ValueError, match="no source secrets"):
        module.redactor_from_pair(RAW, RAW)

def test_binary_only_known_bin_allowed():
    r = redactor()
    assert r.transform(b"\xff\x00data", "probe.bin")[0] == b"\xff\x00data"
    with pytest.raises(ValueError, match="unsupported"):
        r.transform(b"\xff\x00", "unknown.dat")
    with pytest.raises(ValueError, match="secret in binary"):
        r.transform(b"\xff" + AUTH.encode(), "probe.bin")

def prepare_project(tmp_path):
    project = tmp_path / "project"
    for scope in module.SCOPES:
        (project / scope).mkdir(parents=True, exist_ok=True)
    directory = project / module.RAW_DIR
    directory.mkdir(parents=True)
    raw_bytes = json.dumps(RAW).encode()
    raw_files = {
        "redirect-review.json": raw_bytes,
        "requests-started.jsonl": (json.dumps({"location": url()}) + "\n").encode(),
        "requests-completed.jsonl": (json.dumps(RAW) + "\n").encode(),
    }
    refs = []
    for name, data in raw_files.items():
        (directory / name).write_bytes(data)
        refs.append({"path": name, "sha256": module.sha(data), "bytes": len(data)})
    public = dict(PUBLIC, redaction={"raw_sources": refs})
    (directory / "redirect-review-public.json").write_text(json.dumps(public))
    (project / "docs/prefix_io_v1/notes.md").write_text("ordinary text")
    (project.parent / "commands.jsonl").write_text(json.dumps({"stdout": json.dumps(RAW)}) + "\n")
    plan_path = project / "artifacts/prefix_io_v1/new-server-03/delivery-redaction/delivery-plan.json"
    return project, plan_path

def test_plan_and_copy_preserve_sources_exclude_raw(tmp_path):
    project, plan_path = prepare_project(tmp_path)
    before = {str(x): x.read_bytes() for x in project.rglob("*") if x.is_file()}
    parent_before = (project.parent / "commands.jsonl").read_bytes()
    plan = module.make_plan(project, plan_path)
    assert plan["ready"] and plan["totals"]["changed_files"] == 1
    assert module.RAW_EXCLUSIONS <= {x["path"] for x in plan["exclusions"]}
    destination = tmp_path / "delivery-copy"
    module.copy_plan(plan, destination)
    for path, data in before.items():
        assert Path(path).read_bytes() == data
    assert (project.parent / "commands.jsonl").read_bytes() == parent_before
    copied = (destination / "server-root/commands.jsonl").read_text()
    assert AUTH not in copied and "synthetic_cookie_token_123456" not in copied
    for rel in module.RAW_EXCLUSIONS:
        assert not (destination / rel).exists()
    with pytest.raises(ValueError, match="must not exist"):
        module.copy_plan(plan, destination)

def test_changed_source_refuses_before_destination_creation(tmp_path):
    project, plan_path = prepare_project(tmp_path)
    plan = module.make_plan(project, plan_path)
    (project.parent / "commands.jsonl").write_text("changed")
    destination = tmp_path / "copy"
    with pytest.raises(ValueError, match="source changed"):
        module.copy_plan(plan, destination)
    assert not destination.exists()

def test_raw_source_tamper_refuses(tmp_path):
    project, plan_path = prepare_project(tmp_path)
    (project / module.RAW_DIR / "requests-started.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="differs from public provenance"):
        module.make_plan(project, plan_path)

def test_unknown_binary_blocks_and_symlink_refuses(tmp_path):
    project, plan_path = prepare_project(tmp_path)
    (project / "docs/prefix_io_v1/nontext.unknown").write_bytes(b"\xff")
    plan = module.make_plan(project, plan_path)
    assert not plan["ready"] and len(plan["errors"]) == 1
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "docs/prefix_io_v1/link").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink directory"):
        module.make_plan(project, plan_path)

def test_cookie_assignment_does_not_redact_public_tenant_name():
    raw = {"headers": [["Set-Cookie", "tenant=publictenant; Path=/"]]}
    public = {"headers": [["Set-Cookie", "REDACTED"]]}
    redactor = module.redactor_from_pair(raw, public)
    assert redactor.redact("https://publictenant.example/model") == ("https://publictenant.example/model", {})
    assert redactor.redact("tenant=publictenant; Path=/")[0] == "REDACTED"
