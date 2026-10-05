"""CPU redirect tests for the one observed official ModelScope CDN route."""
from __future__ import annotations
import io
import urllib.request
import urllib.parse

import pytest

from test_download_boundaries import dl, Response, rig, forbid_real_network, REV, WEIGHT

CDN = "cdn-lfs-cn-1.modelscope.cn"
ORIGIN = "https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-7B-Instruct/repo"
DIGEST = "b" * 64
BOUND = {"hash_algorithm": "sha256", "hash": DIGEST, "path": WEIGHT, "revision": REV}

def cdn_url(bound):
    digest = bound["hash"]
    query = urllib.parse.urlencode({"namespace": "Qwen", "repository": "Qwen2.5-7B-Instruct",
                                   "revision": bound["revision"], "filename": bound["path"],
                                   "tag": "model", "auth_key": "fresh-fixture-only"})
    return "https://" + CDN + "/prod/lfs-objects/" + digest[:2] + "/" + digest[2:4] + "/" + digest[4:] + "?" + query

FINAL = cdn_url(BOUND)


class Body:
    def __init__(self):
        self.closed = False
    def read(self, *args, **kwargs):
        pytest.fail("redirect body must never be read")
    def close(self):
        self.closed = True


def chain(urls, code=302, start=ORIGIN):
    handler = dl.VerifiedRedirect()
    bodies = []
    requests = []
    class Parent:
        def open(self, req, timeout):
            assert bodies[-1].closed, "close redirect before the next request"
            req.timeout = timeout
            requests.append(req)
            if len(requests) < len(urls):
                body = Body()
                bodies.append(body)
                return handler.http_error_302(req, body, code, "Found",
                                              {"location": urls[len(requests)]})
            return Response(b"done", req.full_url)
    handler.add_parent(Parent())
    req = urllib.request.Request(start)
    req.timeout = 10
    req._prefix_download_expected = dict(BOUND)
    first = Body()
    bodies.append(first)
    return handler, req, bodies, requests


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_observed_official_to_exact_cdn_closes_body_before_following(code):
    handler, req, bodies, requests = chain([FINAL], code)
    result = getattr(handler, "http_error_" + str(code))(
        req, bodies[0], code, "Found", {"location": FINAL})
    assert result.geturl() == FINAL
    assert len(requests) == 1 and bodies[0].closed
    assert requests[0].get_method() == "GET"
    assert requests[0]._prefix_download_redirect_hops == 1
    result.close()


def test_three_approved_hops_allowed_and_fourth_stops_without_body_reads():
    for count in (3, 4):
        urls = [FINAL + "&hop=" + str(i) for i in range(count)]
        handler, req, bodies, requests = chain(urls)
        if count == 3:
            result = handler.http_error_302(req, bodies[0], 302, "Found",
                                            {"location": urls[0]})
            assert len(requests) == 3
            result.close()
        else:
            with pytest.raises(RuntimeError, match="redirect.*limit"):
                handler.http_error_302(req, bodies[0], 302, "Found",
                                       {"location": urls[0]})
            assert len(requests) == 3
        assert all(b.closed for b in bodies)


@pytest.mark.parametrize("target", [
    "https://cdn-lfs-cn-1.modelscope.cn.attacker.example/a",
    "https://cdn-lfs-cn-2.modelscope.cn/a",
    "https://modelscope.cn/another-route",
    "https://huggingface.co/a",
    "http://cdn-lfs-cn-1.modelscope.cn/a",
    "https://user:pass@cdn-lfs-cn-1.modelscope.cn/a",
    "https://cdn-lfs-cn-1.modelscope.cn:444/a",
])
def test_unapproved_target_or_provider_route_never_followed(target):
    handler, req, bodies, requests = chain([target])
    with pytest.raises(RuntimeError):
        handler.http_error_302(req, bodies[0], 302, "Found", {"location": target})
    assert bodies[0].closed and requests == []


def test_huggingface_cannot_use_modelscope_cdn_route():
    handler, req, bodies, requests = chain([FINAL], start="https://huggingface.co/a")
    with pytest.raises(RuntimeError, match="redirect route"):
        handler.http_error_302(req, bodies[0], 302, "Found", {"location": FINAL})
    assert bodies[0].closed and not requests


def test_relative_same_cdn_redirect_preserves_tls_and_hop_count():
    relative = urllib.parse.urlsplit(FINAL).path + "?" + urllib.parse.urlsplit(FINAL).query
    handler, req, bodies, requests = chain([relative], start=FINAL)
    req._prefix_download_redirect_hops = 1
    response = handler.http_error_302(req, bodies[0], 302, "Found",
                                     {"location": relative})
    assert requests[0].full_url == FINAL
    assert requests[0]._prefix_download_redirect_hops == 2
    response.close()


def test_missing_location_is_closed_and_rejected():
    handler, req, bodies, requests = chain([FINAL])
    with pytest.raises(RuntimeError, match="redirect Location"):
        handler.http_error_302(req, bodies[0], 302, "Found", {})
    assert bodies[0].closed and not requests


def test_manifest_cannot_start_at_cdn_even_when_redirect_host_allowed(rig):
    rig.plan["files"][-1]["url"] = FINAL
    rig.save_plan()
    with pytest.raises(RuntimeError, match="must match pinned"):
        rig.run()
    assert not rig.requests


def test_end_to_end_fake_cdn_body_charged_once_and_file_hash_verified(rig):
    rig.plan["source"] = "https://modelscope.cn"
    for rec in rig.plan["files"]:
        rec["url"] = (ORIGIN + "?Revision=" + rig.plan["revision"]
                      + "&FilePath=" + rec["path"])
    rig.save_plan()
    bodies = []
    def response(req, rec):
        if not rec["path"].endswith(".safetensors"):
            return Response(rig.payloads[rec["path"]], req.full_url)
        handler = dl.VerifiedRedirect()
        body = Body()
        bodies.append(body)
        final = cdn_url(dict(rec, revision=rig.plan["revision"]))
        class Parent:
            def open(self, child, timeout):
                assert body.closed
                assert child.full_url == final
                return Response(rig.payloads[rec["path"]], child.full_url)
        handler.add_parent(Parent())
        req.timeout = 60
        return handler.http_error_302(req, body, 302, "Found", {"location": final})
    rig.response_factory = response
    assert rig.run()["status"] == "VERIFIED_LOCAL_FILES"
    assert rig.budget()["model_download_bytes"] == rig.plan["total_bytes"]
    assert rig.budget()["model_payload_received_bytes"] == rig.plan["total_bytes"]
    assert all(b.closed for b in bodies)


@pytest.mark.parametrize("field", ["content_hash", "namespace", "repository", "revision", "filename", "tag", "duplicate_revision"])
def test_wrong_content_address_or_manifest_metadata_rejected_before_cdn_open(field):
    bad = FINAL
    if field == "content_hash":
        bad = bad.replace("/prod/lfs-objects/bb/bb/", "/prod/lfs-objects/aa/bb/")
    elif field == "duplicate_revision":
        bad += "&revision=" + REV
    else:
        parsed = urllib.parse.urlsplit(bad)
        values = urllib.parse.parse_qs(parsed.query)
        values[field] = ["wrong"]
        bad = urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(values, doseq=True)))
    handler, req, bodies, requests = chain([bad])
    with pytest.raises(RuntimeError, match="redirect.*(address|metadata)"):
        handler.http_error_302(req, bodies[0], 302, "Found", {"location": bad})
    assert bodies[0].closed and not requests


def test_missing_manifest_binding_is_rejected_without_following():
    handler, req, bodies, requests = chain([FINAL])
    del req._prefix_download_expected
    with pytest.raises(RuntimeError, match="binding"):
        handler.http_error_302(req, bodies[0], 302, "Found", {"location": FINAL})
    assert bodies[0].closed and not requests
