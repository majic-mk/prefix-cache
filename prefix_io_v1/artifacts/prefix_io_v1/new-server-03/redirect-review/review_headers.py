"""Read only official redirect/HEAD headers. Never read a response body."""
from pathlib import Path
import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
PLAN = ROOT / "artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json"
TOTAL_LIMIT = 64 * 1024
RESPONSE_HEADER_LIMIT = 8 * 1024
used_headers = 0
requests = []


class CapturedRedirect(Exception):
    pass


def durable_append(name, value):
    with (EVIDENCE / name).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def review(url, method, label):
    global used_headers
    if used_headers + RESPONSE_HEADER_LIMIT > TOTAL_LIMIT:
        raise RuntimeError("64 KiB header-review ceiling leaves no full request allowance")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.username or parts.password:
        raise RuntimeError("non-HTTPS or credential-bearing redirect rejected")
    record = {
        "label": label, "method": method, "url": url,
        "started_unix": time.time(), "state": "STARTED",
        "response_header_limit_bytes": RESPONSE_HEADER_LIMIT,
        "response_body_limit_bytes": 0, "body_bytes_read": 0,
        "tls_check_hostname": True, "tls_verify_mode": "CERT_REQUIRED",
        "source_hostname": parts.hostname,
    }
    durable_append("requests-started.jsonl", record)

    def headers_seen(status, headers):
        # Store source headers exactly for provenance; never call response.read().
        nonlocal record
        global used_headers
        header_items = list(headers.items())
        measured = len(("HTTP status " + str(status) + "\r\n").encode()) + sum(
            len((key + ": " + value + "\r\n").encode("latin-1", errors="replace"))
            for key, value in header_items)
        used_headers += measured
        record.update({"status": status, "response_headers": header_items,
                       "measured_header_bytes": measured})
        if measured > RESPONSE_HEADER_LIMIT or used_headers > TOTAL_LIMIT:
            raise RuntimeError("header review limit exceeded; no body was read")

    class StopAtRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            # Intentionally neither call super nor return a follow-up Request.
            try:
                headers_seen(code, headers)
                record["location"] = newurl
                target = urllib.parse.urlsplit(newurl)
                record["location_host"] = target.hostname
                record["location_scheme"] = target.scheme
                record["location_path"] = target.path
                record["location_sha256"] = hashlib.sha256(newurl.encode()).hexdigest()
            finally:
                fp.close()
                record["redirect_fp_closed"] = True
            raise CapturedRedirect("redirect captured; response closed before body read")

    context = ssl.create_default_context()
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context),
                                        StopAtRedirect())
    try:
        request = urllib.request.Request(url, method=method, headers={
            "User-Agent": "prefix-io-v1-header-review/1", "Accept-Encoding": "identity"})
        try:
            response = opener.open(request, timeout=15)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            headers_seen(response.status, response.headers)
            record["final_hostname"] = urllib.parse.urlsplit(response.geturl()).hostname
            # Deliberately no read/readinto/iter_content call.
        record["response_closed"] = True
        record["state"] = "HEADERS_RECEIVED"
    except CapturedRedirect:
        record["state"] = "REDIRECT_CAPTURED"
    except Exception as exc:
        record["state"] = "ERROR"
        record["error"] = repr(exc)
    record["finished_unix"] = time.time()
    durable_append("requests-completed.jsonl", record)
    requests.append(record)
    return record


def main():
    plan = json.loads(PLAN.read_text())
    shards = [item for item in plan["files"] if item["path"].endswith(".safetensors")]
    assert len(shards) == 4
    for item in shards:
        first = review(item["url"], "GET", item["path"] + ":official-first-hop")
        if first["state"] != "REDIRECT_CAPTURED":
            continue
        if first["location_scheme"] != "https":
            continue
        head = review(first["location"], "HEAD", item["path"] + ":exact-location-head")
        headers = {k.lower(): v for k, v in head.get("response_headers", [])}
        head["expected_file_bytes"] = item["bytes"]
        head["content_length_matches"] = headers.get("content-length") == str(item["bytes"])
        # Additional redirects, if any, remain captured and are not followed blindly.
    summary = {
        "requests": requests, "request_count": len(requests),
        "total_measured_header_bytes": used_headers, "header_ceiling_bytes": TOTAL_LIMIT,
        "all_response_body_bytes_read": 0, "weight_body_bytes_read": 0,
        "gpu_operations": 0, "queries_stopped": True,
    }
    (EVIDENCE / "redirect-review.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "request_count": len(requests), "total_measured_header_bytes": used_headers,
        "weight_body_bytes_read": 0,
        "results": [{
            "label": item["label"], "method": item["method"], "status": item.get("status"),
            "state": item["state"], "source_hostname": item["source_hostname"],
            "location_host": item.get("location_host"),
            "expected_file_bytes": item.get("expected_file_bytes"),
            "content_length_matches": item.get("content_length_matches"),
            "content_length": dict((k.lower(),v) for k,v in item.get("response_headers",[])).get("content-length"),
            "error": item.get("error")
        } for item in requests],
    }, indent=2))


if __name__ == "__main__":
    main()
