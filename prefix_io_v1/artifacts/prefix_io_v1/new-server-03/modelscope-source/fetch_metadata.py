import hashlib, json, os, sys, time, urllib.request, urllib.error
from pathlib import Path
root = Path(__file__).resolve().parent
url, name = sys.argv[1:3]
record = {"url": url, "name": name, "started_unix": time.time(), "body_bytes": 0,
          "limit_bytes": 2*1024*1024, "weight_download": False}
completed = [json.loads(line) for line in (root/"metadata-requests.jsonl").read_text().splitlines()] if (root/"metadata-requests.jsonl").exists() else []
started = [json.loads(line) for line in (root/"requests-started.jsonl").read_text().splitlines()] if (root/"requests-started.jsonl").exists() else []
finished_names = {item["name"] for item in completed}
uncertain = sum(item["limit_bytes"] for item in started if item["name"] not in finished_names)
used = sum(item["body_bytes"] for item in completed)
if used + uncertain + record["limit_bytes"] > 64*1024*1024:
    raise RuntimeError("64 MiB metadata ceiling would be exceeded")
with (root/"requests-started.jsonl").open("a") as handle:
    handle.write(json.dumps(dict(record, state="STARTED")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
try:
    request = urllib.request.Request(url, headers={"User-Agent":"prefix-io-v1-source-audit/1",
                                                 "Accept-Encoding":"identity"})
    try:
        response = urllib.request.urlopen(request, timeout=15)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        data = response.read(record["limit_bytes"] + 1)
        record.update({"status": response.status, "effective_url": response.geturl(),
                       "content_type": response.headers.get("Content-Type"),
                       "body_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        if len(data) > record["limit_bytes"]:
            raise RuntimeError("metadata response exceeded 2 MiB limit")
        destination = root / name
        with destination.open("xb") as handle:
            handle.write(data)
except Exception as exc:
    record["error"] = repr(exc)
record["finished_unix"] = time.time()
with (root/"metadata-requests.jsonl").open("a") as handle:
    handle.write(json.dumps(record) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(record, indent=2))
