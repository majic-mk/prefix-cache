# Redirect review delivery copy

Use redirect-review-public.json for local delivery or sharing. Values of the
anonymous provider's temporary Set-Cookie headers and signed URL auth_key query
parameters have been replaced with REDACTED. Hosts, content-addressed paths,
model/revision/filename query fields, HTTP status and size evidence remain intact.

The original per-location location_sha256 values are preserved and refer to the
complete original signed URLs, not the redacted strings. The redaction object
also records SHA256 and byte size of the original server evidence files.

Raw redirect-review.json, requests-started.jsonl and requests-completed.jsonl
remain on the server for audit. Exclude those three raw files from public/local
delivery archives; they have not been deleted or overwritten. allowlist-evidence.json
and content-address-check.json contain no signed query or cookie values.

This redaction performed only local CPU file processing on the server. No further
network request, model download or GPU operation was performed.
