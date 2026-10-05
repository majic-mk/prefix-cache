#!/usr/bin/env python3
"""CPU-only frozen safetensors header audit; never loads tensor payloads or torch."""
import argparse
import collections
import json
import math
from pathlib import Path
import struct

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model-dir", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
args = p.parse_args()
index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
mapping = index["weight_map"]
seen = {}
dtype_counts = collections.Counter()
parameters = 0
payload_bytes = 0
files = []
for name in sorted(set(mapping.values())):
    path = args.model_dir / name
    with path.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        if not 0 < length <= 16 * 1024 ** 2:
            raise RuntimeError("invalid bounded safetensors header")
        header = json.loads(f.read(length))
    offsets = []
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        if key in seen or mapping.get(key) != name:
            raise RuntimeError("tensor/index ownership mismatch")
        if entry["dtype"] != "BF16":
            raise RuntimeError("checkpoint contains non-BF16 tensor: " + key)
        shape = entry["shape"]
        if not shape or any(type(v) is not int or v <= 0 for v in shape):
            raise RuntimeError("invalid tensor dimensions")
        count = math.prod(shape)
        begin, end = entry["data_offsets"]
        if type(begin) is not int or type(end) is not int or begin < 0 or end - begin != count * 2:
            raise RuntimeError("tensor payload size mismatch")
        offsets.append((begin, end))
        seen[key] = name
        dtype_counts[entry["dtype"]] += 1
        parameters += count
        payload_bytes += count * 2
    offsets.sort()
    if not offsets or offsets[0][0] != 0 or any(a[1] != b[0] for a, b in zip(offsets, offsets[1:])):
        raise RuntimeError("safetensors payload spans have gaps/overlap")
    if 8 + length + offsets[-1][1] != path.stat().st_size:
        raise RuntimeError("header and payload do not span complete file")
    files.append({"file": name, "header_bytes": length, "storage_bytes": path.stat().st_size,
                  "tensor_payload_bytes": offsets[-1][1], "tensor_count": len(offsets)})
if seen != mapping:
    raise RuntimeError("incomplete indexed tensor set")
result = {"status": "PASSED_HEADER_AUDIT", "model_dir": str(args.model_dir),
          "tensor_count": len(seen), "dtype_counts": dict(dtype_counts),
          "parameter_elements": parameters, "logical_tensor_payload_bytes": payload_bytes,
          "files": files, "gpu_execution": False, "tensor_payload_loaded": False,
          "scope": "header/index/dtype consistency; payload hashes verified separately by downloader"}
with args.output.open("x") as f:
    json.dump(result, f, indent=2)
    f.write("\n")
print(json.dumps(result, indent=2))
