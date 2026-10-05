"""Bounded CPU fixture consistency; this module supplies no native capture authority.

No framework import, tensor/owner/Future retention, synchronization, filesystem I/O
capture, observer installation or get_finished call is implemented here.  Names of
GPU/I/O stages describe CPU-declared fixture roles, never actual device provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path


LOCK_BYTES = 899396
LOCK_SHA256 = "0325cb500f74051dedf6aefe6b51fd9d0b817f831ee81fe45e1c3942df13336b"
SOURCE_REFS = (
    ("third_party/work/py-kvcache-p4-02-cpu/py_kvcache/transfer.py", 12061,
     "1dcc5f4370e3db694d34924038a8fdd5a2eb12cf7f04ba81072dc969f275c1ba"),
    ("third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py", 169906,
     "2b0ac1e8e68f82adb42ade03b74b49843eabe55361a00f0ac1f7a738ac423e18"),
    ("third_party/work/py-kvcache-p4-02-cpu/py_kvcache/liburing_file.py", 15244,
     "6a8995ca6e5f49ae470e8c49dbf3d98caf2d09dd091cb6f08f9e892c051414f5"),
    ("third_party/work/vllm-author-p4-02-cpu/vllm/v1/kv_offload/base.py", 15330,
     "204d682bf61d3a2c1bbbe8ca1ecfd6fe6b10eca32b2600c39118403db6aae1d2"),
)
ALIGNMENT = 4096
MAX_FILES = 8
MAX_LAYERS = 64
MAX_FACTOR = 32
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_INPUT_BYTES = 512 * 1024 * 1024
STAGES = ("producer_gpu_capture", "d2h_complete", "ssd_write_complete",
          "ssd_payload_capture", "ssd_read_complete", "h2d_complete",
          "consumer_gpu_capture", "consumer_model_compute")
COMPLETION_STAGES = frozenset(("d2h_complete", "ssd_write_complete",
                              "ssd_read_complete", "h2d_complete"))
MISSING_RUNTIME_PROVENANCE = (
    "actual canonical CUDA tensor identity, shape and page capture",
    "original D2H event completion and staging capture provenance",
    "actual SSD path, write/read completion and physical payload capture",
    "original H2D completion and consumer capture before model compute",
    "source-bound live native job/owner registration and capture installer",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: object, minimum: int, maximum: int, name: str) -> None:
    _require(type(value) is int and minimum <= value <= maximum, name)


def _text(value: object, name: str) -> None:
    _require(type(value) is str and 0 < len(value) <= 128
             and value.isascii() and all(32 < ord(c) < 127 for c in value), name)


def _digest(value: object) -> None:
    _require(type(value) is str and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), "exact SHA-256")


def _unique_json(pairs: list[tuple[str, object]]) -> dict:
    out = {}
    for key, value in pairs:
        _require(key not in out, "duplicate JSON key")
        out[key] = value
    return out


@dataclass(frozen=True, slots=True)
class SourceContract:
    """Paths only. A caller's boolean receipt cannot substitute for actual bytes."""
    source_paths: tuple[str, ...]
    lock_path: str


def source_contract(source_root: str | Path, lock_path: str | Path) -> SourceContract:
    root = Path(source_root)
    result = SourceContract(tuple(str(root / ref[0]) for ref in SOURCE_REFS), str(lock_path))
    verify_source_contract(result)
    return result


def verify_source_contract(contract: SourceContract) -> tuple[tuple[str, int, str], ...]:
    _require(type(contract) is SourceContract, "exact source contract required")
    _require(type(contract.source_paths) is tuple and len(contract.source_paths) == 4
             and all(type(p) is str for p in contract.source_paths)
             and type(contract.lock_path) is str, "source path scalars")
    lock_path = Path(contract.lock_path)
    _require(lock_path.is_file() and not lock_path.is_symlink(), "regular source lock")
    _require(lock_path.stat().st_size == LOCK_BYTES, "source lock bytes changed")
    raw_lock = lock_path.read_bytes()
    _require(hashlib.sha256(raw_lock).hexdigest() == LOCK_SHA256, "source lock SHA changed")
    lock = json.loads(raw_lock, object_pairs_hook=_unique_json)
    _require(type(lock) is dict and type(lock.get("files")) is list, "source lock rows")
    rows = {}
    for row in lock["files"]:
        _require(type(row) is dict and type(row.get("path")) is str
                 and row["path"] not in rows, "unique source lock paths")
        rows[row["path"]] = row
    for path_text, (relative, size, sha) in zip(contract.source_paths, SOURCE_REFS):
        row = rows.get(relative)
        _require(type(row) is dict and type(row.get("bytes")) is int
                 and row["bytes"] == size and row.get("sha256") == sha,
                 "fixed source reference missing")
        path = Path(path_text)
        _require(path.is_file() and not path.is_symlink(), "regular original source")
        _require(path.stat().st_size == size, "source bytes changed: " + relative)
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == sha,
                 "source SHA changed: " + relative)
    return SOURCE_REFS


@dataclass(frozen=True, slots=True)
class Geometry:
    gpu_block_tokens: int
    storage_block_tokens: int
    page_bytes: tuple[int, ...]
    row_strides: tuple[int, ...]
    producer_num_blocks: int
    consumer_num_blocks: int
    group_count: int = 1
    dtype: str = "int8"
    element_size: int = 1
    ndim: int = 2
    first_block_index: int = 0


@dataclass(frozen=True, slots=True)
class StageRecord:
    kind: str
    sequence: int
    run_id: str
    job_id: str
    file_index: int
    offload_key: bytes
    block_ids: tuple[int, ...]
    cpu_declared_complete: bool


@dataclass(frozen=True, slots=True)
class FileEvidence:
    file_index: int
    prefix_block_hash: bytes
    offload_key: bytes
    store_job_id: str
    load_job_id: str
    producer_block_ids: tuple[int, ...]
    consumer_block_ids: tuple[int, ...]
    producer_pages: tuple[tuple[bytes, ...], ...]
    d2h_staging_payload: bytes
    ssd_written_payload: bytes
    ssd_read_payload: bytes
    restored_pages: tuple[tuple[bytes, ...], ...]
    declared_sha256: tuple[str, ...]
    stages: tuple[StageRecord, ...]


@dataclass(frozen=True, slots=True)
class CPUFixtureBundle:
    run_id: str
    geometry: Geometry
    files: tuple[FileEvidence, ...]


@dataclass(frozen=True, slots=True)
class CPUByteConsistencyResult:
    status: str
    file_summaries: tuple[tuple[int, int, int, str], ...]
    source_refs: tuple[tuple[str, int, str], ...]
    origin: str = field(default="cpu_fixture", init=False)
    structural_consistency_verified: bool = field(default=True, init=False)
    content_consistency_verified: bool = field(default=True, init=False)
    GPU_verified: bool = field(default=False, init=False)
    production_qualified: bool = field(default=False, init=False)
    effect_verified: bool = field(default=False, init=False)
    real_byte_qualification: bool = field(default=False, init=False)
    missing_runtime_provenance: tuple[str, ...] = field(default=MISSING_RUNTIME_PROVENANCE, init=False)


def _geometry(g: Geometry) -> tuple[int, int]:
    _require(type(g) is Geometry, "exact geometry")
    _require(type(g.group_count) is int and g.group_count == 1
             and type(g.dtype) is str and g.dtype == "int8"
             and type(g.element_size) is int and g.element_size == 1
             and type(g.ndim) is int and g.ndim == 2, "canonical one-group int8 2D geometry")
    _integer(g.gpu_block_tokens, 1, 65536, "GPU block tokens")
    _integer(g.storage_block_tokens, 1, 65536 * MAX_FACTOR, "storage block tokens")
    _require(g.storage_block_tokens % g.gpu_block_tokens == 0, "integral storage factor")
    factor = g.storage_block_tokens // g.gpu_block_tokens
    _integer(factor, 1, MAX_FACTOR, "bounded factor")
    _require(type(g.page_bytes) is tuple and 1 <= len(g.page_bytes) <= MAX_LAYERS
             and type(g.row_strides) is tuple and len(g.row_strides) == len(g.page_bytes),
             "bounded canonical layers")
    for nbytes, stride in zip(g.page_bytes, g.row_strides):
        _integer(nbytes, 1, MAX_FILE_BYTES, "page byte size")
        _require(type(stride) is int and stride == nbytes,
                 "limited contiguous canonical pages only")
    _integer(g.producer_num_blocks, factor, 1000000, "producer block count")
    _integer(g.consumer_num_blocks, factor, 1000000, "consumer block count")
    _require(type(g.first_block_index) is int and g.first_block_index == 0,
             "complete files only; partial skip unsupported")
    logical_bytes = factor * sum(g.page_bytes)
    _require(logical_bytes <= MAX_FILE_BYTES, "bounded file bytes")
    _require(logical_bytes % ALIGNMENT == 0,
             "original staging requires 4096 alignment; padded geometry unsupported")
    return factor, logical_bytes


def _block_ids(ids: object, factor: int, limit: int) -> None:
    _require(type(ids) is tuple and len(ids) == factor, "complete block mapping")
    for item in ids:
        _integer(item, 0, limit - 1, "block ID range/type")
    _require(len(set(ids)) == factor, "duplicate block mapping")


def _pages(pages: object, g: Geometry, factor: int) -> None:
    _require(type(pages) is tuple and len(pages) == len(g.page_bytes), "layer count")
    for layer, nbytes in zip(pages, g.page_bytes):
        _require(type(layer) is tuple and len(layer) == factor, "complete page count")
        _require(all(type(page) is bytes and len(page) == nbytes for page in layer),
                 "immutable exact page bytes")


def _page_digest(pages: tuple[tuple[bytes, ...], ...]) -> str:
    digest = hashlib.sha256()
    for layer in pages:
        for page in layer:
            digest.update(page)
    return digest.hexdigest()


def verify_cpu_kv_bundle(bundle: CPUFixtureBundle, contract: SourceContract) -> CPUByteConsistencyResult:
    """Verify a finite synthetic transcript, never promote it to live evidence.

    A prefix/token block hash is an association key; the KV payload SHA is checked
    independently. File layout is layer-major, then local block within that layer.
    The pinned original staging allocator rejects unaligned logical storage blocks,
    hence the supported physical SSD payload has zero outer alignment padding.
    """
    _require(type(bundle) is CPUFixtureBundle, "exact CPU fixture bundle; no native receipt authority")
    _text(bundle.run_id, "run ID")
    factor, logical_bytes = _geometry(bundle.geometry)
    _require(type(bundle.files) is tuple and 1 <= len(bundle.files) <= MAX_FILES,
             "bounded complete files")
    _require(len(bundle.files) * logical_bytes * 5 <= MAX_INPUT_BYTES, "bounded total captured bytes")
    seen_source, seen_consumer, seen_keys, seen_sequences, seen_jobs = set(), set(), set(), set(), set()
    consumer_capture_sequences, consumer_compute_sequences = [], []
    summaries = []
    # These source reads verify provenance of the formulas, not of fixture payloads.
    refs = verify_source_contract(contract)
    for index, f in enumerate(bundle.files):
        _require(type(f) is FileEvidence, "exact file evidence")
        _require(type(f.file_index) is int and f.file_index == index, "canonical file order")
        _require(type(f.prefix_block_hash) is bytes and 1 <= len(f.prefix_block_hash) <= 64,
                 "bounded prefix block hash")
        _require(type(f.offload_key) is bytes
                 and f.offload_key == f.prefix_block_hash + b"\x00\x00\x00\x00",
                 "original offload key/hash/group association")
        _require(f.offload_key not in seen_keys, "duplicate file key")
        seen_keys.add(f.offload_key)
        _text(f.store_job_id, "store job ID")
        _text(f.load_job_id, "load job ID")
        _require(f.store_job_id != f.load_job_id, "store/load job separation")
        # The limited pilot has one store and one load job spanning all files.
        job_pair = (f.store_job_id, f.load_job_id)
        if seen_jobs:
            _require(job_pair in seen_jobs, "single store/load job pair")
        seen_jobs.add(job_pair)
        _block_ids(f.producer_block_ids, factor, bundle.geometry.producer_num_blocks)
        _block_ids(f.consumer_block_ids, factor, bundle.geometry.consumer_num_blocks)
        _require(not seen_source.intersection(f.producer_block_ids)
                 and not seen_consumer.intersection(f.consumer_block_ids), "cross-file duplicate block mapping")
        seen_source.update(f.producer_block_ids)
        seen_consumer.update(f.consumer_block_ids)
        _pages(f.producer_pages, bundle.geometry, factor)
        _pages(f.restored_pages, bundle.geometry, factor)
        for payload in (f.d2h_staging_payload, f.ssd_written_payload, f.ssd_read_payload):
            _require(type(payload) is bytes and len(payload) == logical_bytes,
                     "exact logical/physical SSD bytes; outer padding unsupported")
        _require(type(f.declared_sha256) is tuple and len(f.declared_sha256) == 5,
                 "five captured payload SHA declarations")
        for sha in f.declared_sha256:
            _digest(sha)
        _require(type(f.stages) is tuple and len(f.stages) == len(STAGES), "complete capture stages")
        previous = 0
        for stage_index, (kind, stage) in enumerate(zip(STAGES, f.stages)):
            _require(type(stage) is StageRecord, "exact stage record")
            _require(type(stage.kind) is str and stage.kind == kind, "original transfer stage order")
            _integer(stage.sequence, 1, 1000000000, "stage sequence")
            _require(stage.sequence > previous and stage.sequence not in seen_sequences,
                     "ordered unique stage sequence")
            previous = stage.sequence
            seen_sequences.add(stage.sequence)
            expected_job = f.store_job_id if stage_index < 4 else f.load_job_id
            expected_blocks = f.producer_block_ids if stage_index < 4 else f.consumer_block_ids
            _require(type(stage.run_id) is str and stage.run_id == bundle.run_id
                     and type(stage.job_id) is str and stage.job_id == expected_job
                     and type(stage.file_index) is int and stage.file_index == index
                     and type(stage.offload_key) is bytes and stage.offload_key == f.offload_key,
                     "stage run/job/file/hash association")
            _block_ids(stage.block_ids, factor,
                       bundle.geometry.producer_num_blocks if stage_index < 4 else bundle.geometry.consumer_num_blocks)
            _require(stage.block_ids == expected_blocks, "stage block association")
            _require(type(stage.cpu_declared_complete) is bool
                     and stage.cpu_declared_complete is (kind in COMPLETION_STAGES),
                     "CPU-declared completion marker; not runtime proof")
            if kind == "consumer_gpu_capture":
                consumer_capture_sequences.append(stage.sequence)
            elif kind == "consumer_model_compute":
                consumer_compute_sequences.append(stage.sequence)
        offset = 0
        for source_layer, restored_layer in zip(f.producer_pages, f.restored_pages):
            for source_page, restored_page in zip(source_layer, restored_layer):
                _require(f.d2h_staging_payload[offset:offset + len(source_page)] == source_page,
                         "D2H layer-major source byte mismatch")
                _require(restored_page == source_page, "H2D restored page byte mismatch")
                offset += len(source_page)
        _require(offset == logical_bytes, "complete layer-major payload")
        _require(f.ssd_written_payload == f.d2h_staging_payload
                 and f.ssd_read_payload == f.ssd_written_payload, "SSD logical payload byte mismatch")
        digests = (_page_digest(f.producer_pages), hashlib.sha256(f.d2h_staging_payload).hexdigest(),
                   hashlib.sha256(f.ssd_written_payload).hexdigest(), hashlib.sha256(f.ssd_read_payload).hexdigest(),
                   _page_digest(f.restored_pages))
        _require(digests == f.declared_sha256, "captured bytes SHA mismatch")
        _require(len(set(digests)) == 1, "all capture content equal")
        summaries.append((index, logical_bytes, 0, digests[0]))
    _require(max(consumer_capture_sequences) < min(consumer_compute_sequences),
             "all files restored and captured before any consumer model compute")
    # Reject source drift during validation too. No raw payload or owner is returned.
    verify_source_contract(contract)
    return CPUByteConsistencyResult("PASS_CPU_STRUCTURAL_AND_CONTENT_CONSISTENCY_ONLY",
                                    tuple(summaries), refs)
