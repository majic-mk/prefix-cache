"""Pure stdlib CPU contracts. No CUDA, framework import or native capture."""
from __future__ import annotations

import argparse
import ast
import __future__
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from typing import NewType
import unittest


HERE = Path(__file__).resolve().parent
PARSER = argparse.ArgumentParser(add_help=False)
PARSER.add_argument("--source-root")
PARSER.add_argument("--lock")
OPTIONS, REMAINING = PARSER.parse_known_args()
spec = importlib.util.spec_from_file_location("g3_cpu_byte_contract", HERE / "g3_kv_byte_evidence.py")
B = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = B
spec.loader.exec_module(B)


def configured_contract():
    candidates = HERE.parent
    lock = (Path(OPTIONS.lock) if OPTIONS.lock else
            candidates / "g2_normal_worker_site_cache_v4_final/gpu-source-lock-candidate.json")
    if OPTIONS.source_root:
        return B.source_contract(OPTIONS.source_root, lock)
    roots = (candidates / "g3_source_readonly", candidates / "g2_source_readonly",
             candidates.parent / "prefix_io_v1_server08_p4_cpu_extended_20261001/verified-v3/source")
    paths = tuple(str(next(root / relative for root in roots if (root / relative).is_file()))
                  for relative, _, _ in B.SOURCE_REFS)
    contract = B.SourceContract(paths, str(lock))
    B.verify_source_contract(contract)
    return contract


CONTRACT = configured_contract()


def fixture(file_count=1):
    geometry = B.Geometry(16, 32, (1024, 1024), (1024, 1024), 32, 64)
    files = []
    for index in range(file_count):
        producer_ids = (index * 2, index * 2 + 1)
        consumer_ids = (20 + index * 2, 21 + index * 2)
        # Unequal pages detect both layer-major and local-block permutations.
        pages = tuple(tuple(bytes((11 + index * 4 + layer * 2 + local,)) * 1024
                            for local in range(2)) for layer in range(2))
        payload = b"".join(page for layer in pages for page in layer)
        sha = hashlib.sha256(payload).hexdigest()
        prefix_hash = bytes((81 + index,)) * 16  # Association key, not raw KV digest.
        key = prefix_hash + b"\0\0\0\0"
        stages = tuple(B.StageRecord(kind, ordinal * file_count + index + 1, "cpu-run", 
                                     "store-one" if ordinal < 4 else "load-one", index, key,
                                     producer_ids if ordinal < 4 else consumer_ids,
                                     kind in B.COMPLETION_STAGES)
                       for ordinal, kind in enumerate(B.STAGES))
        files.append(B.FileEvidence(index, prefix_hash, key, "store-one", "load-one",
                                    producer_ids, consumer_ids, pages, payload, payload, payload,
                                    pages, (sha,) * 5, stages))
    return B.CPUFixtureBundle("cpu-run", geometry, tuple(files))


class ByteEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.bundle = fixture()

    def verify(self, bundle=None, contract=None):
        return B.verify_cpu_kv_bundle(self.bundle if bundle is None else bundle,
                                      CONTRACT if contract is None else contract)

    def file_change(self, **changes):
        return replace(self.bundle, files=(replace(self.bundle.files[0], **changes),))

    def reject(self, bundle, message):
        with self.assertRaisesRegex(ValueError, message):
            self.verify(bundle)

    def stage_change(self, ordinal, **changes):
        stages = list(self.bundle.files[0].stages)
        stages[ordinal] = replace(stages[ordinal], **changes)
        return self.file_change(stages=tuple(stages))

    def test_complete_multifile_layer_major_content_and_different_restore_ids(self):
        result = self.verify(fixture(2))
        self.assertEqual(result.file_summaries[0][1:3], (4096, 0))
        self.assertEqual(len(result.file_summaries), 2)
        self.assertTrue(result.structural_consistency_verified)
        self.assertTrue(result.content_consistency_verified)

    def test_pinned_original_scalar_layout_formula_matches(self):
        raw = Path(CONTRACT.source_paths[0]).read_text(encoding="utf-8")
        tree = ast.parse(raw)
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ParsedKvLayout")
        methods = [n for n in original.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("storage_block_bytes", "gpu_blocks_per_storage_block", "staging_layer_offsets")]
        self.assertEqual(len(methods), 3)
        scalar_class = ast.ClassDef(name="ScalarLayout", bases=[], keywords=[], body=methods, decorator_list=[])
        module = ast.Module(body=[scalar_class], type_ignores=[])
        scope = {"__builtins__": {"__build_class__": __build_class__, "property": property, "sum": sum},
                 "__name__": "cpu_pinned_scalar_layout"}
        exec(compile(ast.fix_missing_locations(module), CONTRACT.source_paths[0], "exec",
                     flags=__future__.annotations.compiler_flag), scope)
        layout = scope["ScalarLayout"]()
        layout.bytes_per_kernel_block = [1024, 1024]
        layout.storage_block_size_factor = 2
        self.assertEqual(layout.storage_block_bytes, 4096)
        self.assertEqual(layout.gpu_blocks_per_storage_block, 2)
        self.assertEqual(layout.staging_layer_offsets, [0, 2048])
        self.assertNotIn("torch", scope)
        self.assertNotIn("numpy", scope)

    def test_pinned_original_offload_key_scalar_formula_matches(self):
        tree = ast.parse(Path(CONTRACT.source_paths[3]).read_text(encoding="utf-8"))
        names = ("make_offload_key", "get_offload_block_hash", "get_offload_group_idx")
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual(len(functions), 3)
        module = ast.Module(body=functions, type_ignores=[])
        scope = {"__builtins__": {"int": int}, "__name__": "cpu_pinned_key",
                 "OffloadKey": NewType("OffloadKey", bytes)}
        exec(compile(ast.fix_missing_locations(module), CONTRACT.source_paths[3], "exec",
                     flags=__future__.annotations.compiler_flag), scope)
        prefix = self.bundle.files[0].prefix_block_hash
        key = scope["make_offload_key"](prefix, 0)
        self.assertEqual(key, self.bundle.files[0].offload_key)
        self.assertEqual(scope["get_offload_block_hash"](key), prefix)
        self.assertEqual(scope["get_offload_group_idx"](key), 0)

    def test_original_alignment_constant_is_pinned_scalar(self):
        tree = ast.parse(Path(CONTRACT.source_paths[2]).read_text(encoding="utf-8"))
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "DIRECT_IO_ALIGNMENT" for t in n.targets))
        self.assertEqual(ast.literal_eval(assignment.value), B.ALIGNMENT)

    def test_d2h_single_byte_corruption_rejected_even_rehashed(self):
        f = self.bundle.files[0]
        payload = b"X" + f.d2h_staging_payload[1:]
        hashes = list(f.declared_sha256)
        hashes[1] = hashlib.sha256(payload).hexdigest()
        self.reject(self.file_change(d2h_staging_payload=payload, declared_sha256=tuple(hashes)), "D2H")

    def test_ssd_write_and_read_corruption_rejected(self):
        for name in ("ssd_written_payload", "ssd_read_payload"):
            with self.subTest(name=name):
                self.reject(self.file_change(**{name: b"X" + self.bundle.files[0].ssd_read_payload[1:]}), "SSD logical")

    def test_restored_page_corruption_rejected(self):
        pages = self.bundle.files[0].restored_pages
        self.reject(self.file_change(restored_pages=((b"X" + pages[0][0][1:], pages[0][1]), pages[1])), "H2D")

    def test_block_major_payload_is_rejected(self):
        pages = self.bundle.files[0].producer_pages
        block_major = b"".join(pages[layer][block] for block in range(2) for layer in range(2))
        sha = hashlib.sha256(block_major).hexdigest()
        self.reject(self.file_change(d2h_staging_payload=block_major, ssd_written_payload=block_major,
                                     ssd_read_payload=block_major, declared_sha256=(sha,) * 5), "D2H")

    def test_layer_or_local_block_permutation_rejected(self):
        pages = self.bundle.files[0].producer_pages
        for altered in (pages[::-1], (pages[0][::-1], pages[1])):
            with self.subTest(altered=altered[0][0][0]):
                self.reject(self.file_change(restored_pages=altered), "H2D")

    def test_declared_sha_is_checked_against_actual_bytes(self):
        self.reject(self.file_change(declared_sha256=("0" * 64,) * 5), "SHA mismatch")

    def test_prefix_hash_is_not_confused_with_payload_sha(self):
        result = self.verify()
        self.assertNotEqual(result.file_summaries[0][3], self.bundle.files[0].prefix_block_hash.hex())

    def test_unsupported_outer_padding_and_unaligned_logical_geometry_rejected(self):
        self.reject(self.file_change(ssd_written_payload=self.bundle.files[0].ssd_written_payload + bytes(4096)),
                    "outer padding unsupported")
        geometry = replace(self.bundle.geometry, page_bytes=(1000, 1000), row_strides=(1000, 1000))
        self.reject(replace(self.bundle, geometry=geometry), "4096 alignment")

    def test_partial_file_and_skip_rejected(self):
        self.reject(self.file_change(producer_block_ids=(0,)), "complete block mapping")
        self.reject(replace(self.bundle, geometry=replace(self.bundle.geometry, first_block_index=1)), "partial skip")

    def test_duplicate_out_of_range_and_bool_blocks_rejected(self):
        for ids in ((0, 0), (0, 32), (0, True)):
            with self.subTest(ids=ids):
                self.reject(self.file_change(producer_block_ids=ids), "block")

    def test_cross_file_mapping_and_key_duplicates_rejected(self):
        bundle = fixture(2)
        f0, f1 = bundle.files
        self.reject(replace(bundle, files=(f0, replace(f1, producer_block_ids=f0.producer_block_ids))), "cross-file")
        self.reject(replace(bundle, files=(f0, replace(f1, prefix_block_hash=f0.prefix_block_hash,
                                                     offload_key=f0.offload_key))), "duplicate file key")

    def test_canonical_file_order_and_key_group_association_rejected(self):
        self.reject(self.file_change(file_index=1), "file order")
        self.reject(self.file_change(offload_key=self.bundle.files[0].prefix_block_hash + b"\0\0\0\1"), "key/hash/group")

    def test_fixed_group_dtype_element_size_and_dimensions(self):
        for changes in ({"group_count": 2}, {"group_count": True}, {"dtype": "float16"},
                        {"element_size": 2}, {"ndim": 3}):
            with self.subTest(changes=changes):
                self.reject(replace(self.bundle, geometry=replace(self.bundle.geometry, **changes)), "canonical")

    def test_noncontiguous_and_nonintegral_factor_rejected(self):
        self.reject(replace(self.bundle, geometry=replace(self.bundle.geometry, row_strides=(1025, 1024))), "contiguous")
        self.reject(replace(self.bundle, geometry=replace(self.bundle.geometry, storage_block_tokens=33)), "integral")

    def test_capture_layer_page_count_and_exact_bytes_types(self):
        pages = self.bundle.files[0].producer_pages
        self.reject(self.file_change(producer_pages=(pages[0],)), "layer count")
        self.reject(self.file_change(producer_pages=((pages[0][0],), pages[1])), "page count")
        self.reject(self.file_change(producer_pages=((bytearray(pages[0][0]), pages[0][1]), pages[1])), "exact page bytes")
        self.reject(self.file_change(d2h_staging_payload=bytearray(self.bundle.files[0].d2h_staging_payload)), "exact logical")

    def test_stage_kind_order_and_h2d_before_compute_required(self):
        self.reject(self.stage_change(5, kind="consumer_model_compute"), "stage order")
        self.reject(self.stage_change(5, sequence=100), "ordered unique")

    def test_all_file_captures_before_any_consumer_compute(self):
        bundle = fixture(2)
        # Each file independently orders its stages, but the consumer reads all files.
        # File-major order lets file0 compute before file1 has restored its pages.
        files = tuple(replace(f, stages=tuple(replace(stage, sequence=index * 8 + ordinal + 1)
                                              for ordinal, stage in enumerate(f.stages)))
                      for index, f in enumerate(bundle.files))
        self.reject(replace(bundle, files=files), "all files restored")

    def test_missing_stage_and_duplicate_sequence_rejected(self):
        self.reject(self.file_change(stages=self.bundle.files[0].stages[:-1]), "complete capture stages")
        self.reject(self.stage_change(2, sequence=2), "ordered unique")

    def test_required_completed_marker_is_strict_cpu_only(self):
        self.reject(self.stage_change(1, cpu_declared_complete=False), "completion marker")
        self.reject(self.stage_change(1, cpu_declared_complete=1), "completion marker")
        self.reject(self.stage_change(0, cpu_declared_complete=True), "completion marker")

    def test_stage_run_job_hash_file_and_block_association(self):
        for changes in ({"run_id": "other"}, {"job_id": "other"}, {"file_index": 1},
                        {"offload_key": b"other-key"}, {"block_ids": (1, 0)}):
            with self.subTest(changes=changes):
                self.reject(self.stage_change(1, **changes), "association")

    def test_single_store_load_job_pair_and_separation(self):
        self.reject(self.file_change(load_job_id="store-one"), "job separation")
        bundle = fixture(2)
        self.reject(replace(bundle, files=(bundle.files[0], replace(bundle.files[1], store_job_id="other"))),
                    "single store/load")

    def test_unknown_owners_future_or_receipt_rejected_without_attribute_read(self):
        class Owner:
            reads = 0
            def __getattribute__(self, name):
                type(self).reads += 1
                raise AssertionError("owner attribute was read")
        owner = Owner()
        self.reject(owner, "exact CPU fixture")
        self.reject(self.file_change(stages=(owner,) + self.bundle.files[0].stages[1:]), "exact stage")
        self.assertEqual(Owner.reads, 0)
        self.reject({"origin": "native", "GPU_verified": True, "bundle": self.bundle}, "exact CPU fixture")
        with self.assertRaisesRegex(ValueError, "exact source contract"):
            self.verify(contract={"verified": True, "source_refs": B.SOURCE_REFS})

    def test_qualified_origin_cannot_be_requested_and_result_has_no_raw_payload(self):
        result = self.verify()
        self.assertEqual(result.origin, "cpu_fixture")
        for name in ("GPU_verified", "production_qualified", "effect_verified", "real_byte_qualification"):
            self.assertIs(getattr(result, name), False)
            with self.assertRaises(FrozenInstanceError):
                setattr(result, name, True)
        self.assertEqual(len(result.missing_runtime_provenance), 5)
        self.assertFalse(hasattr(result, "bundle"))
        self.assertFalse(hasattr(result, "pages"))
        self.assertFalse(hasattr(result, "payload"))
        with self.assertRaises(TypeError):
            B.CPUFixtureBundle("cpu-run", self.bundle.geometry, self.bundle.files, origin="native")

    def test_bounded_factor_layers_files_and_input_bytes(self):
        self.reject(replace(self.bundle, files=self.bundle.files * 9), "bounded complete files")
        self.reject(replace(self.bundle, geometry=replace(self.bundle.geometry, storage_block_tokens=16 * 33)), "bounded factor")
        self.reject(replace(self.bundle, geometry=replace(self.bundle.geometry, page_bytes=(1024,) * 65,
                                                        row_strides=(1024,) * 65)), "bounded canonical layers")
        large = replace(self.bundle.geometry, page_bytes=(16 * 1024 * 1024,), row_strides=(16 * 1024 * 1024,))
        self.reject(replace(self.bundle, geometry=large, files=self.bundle.files * 4), "bounded total")

    def test_hash_scalar_and_integer_subclasses_rejected_without_coercion(self):
        class Poison(str):
            def __str__(self):
                raise AssertionError("coercion")
            def __eq__(self, other):
                raise AssertionError("equality")
        self.reject(self.file_change(declared_sha256=(Poison("0" * 64),) * 5), "exact SHA")
        self.reject(self.stage_change(1, run_id=Poison("cpu-run")), "association")
        self.reject(self.stage_change(1, sequence=True), "stage sequence")

    def test_pinned_source_drift_rejected_even_with_unchanged_contract_type(self):
        with tempfile.TemporaryDirectory(prefix="g3-byte-source-") as directory:
            target = Path(directory) / "transfer.py"
            original = Path(CONTRACT.source_paths[0]).read_bytes()
            target.write_bytes(original[:-1] + bytes((original[-1] ^ 1,)))
            bad = replace(CONTRACT, source_paths=(str(target),) + CONTRACT.source_paths[1:])
            with self.assertRaisesRegex(ValueError, "source SHA changed"):
                self.verify(contract=bad)

    def test_actual_whole_lock_drift_rejected(self):
        with tempfile.TemporaryDirectory(prefix="g3-byte-lock-") as directory:
            target = Path(directory) / "lock.json"
            raw = Path(CONTRACT.lock_path).read_bytes()
            target.write_bytes(raw[:-1] + bytes((raw[-1] ^ 1,)))
            with self.assertRaisesRegex(ValueError, "source lock SHA changed"):
                self.verify(contract=replace(CONTRACT, lock_path=str(target)))

    def test_no_framework_import_no_live_hook_or_sync_api(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("vllm", sys.modules)
        tree = ast.parse((HERE / "g3_kv_byte_evidence.py").read_text(encoding="utf-8"))
        imported = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        imported.update(alias.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names)
        self.assertLessEqual(imported, {"__future__", "dataclasses", "hashlib", "json", "pathlib"})
        forbidden = {"synchronize", "get_finished", "done", "transfer_async", "shutdown"}
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr in forbidden for n in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]] + REMAINING, verbosity=2)
