"""CPU counterexamples only; these tests never establish GPU provenance."""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("native_capture_tested", HERE / "production_kv_capture.py")
C = importlib.util.module_from_spec(spec); sys.modules[spec.name] = C; spec.loader.exec_module(C)

SOURCE = b'''
class IoReactor:
    def _drain_cuda_copies(self):
        for copy in list(self._pending_copies):
            if not copy.end_event.query():
                continue
            self._pending_copies.remove(copy)
            self.original_calls.append("complete")
    def _on_write_complete(self, op, result_nbytes):
        try:
            if result_nbytes != 9: raise ValueError("short write")
            self.file_store.finish_write(fd=op.fd)
            self.original_calls.append("write")
        finally:
            self.original_calls.append("finally")
    def _on_read_complete(self, op, result_nbytes):
        if result_nbytes != 9: raise ValueError("short read")
        self.original_calls.append("read")
class GPUModelRunner:
    def execute_model(self, scheduler_output):
        with self.context:
            model_output = self._model_forward()
        return model_output
'''


def fixture():
    files, records, jobs, prompts = {}, [], {}, {}
    def add(stage, **fields):
        row = dict(sequence=len(records)+1, stage=stage, **fields); records.append(row); return row
    for rep in range(3):
        sid, rid = "store:"+str(rep), "producer"+str(rep)
        prompts[rid] = str(rep)*64
        add("producer_model_forward", request_id=rid, prompt_sha256=prompts[rid], prompt_tokens=129)
        store = dict(is_store=True, request_id=rid, files={}, native_reactor_id=13, native_job_object_id=rep)
        jobs[sid] = store
        for i in range(8):
            key = bytes([rep+1, i+1]).hex()
            files[key] = dict(bytes=C.FILE_BYTES, sha256="a"*64, request_id=rid,
                              payload_file="producer-"+key+".bin", block_ids=[i], job_id=sid, file_index=i)
            row = add("d2h_complete_source_and_staging", key=key, job_id=sid, request_id=rid,
                      file_index=i, block_ids=[i], bytes=C.FILE_BYTES, sha256="a"*64,
                      complete_byte_equality=True, original_event_query_true=True, parent_not_published=True,
                      source_kind="store")
            store["files"][str(i)] = row
            add("ssd_write_published", key=key, job_id=sid, file_index=i, bytes=C.FILE_BYTES,
                sha256="a"*64, complete_byte_equality=True, cqe_nbytes=C.FILE_BYTES)
        lid, consumer = "load:"+str(rep), "consumer"+str(rep)
        load = dict(is_store=False, request_id=consumer, native_reactor_id=13, native_job_object_id=rep+100, files={})
        jobs[lid] = load
        for i in range(8):
            key = bytes([rep+1, i+1]).hex()
            row = add("h2d_complete_dest_and_staging", key=key, job_id=lid, request_id=consumer,
                      file_index=i, block_ids=[i+16], bytes=C.FILE_BYTES, sha256="a"*64,
                      complete_byte_equality=True, original_event_query_true=True, parent_not_published=True,
                      source_kind="cache")
            load["files"][str(i)] = row
        add("before_model_forward", request_id=consumer, job_id=lid, block_ids=list(range(16,24)),
            captured_file_sequences=[x["sequence"] for x in load["files"].values()],
            prompt_sha256=prompts[rid], producer_request_id=rid, prompt_tokens=129)
    return dict(schema=C.SCHEMA, status="PASS_NATIVE_KV_BYTE_DIAGNOSTIC", mode="populate",
                diagnostic_real_bytes_verified=True, failures=[], methods_restored=True,
                production_qualified=False, cost_qualified=False, effect_verified=False, timing_usable=False,
                load_planner="off", geometry=dict(logical_file_bytes=C.FILE_BYTES, storage_factor=1,
                group_count=1, prefix_tokens=128, acquisition_domain=1024, repetitions=3),
                records=records, native_jobs=jobs, native_reactor_id=13, files=files,
                producer_files={}, producer_prompts=prompts, producer_manifest=None)


def paired_fixture():
    producer=fixture(); value=copy.deepcopy(producer)
    value.update(mode="paired",files={},producer_files=producer["files"],records=[],native_jobs={},
                 producer_manifest=dict(path="actual-producer.json",bytes=123,sha256="a"*64))
    def add(stage,**fields):
        row=dict(sequence=len(value["records"])+1,stage=stage,**fields);value["records"].append(row);return row
    for rep in range(3):
        cohort=[k for k,x in producer["files"].items() if x["request_id"]=="producer"+str(rep)]
        for key in cohort:
            add("ssd_read_complete",key=key,bytes=C.FILE_BYTES,sha256="a"*64,cqe_nbytes=C.FILE_BYTES,complete_byte_equality=True)
        for pass_id in range(2):
            jid="load:"+str(rep*2+pass_id);rid="paired"+str(rep*2+pass_id)
            job=dict(is_store=False,request_id=rid,native_reactor_id=13,native_job_object_id=rep*2+pass_id,files={})
            value["native_jobs"][jid]=job
            for i,key in enumerate(cohort):
                row=add("h2d_complete_dest_and_staging",key=key,job_id=jid,request_id=rid,file_index=i,
                        block_ids=[i+16],bytes=C.FILE_BYTES,sha256="a"*64,complete_byte_equality=True,
                        original_event_query_true=True,parent_not_published=True,source_kind="file" if pass_id==0 else "cache")
                job["files"][str(i)]=row
            add("before_model_forward",request_id=rid,job_id=jid,block_ids=list(range(16,24)),prompt_tokens=129,
                captured_file_sequences=[x["sequence"] for x in job["files"].values()],prompt_sha256=str(rep)*64,
                producer_request_id="producer"+str(rep))
    return value


class CaptureTests(unittest.TestCase):
    def test_four_actual_pinned_source_transformations_compile_without_framework(self):
        root = HERE.parents[2]
        mirrors = {
            C.REACTOR: root / "artifacts/prefix_io_v1_server09_candidates/g2_source_readonly" / C.REACTOR,
            C.RUNNER: root / "artifacts/prefix_io_v1_server09_candidates/g2_source_readonly" / C.RUNNER,
        }
        if os.environ.get("KV_CAPTURE_SOURCE_ROOT"):
            mirrors = {relative: Path(os.environ["KV_CAPTURE_SOURCE_ROOT"]) / relative for relative in mirrors}
        for relative, klass, methods in ((C.REACTOR,"IoReactor",("_drain_cuda_copies","_on_write_complete","_on_read_complete")),
                                          (C.RUNNER,"GPUModelRunner",("execute_model",))):
            raw=C.read_exact(mirrors[relative],*C.PINS[relative]); tree=ast.parse(raw)
            ns={node.id:type(node.id,(),{}) for node in ast.walk(tree) if isinstance(node,ast.Name)}
            ns["torch"]=types.SimpleNamespace(inference_mode=lambda:lambda f:f)
            for name in methods:
                code=C.code_at(compile(raw,str(mirrors[relative]),"exec",dont_inherit=True),(klass,name))
                fn=types.FunctionType(code,ns)
                transformed=C.transformed_method(fn,raw,str(mirrors[relative]),klass,name,lambda *x:None)
                self.assertTrue(callable(transformed))

    def test_cpu_receipt_fixture_closure_only(self):
        self.assertTrue(C.validate_receipt(fixture(), mode="populate"))
        self.assertTrue(C.validate_receipt(paired_fixture(), mode="paired"))

    def test_failed_observation_rejected(self):
        value = fixture(); value["failures"] = ["observer exception"]
        with self.assertRaises(ValueError): C.validate_receipt(value, mode="populate")

    def test_missing_file_rejected_even_if_stage_totals_preserved(self):
        value = fixture(); value["native_jobs"]["load:0"]["files"].pop("7")
        with self.assertRaises(ValueError): C.validate_receipt(value, mode="populate")

    def test_foreign_request_rejected(self):
        value = fixture(); value["native_jobs"]["load:0"]["request_id"] = "other"
        with self.assertRaises(ValueError): C.validate_receipt(value, mode="populate")

    def test_failed_cuda_or_short_cqe_rejected(self):
        for field, stage, bad in (("original_event_query_true", "h2d_complete_dest_and_staging", False),
                                  ("cqe_nbytes", "ssd_write_published", 1)):
            value = fixture(); next(x for x in value["records"] if x["stage"] == stage)[field] = bad
            with self.assertRaises(ValueError): C.validate_receipt(value, mode="populate")

    def test_destination_mismatch_and_early_compute(self):
        for mutation in ("destination", "early"):
            value = fixture(); row = next(x for x in value["records"] if x["stage"] == "before_model_forward")
            if mutation == "destination": row["block_ids"][0] = 999
            else: row["sequence"] = 1
            with self.assertRaises(ValueError): C.validate_receipt(value, mode="populate")

    def test_original_event_true_site_only(self):
        ns = {}; exec(compile(SOURCE, "native-fixture.py", "exec", dont_inherit=True), ns)
        owner = ns["IoReactor"](); owner.original_calls = []; calls = []
        ready = types.SimpleNamespace(end_event=types.SimpleNamespace(query=lambda: True))
        pending = types.SimpleNamespace(end_event=types.SimpleNamespace(query=lambda: False))
        owner._pending_copies = [ready, pending]
        def observe(instance, completed):
            self.assertIn(completed, instance._pending_copies)
            self.assertEqual(instance.original_calls, [])
            calls.append(completed)
        method = C.transformed_method(owner._drain_cuda_copies, SOURCE, "native-fixture.py", "IoReactor",
                                      "_drain_cuda_copies", observe)
        method(owner)
        self.assertEqual(calls, [ready]); self.assertEqual(owner._pending_copies, [pending])
        self.assertEqual(owner.original_calls, ["complete"])

    def test_live_callable_drift_rejected(self):
        ns = {}; exec(compile(SOURCE, "native-fixture.py", "exec", dont_inherit=True), ns)
        owner = ns["IoReactor"](); owner._drain_cuda_copies = lambda: None
        with self.assertRaises(ValueError):
            C.transformed_method(owner._drain_cuda_copies, SOURCE, "native-fixture.py", "IoReactor", "_drain_cuda_copies", lambda *x: None)

    def test_callback_failure_safe_wrapper_preserves_original_progress(self):
        ns = {}; exec(compile(SOURCE, "native-fixture.py", "exec", dont_inherit=True), ns)
        owner = ns["IoReactor"](); owner.original_calls=[]
        owner._pending_copies=[types.SimpleNamespace(end_event=types.SimpleNamespace(query=lambda: True))]
        failed=[]
        def safe(*args):
            try: raise RuntimeError("capture-only failure")
            except BaseException as exc: failed.append(str(exc))
        fn=C.transformed_method(owner._drain_cuda_copies,SOURCE,"native-fixture.py","IoReactor","_drain_cuda_copies",safe)
        fn(owner)
        self.assertEqual(failed,["capture-only failure"]); self.assertEqual(owner.original_calls,["complete"])

    def test_original_exception_not_hidden(self):
        ns={};exec(compile(SOURCE,"native-fixture.py","exec",dont_inherit=True),ns)
        owner=ns["IoReactor"](); owner.original_calls=[]; calls=[]
        fn=C.transformed_method(owner._on_read_complete,SOURCE,"native-fixture.py","IoReactor","_on_read_complete",lambda *x:calls.append(True))
        with self.assertRaisesRegex(ValueError,"short read"):fn(owner,None,1)
        self.assertEqual(calls,[True])

    def test_record_overflow_refused(self):
        handle=C.CaptureHandle.__new__(C.CaptureHandle);handle.records=[{}]*C.MAX_RECORDS
        with self.assertRaises(ValueError):handle._record("excess")

    def test_duplicate_source_or_consumer_keys_rejected(self):
        for jid in ("store:0","load:0"):
            value=fixture(); pages=value["native_jobs"][jid]["files"]
            pages["1"]["key"]=pages["0"]["key"]
            with self.assertRaises(ValueError):C.validate_receipt(value,mode="populate")

    def test_paired_medium_order_cannot_be_relabelled(self):
        value=paired_fixture()
        for row in value["records"]:
            if row["stage"]=="h2d_complete_dest_and_staging":row["source_kind"]="cache"
        with self.assertRaises(ValueError):C.validate_receipt(value,mode="paired")


if __name__ == "__main__": unittest.main(verbosity=2)
