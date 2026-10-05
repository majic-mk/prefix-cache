"""Append V5 observer diagnostics only; no model or GPU code is imported."""
from pathlib import Path
import hashlib

PREP = Path(__file__).resolve().parent.parent
SOURCE = PREP / "runner/strong_native_cost_runner_v4.py"
TARGET = PREP / "runner/strong_native_cost_runner_v5.py"
raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "3f97053f9638a9e58faf8cfa3f560762e294015111b1c92e8b88a379dbc9193a"
text = raw.decode("utf-8")
marker = "    new_tree = ast.parse(new_source)"
assert text.count(marker) == 1
injection = '''    # Persist actual failures before any assertion or observer detachment.
    # This records copied CPU metadata only, never qualifies an invalid stream.
    diagnostic_source = """
    rid=scalar_adapter=frontend=capture=None
    def preserve_actual_capture(phase, exported=None):
        if active_capture is None or rid is None:
            return
        row=dict(schema='strong_actual_capture_failure_diagnostic_v1', phase=phase,
            run_id=rid, origin='native_gpu_recording', production_qualified=False,
            performance_claim=False, expected_frame_count=128, diagnostic_errors=[])
        # Each read is best effort, so recording cannot prevent original cleanup.
        try:
            actual_export=exported if type(exported) is dict else active_capture.export()
            row['actual_capture_export']=actual_export
        except Exception as diagnostic_exc:
            row['diagnostic_errors'].append('export:'+type(diagnostic_exc).__name__+':'+str(diagnostic_exc)[:256])
        try:
            observer=active_capture.observer
            scalar=observer.scalar
            adapter=scalar._adapter
            actual_frames=list(adapter.frames)
            row['scalar_adapter']=dict(initial_identity=id(scalar_adapter),current_identity=id(adapter),
                same_object=adapter is scalar_adapter,actual_frame_count=len(actual_frames),
                initial_native_source_sha256=getattr(scalar_adapter,'native_source_sha256',None),
                current_native_source_sha256=adapter.native_source_sha256,
                actual_expected_native_source_ref=refs[REACTOR],
                valid=adapter.valid,enabled=adapter.enabled,last_reason=adapter.last_reason,
                frames=[dataclasses.asdict(frame) for frame in actual_frames[:128]],
                frames_truncated=len(actual_frames)>128)
            pending=adapter._pending
            if type(pending) is dict:
                row['actual_prepared_pending']=dict(ordinal=pending.get('ordinal'),phase=pending.get('phase'),
                    start_ns=pending.get('start_ns'),frame=(dataclasses.asdict(pending['frame'])
                    if dataclasses.is_dataclass(pending.get('frame')) else None))
            else:
                row['actual_prepared_pending']=None
            row['observer_stats']=dict(enabled=observer.enabled,status=observer.status,reason=observer.reason,
                scalar_enabled=scalar.enabled,scalar_status=scalar.status,scalar_last_reason=scalar.last_reason,
                scalar_adapter_last_reason=adapter.last_reason,
                event_valid=observer.events.valid,event_last_reason=observer.events.reason,
                pending_event_pairs=len(observer.events.pending),open_event_pair=observer.events.active is not None,
                actual_capture_valid=active_capture.valid,actual_capture_failures=list(active_capture.failures))
        except Exception as diagnostic_exc:
            row['diagnostic_errors'].append('scalar:'+type(diagnostic_exc).__name__+':'+str(diagnostic_exc)[:256])
        try:
            if frontend is not None:
                row['actual_frontend']=frontend
            filename=rid+'-'+phase+'-capture-diagnostic.json'
            write_json(out/filename,row)
            result.setdefault('actual_capture_diagnostics',[]).append(dict(phase=phase,path=filename))
        except Exception as diagnostic_exc:
            result.setdefault('capture_diagnostic_write_errors',[]).append(
                type(diagnostic_exc).__name__+':'+str(diagnostic_exc)[:256])
"""
    before = "    try:\\n        sys.path[:0]="
    require(new_source.count(before) == 1, "original window setup diagnostics preimage")
    new_source = new_source.replace(before, diagnostic_source + before)
    before = "                capture=active_capture.export()\\n                require(active_capture.observer.scalar._adapter is scalar_adapter and"
    after = "                capture=active_capture.export()\\n                preserve_actual_capture('preassert',capture)\\n                require(active_capture.observer.scalar._adapter is scalar_adapter and"
    require(new_source.count(before) == 1, "original frame assertion diagnostics preimage")
    new_source = new_source.replace(before, after)
    before = "        if active_capture is not None:\\n            try: active_capture.detach()"
    after = "        if active_capture is not None:\\n            preserve_actual_capture('finally',capture)\\n            try: active_capture.detach()"
    require(new_source.count(before) == 1, "original capture cleanup diagnostics preimage")
    new_source = new_source.replace(before, after)
'''
updated = text.replace(marker, injection + marker)
with TARGET.open("xb") as stream:
    stream.write(updated.encode("utf-8"))
print(TARGET.name, len(TARGET.read_bytes()), hashlib.sha256(TARGET.read_bytes()).hexdigest())
