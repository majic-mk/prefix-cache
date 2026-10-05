"""Validate a frozen P0 batch; --execute explicitly runs it on the local server.

No SSH, downloads, rental, credential handling, arbitrary factory, or auto-resume.
This entry is request/lifecycle evidence, not the numerical E/M qualification suite.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from probekv.p0_batch_v2 import validate_p0_batch, run_p0_native_batch
from probekv.p0_evidence_v2 import write_new_json
from probekv.p0_stage_readiness_v2 import validate_p0_model_geometry
from probekv.source_comparison_v2 import runtime_binding_digest
from probekv.source_manifest_v2 import RequestManifestRegistry
from probekv.source_store_v2 import TargetSourceStoreV2, parse_target_catalog_v2
from probekv.v8_schema10_storage import file_digest
from probekv.v8_schema10_native_factory import validate_native_attachment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--native-manifest',type=Path,required=True)
    parser.add_argument('--store-root',type=Path,required=True)
    parser.add_argument('--registry-root',type=Path,required=True)
    parser.add_argument('--instance-id',required=True,help='current explicitly authorized instance, not a historical default')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    started = time.perf_counter_ns()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    native = json.loads(args.native_manifest.read_text(encoding='utf-8'))
    if file_digest(args.native_manifest) != manifest['native_manifest_sha256']:
        raise ValueError('native environment manifest file changed')
    runtime = validate_native_attachment(native,allow_unmeasured=True)
    model_geometry = validate_p0_model_geometry(manifest, runtime)
    # Existing isolated store only. This command must not create a new pool in
    # place of missing initial state or silently use the legacy exact pool.
    catalog_path = args.store_root/'catalog.json'
    catalog = parse_target_catalog_v2(json.loads(catalog_path.read_text(encoding='utf-8')))
    config = catalog['config']
    source = runtime['source_provenance']
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source['code_commit'] != commit or native['binding']['code_commit'] != commit:
        raise ValueError('actual checkout is not the bound native revision')
    actual = dict(code_commit=commit,runtime_digest=runtime_binding_digest(),
        patch_sha256=native['binding']['patch_sha256'],model_signature=source['model_signature'],
        tokenizer_hash=source['tokenizer_hash'],instance_id=args.instance_id,
        gpu_uuid=runtime['cost_provenance']['gpu'],
        input_manifest_sha256=manifest['binding']['input_manifest_sha256'],
        initial_pool_sha256=file_digest(catalog_path),
        initial_registry_sha256=file_digest(args.registry_root/'registry.json'))
    validation = validate_p0_batch(manifest,actual_binding=actual,now_unix=time.time())
    if (config['model_signature'] != source['model_signature'] or config['tokenizer_hash'] != source['tokenizer_hash']
            or not args.registry_root.is_dir()):
        raise ValueError('existing v2 pool/registry/model binding missing')
    if args.execute:
        uuids = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
                                       text=True,timeout=10).splitlines()
        if len(uuids)!=1 or uuids[0].strip()!=actual['gpu_uuid']:
            raise ValueError('actual GPU differs from scoped current authorization')
    args.output.mkdir(parents=True,exist_ok=False)
    write_new_json(args.output/'preflight.json',dict(validation,execute_requested=args.execute,
        actual_binding=actual,instance_identity_source='explicit_current_operator_argument',
        gpu_identity_observed=args.execute,native_numeric_qualification='NOT_RUN',
        model_geometry=model_geometry))
    if not args.execute:
        print(json.dumps(dict(status='INPUTS_VALIDATED_GPU_NOT_RUN',output=str(args.output))))
        return 0
    store = None
    try:
        registry_cfg = manifest['registry_budget']
        registry = RequestManifestRegistry(persistent_root=args.registry_root,
            max_bytes=registry_cfg['max_bytes'],max_manifest_bytes=registry_cfg['max_manifest_bytes'])
        store = TargetSourceStoreV2(args.store_root,registry=registry,
            **{k:v for k,v in config.items() if k!='purpose'})
        # Only the fixed built-in factory is callable. It verifies model files,
        # installed patch sources, GPU and frozen server dependency versions.
        from probekv.v8_schema10_native_factory import create_native_backend, create_native_measurement_backend
        has_costs = bool(runtime.get('cost_table_path') and runtime.get('cost_table_sha256'))
        backend = create_native_backend(native) if has_costs else create_native_measurement_backend(native)
        adapter = backend.adapters['legacy_multicheckpoint']
        if not has_costs:
            adapter.costs = None  # Explicit unsupported; never a zero-cost table.
        report = run_p0_native_batch(adapter,store,manifest,actual_binding=actual,
            output=args.output/'batch',session_started_ns=started)
        print(json.dumps(report,ensure_ascii=False))
        return 0 if report['status']=='COMPLETED' else 1
    except BaseException as exc:
        write_new_json(args.output/'failed.json',dict(error_type=type(exc).__name__,detail=str(exc),
            elapsed_seconds=(time.perf_counter_ns()-started)/1e9,native_runtime_qualified=False))
        raise
    finally:
        if store is not None:
            store.close()  # Fails closed if in-flight/quarantined leases remain.


if __name__ == '__main__':
    raise SystemExit(main())
