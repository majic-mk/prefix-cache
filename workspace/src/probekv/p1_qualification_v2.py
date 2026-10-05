"""Read-only P0 evidence + P1 build-plan qualification, never GPU permission.

This consumer has its own digest. It does not relabel historical evidence or
change the model execution digest. Passing prepares bounded Source builds;
unbuilt P1 artifacts, native QA dispatch and authority remain separate gates.
"""
from pathlib import Path
from .p0_acceptance_v2 import assess_numerical_correctness, file_sha, verify_batch, verify_lineage_scope
from .p0_stage_readiness_v2 import _read_ref
from .p1_input_consumer_v2 import load_frozen_input_graph
from .v8_schema10_execution import digest_json

IDENTITY = ('code_commit', 'runtime_digest', 'patch_sha256', 'model_signature', 'tokenizer_hash')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _batch(spec, identity):
    binding = spec['binding']
    _require(all(binding.get(k) == identity[k] for k in IDENTITY),
             'P0 runtime/model/patch identity differs; no implicit compatibility')
    return verify_batch(spec['root'], manifest_sha256=spec['manifest_sha256'], expected_binding=binding)


def verify_birth_witness(witness, batches, identity):
    """Re-open a quiescent real store and bind its Source to a verified action.

    No request snapshot, comparison, LRU touch or publication is issued. Missing
    stores are rejected before constructors can initialize a new empty pool.
    """
    from .source_store_v2 import TargetSourceStoreV2, parse_target_catalog_v2
    from .source_manifest_v2 import RequestManifestRegistry, ManifestReference, request_input_digest
    root = str(Path(witness['batch_root']).resolve())
    _require(root in batches, 'Source witness has no verified birth batch')
    batch = batches[root]
    jobs = {j['action_id']: j for j in batch['manifest']['jobs']}
    job = jobs[witness['action_id']]; q = job['request']; sid = witness['target_id']
    _require(job.get('operation') in ('source_request', 'controlled_lineage_birth')
             and not q.get('capture_logits') and 'teacher_token_ids' not in q
             and q.get('max_new_tokens') == 1, 'diagnostic teacher/reference cannot supply Source')
    audit = batch['audits'][job['action_id']]; publication = audit['publication']
    event = publication['targets'][sid]
    _require(event['publication_performed'] is True and event['source_id'] == witness['source_id']
             and publication['extra_forward_count'] == 0 and publication['prefix_shadow_created'] is False,
             'Source not published by this ordinary action or unexpected shadow/forward')
    target = next(s for s in q['segments'] if s['segment_id'] == sid)
    _require(len(target['token_ids']) == 512, 'P1 preparation witness must have exact 512-token target')
    catalog_doc, cp = _read_ref(witness['catalog'], 'catalog')
    _, rp = _read_ref(witness['registry'], 'registry')
    spec, _ = _read_ref(witness['pool_spec'], 'pool_spec')
    _require(cp.name == 'catalog.json' and rp.name == 'registry.json'
             and (cp.parent/'writer.lock').is_file() and (cp.parent/'writer.lock').stat().st_size > 0
             and (rp.parent/'manifests').is_dir(), 'existing quiescent store required')
    cat = parse_target_catalog_v2(catalog_doc)
    _require(cat['config']['model_signature'] == identity['model_signature']
             and cat['config']['tokenizer_hash'] == identity['tokenizer_hash'], 'store identity differs')
    before = (file_sha(cp), file_sha(rp))
    registry = RequestManifestRegistry(persistent_root=rp.parent, **spec['registry_budget'])
    with TargetSourceStoreV2(cp.parent, registry=registry,
                            **{k:v for k,v in cat['config'].items() if k != 'purpose'}) as store:
        row = store._catalog['rows'][witness['source_id']]
        _require(row['birth_request_id'] == q['request_id'] and row['token_ids'] == target['token_ids']
                 and row['birth_target_positions'] == target['positions']
                 and row['generation'] == event['generation'] and row['origin'] == event['origin']
                 and row['publication_epoch'] == event['publication_epoch']
                 and row['generation'] in (0, 1)
                 and row['parent_owned_kv_bytes'] == row['prefix_shadow_bytes'] == 0,
                 'Source birth/target/lineage/publication differs')
        reference = ManifestReference(**row['manifest_reference'])
        manifest, record = store._manifest_record(reference, row['token_ids'], row['birth_target_positions'])
        positions = list(range(len(q['token_ids'])))
        _require(manifest['token_ids'] == q['token_ids'] and manifest['absolute_positions'] == positions
                 and manifest['input_digest'] == request_input_digest(q['token_ids'], positions)
                 and row['layer_count'] == manifest['num_layers'] == record['proof']['expected_layers'],
                 'full historical context or complete layer geometry differs')
        if job['operation'] == 'controlled_lineage_birth':
            proof = audit['target_proofs'][sid] if 'target_ids' in job else audit['target_proof']
            _require(record['proof'] == proof, 'stored proof differs from raw GPU lineage audit')
        store._verify_row(row, full=True)
        store._guard()
        result = dict(source_id=row['source_id'], artifact_digest=row['artifact_digest'],
            generation=row['generation'], origin=row['origin'], target_tokens=512,
            layer_count=row['layer_count'], manifest_id=reference.manifest_id,
            proof_digest=record['proof_digest'], birth_action_id=job['action_id'],
            parent_owned_kv_bytes=0, prefix_shadow_bytes=0, full_backing_verified=True,
            evidence_scope='P0_representative_birth_not_a_P1_build_receipt')
    _require(before == (file_sha(cp), file_sha(rp)), 'store changed during read-only verification')
    return result


def assess_p1_preparation(contract, *, observed_runtime_digest):
    """Derive component verdicts from raw evidence. Never accepts PASS summaries."""
    checks = {}; blockers = []; details = {}
    def check(name, fn):
        try:
            value = fn(); checks[name] = True; details[name] = value
            return value
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, MemoryError, StopIteration) as exc:
            checks[name] = False; blockers.append(dict(code=name, detail=str(exc)))
            return None

    def header():
        _require(contract.get('kind') == 'P1_preparation_evidence_v2', 'unknown qualification contract')
        identity = contract['runtime_identity']
        _require(set(identity) == set(IDENTITY) and all(type(identity[k]) is str and identity[k] for k in IDENTITY),
                 'explicit complete runtime identity required')
        _require(identity['runtime_digest'] == observed_runtime_digest, 'actual runtime differs from contract')
        _require(contract['prefix_mode'] == 'off', 'Prefix-on requires independent qualification')
        _require(contract['locked_test_accessed'] is False, 'locked test is out of scope')
        return identity
    identity = check('RUNTIME_SCOPE', header)
    batches = {}
    def numerical():
        specs = contract['p0_batches']
        _require(type(specs) is list and 0 < len(specs) <= 32, 'bounded raw P0 batch list required')
        for spec in specs:
            key = str(Path(spec['root']).resolve())
            _require(key not in batches, 'duplicate P0 batch cannot amplify evidence')
            batches[key] = _batch(spec, identity)
        report = assess_numerical_correctness(specs)
        _require(report['status'] == 'PASS', 'P0 raw numerical prerequisites failed: '+str(report['blockers']))
        lineage = [row for b in batches.values() for row in verify_lineage_scope(b)]
        generations = {g for row in lineage for g in row['expected_generation'].values()}
        _require({1, 2} <= generations, 'mixed publication and G2 rejection raw evidence required')
        return dict(counts=report['counts'], lineage=lineage,
            batches=[dict(root=k, manifest_sha256=b['manifest_file_sha256'], events_sha256=b['events_sha256'])
                     for k,b in batches.items()])
    numeric = check('P0_RAW_NUMERICS_AND_LINEAGE', numerical) if identity else None

    def inputs():
        args = contract['frozen_inputs']; graph = load_frozen_input_graph(**args)
        _require(graph['model_signature'] == identity['model_signature']
                 and graph['runtime_tokenizer_identity'] == identity['tokenizer_hash'], 'data/model identity differs')
        _require(graph['graph_sha256'] == contract['expected_input_graph_sha256'], 'frozen input graph differs')
        return graph
    graph = check('FROZEN_INPUT_GRAPH', inputs) if identity else None

    def stores():
        witnesses = contract['p0_source_witnesses']
        _require(type(witnesses) is list and 2 <= len(witnesses) <= 16, 'bounded exact and mixed witnesses required')
        rows = [verify_birth_witness(w, batches, identity) for w in witnesses]
        _require({r['generation'] for r in rows} == {0,1}, 'both G0 and G1 stored birth proofs required')
        _require(len({r['source_id'] for r in rows}) == len(rows), 'duplicate birth witness')
        return rows
    if numeric: check('P0_SOURCE_STORAGE', stores)
    ready = not blockers and all(checks.get(n) for n in
        ('RUNTIME_SCOPE','P0_RAW_NUMERICS_AND_LINEAGE','FROZEN_INPUT_GRAPH','P0_SOURCE_STORAGE'))
    if graph:
        # P0 witnesses are NOT receipts for all P1 historical/S0/E/M1 builds.
        details['FROZEN_INPUT_GRAPH'] = dict(graph_sha256=graph['graph_sha256'],
            partition_digest=graph['partition_digest'], file_index_sha256=graph['file_index_sha256'],
            source_builds_pending=[b['build_id'] for b in graph['source_builds']],
            consumer_actions=len(graph['consumer_actions']),
            maximum_build_seconds=graph['maximum_build_seconds'],
            maximum_consumer_seconds=graph['maximum_consumer_seconds'])
    report = dict(kind='P1_preparation_qualification_v2',
        status='BUILD_PLAN_PREPARED_EXECUTION_BLOCKED' if ready else 'BLOCKED',
        build_plan_preparation_ready=bool(ready), checks=checks, blockers=blockers, details=details,
        contract_sha256=digest_json(contract), qualification_consumer_sha256=file_sha(__file__),
        execution_blockers=['BOUND_NATIVE_P1_DISPATCH','ACTUAL_P1_SOURCE_BUILD_RECEIPTS',
                            'CURRENT_P1_AUTHORIZATION_AND_PREFLIGHT'],
        full_P0_complete=False, P1_execution_allowed=False, GPU_execution_allowed=False,
        P1_QA_execution_allowed=False, source_artifacts_for_P1_verified=False,
        QA_evaluated=False, paper_evidence=False, locked_test_accessed=False)
    report['report_sha256'] = digest_json(report)
    return report
