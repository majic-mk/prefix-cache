"""Read-only, metadata-only exposure annotation; never a data qualification gate.

An absent edge means unknown, not fresh. Old partition rows are never relabelled;
the graph only joins already-recorded content/origin/lineage identities. Actual
raw prompts, locked examples, model answers and KV tensors are not inputs.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError('missing metadata identity: ' + label)
    return value


def _sha(value, label):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('invalid SHA256: ' + label)
    return value


def _dataset(value):
    name = _text(value, 'dataset').lower()
    names = {'2wiki': '2wiki', '2wikimultihopqa': '2wiki',
             'hotpotqa': 'hotpotqa', 'musique': 'musique'}
    if name not in names:
        raise ValueError('unknown dataset namespace')
    return names[name]


def _group(value, dataset):
    value = _text(value, 'group_id')
    prefix, sep, suffix = value.partition(':')
    if not sep or not suffix or _dataset(prefix) != dataset:
        raise ValueError('group/dataset namespace mismatch')
    return ('group', dataset, suffix)


def _origins(values, dataset, label):
    if not isinstance(values, list):
        raise ValueError('metadata origin list required: ' + label)
    return {('origin', dataset, _text(v, label)) for v in values}


def _row_identities(row, *, prior):
    dataset = _dataset(row['source_dataset' if prior else 'dataset'])
    group = _group(row['group_id'], dataset)
    case_id = _text(row['case_id'], 'case_id')
    origin, sep, _ = case_id.partition(':')
    if not sep or not origin:
        raise ValueError('historical case_id must preserve origin before colon')
    nodes = {group, ('origin', dataset, origin)}
    if prior:
        if (row.get('partition_role') != 'development_profile_freeze'
                or row.get('source_split') != 'calibration'
                or row.get('locked_test_accessed') is not False):
            raise ValueError('only recorded non-locked development metadata allowed')
    else:
        if row.get('source_ids_are_origin_ids_not_artifact_ids') is not True:
            raise ValueError('historical origin IDs must not be artifact IDs')
        nodes.update(_origins(row['source_origin_ids'], dataset, 'source_origin_ids'))
        if not isinstance(row['targets'], list):
            raise ValueError('target metadata list required')
        nodes.update(_origins([t['origin_example_id'] for t in row['targets']],
                              dataset, 'target origin'))
        component = row.get('known_dependency_component')
        if component is not None:
            nodes.add(_group(component, dataset))
    # Optional explicit ancestry is metadata, not inferred from matching text.
    nodes.update(_origins(row.get('lineage_origin_ids', []), dataset, 'lineage origins'))
    lineage_groups = row.get('lineage_group_ids', [])
    if not isinstance(lineage_groups, list):
        raise ValueError('metadata lineage group list required')
    for value in lineage_groups:
        nodes.add(_group(value, dataset))
    return group, nodes


def audit_partition_overlap(snapshot, census_rows, *, metadata_snapshot_sha256,
                            census_sha256):
    """Annotate known exposure from supplied metadata without modifying inputs."""
    _sha(metadata_snapshot_sha256, 'metadata snapshot')
    _sha(census_sha256, 'census')
    if (snapshot.get('kind') != 'readonly_server_metadata_audit'
            or snapshot.get('locked_test_accessed') is not False
            or snapshot.get('raw_dataset_rows_read') is not False
            or snapshot.get('mixed_split_cases_read') is not False):
        raise ValueError('expected metadata-only read-only server audit')
    partition = snapshot['old_development_partition']
    partition_sha = _sha(partition['sha256'], 'declared original partition')
    prior = partition['rows']
    if not isinstance(prior, list) or not prior or not isinstance(census_rows, list):
        raise ValueError('nonempty prior metadata and census list required')
    if len({r['case_id'] for r in census_rows}) != len(census_rows):
        raise ValueError('duplicate census case identity')
    for row in census_rows:
        if row.get('input_sha256', {}).get('partition') != partition_sha:
            raise ValueError('census references a different historical partition')
        if type(row.get('known_previously_observed')) is not bool:
            raise ValueError('known exposure flag must be explicit boolean')

    prior_nodes = [_row_identities(r, prior=True) for r in prior]
    current_nodes = [_row_identities(r, prior=False) for r in census_rows]
    parents = {}

    def find(node):
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(nodes):
        roots = sorted({find(n) for n in nodes})
        for root in roots[1:]:
            parents[root] = roots[0]

    for _, nodes in prior_nodes + current_nodes:
        union(nodes)
    prior_components = {find(group) for group, _ in prior_nodes}
    observed_components = {find(group) for row, (group, _) in zip(census_rows, current_nodes)
                           if row['known_previously_observed']}
    all_prior_nodes = set().union(*(nodes for _, nodes in prior_nodes))
    all_prior_groups = {group for group, _ in prior_nodes}
    annotations = []
    for row, (group, nodes) in zip(census_rows, current_nodes):
        component = find(group)
        direct_origins = sorted(n[2] for n in nodes & all_prior_nodes if n[0] == 'origin')
        reasons = []
        if group in all_prior_groups:
            reasons.append('DIRECT_OLD_DEVELOPMENT_CONTENT_OVERLAP')
        if direct_origins:
            reasons.append('DIRECT_OLD_DEVELOPMENT_ORIGIN_OVERLAP')
        if component in prior_components:
            reasons.append('KNOWN_COMPONENT_TOUCHES_OLD_DEVELOPMENT')
        if component in observed_components:
            reasons.append('KNOWN_COMPONENT_TOUCHES_PREVIOUSLY_OBSERVED_CENSUS_GROUP')
        annotations.append({
            'case_id': row['case_id'], 'group_id': row['group_id'],
            'known_component': list(component),
            'direct_old_partition_origin_ids': direct_origins,
            'known_exposure_detected': bool(reasons), 'reasons': reasons,
            'freshness_established': False, 'full_lineage_verified': False,
            'new_v2_qualified': False,
            'status': 'KNOWN_EXPOSURE' if reasons else 'UNKNOWN_NOT_PROVEN_FRESH',
            'full_support_pairs': sum(t.get('support_stratum') == 'full_support_sentence'
                                      for t in row['targets']),
        })
    return {
        'kind': 'p0_metadata_partition_overlap_v2',
        'evidence_scope': 'known_metadata_edges_only_not_data_qualification',
        'binding': {'metadata_snapshot_sha256': metadata_snapshot_sha256,
                    'census_sha256': census_sha256,
                    'declared_original_partition_sha256': partition_sha,
                    'original_partition_path': partition['path']},
        'partition_rows': len(prior),
        'partition_unique_groups': len(all_prior_groups),
        'partition_unique_case_origins': len({(_dataset(r['source_dataset']), r['case_id'].split(':', 1)[0])
                                             for r in prior}),
        'partition_dataset_rows': dict(sorted(Counter(_dataset(r['source_dataset']) for r in prior).items())),
        'census_rows': len(census_rows),
        'known_component_count': len({find(g) for g, _ in current_nodes}),
        'known_components_are_not_split_proof': True,
        'direct_group_overlap_count': sum('DIRECT_OLD_DEVELOPMENT_CONTENT_OVERLAP' in a['reasons'] for a in annotations),
        'direct_origin_overlap_count': sum(bool(a['direct_old_partition_origin_ids']) for a in annotations),
        'known_exposure_census_rows': sum(a['known_exposure_detected'] for a in annotations),
        'support_pairs_without_known_exposure': sum(a['full_support_pairs'] for a in annotations if not a['known_exposure_detected']),
        'annotations': annotations,
        'original_partition_bytes_reverified': False,
        'old_partitions_modified': False, 'new_split_created': False,
        'raw_dataset_rows_read': False, 'locked_test_accessed': False,
        'qualified_target_count': 0, 'qualified_source_artifact_count': 0,
        'gpu_execution_allowed': False, 'P1_execution_allowed': False,
        'paper_evidence': False,
        'limitations': [
            'Embedded development metadata does not expose all historical source ancestry.',
            'Missing metadata edges cannot establish freshness or group independence.',
            'Origin parsed before first colon follows recorded historical case-ID recipe.',
            'No original prompt, token, chronology, raw-file or Source-build qualification.',
            'No original partition bytes were re-read by this helper.',
        ],
    }


def audit_partition_overlap_files(metadata_snapshot_path, census_path, *,
                                  expected_metadata_snapshot_sha256, expected_census_sha256):
    """Read only the two explicitly named metadata files, with immutable bindings."""
    metadata_bytes = Path(metadata_snapshot_path).read_bytes()
    census_bytes = Path(census_path).read_bytes()
    for raw, expected, label in ((metadata_bytes, expected_metadata_snapshot_sha256, 'snapshot'),
                                  (census_bytes, expected_census_sha256, 'census')):
        if hashlib.sha256(raw).hexdigest() != _sha(expected, label):
            raise ValueError(label + ' file digest mismatch')
    return audit_partition_overlap(
        json.loads(metadata_bytes), [json.loads(line) for line in census_bytes.splitlines() if line.strip()],
        metadata_snapshot_sha256=expected_metadata_snapshot_sha256,
        census_sha256=expected_census_sha256)
