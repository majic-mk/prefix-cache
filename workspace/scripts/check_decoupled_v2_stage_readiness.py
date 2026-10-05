"""Read-only P0/P1 input checks and a fresh report; never starts a GPU task.

Evidence JSON names concrete {path, sha256} records consumed by
assess_stage_readiness. Paths use the current working directory when relative.
No example authorization, manifest, identity, price or numeric limit is supplied.
The output directory must not exist; previous reports are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from probekv.p0_stage_readiness_v2 import STAGES, assess_stage_readiness


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=STAGES)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--evidence', type=Path, required=True,
                        help='JSON of actual named evidence references, not a passed=true certificate')
    parser.add_argument('--output', type=Path, required=True, help='new report directory, must not exist')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('readiness reports are immutable; choose a fresh output directory')
    source = args.evidence.resolve()
    if not source.is_file() or source.stat().st_size > 32 * 1024 * 1024:
        raise ValueError('actual bounded evidence-reference JSON file required')
    raw = source.read_bytes()
    refs = json.loads(raw)
    if not isinstance(refs, dict):
        raise ValueError('evidence-reference JSON must be an object')
    started = time.time()
    report = assess_stage_readiness(args.stage, workspace=args.workspace.resolve(),
                                    evidence=refs, now_unix=started)
    report['inspection'] = dict(started_at_unix=started, completed_at_unix=time.time(),
        workspace=str(args.workspace.resolve()), evidence_path=str(source),
        evidence_sha256=hashlib.sha256(raw).hexdigest(),
        network_access_performed=False, model_loaded=False, gpu_actions_executed=0,
        input_files_modified=False)
    output.mkdir(parents=True, exist_ok=False)
    destination = output / 'stage_readiness.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], stage=args.stage,
        blockers=len(report['blockers']), output=str(destination),
        gpu_execution_allowed=False, automatic_rental_allowed=False), ensure_ascii=False))
    return 0 if report['status'] == 'INPUTS_VERIFIED_RUNTIME_RECHECK_REQUIRED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
