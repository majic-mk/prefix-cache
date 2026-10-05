"""Compile a frozen natural-input P0 recipe without loading weights or CUDA."""
import argparse
import json
from pathlib import Path

from probekv.p0_evidence_v2 import write_new_json
from probekv.p0_natural_input_v2 import compile_natural_p0_controls
from probekv.p1_input_consumer_v2 import load_frozen_documents
from probekv.source_manifest_v2 import RequestManifestRegistry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-root', type=Path, required=True)
    parser.add_argument('--file-index-sha256', required=True)
    parser.add_argument('--spec', type=Path, required=True,
                        help='identity/build_id/teacher_token_ids/comparison_profile/upper_seconds_by_arm JSON')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('fresh output required; never replace previous evidence')
    if args.spec.stat().st_size > 1024 * 1024:
        raise ValueError('bounded explicit diagnostic spec required')
    spec = RequestManifestRegistry._parse(args.spec.read_bytes())
    documents = load_frozen_documents(args.frozen_root, index_sha256=args.file_index_sha256)
    controls = compile_natural_p0_controls(documents, **spec)
    args.output.mkdir(parents=True, exist_ok=False)
    write_new_json(args.output/'controls.json', controls)
    write_new_json(args.output/'input_binding.json', dict(
        frozen_root=str(args.frozen_root.resolve()), file_index_sha256=args.file_index_sha256,
        controls_sha256=controls['controls_sha256'], model_loaded=False, gpu_actions_executed=0,
        P1_execution_allowed=False))
    print(json.dumps(dict(status=controls['status'], actions=controls['maximum_actions'],
                          gpu_execution_allowed=False, output=str(args.output)), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
