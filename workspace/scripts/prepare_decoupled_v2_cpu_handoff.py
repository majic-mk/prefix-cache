"""Build or verify a CPU evidence-bound portable snapshot, never deploy or run GPU.

All storage budgets are explicit. Existing archives are immutable. The build
keeps actual dirty-worktree bytes and does not invent a new Git commit. It does
not include .git; a receiver needs the same base commit in its own fresh clone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.p0_cpu_handoff_v2 import build_cpu_handoff, verify_cpu_handoff_archive
from probekv.source_manifest_v2 import RequestManifestRegistry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    build = commands.add_parser('build')
    verify = commands.add_parser('verify')
    for command in (build, verify):
        command.add_argument('--archive', required=True, type=Path)
        command.add_argument('--max-archive-bytes', required=True, type=int)
        command.add_argument('--max-uncompressed-bytes', required=True, type=int)
        command.add_argument('--max-members', required=True, type=int)
    build.add_argument('--workspace', required=True, type=Path)
    build.add_argument('--evidence', required=True, type=Path,
                       help='actual five CPU evidence references with file SHA256')
    verify.add_argument('--sha256', required=True, help='expected archive SHA from a separate trusted record')
    args = parser.parse_args(argv)
    limits = dict(max_archive_bytes=args.max_archive_bytes,
                  max_uncompressed_bytes=args.max_uncompressed_bytes, max_members=args.max_members)
    if args.operation == 'build':
        if not args.evidence.is_file() or args.evidence.is_symlink() or args.evidence.stat().st_size > 32768:
            raise ValueError('actual bounded non-symlink evidence reference file required')
        refs = RequestManifestRegistry._parse(args.evidence.read_bytes())
        result = build_cpu_handoff(args.workspace, evidence_refs=refs, output_archive=args.archive, **limits)
        sha = result['archive_sha256']
        metadata = result['metadata']
    else:
        sha = args.sha256
        metadata = verify_cpu_handoff_archive(args.archive, expected_sha256=sha, **limits)
    print(json.dumps(dict(status='CPU_HANDOFF_' + ('BUILT' if args.operation == 'build' else 'VERIFIED'),
        archive=str(args.archive.resolve()), archive_sha256=sha,
        base_commit=metadata['base_commit'], worktree_digest=metadata['worktree_digest'],
        member_count=len(metadata['members']), omitted_worktree_files=len(metadata['excluded_worktree_members']),
        dirty_worktree_snapshot=True, new_git_commit_created=False,
        gpu_execution_allowed=False, remote_deployment_performed=False), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
