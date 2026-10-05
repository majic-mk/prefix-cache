"""Explicit GPU start; supervised timeout; new output directory on every attempt."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pacekv.audit import verify_audit, write_new_json
from pacekv.plan import verify_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.execute_gpu or args.max_seconds <= 0:
        parser.error("explicit --execute-gpu and positive --max-seconds required")
    source = Path(args.preflight)
    audit = json.loads((source / "cpu_preflight.json").read_text(encoding="utf-8"))
    plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    verify_plan(plan)
    out = Path(args.output)
    if args.worker:
        verify_audit(audit, ROOT)
        from pacekv.gpu_pilot import run
        run(plan, audit, out)
        return 0
    out.mkdir(parents=True, exist_ok=False)
    write_new_json(out / "launch.json", dict(plan_sha256=plan["plan_sha256"],
                   audit_sha256=audit["audit_sha256"], max_seconds=args.max_seconds,
                   gpu_authorized_by_this_explicit_invocation=True,
                   instance_shutdown_or_rental_authorized=False))
    cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    try:
        with (out / "worker.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.max_seconds)
        if result.returncode:
            raise RuntimeError("GPU worker failed with exit code %s" % result.returncode)
        if not (out / "summary.json").is_file():
            raise RuntimeError("missing raw-evidence summary")
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        write_new_json(out / "failed.json", dict(reason=str(exc),
                       status="TIMEOUT" if isinstance(exc, subprocess.TimeoutExpired) else "FAILED",
                       completed=False, preserve_successful_prefix=True))
        print(str(exc))
        return 2
    print("Controlled pilot finished; native serving/system benefit remains unvalidated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
