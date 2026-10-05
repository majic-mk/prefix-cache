"""Finite server CPU replay. No GPU launch, author edit or calibration export."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--drain-source", required=True, type=Path)
    parser.add_argument("--connector-source", required=True, type=Path)
    parser.add_argument("--adapter-source", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "explicit no-GPU CPU environment required")
    root = args.root.resolve(strict=True)
    here = Path(__file__).resolve().parent
    require(root.as_posix() == "/root/autodl-tmp/prefix-io-v1-handoff/project", "intended current server project required")
    require(args.receipt.resolve().parent == here and not args.receipt.exists(), "new receipt inside this CPU artifact required")
    lock_raw = args.lock.read_bytes()
    lock_sha = hashlib.sha256(lock_raw).hexdigest()
    require(lock_sha == "0325cb500f74051dedf6aefe6b51fd9d0b817f831ee81fe45e1c3942df13336b", "prior approved G2 source lock required")
    locked = json.loads(lock_raw)
    inputs = json.loads(args.inputs.read_bytes())
    require(inputs["expected_context"]["source_lock_sha256"] == lock_sha, "context/source lock differs")
    source_refs = {row["path"]: row for row in locked["files"]}
    require(len(source_refs) == 4050, "complete prior source lock required")
    tests = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               PREFIX_G3_CPU_NATIVE_ROOT=str(root / "third_party/work/py-kvcache-p4-02-cpu"),
               PREFIX_G3_CPU_LOCK_PATH=str(args.lock))
    for filename, extra in (
        ("test_g3_native_pilot_contract.py", []),
        ("test_g3_curve_context_audit.py", []),
        ("test_g3_native_owner_observer.py", ["--source-root", str(root), "--support-root", str(root),
           "--drain-source", str(args.drain_source), "--connector-source", str(args.connector_source),
           "--adapter-source", str(args.adapter_source)]),
        ("test_g3_kv_byte_evidence.py", ["--source-root", str(root), "--lock", str(args.lock)]),
    ):
        require((here / filename).is_file(), "complete CPU test set required: " + filename)
        command = [sys.executable, "-B", "-I", "-S", str(here / filename), *extra]
        started = time.monotonic()
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
        elapsed = time.monotonic() - started
        log = result.stdout + result.stderr
        match = re.search(r"Ran (\d+) tests? in ([0-9.]+)s", log)
        count = int(match.group(1)) if match else None
        log_path = args.receipt.with_name(args.receipt.stem + "-" + filename + ".log")
        with log_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(log)
        tests.append(dict(file=filename, command=command, exit=result.returncode,
                          tests=count, passed=result.returncode == 0 and count is not None and bool(re.search(r"\nOK\s*$", log)),
                          elapsed_s=elapsed, log=log_path.name,
                          log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest()))
    context_audit = load_module("g3_curve_context_cpu_replay", here / "g3_curve_context_audit.py")
    contract = load_module("g3_contract_cpu_replay", here / "g3_native_pilot_contract.py")
    native_root = root / "third_party/work/py-kvcache-p4-02-cpu"
    refs = {name: source_refs["third_party/work/py-kvcache-p4-02-cpu/py_kvcache/" + name]
            for name in contract.SOURCE_FILES}
    probe = contract.BoundOriginalCPUProbe(native_root, refs)
    curve_results = []
    try:
        for ref in inputs["existing_curves"]:
            relative = Path(ref["path"])
            require(not relative.is_absolute() and ".." not in relative.parts, "project-relative curve required")
            path = root / relative
            require(root in path.resolve(strict=True).parents, "curve escaped project")
            audit = context_audit.inspect_curve_context(path, ref, inputs["expected_context"])
            # Deliberately pass the actual engine model path; never substitute the
            # historical logical model alias merely to get an admission result.
            try:
                candidate = contract.probe_cold_consumer(path, ref,
                    model_name=inputs["expected_context"]["model_name"], kv_dtype="auto",
                    prompt_tokens=128, max_model_len=256,
                    layout_summary=dict(kv_bytes_per_token=57344, storage_block_tokens=16, max_preload_slots=17),
                    probe=probe)
                pricing = dict(status="CPU_PREDICTION_ONLY", result=candidate)
            except ValueError as error:
                pricing = dict(status="EXPECTED_CURRENT_CONTEXT_REJECTION", reason=str(error))
            curve_results.append(dict(path=ref["path"], context_audit=audit, original_cpu_probe=pricing))
    finally:
        probe.close()
    require(not any(name.split(".")[0] in ("torch", "vllm", "numpy", "triton", "cupy", "py_kvcache")
                    for name in sys.modules), "unexpected real backend import")
    receipt = dict(status="PASS_SERVER_CPU_G3_PREPARATION" if all(row["passed"] for row in tests) else "FAIL_SERVER_CPU_G3_PREPARATION",
                   python=sys.version, CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
                   source_lock_sha256=lock_sha, tests=tests,
                   total_tests=sum(row["tests"] or 0 for row in tests), curve_results=curve_results,
                   current_environment_recalibration_required=True,
                   byte_capture_installed=False, native_owner_hook_installed=False,
                   GPU_operations=0, gpu_authorized=False, production_qualified=False,
                   SSD_restore_verified=False, performance_claim=False)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(dict(status=receipt["status"], total_tests=receipt["total_tests"],
                          tests=[dict(file=r["file"], tests=r["tests"], passed=r["passed"]) for r in tests],
                          existing_curves=len(curve_results), blocked_curves=sum(r["context_audit"]["status"] == "CPU_EXISTING_CURVE_BLOCKED" for r in curve_results),
                          receipt=str(args.receipt), GPU_operations=0)))
    return 0 if receipt["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
