"""Create the private calibration copy by four explicit, audited insertions."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
original = (HERE / "original_scripts/run_concurrent_pilot_p316.py").read_text(encoding="utf-8")
helper = (HERE / "dr_p316_calibration_input_v1.py").read_text(encoding="utf-8")
changes = [
    ('        item=permit["manifests"][manifest["profile"]]\n'
     '        base.require(item["sha256"]==hashlib.sha256(a.manifest.read_bytes()).hexdigest(),"workload changed")',
     '        calibration_registration=validate_calibration_registration(\n'
     '            Path(__file__).resolve().parents[3],permit,manifest,a.manifest,out)'),
    ('    engine_config=validate(manifest,candidate,os.environ.get("CUDA_VISIBLE_DEVICES"))',
     '    engine_config=validate(calibration_registration["parent_manifest"],candidate,os.environ.get("CUDA_VISIBLE_DEVICES"))'),
    ('    report["disk_contract"]=disk_contract',
     '    report["calibration_development_registration"]=calibration_registration["record"]\n'
     '    report["disk_contract"]=disk_contract'),
    ('        os.chdir(out)\n        from vllm import LLM,SamplingParams',
     '        validate_calibration_adapter_arm(json.loads((out/"frozen-config.json").read_text()))\n'
     '        os.chdir(out)\n        from vllm import LLM,SamplingParams'),
]
modified = original
for before, after in changes:
    if modified.count(before) != 1:
        raise ValueError("original runner insertion boundary differs")
    modified = modified.replace(before, after)
if modified.count("def main():") != 1:
    raise ValueError("one original entry required")
modified = modified.replace("def main():", helper + "\n\ndef main():", 1)
target = HERE / "run_concurrent_pilot_p316_single_family_cal_v1.py"
if target.exists():
    raise ValueError("fresh private calibration copy required")
target.write_text(modified, encoding="utf-8", newline="\n")
