"""CPU fixtures only: no RPC, query, device opening or GPU qualification."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import stat
import sys
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "uuid_minor_device.py"
spec = importlib.util.spec_from_file_location("_cpu_uuid_minor_fixture", SOURCE)
B = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = B
spec.loader.exec_module(B)
GPU = "GPU-b2de2c25-cdc7-a350-267f-56e7763a287f"
OTHER = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def xml(uuid=GPU, minor="1", *, index="0", extra=""):
    return (f'<?xml version="1.0"?><!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd">'
            f'<nvidia_smi_log><gpu id="0000:01:00.0"><uuid>{uuid}</uuid><minor_number>{minor}</minor_number>'
            f'<index>{index}</index></gpu>{extra}</nvidia_smi_log>').encode()


class DeviceMinorCPU(unittest.TestCase):
    def test_index_zero_can_have_physical_minor_one(self):
        value = B.parse_nvidia_smi_xml(xml(index="0", minor="1"), expected_uuid=GPU)
        self.assertEqual(value.device_minor, 1)
        self.assertEqual(value.device_path, "/dev/nvidia1")
        self.assertFalse(value.production_qualified)

    def test_select_same_uuid_among_multiple_physical_gpus(self):
        extra = f"<gpu><uuid>{OTHER}</uuid><minor_number>7</minor_number><index>1</index></gpu>"
        value = B.parse_nvidia_smi_xml(xml(extra=extra), expected_uuid=OTHER)
        self.assertEqual(value.device_path, "/dev/nvidia7")
        with self.assertRaisesRegex(ValueError, "selected live UUID"):
            B.parse_nvidia_smi_xml(xml(), expected_uuid=OTHER)
        with self.assertRaisesRegex(ValueError, "duplicate physical UUID"):
            B.parse_nvidia_smi_xml(xml(extra=extra.replace(OTHER, GPU)), expected_uuid=GPU)

    def test_missing_duplicate_unknown_and_noninteger_minor_rejected(self):
        for raw in (xml().replace(b"<minor_number>1</minor_number>", b""),
                    xml().replace(b"<minor_number>1</minor_number>", b"<minor_number>1</minor_number>" * 2),
                    xml(minor="N/A"), xml(minor="-1"), xml(minor="255"), xml(minor="256"), xml(minor="1.0")):
            with self.assertRaises(ValueError):
                B.parse_nvidia_smi_xml(raw, expected_uuid=GPU)

    def test_ctl_uvm_or_unrelated_node_cannot_satisfy_gpu(self):
        value = B.parse_nvidia_smi_xml(xml(), expected_uuid=GPU)
        for path in ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia0", "/dev/nvidia2"):
            with self.assertRaisesRegex(ValueError, "exact GPU node"):
                B.validate_node_metadata(value, path=path, lstat_mode=stat.S_IFCHR,
                    stat_mode=stat.S_IFCHR, device_major=195, device_minor=1)

    def test_symlink_regular_and_wrong_rdev_rejected(self):
        value = B.parse_nvidia_smi_xml(xml(), expected_uuid=GPU)
        cases = ((stat.S_IFLNK, stat.S_IFCHR, 195, 1), (stat.S_IFREG, stat.S_IFREG, 195, 1),
                 (stat.S_IFCHR, stat.S_IFCHR, 195, 0), (stat.S_IFCHR, stat.S_IFCHR, 511, 1))
        for lmode, mode, major, minor in cases:
            with self.assertRaises(ValueError):
                B.validate_node_metadata(value, path="/dev/nvidia1", lstat_mode=lmode,
                    stat_mode=mode, device_major=major, device_minor=minor)
        parsed = B.validate_node_metadata(value, path="/dev/nvidia1", lstat_mode=stat.S_IFCHR,
            stat_mode=stat.S_IFCHR, device_major=195, device_minor=1)
        self.assertEqual(parsed["actual_GPU_runs"], 0)
        self.assertFalse(parsed["production_qualified"])

    def test_proc_minor_is_not_bus_or_index_and_uuid_is_required(self):
        raw = f"Model: NVIDIA RTX 5090\nGPU UUID: {GPU}\nBus Location: 0000:01:00.0\nDevice Minor: 1\n".encode()
        value = B.parse_proc_information(raw, expected_uuid=GPU)
        self.assertEqual(value.device_path, "/dev/nvidia1")
        with self.assertRaises(ValueError):
            B.parse_proc_information(raw, expected_uuid=OTHER)
        with self.assertRaises(ValueError):
            B.parse_proc_information(raw + b"Device Minor: 0\n", expected_uuid=GPU)

    def test_xml_entity_malformed_and_unbounded_inputs_rejected(self):
        for raw in (b"<!ENTITY expand 'x'>" + xml(), b"<gpu>", b"x" * (B.MAX_XML_BYTES + 1)):
            with self.assertRaises(ValueError):
                B.parse_nvidia_smi_xml(raw, expected_uuid=GPU)

    def test_v2_only_changes_device_binding_function_and_keeps_sealed_siblings(self):
        runner = HERE.parent / "runner"
        original = runner / "strong_native_cost_runner.py"
        revised = runner / "strong_native_cost_runner_v2.py"
        old, new = ast.parse(original.read_bytes()), ast.parse(revised.read_bytes())
        changed = [getattr(a, "name", type(a).__name__) for a, b in zip(old.body, new.body)
                   if ast.dump(a, include_attributes=False) != ast.dump(b, include_attributes=False)]
        self.assertEqual(len(old.body), len(new.body))
        self.assertEqual(changed, ["bind_live_plan"])
        functions = {node.name: node for node in new.body if isinstance(node, ast.FunctionDef)}
        self.assertIn('HERE / "strong_trace_runner.py"', revised.read_text())
        self.assertEqual(ast.dump(functions["prepare_plan"], include_attributes=False),
                         ast.dump(next(n for n in old.body if getattr(n, "name", None) == "prepare_plan"), include_attributes=False))
        self.assertNotIn("/dev/nvidia0", ast.unparse(functions["bind_live_plan"]))
        self.assertIn("inspect_actual_binding", ast.unparse(functions["bind_live_plan"]))
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),
                         "524ff8a93e84609865a8e6adf7c89dc2d941906f968f7285d87f06e0b2e1cdcc")
        manifest = json.loads((runner / "RUNNER_SOURCE_MANIFEST.json").read_bytes())
        self.assertEqual(len(manifest["files"]), 16)
        for row in manifest["files"]:
            raw = (HERE.parent / row["path"]).read_bytes()
            self.assertEqual(len(raw), row["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
