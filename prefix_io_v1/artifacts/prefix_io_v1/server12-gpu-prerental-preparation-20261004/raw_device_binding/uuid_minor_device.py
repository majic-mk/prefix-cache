"""Read-only UUID-to-minor device binding; CPU parsing grants no GPU authority.

GPU display index and CUDA logical index are deliberately never used as
device-node minor numbers. The caller must still enforce the original guard,
permission, source, budget and completed-off gates.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
import xml.etree.ElementTree as ET

UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
DECIMAL = re.compile(r"(?:0|[1-9][0-9]{0,5})")
MAX_XML_BYTES = 1024 * 1024
MAX_PROC_BYTES = 65536
NVIDIA_DEVICE_MAJOR = 195


def require(value, reason):
    if not value:
        raise ValueError("UUID_MINOR_DEVICE_REJECTED: " + reason)


def uuid(value):
    require(type(value) is str and UUID.fullmatch(value), "exact GPU UUID")
    return "GPU-" + value[4:].lower()


def minor_number(value):
    require(type(value) is str and DECIMAL.fullmatch(value.strip()), "actual integer Device Minor")
    number = int(value.strip())
    require(0 <= number < 255, "physical NVIDIA GPU minor range; control minor excluded")
    return number


@dataclass(frozen=True)
class ParsedDeviceIdentity:
    gpu_uuid: str
    device_minor: int
    source_kind: str

    @property
    def device_path(self):
        return "/dev/nvidia" + str(self.device_minor)

    @property
    def production_qualified(self):
        return False


def parse_nvidia_smi_xml(raw, *, expected_uuid):
    """Parse actual query bytes, or explicitly labelled CPU fixtures only.

    nvidia-smi XML may declare its external DTD. ElementTree does not retrieve
    that DTD; entity declarations are rejected and input size stays bounded.
    """
    expected = uuid(expected_uuid)
    require(type(raw) is bytes and 0 < len(raw) <= MAX_XML_BYTES, "bounded actual XML bytes")
    require(b"<!ENTITY" not in raw.upper(), "XML entity declarations refused")
    try:
        tree = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError("UUID_MINOR_DEVICE_REJECTED: malformed XML") from exc
    require(tree.tag == "nvidia_smi_log", "original nvidia-smi XML root")
    devices = tree.findall("gpu")
    require(1 <= len(devices) <= 64, "bounded physical GPU records")
    selected, observed = [], set()
    for device in devices:
        uuid_fields = device.findall("uuid")
        require(len(uuid_fields) == 1 and type(uuid_fields[0].text) is str, "one actual UUID per GPU")
        current = uuid(uuid_fields[0].text.strip())
        require(current not in observed, "duplicate physical UUID")
        observed.add(current)
        if current == expected:
            fields = device.findall("minor_number")
            require(len(fields) == 1 and type(fields[0].text) is str, "one actual selected GPU minor")
            selected.append(ParsedDeviceIdentity(current, minor_number(fields[0].text), "nvidia_smi_xml_uuid_minor"))
    require(len(selected) == 1, "exactly one selected live UUID")
    return selected[0]


def parse_proc_information(raw, *, expected_uuid):
    """Pure alternate parser of actual /proc driver information bytes.

    Runtime v2 uses the nvidia-smi XML query; this alternate parser does not
    silently substitute unrelated /proc information after a query failure.
    """
    require(type(raw) is bytes and 0 < len(raw) <= MAX_PROC_BYTES, "bounded driver information")
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("UUID_MINOR_DEVICE_REJECTED: driver information encoding") from exc
    fields = {}
    for line in lines:
        key, separator, value = line.partition(":")
        if separator and key.strip() in ("GPU UUID", "Device Minor"):
            key = key.strip()
            require(key not in fields, "duplicate proc UUID/minor")
            fields[key] = value.strip()
    require(set(fields) == {"GPU UUID", "Device Minor"}, "original proc UUID/minor pair")
    current = uuid(fields["GPU UUID"])
    require(current == uuid(expected_uuid), "selected proc UUID")
    return ParsedDeviceIdentity(current, minor_number(fields["Device Minor"]), "proc_driver_uuid_device_minor")


def validate_node_metadata(identity, *, path, lstat_mode, stat_mode, device_major, device_minor):
    """Pure strict metadata check; fake CPU stats never imply qualification."""
    require(type(identity) is ParsedDeviceIdentity, "parsed actual UUID/minor type")
    require(type(path) is str and path == identity.device_path, "minor selects only exact GPU node")
    require(type(lstat_mode) is int and type(stat_mode) is int and not stat.S_ISLNK(lstat_mode), "no device symlink")
    require(stat.S_ISCHR(lstat_mode) and stat.S_ISCHR(stat_mode), "actual character GPU device")
    require(type(device_major) is int and type(device_minor) is int and
            device_major == NVIDIA_DEVICE_MAJOR and device_minor == identity.device_minor,
            "actual NVIDIA character major/minor must agree")
    return dict(device_path=path, device_major=device_major, device_minor=device_minor,
                actual_GPU_runs=0, production_qualified=False)


def inspect_actual_binding(expected_uuid):
    """Actual bounded read-only query + non-symlink device metadata, no kernels."""
    require(os.name == "posix", "Linux GPU-mode device metadata")
    selected = uuid(expected_uuid)
    command = ["nvidia-smi", "-i", selected, "-q", "-x"]
    completed = subprocess.run(command, capture_output=True, check=False, timeout=10)
    require(completed.returncode == 0, "actual selected GPU XML query succeeded")
    identity = parse_nvidia_smi_xml(completed.stdout, expected_uuid=selected)
    path = Path(identity.device_path)
    first = path.lstat()
    require(not stat.S_ISLNK(first.st_mode), "actual device symlink refused")
    actual = path.stat(follow_symlinks=False)
    require((first.st_dev, first.st_ino, first.st_mode, first.st_rdev) ==
            (actual.st_dev, actual.st_ino, actual.st_mode, actual.st_rdev), "actual node metadata drift")
    checked = validate_node_metadata(identity, path=str(path), lstat_mode=first.st_mode,
        stat_mode=actual.st_mode, device_major=os.major(actual.st_rdev), device_minor=os.minor(actual.st_rdev))
    return dict(schema="actual_readonly_uuid_minor_node_binding_v1", origin="actual_nvidia_smi_and_lstat",
        gpu_uuid=identity.gpu_uuid, device_path=checked["device_path"], device_major=checked["device_major"],
        device_minor=checked["device_minor"], query_command=command,
        raw_query_xml=completed.stdout.decode("utf-8", errors="strict"),
        raw_query_sha256=hashlib.sha256(completed.stdout).hexdigest(),
        node_device_id=actual.st_dev, node_inode=actual.st_ino,
        GPU_index_used_as_minor=False, CUDA_logical_index_used_as_minor=False,
        GPU_compute_operations=0, production_qualified=False, caller_original_gates_required=True)
