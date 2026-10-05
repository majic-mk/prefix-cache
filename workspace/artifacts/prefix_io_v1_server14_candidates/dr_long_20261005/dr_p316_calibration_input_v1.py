"""Strict ancestry for one independent single-family development calibration.

This admits no effect run and does not extend the historical qualification.
The original contract still validates the unchanged historical parent.
"""
import hashlib
import json
from pathlib import Path

CALIBRATION_SCOPE = "CAL_DEV_SINGLE_FAMILY"
PARENT_PATH = "artifacts/prefix_io_v1/server08-p3-15/all_hit-manifest.json"
PARENT_SHA256 = "1dc8f14dcd076f41ce05dacb0244444c4919693c34dda454dc80b9a5cf617d48"
PARENT_BYTES = 1122310


def _cal_require(ok, message):
    if not ok:
        raise ValueError(message)


def _cal_read(path):
    raw = Path(path).read_bytes()
    _cal_require(0 < len(raw) <= 2 * 1024**2, "bounded calibration manifest")
    def unique(items):
        value = {}
        for key, part in items:
            _cal_require(key not in value, "duplicate calibration input key")
            value[key] = part
        return value
    return raw, json.loads(raw, object_pairs_hook=unique)


def validate_calibration_registration(root, permit, manifest, manifest_path, out):
    """Check actual parent bytes and only the declared request-family change."""
    root = Path(root).resolve(strict=True)
    manifest_path, out = Path(manifest_path).resolve(), Path(out).resolve()
    _cal_require(out.parent.name.startswith("server14-dr-cal"), "calibration labels only")
    _cal_require(manifest_path.is_relative_to(root) and out.is_relative_to(root),
                 "calibration inputs and output must stay in project")
    row = permit["manifests"]["all_hit"]
    _cal_require(row == {"path": PARENT_PATH, "sha256": PARENT_SHA256},
                 "unchanged original all-hit parent required")
    parent_path = root / PARENT_PATH
    cursor = parent_path
    while cursor != root:
        _cal_require(not cursor.is_symlink(), "parent manifest symlink")
        cursor = cursor.parent
    parent_raw, parent = _cal_read(parent_path)
    _cal_require(len(parent_raw) == PARENT_BYTES and
                 hashlib.sha256(parent_raw).hexdigest() == PARENT_SHA256,
                 "original all-hit parent bytes changed")
    current_raw, current = _cal_read(manifest_path)
    _cal_require(current == manifest and current["scope"] == CALIBRATION_SCOPE and
                 current["profile"] == parent["profile"] == "all_hit",
                 "explicit single-family calibration scope required")
    _cal_require({key:value for key,value in current.items() if key not in ("scope", "requests")} ==
                 {key:value for key,value in parent.items() if key not in ("scope", "requests")},
                 "calibration changed model, engine, family tokens or other frozen fields")
    _cal_require(len(current["requests"]) == len(parent["requests"]) == 10,
                 "calibration must retain all ten complete requests")
    for child, original in zip(current["requests"], parent["requests"]):
        _cal_require(set(child) == set(original) and type(child["family_index"]) is int and
                     child["family_index"] == 0 and
                     {key:value for key,value in child.items() if key != "family_index"} ==
                     {key:value for key,value in original.items() if key != "family_index"},
                     "calibration changed request identity, arrival or non-family fields")
    _cal_require(current["families"][0]["initial_ssd_present"] is True and
                 len(current["families"][0]["tokens"]) == 16257 and current["output_tokens"] == 128,
                 "calibration needs original full-length SSD family and complete output")
    return dict(parent_manifest=parent, record=dict(scope=CALIBRATION_SCOPE,
        qualification_scope="historical parent only; new stream is independent development calibration",
        parent_manifest=dict(path=PARENT_PATH, bytes=len(parent_raw), sha256=PARENT_SHA256),
        calibration_manifest=dict(path=manifest_path.relative_to(root).as_posix(),
            bytes=len(current_raw), sha256=hashlib.sha256(current_raw).hexdigest()),
        differences=["scope", "requests[*].family_index=0"],
        request_count=10, final_input_ids_per_request=16257, complete_output_tokens=128,
        gpu_hot_prewarm="absent; original ordinary SSD recovery and cost admission retained"))


def validate_calibration_adapter_arm(frozen_config):
    _cal_require(frozen_config["current_device_adapter"]["arm"] == "CAL",
                 "independent calibration runner cannot execute U or method effects")
