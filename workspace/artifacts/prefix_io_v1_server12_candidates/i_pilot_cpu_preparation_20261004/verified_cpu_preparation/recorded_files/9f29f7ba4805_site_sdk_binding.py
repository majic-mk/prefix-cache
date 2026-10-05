"""Explicit read-only reuse of the real 580-driver CUDA13 CPU asset receipts.

The unmodified original SDK helper verifies every existing asset and creates
only the new guarded child's private layout. This module does not compile,
load a shared library, import a GPU framework, change the original common
module, or grant authority. Historical CPU receipts are revalidated, not refit.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time

ENTRY = 'artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/calibration'
SOURCE = ENTRY + '/site_sdk_binding.py'
HELPER = 'artifacts/prefix_io_v1/server09-cuda13-toolchain-v2-20261001/cuda13_sdk_overlay.py'
SITE = 'artifacts/prefix_io_v1/server10-cuda13-cpu-20261003'
INVENTORY = SITE + '/CUDA13_SOURCE_INVENTORY.json'
PROOF = SITE + '/CPU_COMPILE_LINK_RESULT.json'
RESULT = SITE + '/SDK_REBIND_RESULT.json'
SDK_EVIDENCE = (INVENTORY, PROOF, RESULT)
PINS = {
    HELPER: (16929, 'ed846361788e9bdde853e6e0aa16e361329e3f2dffbfbe2c4f396acf9e9d1ecf'),
    INVENTORY: (692329, '0e7cc1df1d8d7aed7271841d6e9f1e4188785f40ad4c74b4c6c78f670b7794b6'),
    PROOF: (5588, 'ef7a7482cc1cae539e2425d6aefc82ea8160e398ab7d8b14926f76537e614a25'),
    RESULT: (1613, '98ff8bf63963109ffd6e1b3aa87e8802ba5c74d910a5a37a34aad72c279d69bc'),
}
DRIVER = dict(path='/usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05', bytes=96276264,
    sha256='f27223c58d4c0d2ead3c2d747eb30a530c449f89c42e16b23e9bef04f3e6dc2e')
PROCESS_KEYS = ('CUDA_HOME', 'CUDA_PATH', 'FLASHINFER_NVCC', 'CUDACXX', 'PATH', 'LD_LIBRARY_PATH')


def require(value, reason):
    if not value:
        raise ValueError('SITE_SDK_REJECTED: ' + reason)


def safe(root, relative):
    root = Path(root).resolve(strict=True)
    require(type(relative) is str and relative and ':' not in relative and '\\' not in relative
        and not relative.startswith('/') and all(part not in ('', '.', '..') for part in relative.split('/')),
        'project-relative asset path')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'source/receipt symlink refused')
    require(path.resolve().is_relative_to(root), 'asset outside project')
    return path


def source_ref(root, relative):
    path = safe(root, relative)
    st = path.stat()
    require(stat.S_ISREG(st.st_mode) and 0 < st.st_size <= 32 * 1024**2, 'bounded existing SDK source/receipt')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(path.stat().st_size == st.st_size, 'asset changed while hashing')
    return dict(path=relative, bytes=st.st_size, sha256=digest)


def checked(root, refs, relative):
    require(type(refs) is dict and relative in refs, 'site SDK source missing from frozen closure')
    row = refs[relative]
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}
        and row['path'] == relative and type(row['bytes']) is int and row['bytes'] > 0
        and type(row['sha256']) is str and len(row['sha256']) == 64
        and all(value in '0123456789abcdef' for value in row['sha256']), 'exact immutable source ref')
    require(source_ref(root, relative) == row, 'site SDK source/receipt byte drift')
    if relative in PINS:
        size, digest = PINS[relative]
        require(row['sha256'] == digest and (size is None or row['bytes'] == size),
            'explicit real CPU helper/inventory/proof provenance')
    return dict(row)


def read(root, relative):
    def unique(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate CPU receipt key')
            value[key] = item
        return value
    return json.loads(safe(root, relative).read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda _value: require(False, 'nonfinite CPU receipt'))


def absolute_ref(root, row):
    return dict(row, path=str(safe(root, row['path'])))


def validate_receipts(root, rows, inventory, proof, result):
    """Check actual historical receipt identity before calling the old helper."""
    expected_inventory = absolute_ref(root, rows[INVENTORY])
    expected_proof = absolute_ref(root, rows[PROOF])
    require(type(inventory) is dict and inventory.get('driver') == DRIVER
        and type(proof) is dict and proof.get('driver_ref') == DRIVER
        and proof.get('manifest_ref') == expected_inventory,
        'real 580 inventory/proof/driver identity')
    expected = dict(status='PASS_SERVER10_CPU_SDK_REBIND_EXISTING_HELPER_VALIDATED',
        inventory_ref=expected_inventory, compiler_proof_ref=expected_proof, driver_ref=DRIVER,
        source_files=1934, source_bytes=217111529, compiler_executions=3, binutils_executions=1,
        GPU_operations=0, shared_object_loaded=False, old_assets_modified=False,
        SDK_trees_copied=False, GPU_qualification=False)
    require(type(result) is dict and all(type(result.get(key)) is type(value) and result.get(key) == value
        for key, value in expected.items()), 'real compiler-only SDK rebind receipt semantics')
    toolkit = safe(root, '.venv/lib/python3.12/site-packages/nvidia/cu13')
    require(inventory.get('toolkit_root') == str(toolkit), 'original installed CUDA13 toolkit location')
    outputs = proof.get('output_refs')
    require(type(outputs) is list and len(outputs) == 3
        and {Path(row.get('path', '')).name for row in outputs} == {'probe.cu', 'probe.o', 'probe.so'},
        'all three distinct original CPU probe outputs retained')
    for row in outputs:
        require(type(row) is dict and type(row.get('path')) is str,
            'actual original CPU probe reference')
        try:
            relative = Path(row['path']).relative_to(Path(root).resolve(strict=True)).as_posix()
        except ValueError:
            raise ValueError('SITE_SDK_REJECTED: CPU probe outside project') from None
        require(relative.startswith(SITE + '/') and relative.rsplit('/', 1)[-1] in ('probe.cu', 'probe.o', 'probe.so'),
            'original bounded CPU probe artifact')
    return expected_inventory, expected_proof


def required_source_refs(root):
    """Freeze small real site evidence without rehashing the inherited tree.

    The 1,934 SDK tree assets remain in the original ancestry. Their complete
    validation is still performed by the unchanged helper at asset preflight
    and guarded layout creation. The external driver stays a host asset pin.
    """
    root = Path(root).resolve(strict=True)
    rows = {relative: source_ref(root, relative) for relative in PINS}
    for relative, row in rows.items():
        checked(root, rows, relative)
    inventory, proof, result = (read(root, relative) for relative in SDK_EVIDENCE)
    validate_receipts(root, rows, inventory, proof, result)
    for external_row in [*proof['output_refs'], inventory['cudart']]:
        require(type(external_row) is dict and set(external_row) == {'path', 'bytes', 'sha256'}
            and type(external_row['path']) is str, 'original project CPU/SDK file reference')
        try:
            relative = Path(external_row['path']).relative_to(root).as_posix()
        except ValueError:
            raise ValueError('SITE_SDK_REJECTED: immutable SDK file outside project') from None
        row = dict(external_row, path=relative)
        checked(root, {relative: row}, relative)
        require(relative not in rows or rows[relative] == row, 'conflicting SDK source reference')
        rows[relative] = row
    return [rows[relative] for relative in sorted(rows)]


def original_helper(root, reference):
    path = safe(root, reference['path'])
    require(source_ref(root, reference['path']) == reference, 'unchanged SDK helper before import')
    name = '_server12_original_sdk_overlay_' + str(time.monotonic_ns())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), module.__dict__)
        require(source_ref(root, reference['path']) == reference, 'SDK helper drift during import')
        return module
    finally:
        sys.modules.pop(name, None)


def load_site_assets(root, refs):
    """Read-only complete actual asset audit; no compiler or library loading."""
    root = Path(root).resolve(strict=True)
    rows = {relative: checked(root, refs, relative) for relative in PINS}
    inventory, proof, result = (read(root, relative) for relative in (INVENTORY, PROOF, RESULT))
    inventory_ref, proof_ref = validate_receipts(root, rows, inventory, proof, result)
    helper = original_helper(root, rows[HELPER])
    # The original API independently verifies the entire SDK trees, runtime,
    # current 580 driver and all original source/object/shared-object bytes.
    pin = helper.load_audited_assets(inventory_ref, proof_ref)
    require(pin['driver'] == DRIVER and pin['inventory_ref'] == inventory_ref
        and pin['compiler_cpu_proof'] == proof_ref, 'unchanged helper returned the real site asset binding')
    for relative in PINS:
        require(checked(root, refs, relative) == rows[relative], 'site receipt drift after full audit')
    return helper, pin, rows


def prepare_site_sdk(root, refs, out):
    """Use the old layout/environment API in a fresh guarded child only.

    The caller owns the original active GPU guard; asset qualification alone
    neither authorizes nor launches a GPU job. Ninja setup remains unchanged.
    """
    root = Path(root).resolve(strict=True)
    out = Path(out)
    require(out.is_absolute() and out.resolve(strict=True).is_relative_to(root)
        and out.name == 'details' and not out.is_symlink(), 'real guarded child details directory')
    checked(root, refs, SOURCE)
    helper, pin, rows = load_site_assets(root, refs)
    evidence = helper.prepare_overlay(approved_run_root=out.parent, runtime_cache=out/'runtime-cache',
        pin=pin, inherited_environment=dict(os.environ))
    selected = dict(evidence.environment)
    # Only this already-guarded child's process environment changes. No change
    # is made to common module attributes, installed packages or system files.
    os.environ.update(selected)
    return dict(overlay=evidence.overlay, links=evidence.links,
        selected_environment={key: selected[key] for key in PROCESS_KEYS},
        inventory_ref=rows[INVENTORY], compiler_proof_ref=rows[PROOF],
        helper_ref=rows[HELPER], site_sdk_adapter_ref=refs[SOURCE],
        existing_rebind_receipt_ref=rows[RESULT], current_driver_ref=pin['driver'],
        CPU_assets_verified=True, GPU_model_or_JIT_runtime_qualified=False,
        production_qualified=False, stubs_on_runtime_library_path=False,
        compiler_executions_this_action=0, GPU_operations_this_action=0,
        shared_objects_loaded_this_action=False)
