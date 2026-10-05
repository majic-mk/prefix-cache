"""Append a real CPU-only SDK receipt for the cloned server10 installation.

No GPU libraries are imported or loaded. Four compiler/binutils commands run;
the emitted probe shared object is never loaded or executed. Existing SDK trees
are linked, not copied. Old receipts, packages, and system assets are read-only.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
DEST = ROOT / 'artifacts/prefix_io_v1/server10-cuda13-cpu-20261003'
OLD = ROOT / 'artifacts/prefix_io_v1/server09-cuda13-cpu-20261001'
HELPER = ROOT / 'artifacts/prefix_io_v1/server09-cuda13-toolchain-v2-20261001/cuda13_sdk_overlay.py'
PINS = {
    'helper': dict(path=str(HELPER), bytes=16929, sha256='ed846361788e9bdde853e6e0aa16e361329e3f2dffbfbe2c4f396acf9e9d1ecf'),
    'inventory': dict(path=str(OLD / 'CUDA13_SOURCE_INVENTORY.json'), bytes=692329, sha256='2ed079701c53d3e37005cf33432937e6c442f06cad16c5aded06efffcc7c36e0'),
    'proof': dict(path=str(OLD / 'CPU_COMPILE_LINK_RESULT.json'), bytes=5589, sha256='3c3f9b092aea3ca89e07f31530a2ae3b38ab65cf0a8868821031f1d7f8d14e3f'),
    'driver': dict(path='/usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05', bytes=96276264, sha256='f27223c58d4c0d2ead3c2d747eb30a530c449f89c42e16b23e9bef04f3e6dc2e'),
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def file_ref(path):
    require(path.is_absolute() and not path.is_symlink() and path.is_file()
            and path.resolve(strict=True) == path, 'canonical regular file required')
    h = hashlib.sha256()
    size = path.stat().st_size
    require(0 <= size <= 256 * 1024 * 1024, 'bounded asset')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    require(path.stat().st_size == size, 'size changed during read')
    return dict(path=str(path), bytes=size, sha256=h.hexdigest())


def verify_ref(pin):
    require(set(pin) == {'path', 'bytes', 'sha256'}, 'exact file ref')
    require(type(pin['bytes']) is int and file_ref(Path(pin['path'])) == pin, 'source byte or SHA drift')


def write_json(path, document):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(document, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write('\n')


def cpu_environment():
    # Fixed standard tool search; no inherited CUDA/compiler injection.
    return dict(PATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
                HOME='/root', LANG='C.UTF-8', LC_ALL='C.UTF-8', CUDA_VISIBLE_DEVICES='')


def execute(command, cwd):
    start = time.monotonic()
    result = subprocess.run(command, cwd=cwd, env=cpu_environment(),
                            text=True, capture_output=True, timeout=60, check=False)
    return dict(command=command, exit=result.returncode, stdout=result.stdout,
                stderr=result.stderr, wall_seconds=time.monotonic() - start)


def verify_old_trees(helper, inventory):
    toolkit = Path(inventory['toolkit_root'])
    require(toolkit == ROOT / '.venv/lib/python3.12/site-packages/nvidia/cu13', 'same installed CUDA13 root')
    require(set(inventory['directories']) == {'bin', 'include', 'nvvm'}, 'same three SDK trees')
    for name, directory in inventory['directories'].items():
        require(directory['root'] == str(toolkit / name), 'coherent original tree')
        rows = directory['files']
        require(directory['file_count'] == len(rows), 'original inventory count')
        require(helper.tree_manifest_digest(rows) == directory['inventory_sha256'], 'original inventory SHA')
        converted = []
        for row in rows:
            require(row['path'] == str(toolkit / name / helper.relative_file(row['relative'])), 'original tree path binding')
            converted.append(dict(path=row['relative'], bytes=row['bytes'], sha256=row['sha256']))
        converted.sort(key=lambda row: row['path'])
        helper.verify_tree(dict(path=str(toolkit / name), bytes=directory['total_bytes'],
                                sha256=helper.tree_manifest_digest(converted), files=converted))
    verify_ref(inventory['cudart'])
    return toolkit


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    require(os.name == 'posix' and ROOT.resolve(strict=True) == ROOT, 'expected Linux project root')
    require(not DEST.exists() and not DEST.is_symlink(), 'append-only new proof directory')
    require(DEST.parent.resolve(strict=True) == DEST.parent, 'canonical artifact parent')
    require(shutil.disk_usage(DEST.parent).free >= 8 * 1024**3 + 2 * 1024**3 + 32 * 1024**2,
            'preserve eight GiB free floor plus two GiB GPU reserve and 32 MiB CPU allowance')
    for pin in PINS.values():
        verify_ref(pin)
    spec = importlib.util.spec_from_file_location('_server10_cpu_sdk_helper', HELPER)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    inventory = helper.read_json(Path(PINS['inventory']['path']))
    old_proof = helper.read_json(Path(PINS['proof']['path']))
    require(old_proof['manifest_ref'] == PINS['inventory'], 'old compiler proof inventory binding')
    require(old_proof['driver_ref'] == inventory['driver'], 'old driver evidence binding')
    toolkit = verify_old_trees(helper, inventory)
    source = old_proof['output_refs'][0]
    require(source['path'] == str(OLD / 'probe.cu') and source['bytes'] == 54
            and source['sha256'] == '16c29eab5fcf6bd63d98acd2145779c8f131c3af1938238d0635414f5d39169a', 'original small probe source')
    verify_ref(source)
    new_inventory = json.loads(json.dumps(inventory))
    new_inventory['driver'] = dict(PINS['driver'])
    # The same strict inventory format remains accepted by the unmodified helper.
    DEST.mkdir(mode=0o755, exist_ok=False)
    commands = []
    try:
        inventory_path = DEST / 'CUDA13_SOURCE_INVENTORY.json'
        write_json(inventory_path, new_inventory)
        manifest_ref = file_ref(inventory_path)
        overlay = DEST / 'sdk-overlay'
        overlay.mkdir()
        (overlay / 'lib64').mkdir()
        (overlay / 'lib64/stubs').mkdir()
        for name in ('bin', 'include', 'nvvm'):
            os.symlink(toolkit / name, overlay / name, target_is_directory=True)
        for name in ('libcudart.so', 'libcudart.so.13'):
            os.symlink(inventory['cudart']['path'], overlay / 'lib64' / name)
        os.symlink(PINS['driver']['path'], overlay / 'lib64/stubs/libcuda.so')
        probe = DEST / 'probe.cu'
        with probe.open('xb') as stream:
            stream.write(Path(source['path']).read_bytes())
        obj, so = DEST / 'probe.o', DEST / 'probe.so'
        nvcc = str(overlay / 'bin/nvcc')
        command_list = [
            [nvcc, '--version'],
            [nvcc, '-gencode=arch=compute_120f,code=sm_120f', '-Xcompiler=-fPIC', '-c', str(probe), '-o', str(obj)],
            ['c++', '-shared', str(obj), '-L' + str(overlay / 'lib64'), '-L' + str(overlay / 'lib64/stubs'), '-lcudart', '-lcuda', '-o', str(so)],
            ['readelf', '-d', str(so)],
        ]
        for command in command_list:
            row = execute(command, DEST)
            commands.append(row)
            require(row['exit'] == 0, 'CPU compiler/link/readelf command failed')
        require('release 13.0, V13.0.88' in commands[0]['stdout'], 'same CUDA13 compiler version')
        require('[libcudart.so.13]' in commands[3]['stdout'], 'linked CUDA13 runtime SONAME')
        proof = dict(schema_version=1, status='PASS_CPU_CUDA13_SM120F_COMPILE_AND_HOST_LINK_ONLY',
                     manifest_ref=manifest_ref,
                     source_inventory_counts={name: row['file_count'] for name, row in inventory['directories'].items()},
                     source_inventory_bytes=sum(row['total_bytes'] for row in inventory['directories'].values()),
                     driver_ref=dict(PINS['driver']), cudart_ref=inventory['cudart'], compiler_commands=commands,
                     output_refs=[file_ref(p) for p in (probe, obj, so)], GPU_operations=0, GPU_kernels_executed=0,
                     framework_imports=0, shared_object_loaded=False, system_or_driver_modified=False,
                     downloads=False, stubs_on_runtime_library_path=False)
        proof_path = DEST / 'CPU_COMPILE_LINK_RESULT.json'
        write_json(proof_path, proof)
        proof_ref = file_ref(proof_path)
        pin = helper.load_audited_assets(manifest_ref, proof_ref)
        result = dict(status='PASS_SERVER10_CPU_SDK_REBIND_EXISTING_HELPER_VALIDATED',
                      inventory_ref=manifest_ref, compiler_proof_ref=proof_ref, driver_ref=pin['driver'],
                      source_files=sum(len(t['files']) for t in pin['trees'].values()),
                      source_bytes=sum(t['bytes'] for t in pin['trees'].values()),
                      compiler_executions=3, binutils_executions=1, GPU_operations=0,
                      shared_object_loaded=False, old_assets_modified=False, SDK_trees_copied=False,
                      GPU_qualification=False, original_inventory_ref=PINS['inventory'], original_proof_ref=PINS['proof'])
        write_json(DEST / 'SDK_REBIND_RESULT.json', result)
        print(json.dumps(result, ensure_ascii=True))
        return 0
    except Exception as exc:
        failure = dict(status='FAIL_SERVER10_CPU_SDK_REBIND', reason=str(exc), compiler_commands=commands,
                       GPU_operations=0, shared_object_loaded=False, old_assets_modified=False)
        write_json(DEST / 'SDK_REBIND_FAILURE.json', failure)
        print(json.dumps(failure, ensure_ascii=True))
        return 78


if __name__ == '__main__':
    raise SystemExit(main())
