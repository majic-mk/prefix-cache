"""Private process-only server migration of the unmodified reference runtime.

The caller owns source-lock, human scope and original budget-guard validation.
This adapter changes identity/configuration globals, never model/cache logic.
Its receipt is separate so every original runtime receipt remains untouched.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

SOURCE = "artifacts/prefix_io_v1/server10-reference-migration-v1-20261003/server10_reference_runtime.py"
ORIGINAL = "artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002/g3_reference_runtime.py"
ORIGINAL_REF = dict(path=ORIGINAL, bytes=14377, sha256="4adc621057fde015e4483d2140601c1d8f68936c1e2b4dd2f828f3c04ef50ab3")
GPU_UUID = "GPU-4b4d17ec-95a3-4efd-2bc4-1613333e949f"
PERMISSIONS = "experiments/prefix_io_v1/configs/permissions.server10.reference.yaml"
JOB_NAMES = {"cold": "server10-g3-reference-native-01", "paired": "server10-g3-reference-paired-01"}
STORAGE = {
    "cold": "experiments/prefix_io_v1/runs/server10-g3-reference-native-01-unused-storage",
    "paired": "experiments/prefix_io_v1/runs/server09-g3-calibration-02-private-storage",
}
SDK_DIRECTORY = "artifacts/prefix_io_v1/server10-cuda13-cpu-20261003/"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def checked_source(root, refs, relative):
    require(type(relative) is str and relative and not Path(relative).is_absolute()
            and "\\" not in relative and all(p not in ("", ".", "..") for p in relative.split("/")),
            "safe migration-relative source")
    row = refs.get(relative)
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"} and row["path"] == relative
            and type(row["bytes"]) is int and 0 < row["bytes"] <= 4 * 1024**2
            and type(row["sha256"]) is str and len(row["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in row["sha256"]), "exact bounded migration source ref")
    path = root / relative
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), "migration source symlink")
        cursor = cursor.parent
    require(path.resolve().is_relative_to(root) and path.is_file()
            and path.stat().st_size == row["bytes"], "migration source location/bytes drift")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == row["sha256"], "migration source SHA drift")
    return path, raw


def validated_config(root, refs, config):
    keys = {"schema_version", "gpu_uuid", "job_names", "storage", "permissions_ref", "sdk_inventory_ref", "sdk_proof_ref"}
    require(type(config) is dict and keys <= set(config), "frozen migration configuration fields")
    selected = {key: config[key] for key in keys}
    require(type(selected["schema_version"]) is int and selected["schema_version"] == 1
            and selected["gpu_uuid"] == GPU_UUID and selected["job_names"] == JOB_NAMES
            and selected["storage"] == STORAGE, "fixed server10 reference identity")
    for key in ("permissions_ref", "sdk_inventory_ref", "sdk_proof_ref"):
        row = selected[key]
        require(type(row) is dict and type(row.get("path")) is str and refs.get(row["path"]) == row,
                "migration config must match complete frozen lock")
        checked_source(root, refs, row["path"])
    require(selected["permissions_ref"]["path"] == PERMISSIONS, "fixed server10 effective permission path")
    require(all(selected[key]["path"].startswith(SDK_DIRECTORY) for key in ("sdk_inventory_ref", "sdk_proof_ref"))
            and selected["sdk_inventory_ref"]["path"] != selected["sdk_proof_ref"]["path"],
            "separate server10 SDK inventory/proof source paths")
    return json.loads(json.dumps(selected, allow_nan=False))


class MigratedRuntime:
    def __init__(self, root, refs, config):
        self.root = Path(root).resolve(strict=True)
        require(type(refs) is dict, "complete reference source mapping")
        # Take value copies; the caller cannot change the in-flight binding.
        self.refs = {key: dict(value) for key, value in refs.items()}
        own, _ = checked_source(self.root, self.refs, SOURCE)
        require(Path(__file__).resolve() == own.resolve(), "migration adapter must be loaded from locked project path")
        require(self.refs.get(ORIGINAL) == ORIGINAL_REF, "unchanged original reference runtime pin")
        checked_source(self.root, self.refs, ORIGINAL)
        self.config = validated_config(self.root, self.refs, config)
        self.last_migration_evidence = None
        self._active = False

    def _arguments(self, root, refs):
        require(Path(root).resolve(strict=True) == self.root and refs == self.refs,
                "same frozen project/source mapping for migrated runtime")
        validated_config(self.root, self.refs, self.config)

    @contextmanager
    def _session(self):
        require(not self._active, "serial fresh private migration session")
        self._active = True
        saved = []
        loaded = []
        prior = dict(sys.modules)
        evidence = dict(status="ACTIVE_PRIVATE_GLOBAL_BINDINGS", gpu_uuid=GPU_UUID,
                        original_reference_ref=dict(ORIGINAL_REF), configuration_only=True,
                        original_source_bytes_changed=False, GPU_authorization_granted=False,
                        changed_bindings=[], restored_bindings=[], private_modules_unloaded=[],
                        all_bindings_restored=False, all_new_private_modules_unloaded=False)

        def patch(module, key, value):
            old = getattr(module, key)
            saved.append((module, key, old, value))
            setattr(module, key, value)
            evidence["changed_bindings"].append(dict(module=module.__name__, global_name=key))

        def common_binding(common):
            require(common.load_sdk_assets.__globals__ is common.__dict__
                    and common.prepare_process_sdk.__globals__ is common.__dict__, "actual common function globals")
            replacement = {common.SDK_INVENTORY: self.config["sdk_inventory_ref"],
                           common.SDK_PROOF: self.config["sdk_proof_ref"]}
            require(set(replacement) <= {row[0] for row in common.SDK_SOURCE_REFS}, "original SDK inventory/proof pins")
            sdk_refs = tuple((replacement[p]["path"], replacement[p]["bytes"], replacement[p]["sha256"])
                             if p in replacement else (p, n, sha) for p, n, sha in common.SDK_SOURCE_REFS)
            for key, value in (("GPU_UUID", GPU_UUID), ("PERMISSIONS", PERMISSIONS),
                               ("SDK_INVENTORY", self.config["sdk_inventory_ref"]["path"]),
                               ("SDK_PROOF", self.config["sdk_proof_ref"]["path"]), ("SDK_SOURCE_REFS", sdk_refs)):
                patch(common, key, value)
            return common

        try:
            path, raw = checked_source(self.root, self.refs, ORIGINAL)
            name = "_server10_original_reference_" + hashlib.sha256(str(path).encode()).hexdigest()[:20]
            require(name not in sys.modules, "fresh original reference private module")
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            loaded.append(module)
            exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
            original_load_delegate = module._load_delegate

            def load_delegate(root, refs):
                delegate = original_load_delegate(root, refs)
                loaded.append(delegate)
                require(delegate.execute_original_acquisition.__globals__ is delegate.__dict__, "actual delegate function globals")
                patch(delegate, "GPU_UUID", GPU_UUID)
                original_load_source = delegate.load_source

                def load_source(root, refs, relative):
                    common = original_load_source(root, refs, relative)
                    loaded.append(common)
                    return common_binding(common) if relative == delegate.COMMON else common

                patch(delegate, "load_source", load_source)
                return delegate

            for key, value in (("GPU_UUID", GPU_UUID), ("JOB_NAMES", dict(JOB_NAMES)),
                               ("STORAGE", dict(STORAGE)), ("_load_delegate", load_delegate)):
                patch(module, key, value)
            require(module.validate_guard.__globals__ is module.__dict__
                    and module.execute_original_acquisition.__globals__ is module.__dict__, "actual reference function globals")
            yield module
        finally:
            for module, key, original, replacement in reversed(saved):
                # Nested original runtime legitimately restores its own overrides;
                # ours must still be the exact values installed in these slots.
                was_expected = getattr(module, key) is replacement
                setattr(module, key, original)
                evidence["restored_bindings"].append(dict(module=module.__name__, global_name=key,
                    installed_binding_preserved=was_expected, restored=getattr(module, key) is original))
            for module in reversed(loaded):
                if sys.modules.get(module.__name__) is module:
                    sys.modules.pop(module.__name__)
            # Existing G2 helper loader memoizes private source-only modules.
            # Remove only newly created private entries; preserve all pre-existing.
            for name, module in tuple(sys.modules.items()):
                if name not in prior and name.startswith(("_g2_lifecycle_", "_g3_calibration_", "_g3_reference_delegate_", "_server10_original_reference_")):
                    sys.modules.pop(name)
                    evidence["private_modules_unloaded"].append(name)
            for module in loaded:
                evidence["private_modules_unloaded"].append(module.__name__)
            evidence["private_modules_unloaded"] = sorted(set(evidence["private_modules_unloaded"]))
            evidence["all_bindings_restored"] = all(row["restored"] and row["installed_binding_preserved"] for row in evidence["restored_bindings"])
            evidence["all_new_private_modules_unloaded"] = all(name not in sys.modules for name in evidence["private_modules_unloaded"])
            evidence["preexisting_modules_preserved"] = all(sys.modules.get(name) is module for name, module in prior.items())
            evidence["status"] = "RESTORED_PRIVATE_GLOBAL_BINDINGS" if all(evidence[key] for key in (
                "all_bindings_restored", "all_new_private_modules_unloaded", "preexisting_modules_preserved")) else "MIGRATION_RESTORATION_FAILED"
            self.last_migration_evidence = evidence
            self._active = False

    def preflight_runtime(self, root, refs):
        self._arguments(root, refs)
        with self._session() as original:
            result = original.preflight_runtime(root, refs)
        require(self.last_migration_evidence["status"] == "RESTORED_PRIVATE_GLOBAL_BINDINGS", "migration CPU cleanup")
        return result

    def execute_original_acquisition(self, root, mode, out, storage, refs, guard_witness):
        self._arguments(root, refs)
        require(type(guard_witness) is dict and guard_witness.get("permissions_ref") == self.config["permissions_ref"],
                "actual guard uses the frozen server10 permission")
        with self._session() as original:
            result = original.execute_original_acquisition(root, mode, out, storage, refs, guard_witness)
        require(self.last_migration_evidence["status"] == "RESTORED_PRIVATE_GLOBAL_BINDINGS", "migration execution cleanup")
        return result


def create_runtime(root, refs, migration_config):
    """Return API-compatible methods; caller persists last_migration_evidence."""
    return MigratedRuntime(root, refs, migration_config)
