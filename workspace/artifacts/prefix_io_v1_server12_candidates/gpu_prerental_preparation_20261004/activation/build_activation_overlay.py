"""Append a minimal project CostTable overlay; do not edit any frozen input."""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-bundle", type=Path, required=True)
    parser.add_argument("--native-validation-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.frozen_bundle / "FROZEN_CPU_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest["files"] if row["path"].startswith("frozen/control/")]
    source = args.frozen_bundle / "frozen" / "control"
    destination = args.output / "source" / "prefix_io_control"
    destination.mkdir(parents=True, exist_ok=True)
    before = []
    for row in rows:
        path = args.frozen_bundle / row["path"]
        require(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], "original frozen control source drift")
        before.append({"path": str(path), "sha256": digest(path)})
    raw = (source / "p4_cost_table.py").read_text(encoding="utf-8")
    replacements = [
        ('__slots__ = ("cells", "scope", "source_sha256")', '__slots__ = ("cells", "scope", "source_sha256", "_gpu_proof")'),
        ('    def production_qualified(self):\n        return False', '    def production_qualified(self):\n        if self.scope != "gpu_verified_exact_cells":\n            return False\n        from .gpu_cell_issuer import _table_proof_valid\n        return _table_proof_valid(self._gpu_proof, self.cells, self.source_sha256)'),
        ('        object.__setattr__(self, "source_sha256", source_sha256)', '        object.__setattr__(self, "source_sha256", source_sha256)\n        object.__setattr__(self, "_gpu_proof", None)'),
        ('        if execution == "production" or self.scope != "mock_only":\n            return None', '        if execution == "production":\n            if not self.production_qualified:\n                return None\n        elif self.scope != "mock_only":\n            return None'),
        ('                            c.uncertainty_ns, self.source_sha256)', '                            c.uncertainty_ns, self.source_sha256,\n                            mock_only=execution == "cpu_mock",\n                            production_qualified=execution == "production" and self.production_qualified)'),
    ]
    modified = raw
    for old, new in replacements:
        require(modified.count(old) == 1, "minimal CostTable patch anchor mismatch")
        modified = modified.replace(old, new)
    modified += '''\n\ndef _from_verified_gpu_capability(proof):
    """Private entry; a JSON flag or CPU candidate cannot provide this capability."""
    from .gpu_cell_issuer import _table_material
    cells, source_sha256, identity = _table_material(proof)
    if type(cells) is not tuple or not 1 <= len(cells) <= 8:
        raise ValueError("finite verified GPU cells required")
    table = CostTable(cells, scope="conditional", source_sha256=source_sha256)
    object.__setattr__(table, "scope", "gpu_verified_exact_cells")
    object.__setattr__(table, "_gpu_proof", proof)
    if not table.production_qualified:
        raise ValueError("raw GPU issuer capability rejected")
    return table
'''
    for row in rows:
        target = destination / Path(row["path"]).name
        payload = modified.encode("utf-8") if target.name == "p4_cost_table.py" else (source / target.name).read_bytes()
        with target.open("xb") as handle:
            handle.write(payload)
    issuer = Path(__file__).with_name("gpu_cell_issuer.py")
    with (destination / "gpu_cell_issuer.py").open("xb") as handle:
        handle.write(issuer.read_bytes())
    validation_target = args.output / "native_conditional_cost.py"
    require(digest(args.native_validation_source) == "675be4f91821863e23cc82128aadc040d8a9513c3aaf9b407d6c3fccd978ccb8", "original raw validator source pin")
    with validation_target.open("xb") as handle:
        handle.write(args.native_validation_source.read_bytes())
    diff = "".join(difflib.unified_diff(raw.splitlines(True), modified.splitlines(True), fromfile="frozen/control/p4_cost_table.py", tofile="activation/source/prefix_io_control/p4_cost_table.py"))
    with (args.output / "COST_TABLE_OVERLAY.patch").open("x", encoding="utf-8") as handle:
        handle.write(diff)
    require(before == [{"path": str(args.frozen_bundle / row["path"]), "sha256": digest(args.frozen_bundle / row["path"])} for row in rows], "frozen original source mutated")
    report = {"schema": "finite_exact_cell_activation_overlay_build_v1", "status": "PASS_MINIMAL_PROJECT_OVERLAY_CREATED",
              "original_sources_unchanged": True, "copied_original_control_modules": len(rows), "patched_original_modules": ["p4_cost_table.py"],
              "P4Policy_exact_type_check_unchanged": True, "CPU_mock_and_conditional_production_closed": True,
              "actual_GPU_operations": 0, "actual_GPU_qualified_cells": 0,
              "original_inputs": before, "files": [{"path": str(path.relative_to(args.output)), "bytes": path.stat().st_size, "sha256": digest(path)}
                   for path in sorted(args.output.rglob("*.py"))]}
    with (args.output / "ACTIVATION_OVERLAY_BUILD.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "control_modules": len(rows), "GPU_operations": 0}))


if __name__ == "__main__":
    main()
