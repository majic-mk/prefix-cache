"""Assemble a project-private CUDA SDK view from installed NVIDIA wheels."""
from pathlib import Path
import json
PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / ".venv/lib/python3.12/site-packages/nvidia/cu13"
DEST = PROJECT / "experiments/prefix_io_v1/cuda-toolkit"
DEST.mkdir(exist_ok=True)
links = {}
def link(dst, src):
    assert src.exists(), str(src)
    if dst.is_symlink():
        assert dst.resolve() == src.resolve(), str(dst)
    else:
        assert not dst.exists(), str(dst)
        dst.symlink_to(src)
    links[str(dst.relative_to(PROJECT))] = str(src)
for name in ("bin", "include", "lib"):
    target = DEST / name
    target.mkdir(exist_ok=True)
    for src in sorted((SOURCE / name).iterdir()):
        link(target / src.name, src)
for name in ("nvvm",):
    link(DEST / name, SOURCE / name)
assert (DEST / "include/cccl").exists()
for src in sorted((SOURCE / "lib").glob("*.so.*")):
    name = src.name.split(".so.")[0] + ".so"
    # Only the default nvrtc variant belongs under the unversioned name.
    if not (DEST / "lib" / name).exists():
        link(DEST / "lib" / name, src)
link(DEST / "lib/libcuda.so", Path("/usr/lib/x86_64-linux-gnu/libcuda.so"))
link(DEST / "lib64", DEST / "lib")
out = PROJECT / "artifacts/prefix_io_v1/new-server-02/cuda-toolkit-links.json"
out.write_text(json.dumps(links, indent=2)+"\n")
print(json.dumps({"root": str(DEST), "links": len(links), "gpu_execution": False}))
