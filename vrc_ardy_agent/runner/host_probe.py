"""Standalone probe of the selected runtime interpreter; never loads a model."""
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import sys


def inspect():
    packages = {}
    for name in ("vrc-ardy-agent", "ardy", "numpy", "scipy", "torch", "websockets", "faster-whisper", "ctranslate2"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    result = {"python": platform.python_version(), "executable": sys.executable,
              "platform": platform.system(), "packages": packages,
              "ardy_importable": importlib.util.find_spec("ardy") is not None}
    if packages["torch"]:
        import torch
        result["devices"] = {"cpu": True, "cuda": torch.cuda.is_available(),
                             "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available()}
    else:
        result["devices"] = {"cpu": True, "cuda": False, "mps": False}
    spec = importlib.util.find_spec("ardy")
    if spec and spec.origin:
        result["ardy_source"] = {"module": spec.origin, "git_revision": None, "dirty": None}
        for directory in Path(spec.origin).parents:
            if (directory / ".git").exists():
                revision = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True)
                changes = subprocess.run(["git", "-C", str(directory), "status", "--porcelain"], capture_output=True, text=True, timeout=5, check=True)
                result["ardy_source"].update(git_revision=revision.stdout.strip(), dirty=bool(changes.stdout.strip()))
                break
    return result


if __name__ == "__main__":
    print(json.dumps(inspect()))
