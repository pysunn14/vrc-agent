"""A snapshot is complete only after all requested files have been read."""
import hashlib

from .avatar_setup import asset_files
from .diagnostics import runtime_probe
from .launch import local_path
from .profile import ProfileStore, RevisionConflict
from .settings import atomic_json


def digest_file(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024*1024): digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {path.name}")
    return {"sha256": digest.hexdigest(), "bytes": after.st_size}


def snapshot(profile, revision, *, progress):
    store = ProfileStore(profile.path)
    target = store.path.with_name(store.path.stem + ".lock.json")
    pending = target.with_name(target.name + ".pending")
    if profile.has_agent:
        files = asset_files(profile)
    else:
        from ..avatar_config import rig_path
        files = {'rig': rig_path({'rig': profile.data['bridge']['avatar_rig']}, profile.path.parent)}
    total = 3 + len(files)
    result = {"complete": False, "completed": 0, "total": total, "profile_revision": revision,
              "runtime": None, "role": profile.data["role"], "bindings": profile.data.get("bindings", {}), "assets": {}, "dependency_locks": {}}
    atomic_json(pending, result)
    if profile.has_agent:
        result["runtime"] = runtime_probe(profile)
    else:
        from .bridge_diagnostics import bridge_probe
        result["runtime"] = bridge_probe(profile)
    result["completed"] = 1
    atomic_json(pending, result); progress(1, total)
    for index, (key, path) in enumerate(files.items(), 2):
        result["assets"][key] = digest_file(path)
        result["completed"] = index
        atomic_json(pending, result); progress(index, total)
    root = local_path(profile, profile.data["runtime" if profile.has_agent else "bridge"]["project_dir"])
    for name in ("uv.lock", "package-lock.json"):
        result["dependency_locks"][name] = digest_file(root / name)
        result["completed"] += 1
        atomic_json(pending, result); progress(result["completed"], total)
    if store.read()["revision"] != revision: raise RevisionConflict("profile changed during snapshot; retry")
    result["complete"] = True
    atomic_json(target, result)
    pending.unlink()
    return {"path": str(target), **result}
