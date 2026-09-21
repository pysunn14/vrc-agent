"""Append-only experiment events and deterministic, resume-safe manifests."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class EventStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lockfile = (self.directory/'run.lock').open('a')
        try:
            fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lockfile.close()
            raise RuntimeError('another benchmark owns this output directory')
        self._lock = threading.Lock()
        self._stream = (self.directory/'events.jsonl').open('a', buffering=1)

    def append(self, row):
        with self._lock:
            self._stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
            self._stream.flush()
            os.fsync(self._stream.fileno())

    def close(self):
        self._stream.close()
        self._lockfile.close()


def read_events(directory):
    path = Path(directory)/'events.jsonl'
    if not path.exists(): return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try: rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            # Do not silently drop a possibly charged call after a crash.
            raise ValueError(f'event log has an incomplete/corrupt row {number}; inspect before resuming') from exc
    return rows


def code_identity(root):
    root = Path(root)
    files = sorted([*root.glob('vrc_ardy_agent/*.py'), *root.glob('scripts/*.py'),
                    *root.glob('benchmarks/observation/*.py')])
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(root)).encode()); h.update(b'\0'); h.update(path.read_bytes())
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
    return dict(revision=git('rev-parse','HEAD'), dirty=bool(git('status','--porcelain')),
                source_sha256=h.hexdigest())
