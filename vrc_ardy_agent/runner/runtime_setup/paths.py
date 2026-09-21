"""Path choices at the caller boundary, without dereferencing venv Python."""
import os
from pathlib import Path
import tempfile

from platformdirs import user_data_path


def absolute_path(value, base=None):
    if not isinstance(value, (str, Path)) or not str(value).strip() or '\0' in str(value):
        raise ValueError('path must be a non-empty string')
    path = Path(value).expanduser()
    if not path.is_absolute(): path = Path(base or Path.cwd()) / path
    # resolve() would turn venv/bin/python into the system interpreter.
    return Path(os.path.abspath(path))


def default_root():
    return user_data_path('vrc-agent', appauthor=False) / 'ardy'


def python_in(environment):
    return Path(environment) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def discover_python(root):
    root = absolute_path(root)
    if root.is_file(): return root
    candidates = [python_in(root), python_in(root / '.venv'), python_in(root / 'runtime'), python_in(root / 'venv')]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ValueError('multiple Python environments found; choose an executable' if found else
                         'Python environment not found; choose its executable')
    return found[0]


def installation_paths(root, models=None):
    root = absolute_path(root)
    models = absolute_path(models, root) if models else root / 'models'
    for reserved in (root / 'source', root / 'runtime'):
        left, right = models.resolve(), reserved.resolve()
        if left == right or left in right.parents or right in left.parents:
            raise ValueError('model directory must not overlap source or runtime')
    return {'root': str(root), 'source': str(root / 'source'), 'environment': str(root / 'runtime'),
            'python': str(python_in(root / 'runtime')), 'models': str(models)}


def existing_parent(path):
    path = Path(path)
    while not path.exists():
        if path.is_symlink(): raise ValueError(f'storage is unavailable: {path}')
        path = path.parent
    if not path.is_dir(): raise ValueError(f'not a directory: {path}')
    return path


def writable_directory(path):
    path = Path(path)
    parent = existing_parent(path)
    if not os.access(parent, os.W_OK): raise ValueError(f'directory is not writable: {parent}')
    return parent


def confirm_writable(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path) as stream:
        stream.write(b'vrc-agent'); stream.flush()
