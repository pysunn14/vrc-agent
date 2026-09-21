"""Install stages are repeatable and validate source and dependency state."""
import json
from pathlib import Path
import re

from ..settings import atomic_json
from .paths import confirm_writable
from .plan import check_ownership
from .recipe import ASSETS


def claim(plan):
    paths = plan['paths']
    check_ownership(paths)
    for key in ('root', 'models'): confirm_writable(paths[key])
    marker = Path(paths['root']) / '.vrc-agent-runtime.json'
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get('recipe') != plan['recipe']['digest']:
            raise ValueError('installation belongs to a different recipe; select a new directory')
    else:
        atomic_json(marker, {'owner': 'vrc-agent', 'root': paths['root'], 'recipe': plan['recipe']['digest']})


def source_stage(plan, run):
    recipe, root = plan['recipe'], Path(plan['paths']['source'])
    root.mkdir(exist_ok=True)
    if not (root / '.git').exists():
        if any(root.iterdir()): raise ValueError('incomplete source directory contains unknown files')
        run(['git', 'init', str(root)])
        run(['git', '-C', str(root), 'remote', 'add', 'origin', recipe['source_url']])
    observed = run(['git', '-C', str(root), 'remote', 'get-url', 'origin']).strip()
    if observed != recipe['source_url']: raise ValueError('source remote differs from recipe')
    # Fetching the pinned object is repeatable; never checkout/reset a modified tree.
    if not (root / 'pyproject.toml').exists():
        run(['git', '-C', str(root), 'fetch', '--depth', '1', 'origin', recipe['source_revision']])
        run(['git', '-C', str(root), 'checkout', '--detach', recipe['source_revision']])
    if run(['git', '-C', str(root), 'rev-parse', 'HEAD']).strip() != recipe['source_revision']:
        raise ValueError('source revision changed; select a new installation directory')
    patch = (ASSETS / 'mps.patch').read_text()
    observed = run(['git', '-C', str(root), 'diff', '--binary', 'HEAD', '--', 'ardy'])
    if not observed.strip():
        run(['git', '-C', str(root), 'apply', '--index', str(ASSETS / 'mps.patch')])
        observed = run(['git', '-C', str(root), 'diff', '--binary', 'HEAD', '--', 'ardy'])
    if observed.strip() != patch.strip(): raise ValueError('ARDY source differs from the pinned patch')
    # Include all tracked paths; a user edit outside the patch must not be ignored.
    if run(['git', '-C', str(root), 'diff', '--binary', 'HEAD']).strip() != patch.strip():
        raise ValueError('ARDY source has unrecognized changes')


def environment_stage(plan, run):
    paths, recipe = plan['paths'], plan['recipe']
    environment = Path(paths['environment'])
    if not (environment / 'pyvenv.cfg').exists():
        if environment.exists() and any(environment.iterdir()): raise ValueError('incomplete runtime directory; choose another installation location')
        run(['uv', 'venv', '--python', recipe['python'], str(environment)])
    actual = run([paths['python'], '-c', 'import platform; print(platform.python_version())']).strip()
    if actual != recipe['python']: raise ValueError('runtime Python version differs from recipe')
    lock = ASSETS / (recipe['id'] + '.lock')
    # Sync the complete lock before building ARDY, avoiding floating build tools.
    command = ['uv', 'pip', 'sync', '--python', paths['python'], '--require-hashes', str(lock)]
    if recipe.get('torch_backend'): command += ['--torch-backend', recipe['torch_backend']]
    run(command)
    run(['uv', 'pip', 'install', '--python', paths['python'], '--no-deps', '--no-build-isolation', '-e', paths['source']])
    run(['uv', 'pip', 'check', '--python', paths['python']])


def installed_environment_matches(plan, run):
    """Observe installed versions and import the native module before reuse."""
    python = Path(plan['python'])
    if not python.is_file(): return False
    code = ('import importlib.metadata as m, json, platform; import ardy, motion_correction; '
            'print(json.dumps({"python": platform.python_version(), "packages": {d.metadata["Name"].lower().replace("_", "-"): d.version for d in m.distributions()}}))')
    try: observed = json.loads(run([str(python), '-c', code]))
    except (RuntimeError, ValueError): return False
    if observed['python'] != plan['recipe']['python']: return False
    text = (ASSETS / (plan['recipe']['id'] + '.lock')).read_text()
    pins = dict(re.findall(r'^([A-Za-z0-9_.-]+)==([^\s;\\]+)', text, re.M))
    return all(observed['packages'].get(name.lower().replace('_', '-')) == version for name, version in pins.items())
