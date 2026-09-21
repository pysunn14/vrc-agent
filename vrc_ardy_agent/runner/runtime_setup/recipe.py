"""Versioned source, dependency and model inputs; no floating model revisions."""
import hashlib
import json
from pathlib import Path

ASSETS = Path(__file__).with_name('recipes')
SOURCE_URL = 'https://github.com/nv-tlabs/ardy.git'
SOURCE_REVISION = '693f74d13b3d04a0a22ce127ee79c929dd89756b'
PYTHON_VERSION = '3.11.16'
MODEL_NAME = 'ARDY-Core-RP-20FPS-Horizon40'
RECIPES = {
    ('macos', 'arm64'): {'id': 'macos-arm64', 'device': 'mps', 'validation': 'local-source-tested'},
    ('linux', 'x86_64'): {'id': 'linux-cuda', 'device': 'cuda', 'validation': 'hardware-test-pending'},
    ('windows', 'x86_64'): {'id': 'windows-cuda', 'device': 'cuda', 'validation': 'hardware-test-pending'},
}


def models():
    return json.loads((ASSETS / 'models.json').read_text())


def recipe_for(host):
    key = (host['os'], host['arch'])
    if key not in RECIPES: raise ValueError(f'no managed ARDY recipe for {key}; connect an existing environment')
    result = dict(RECIPES[key])
    result.update(python=PYTHON_VERSION, source_url=SOURCE_URL, source_revision=SOURCE_REVISION,
                  model=MODEL_NAME, torch_backend='cu128' if result['device'] == 'cuda' else None)
    result['inputs'] = {name: hashlib.sha256((ASSETS / name).read_bytes()).hexdigest()
                        for name in (result['id'] + '.lock', 'mps.patch', 'models.json')}
    result['digest'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def model_layout(models_dir):
    # Each model manifest owns its cache refs. Different releases cannot change
    # the default refs used by an already configured, offline runtime.
    version = hashlib.sha256((ASSETS / 'models.json').read_bytes()).hexdigest()[:16]
    root = Path(models_dir) / version
    return {'checkpoints_dir': str(root / 'checkpoints'), 'hf_cache_dir': str(root / 'hub')}
