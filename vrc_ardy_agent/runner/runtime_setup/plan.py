"""Read-only installation plans; a plan never starts a download."""
import json
from pathlib import Path
import platform
import shutil

from .paths import absolute_path, default_root, discover_python, installation_paths, writable_directory
from .recipe import models, model_layout, recipe_for


def host_info():
    system = {'Darwin': 'macos', 'Linux': 'linux', 'Windows': 'windows'}.get(platform.system(), platform.system().lower())
    arch = {'aarch64': 'arm64', 'amd64': 'x86_64'}.get(platform.machine().lower(), platform.machine().lower())
    import psutil
    tools = {name: shutil.which(name) for name in ('uv', 'git', 'c++', 'cl', 'g++', 'nvidia-smi')}
    return {'os': system, 'arch': arch, 'tools': tools, 'memory_bytes': psutil.virtual_memory().total,
            'device': 'mps' if system == 'macos' and arch == 'arm64' else 'cuda' if tools['nvidia-smi'] else 'cpu'}


def check_ownership(paths):
    root = Path(paths['root'])
    marker = root / '.vrc-agent-runtime.json'
    if marker.exists():
        data = json.loads(marker.read_text())
        if data.get('owner') != 'vrc-agent' or data.get('root') != str(root):
            raise ValueError('installation directory is not owned by this runner')
    elif any((root / name).exists() for name in ('source', 'runtime')):
        raise ValueError('existing source/runtime is not owned by this runner; use existing environment connection')


def make_plan(request):
    allowed = {'mode', 'root', 'base_dir', 'models_dir', 'python', 'device', 'model', 'project_dir', 'checkpoints_dir', 'hf_cache_dir'}
    if set(request) - allowed: raise ValueError('unknown runtime setup option')
    mode = request.get('mode', 'install')
    if mode not in ('install', 'connect', 'verify'): raise ValueError('invalid runtime setup mode')
    base = absolute_path(request.get('base_dir', str(Path.cwd())))
    inferred_root = Path(absolute_path(request['python'], base)).parent.parent if request.get('python') else default_root()
    root = absolute_path(request.get('root') or inferred_root, base)
    if mode == 'install' and request.get('python'): raise ValueError('use existing environment connection to select a Python executable')
    supplied_models = request.get('models_dir')
    selected_python = root if mode != 'install' and root.is_file() else None
    if selected_python: root = root.parent.parent
    root = root.resolve()
    paths = installation_paths(root, absolute_path(supplied_models, base) if supplied_models else None)
    host = host_info()
    plan = {'mode': mode, 'paths': paths, 'host': host, 'device': request.get('device') or host['device'],
            'recipe': None, 'model': request.get('model') or 'ARDY-Core-RP-20FPS-Horizon40',
            'project_dir': str(absolute_path(request.get('project_dir') or Path(__file__).parents[3], base)),
            'checks': [], 'model_bytes': 0}
    if plan['device'] not in ('mps', 'cuda', 'cpu'): raise ValueError('select an explicit runtime device')
    if mode == 'install':
        check_ownership(paths)
        plan['recipe'] = recipe_for(host)
        if plan['device'] != plan['recipe']['device']:
            raise ValueError(f"managed recipe requires {plan['recipe']['device']}; connect an existing environment for other devices")
        plan['model'] = plan['recipe']['model']
        plan['model_bytes'] = sum(f['bytes'] for model in models() for f in model['files'])
        for name in ('uv', 'git'):
            plan['checks'].append({'check': name, 'result': 'pass' if host['tools'].get(name) else 'error', 'detail': f'{name} must be installed and on PATH'})
        compiler = any(host['tools'].get(name) for name in ('c++', 'cl', 'g++'))
        plan['checks'].append({'check': 'compiler', 'result': 'pass' if compiler else 'error',
                               'detail': 'C++17 compiler required: Xcode Command Line Tools, build-essential, or a Visual Studio developer terminal'})
        if plan['device'] == 'cuda':
            plan['checks'].append({'check': 'gpu', 'result': 'pass' if host['tools'].get('nvidia-smi') else 'error',
                                   'detail': 'NVIDIA GPU/driver required; availability is verified after installation'})
        for location, budget in ((paths['root'], 8*1024**3), (paths['models'], plan['model_bytes'])):
            parent = writable_directory(location)
            free = shutil.disk_usage(parent).free
            plan['checks'].append({'check': 'disk', 'result': 'pass' if free >= budget else 'error',
                                   'detail': f'{location}: {free} free bytes, up to {budget} required bytes'})
        plan.update(model_layout(paths['models']))
        plan['python'] = paths['python']
    else:
        plan['python'] = str(absolute_path(request['python'], base) if request.get('python') else selected_python or discover_python(root))
        plan['checkpoints_dir'] = str(absolute_path(request['checkpoints_dir'], base)) if request.get('checkpoints_dir') else ''
        plan['hf_cache_dir'] = str(absolute_path(request['hf_cache_dir'], base)) if request.get('hf_cache_dir') else ''
        if supplied_models: plan.update(model_layout(paths['models']))
        if not Path(plan['python']).is_file(): raise ValueError('selected Python executable does not exist')
    if selected_python: request = {**request, 'python': str(selected_python)}
    plan['request'] = {**request, 'mode': mode, 'root': str(root), 'base_dir': str(base), 'device': plan['device'], 'project_dir': plan['project_dir']}
    return plan


def options():
    host = host_info()
    try: recipe = recipe_for(host)
    except ValueError: recipe = None
    return {'host': host, 'default_root': str(default_root()), 'recipe': recipe,
            'model_bytes': sum(f['bytes'] for m in models() for f in m['files']),
            'project_dir': str(Path(__file__).parents[3]),
            'auth_url': 'https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct'}
