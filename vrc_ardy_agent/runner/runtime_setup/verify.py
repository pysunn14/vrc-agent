"""Readiness evidence is tied to the exact interpreter and files inspected."""
from dataclasses import fields
from pathlib import Path
import os
import time


def fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'device': stat.st_dev, 'inode': stat.st_ino}


def evidence_current(result):
    files = result.get('evidence', [])
    if not files: return False
    try: return all(fingerprint(item['path']) == item for item in files)
    except OSError: return False


def runtime_environment(runtime):
    env = os.environ.copy()
    # Generation must use installed assets. Install-time downloads are explicit.
    for key in ('TEXT_ENCODERS_DIR', 'TEXT_ENCODER_DEVICE', 'TEXT_ENCODER_URL', 'TEXT_ENCODER', 'CHECKPOINTS_DIR'):
        env.pop(key, None)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TEXT_ENCODER_MODE='local', PYTHONDONTWRITEBYTECODE='1')
    if runtime.get('hf_cache_dir'): env['HF_HUB_CACHE'] = runtime['hf_cache_dir']
    if runtime.get('checkpoints_dir'): env['CHECKPOINTS_DIR'] = runtime['checkpoints_dir']
    return env


def check_chunk(chunk, expected_joints=None):
    import numpy as np
    if chunk.frame_count <= 0 or not np.isfinite(chunk.fps) or chunk.fps <= 0:
        raise ValueError('ARDY returned invalid frames or FPS')
    shape = chunk.posed_joints.shape
    if len(shape) != 3 or shape[0] != chunk.frame_count or shape[-1] != 3:
        raise ValueError('ARDY returned an invalid skeleton shape')
    if expected_joints is not None and shape[1] != expected_joints:
        raise ValueError('ARDY skeleton does not match the selected model')
    if chunk.global_rot_mats.shape != (chunk.frame_count, shape[1], 3, 3):
        raise ValueError('ARDY returned invalid joint rotations')
    for field in fields(chunk):
        value = getattr(chunk, field.name)
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise ValueError(f'ARDY returned non-finite {field.name}')
    return {'frames': chunk.frame_count, 'joints': shape[1], 'fps': chunk.fps,
            'generation_seconds': chunk.generation_seconds,
            'motion_seconds': chunk.frame_count / chunk.fps}


def verify_runtime(plan):
    # Executed only inside the selected model interpreter, never the UI core.
    import importlib.metadata
    import sys
    from huggingface_hub.constants import HF_HUB_CACHE
    from vrc_ardy_agent.ardy_runtime import ArdyHeadlessRuntime
    started = time.monotonic()
    runtime = ArdyHeadlessRuntime.load(model_name=plan['model'], device=plan['device'],
        checkpoints_dir=plan.get('checkpoints_dir') or None, text_encoder_mode='local')
    runtime.set_prompt('A person stands calmly.')
    result = check_chunk(runtime.generate_next(), expected_joints=27 if 'Core' in plan['model'] or plan['model'].startswith('core') else None)
    second = check_chunk(runtime.generate_next(), expected_joints=result['joints'])
    result['continuation_frames'] = second['frames']
    result['elapsed_seconds'] = time.monotonic() - started
    result['executed_in_game'] = False
    config = {'python': plan['python'], 'project_dir': plan['project_dir'], 'device': plan['device'],
              'model': plan['model'], 'text_encoder_mode': 'local',
              'hf_cache_dir': plan.get('hf_cache_dir') or HF_HUB_CACHE}
    if plan.get('checkpoints_dir'): config['checkpoints_dir'] = plan['checkpoints_dir']
    files = [Path(sys.executable)]
    import ardy
    import motion_correction._motion_correction as native
    files += [Path(native.__file__)]
    files += list(Path(ardy.__file__).parent.rglob('*.py'))
    files += [Path(__file__).parents[2] / 'ardy_runtime.py']
    cfg = Path(sys.executable).parent.parent / 'pyvenv.cfg'
    if cfg.is_file(): files.append(cfg)
    for name in ('ardy', 'torch', 'transformers', 'numpy'):
        dist = importlib.metadata.distribution(name)
        for item in dist.files or []:
            if str(item).endswith('.dist-info/METADATA'): files.append(Path(dist.locate_file(item)))
    from .recipe import models
    model_roots = [Path(config['hf_cache_dir']) / ('models--' + model['repo'].replace('/', '--')) for model in models()]
    if config.get('checkpoints_dir'): model_roots.append(Path(config['checkpoints_dir']))
    for root in model_roots:
        if root.exists():
            files += [p for p in root.rglob('*') if p.is_file() and (p.suffix in ('.safetensors', '.npy', '.yaml', '.json') or p.parent.name == 'refs')]
    result.update(runtime=config, evidence=[fingerprint(p) for p in dict.fromkeys(files)],
                  packages={name: importlib.metadata.version(name) for name in ('ardy', 'torch', 'transformers', 'numpy')})
    return result
