"""Pinned downloads and cache reuse, executed in the selected model environment."""
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from .recipe import models


def event(**data):
    print('VRC_RUNTIME_EVENT ' + json.dumps(data), flush=True)


def valid_file(path, spec, progress=None):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != spec['bytes']: return False
    digest = hashlib.sha256() if spec.get('sha256') else hashlib.sha1()
    if not spec.get('sha256'): digest.update(f"blob {spec['bytes']}\0".encode())
    before = path.stat()
    with path.open('rb') as stream:
        read, last = 0, 0.
        while chunk := stream.read(4*1024*1024):
            digest.update(chunk); read += len(chunk)
            if progress and time.monotonic() - last >= 1:
                progress(read); last = time.monotonic()
    after = path.stat()
    return (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns) and digest.hexdigest() == (spec.get('sha256') or spec['git_sha1'])


def reuse_file(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.partial')
    temporary.unlink(missing_ok=True)
    try: os.link(source.resolve(), temporary)
    except OSError as exc:
        # Separate disks and filesystems without hardlinks need a byte copy.
        # The destination is still verified before it can become usable.
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP): raise
        event(detail=f'Copying cached file: {destination.name}')
        shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _prepare_models(plan):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.errors import LocalEntryNotFoundError, GatedRepoError
    cache = Path(plan['hf_cache_dir'])
    done = 0
    total = sum(f['bytes'] for m in models() for f in m['files'])
    for model in models():
        repo, revision = model['repo'], model['revision']
        storage = cache / ('models--' + repo.replace('/', '--'))
        for spec in model['files']:
            relative = Path(spec['name'])
            target = storage / 'snapshots' / revision / relative
            report = lambda read: event(detail=f"Verifying {repo}/{relative}", bytes_completed=done + read, bytes_total=total)
            if not valid_file(target, spec, report):
                try:
                    cached = Path(hf_hub_download(repo, spec['name'], revision=revision, cache_dir=HF_HUB_CACHE, local_files_only=True))
                except LocalEntryNotFoundError:
                    cached = None
                if cached is not None and valid_file(cached, spec, report):
                    reuse_file(cached, target)
                else:
                    event(detail=f'Downloading {repo}/{relative}', bytes_completed=done, bytes_total=total)
                    try:
                        downloaded = Path(hf_hub_download(repo, spec['name'], revision=revision, cache_dir=cache,
                                                         force_download=target.exists()))
                    except GatedRepoError as exc:
                        raise RuntimeError('authentication required: grant Llama model access and run hf auth login or set HF_TOKEN; then resume') from exc
                    if not valid_file(downloaded, spec, report): raise ValueError(f'model integrity check failed: {repo}/{relative}')
            if repo.startswith('nvidia/ARDY-'):
                projected = Path(plan['checkpoints_dir']) / repo.split('/')[1] / relative
                if not valid_file(projected, spec): reuse_file(target, projected)
            done += spec['bytes']
            event(detail=f"Ready: {repo}/{relative}", bytes_completed=done, bytes_total=total)
        # Only this versioned, runner-owned cache is changed. Upstream cache
        # snapshots and the user's global main refs are never modified.
        refs = storage / 'refs'; refs.mkdir(parents=True, exist_ok=True)
        temporary = refs / 'main.partial'; temporary.write_text(revision)
        os.replace(temporary, refs / 'main')
    return {'bytes': done, 'repositories': len(models())}


def prepare_models(plan):
    from filelock import FileLock
    cache = Path(plan['hf_cache_dir'])
    cache.parent.mkdir(parents=True, exist_ok=True)
    event(detail='Waiting for the shared model cache')
    # Multiple installations can share models, but projections and refs must
    # be published by one writer at a time, including when caches are reused.
    with FileLock(str(cache.parent / '.download.lock')):
        return _prepare_models(plan)
