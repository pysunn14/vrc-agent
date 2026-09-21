"""Shared command boundary for setup UI and headless callers."""
from pathlib import Path
import json

from .jobs import SetupJobs
from .plan import make_plan, options
from .verify import evidence_current


def runtime_action(action, payload, *, store, state_dir):
    saved = store.read()
    existing = saved['profile']
    setup_profile = existing
    if existing and existing['role'] == 'bridge':
        try: draft = json.loads(store.draft_path.read_text(encoding='utf-8'))
        except FileNotFoundError: draft = None
        # A role change is prepared in the wizard before replacing the active
        # profile. Only a current draft may prepare its new agent environment.
        if draft and draft.get('revision') == saved['revision'] and draft.get('profile', {}).get('role') in ('agent', 'combined'):
            setup_profile = draft['profile']
    # File/path discovery is shared by both roles; model operations belong to
    # the agent. This guard also applies to headless callers, not just the UI.
    if setup_profile and setup_profile['role'] == 'bridge' and action not in ('runtime.paths', 'runtime.discover'):
        raise ValueError('this operation requires an agent role; run it on the agent computer')
    jobs = SetupJobs(state_dir)
    if action == 'runtime.options': return options()
    if action == 'runtime.discover':
        from .paths import absolute_path, discover_python
        return {'python': str(discover_python(absolute_path(payload['root'], payload.get('base_dir'))))}
    if action == 'runtime.paths':
        from .paths import absolute_path
        result = {}
        for key, value in payload.get('paths', {}).items():
            if not value: result[key] = ''; continue
            path = absolute_path(value, payload.get('base_dir'))
            if not path.is_file(): raise ValueError(f'file does not exist: {path}')
            result[key] = str(path)
        return result
    if action == 'runtime.plan': return make_plan(payload)
    if action == 'runtime.setup':
        plan = make_plan(payload)
        existing = store.read()['profile']
        if plan['mode'] == 'install' and existing and existing['role'] != 'bridge':
            from ..core import Runner
            from ..launch import runner_config
            from ..profile import Profile
            selected = Path(existing['runtime']['python'])
            environment = Path(plan['paths']['environment'])
            if environment == selected.parent.parent and Runner(runner_config(Profile.parse(existing, store.path)), state_dir).observe('agent', check_health=False)['owned']:
                raise ValueError('stop agent before preparing its ARDY environment')
        return jobs.start(payload)
    if action == 'runtime.status': return jobs.status(payload.get('id'))
    if action == 'runtime.resume': return jobs.resume(payload.get('id'))
    if action == 'runtime.cancel': return jobs.cancel(payload.get('id'))
    if action == 'runtime.auth':
        import os
        state = jobs.status(payload.get('id'))
        if not state.get('plan'): raise ValueError('prepare an ARDY environment before authenticating')
        executable = Path(state['plan']['python']).parent / ('hf.exe' if os.name == 'nt' else 'hf')
        if not executable.is_file(): raise ValueError('Hugging Face CLI is not installed in the selected environment')
        return {'command': [str(executable), 'auth', 'login'], 'url': options()['auth_url']}
    if action == 'runtime.logs': return jobs.logs(payload.get('id'))
    if action == 'runtime.verify':
        if payload.get('python') or payload.get('root'):
            return jobs.start({**payload, 'mode': 'verify'})
        profile = setup_profile
        if not profile: raise ValueError('connect or install ARDY first with runtime setup')
        runtime = profile['runtime']
        request = {k: runtime[k] for k in ('python', 'project_dir', 'device', 'model', 'checkpoints_dir', 'hf_cache_dir') if runtime.get(k)}
        request.update(mode='verify', root=runtime.get('installation_root') or str(Path(runtime['python']).parent.parent))
        if request.get('device') == 'auto': request.pop('device')
        return jobs.start(request)
    if action == 'runtime.result':
        state = jobs.status(payload.get('id'))
        if state['state'] != 'complete' or not evidence_current(state['result']):
            raise ValueError('runtime verification has not completed or its files changed')
        return {**state['result']['runtime'], 'installation_root': state['result']['installation_root']}
    raise ValueError('unknown runtime setup operation')
