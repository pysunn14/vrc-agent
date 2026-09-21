"""Detached setup jobs, with PID identity and durable checkpoints."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from filelock import FileLock
import psutil

from ..records import owned_process, read_record
from ..settings import atomic_json, state_directory
from .plan import make_plan

ACTIVE = ('starting', 'running', 'canceling')


class SetupJobs:
    def __init__(self, state_dir=None):
        self.directory = Path(state_dir or state_directory()) / 'runtime-setup'

    def path(self, job_id):
        if not isinstance(job_id, str) or not job_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in job_id):
            raise ValueError('invalid runtime job ID')
        return self.directory / f'{job_id}.json'

    def status(self, job_id=None):
        if not job_id:
            pointer = read_record(self.directory / 'latest.json')
            job_id = pointer.get('id')
        if not job_id: return {'state': 'not-configured', 'id': None}
        result = read_record(self.path(job_id))
        if not result: raise ValueError('runtime job not found')
        result = dict(result)
        process = owned_process(result.get('pid'), result.get('created'))
        if result['state'] in ACTIVE and process is None:
            result.update(state='interrupted', error='setup worker is no longer running; resume to recheck completed steps')
        child = owned_process(result.get('child_pid'), result.get('child_created'))
        if process is None and child is not None:
            result.update(state='orphaned', error='setup worker exited but its child is alive; run runtime cancel before resuming')
        result['child_alive'] = child is not None
        result['worker_alive'] = process is not None
        result['log_path'] = str(self.path(job_id).with_suffix('.log'))
        if result.get('state') == 'complete':
            from .verify import evidence_current
            result['verification_current'] = evidence_current(result.get('result', {}))
            if not result['verification_current']:
                result.update(state='changed', error='runtime files changed since verification; run runtime verify')
        return result

    def start(self, request, *, resume=False):
        plan = make_plan(request)
        self.directory.mkdir(parents=True, exist_ok=True)
        job_id = hashlib.sha256(plan['paths']['root'].encode()).hexdigest()[:24]
        path = self.path(job_id)
        with FileLock(str(path) + '.lock', timeout=5):
            existing = read_record(path)
            if existing and owned_process(existing.get('pid'), existing.get('created')):
                if existing.get('plan', {}).get('request') != plan['request']:
                    raise ValueError('another runtime operation is using this installation')
                return self.status(job_id)
            if existing and existing.get('plan', {}).get('request') == plan['request'] and plan['mode'] == 'install':
                observed = self.status(job_id)
                if observed['state'] == 'complete': return observed
            if existing and owned_process(existing.get('child_pid'), existing.get('child_created')):
                raise ValueError('previous setup child is still alive; run runtime cancel before resuming')
            if resume and not existing: raise ValueError('nothing to resume')
            ticket = uuid.uuid4().hex
            value = {'id': job_id, 'ticket': ticket, 'state': 'starting', 'plan': plan,
                     'pid': None, 'created': None, 'heartbeat_at': time.time(), 'step': 'preflight',
                     'completed': 0, 'total': 4 if plan['mode'] == 'install' else 1,
                     'cancel_requested': False, 'error': None, 'started_at': time.time()}
            atomic_json(path, value)
            with path.with_suffix('.log').open('ab') as log:
                try:
                    child = subprocess.Popen([sys.executable, '-m', 'vrc_ardy_agent.runner.runtime_setup.worker', str(path), ticket],
                        cwd=Path(__file__).parents[3], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=os.name != 'nt',
                        creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if os.name == 'nt' else 0)
                    process = psutil.Process(child.pid)
                    value.update(pid=process.pid, created=process.create_time())
                    atomic_json(path, value)
                except Exception as exc:
                    value.update(state='failed', error=str(exc)); atomic_json(path, value)
                    raise
        atomic_json(self.directory / 'latest.json', {'id': job_id})
        return self.status(job_id)

    def resume(self, job_id=None):
        state = self.status(job_id)
        if not state.get('plan'): raise ValueError('nothing to resume')
        return self.start(state['plan']['request'], resume=True)

    def cancel(self, job_id=None):
        state = self.status(job_id)
        if state['state'] == 'orphaned':
            path = self.path(state['id'])
            with FileLock(str(path) + '.lock', timeout=5):
                current = read_record(path)
                child = owned_process(current.get('child_pid'), current.get('child_created'))
                if child:
                    children = child.children(recursive=True) + [child]
                    for process in reversed(children):
                        try: process.terminate()
                        except psutil.NoSuchProcess: pass
                    _, alive = psutil.wait_procs(children, timeout=3)
                    for process in alive:
                        try: process.kill()
                        except psutil.NoSuchProcess: pass
                    psutil.wait_procs(alive, timeout=3)
                current.update(state='canceled', child_pid=None, child_created=None)
                atomic_json(path, current)
            return self.status(state['id'])
        if state['state'] not in ACTIVE: return state
        path = self.path(state['id'])
        with FileLock(str(path) + '.lock', timeout=5):
            value = read_record(path)
            if owned_process(value.get('pid'), value.get('created')):
                value.update(cancel_requested=True, state='canceling'); atomic_json(path, value)
        return self.status(state['id'])

    def logs(self, job_id=None):
        state = self.status(job_id)
        if not state.get('id'): return {'output': ''}
        path = self.path(state['id']).with_suffix('.log')
        if not path.exists(): return {'output': ''}
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 16_384))
            return {'output': stream.read().decode('utf-8', errors='replace')}
