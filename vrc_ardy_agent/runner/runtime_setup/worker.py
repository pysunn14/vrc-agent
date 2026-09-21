"""One finite setup job. The terminal only observes this worker."""
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

import psutil

from filelock import FileLock, Timeout

from ..records import read_record, update_record
from ..settings import atomic_json, state_directory
from ..worker import terminate_tree
from .install import claim, environment_stage, installed_environment_matches, source_stage


class Canceled(Exception): pass


class Job:
    def __init__(self, path, ticket):
        self.path, self.ticket = Path(path), ticket
        self.plan = read_record(self.path)['plan']
        self.stopped = threading.Event()

    def update(self, **data):
        if 'step' in data:
            data.setdefault('detail', '')
            data.setdefault('bytes_completed', None)
            data.setdefault('bytes_total', None)
        return update_record(self.path, self.ticket, heartbeat_at=time.time(), **data)

    def check_cancel(self):
        if self.stopped.is_set() or read_record(self.path).get('cancel_requested'): raise Canceled('setup canceled; completed files are retained for resume')

    def run(self, args, *, env=None, timeout=1800):
        self.check_cancel()
        print('Running: ' + json.dumps(args), flush=True)
        environment = dict(env or os.environ)
        bin_dir = str(Path(self.plan['python']).parent)
        environment['PATH'] = bin_dir + os.pathsep + environment.get('PATH', '')
        # Set no global environment variables or shell files.
        process = subprocess.Popen(args, cwd=self.plan['project_dir'], env=environment,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=os.name != 'nt', creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        identity = psutil.Process(process.pid)
        self.update(child_pid=identity.pid, child_created=identity.create_time())
        lines = queue.Queue()
        def consume():
            try:
                for raw in iter(process.stdout.readline, b''): lines.put(raw.decode('utf-8', errors='replace'))
            finally: lines.put(None)
        reader = threading.Thread(target=consume, daemon=True); reader.start()
        start = time.monotonic(); captured = ''; ended = False
        try:
            while not ended or process.poll() is None:
                self.check_cancel()
                if time.monotonic() - start > timeout: raise RuntimeError('setup command timed out; inspect runtime logs and resume')
                try: line = lines.get(timeout=.2)
                except queue.Empty: continue
                if line is None: ended = True; continue
                if line.startswith('VRC_RUNTIME_EVENT '): self.update(**json.loads(line.removeprefix('VRC_RUNTIME_EVENT ')))
                else:
                    captured = (captured + line)[-2*1024*1024:]
                    print(line, end='', flush=True)
            code = process.wait()
            if code: raise RuntimeError(f'command exited {code}: {captured[-4000:]}')
            return captured
        finally:
            if process.poll() is None: terminate_tree(process)
            reader.join(timeout=2)
            process.stdout.close()
            self.update(child_pid=None, child_created=None)

    def task(self, action):
        plan_file = self.path.with_suffix('.plan.json')
        output_file = self.path.with_suffix(f'.{action}.json')
        output_file.unlink(missing_ok=True)
        atomic_json(plan_file, self.plan)
        self.run([self.plan['python'], '-m', 'vrc_ardy_agent.runner.runtime_setup.task', action, str(plan_file), str(output_file)], timeout=6*3600 if action == 'download' else 1800)
        return json.loads(output_file.read_text())

    def execute(self):
        for signum in (signal.SIGTERM, signal.SIGINT): signal.signal(signum, lambda *_: self.stopped.set())
        timer_stop = threading.Event()
        def heartbeat():
            while not timer_stop.wait(1): self.update()
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        # Root identity is canonical even if two callers use different symlink paths
        # or --state-dir. Locks live in app state, never inside an adopted environment.
        key = hashlib.sha256(str(Path(self.plan['paths']['root']).resolve()).encode()).hexdigest()
        lock_dir = state_directory() / 'runtime-locks'; lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(lock_dir / key), timeout=0):
                self.update(state='running'); heartbeat_thread.start()
                if self.plan['mode'] == 'install':
                    errors = [item for item in self.plan['checks'] if item['result'] == 'error']
                    if errors: raise ValueError('preflight failed: ' + json.dumps(errors))
                    self.update(step='source', completed=0)
                    claim(self.plan); source_stage(self.plan, self.run)
                    self.update(step='environment', completed=1)
                    if not installed_environment_matches(self.plan, self.run): environment_stage(self.plan, self.run)
                    self.update(step='models', completed=2)
                    self.task('download')
                    self.update(step='verification', completed=3)
                else: self.update(step='verification', completed=0)
                result = self.task('verify')
                self.check_cancel()
                result['installation_root'] = self.plan['paths']['root']
                if self.plan['recipe']: result['recipe'] = self.plan['recipe']
                self.update(state='complete', step='complete', completed=read_record(self.path)['total'],
                            result=result, finished_at=time.time())
        except Canceled as exc:
            self.update(state='canceled', error=str(exc), finished_at=time.time())
        except Timeout:
            self.update(state='failed', error='another setup job owns this installation; observe it before retrying', finished_at=time.time())
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}'
            self.update(state='needs-auth' if 'authentication required' in message else 'failed', error=message, finished_at=time.time())
            print(message, flush=True)
        finally:
            timer_stop.set()
            if heartbeat_thread.is_alive(): heartbeat_thread.join(timeout=2)


if __name__ == '__main__': Job(Path(sys.argv[1]), sys.argv[2]).execute()
