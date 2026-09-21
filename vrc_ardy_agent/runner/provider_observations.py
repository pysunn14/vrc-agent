"""Historical connection observations, never a declaration of current health."""
import hashlib
import json
from pathlib import Path
import time
import psutil
from uuid import uuid4
from filelock import FileLock
from .settings import atomic_json, state_directory
from ..providers.catalog import definition, validate_connection, validate_binding


# Increment when request/validation semantics change without a config change.
PROBE_CONTRACT_VERSION = 1


def fingerprint(provider, config, capability, settings, *, runtime=None):
    config=validate_connection(provider,config)
    binding=validate_binding(provider,capability,settings,config=config)
    spec=definition(provider,config)
    value=[PROBE_CONTRACT_VERSION,provider,config,capability,binding,spec['protocol'],spec['extension'],runtime]
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


class ObservationStore:
    def __init__(self,root=None):
        self.root=Path(root) if root is not None else state_directory()/'provider-tests'

    def path(self,key):
        if len(key)!=64 or any(c not in '0123456789abcdef' for c in key): raise ValueError('invalid observation key')
        return self.root/(key+'.json')

    def read(self,key):
        try: value=json.loads(self.path(key).read_text())
        except FileNotFoundError: return None
        if value.get('state')=='running':
            try: alive=psutil.Process(value['pid']).create_time()==value['created']
            except psutil.NoSuchProcess: alive=False
            if not alive: value=value|{'state':'interrupted'}
        return value

    def begin(self,key):
        path=self.path(key); path.parent.mkdir(parents=True,exist_ok=True)
        token=uuid4().hex
        with FileLock(str(path)+'.lock',timeout=5):
            owner=psutil.Process()
            atomic_json(path,{'ticket':token,'started_at':time.time(),'state':'running','pid':owner.pid,'created':owner.create_time()})
        return token

    def finish(self,key,token,result):
        path=self.path(key)
        with FileLock(str(path)+'.lock',timeout=5):
            current=self.read(key)
            if not current or current['ticket']!=token:return False
            atomic_json(path,current|{'state':'finished','finished_at':time.time(),'result':result})
        return True
