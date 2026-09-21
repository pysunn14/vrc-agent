"""Shared headless setup tests and observations, usable before profile creation."""
from pathlib import Path
import json
import subprocess
import os

from ..providers.catalog import definition, validate_connection, validate_binding
from ..providers.probe import probe_connection
from .provider_observations import ObservationStore, fingerprint


def connection_test(action, payload, *, state_dir=None):
    provider=payload['provider']
    config=validate_connection(provider,payload['config'])
    capability=payload['capability']
    binding=validate_binding(provider,capability,payload['settings'],config=config)
    local=definition(provider,config)['protocol']=='whisper-local'
    runtime=payload.get('runtime') if local else None
    key=fingerprint(provider,config,capability,binding,runtime=runtime)
    store=ObservationStore(Path(state_dir)/'provider-tests' if state_dir else None)
    if action=='connection.observation':
        return {'fingerprint':key,'historical':True,'observation':store.read(key)}
    token=store.begin(key)
    args={'provider':provider,'config':config,'capability':capability,'settings':binding,
          **{k:payload[k] for k in ('text','audio_file') if k in payload}}
    try:
        if local:
            if not isinstance(runtime,dict) or not runtime.get('python') or not runtime.get('project_dir'):
                result={'state':'failed','reason':'runtime-unavailable','checks':{}}
            else:
                command=[runtime['python'],'-m','vrc_ardy_agent.providers.probe']
                completed=subprocess.run(command,cwd=runtime['project_dir'],input=json.dumps(args),text=True,
                                         capture_output=True,timeout=120,env=os.environ | {"VRC_AGENT_PROBE_PARENT":str(os.getpid())})
                result=json.loads(completed.stdout) if completed.returncode==0 else {'state':'failed','reason':'runtime-error','checks':{}}
        else: result=probe_connection(**args)
    except subprocess.TimeoutExpired:
        result={'state':'failed','reason':'timeout','checks':{}}
    except (OSError,ValueError):
        result={'state':'failed','reason':'runtime-error','checks':{}}
    recorded=store.finish(key,token,result)
    return result|{'fingerprint':key,'recorded':recorded}
