"""Credential references shared by setup, discovery, doctor and inference.

Secrets never enter profile/draft JSON. OS storage is explicit, with no plaintext
fallback. Local Hermes reuse references its existing file, only for loopback
endpoints; an unset shell variable need not force the user to copy the key.
"""
import hashlib
import json
import os
import re
from uuid import uuid4
from pathlib import Path
import shlex
import sys
from urllib.parse import urlsplit

SERVICE = 'vrc-agent.providers'

class CredentialError(ValueError):
    def __init__(self, message, state='needs-auth'):
        self.state = state
        super().__init__(message)


def system_store():
    try:
        if sys.platform == 'darwin':
            from keyring.backends.macOS import Keyring
        elif sys.platform == 'win32':
            from keyring.backends.Windows import WinVaultKeyring as Keyring
        else:
            from keyring.backends.SecretService import Keyring
        store = Keyring()
        if store.priority <= 0: raise RuntimeError()
        return store
    except Exception:
        raise CredentialError('OS credential storage is unavailable. Install the project dependencies and unlock the system credential store.', 'credential-store-error') from None


def account_id(provider, config):
    identity = json.dumps([provider, config['base_url'].rstrip('/'), config.get('api_key_env', '')])
    return hashlib.sha256(identity.encode()).hexdigest()


def loopback(config):
    return urlsplit(config['base_url']).hostname in ('localhost', '127.0.0.1', '::1')


def prepare(provider, config, *, environment=None, hermes_home=None):
    environment = os.environ if environment is None else environment
    config = dict(config)
    # Only absence permits discovery. A supplied but rejected key must be
    # explicitly replaced, never quietly exchanged for another identity.
    if (config.get('credential_source', 'environment') == 'environment' and provider == 'hermes'
            and loopback(config) and not environment.get(config.get('api_key_env', ''), '').strip()
            and config.get('api_key_env') == 'API_SERVER_KEY'):
        home = Path(hermes_home) if hermes_home is not None else Path(environment.get('HERMES_HOME') or Path.home()/'.hermes')
        path = home / '.env'
        if path.is_file():
            config.update(credential_source='hermes', credential_ref=str(path.absolute()))
            return {'state':'ready', 'config':config, 'source':'hermes'}
    return {'state':'needs-auth', 'config':config}


def resolve_key(provider, config, *, environment=None):
    environment = os.environ if environment is None else environment
    source = config.get('credential_source', 'environment')
    if source == 'environment':
        name = config.get('api_key_env', '')
        if not name: return ''
        key = environment.get(name, '').strip()
        if not key: raise CredentialError(f'Register an API key in setup or configure environment variable {name}.')
        return key
    if source == 'keyring':
        if not re.fullmatch(account_id(provider, config) + r'_[0-9a-f]{32}', config.get('credential_ref', '')):
            raise CredentialError('The connection address or account changed. Register a key for this connection.')
        forwarded = environment.get('VRC_AGENT_CREDENTIAL_' + config['credential_ref'])
        if forwarded: return forwarded
        try: key = system_store().get_password(SERVICE, config['credential_ref'])
        except CredentialError: raise
        except Exception: raise CredentialError('Cannot read the OS credential store. Unlock it and retry.', 'credential-store-error') from None
        if not key: raise CredentialError('The saved API key is missing. Register it again in setup.')
        return key
    if source == 'hermes':
        if provider != 'hermes' or not loopback(config):
            raise CredentialError('Local Hermes credentials can only be used with a local Hermes endpoint.')
        path = Path(config.get('credential_ref', ''))
        try:
            if not path.is_absolute() or not path.is_file() or path.stat().st_size > 1024*1024: raise ValueError()
            for line in path.read_text().splitlines():
                line = line.strip().removeprefix('export ')
                name, separator, raw = line.partition('=')
                if separator and name.strip() == 'API_SERVER_KEY':
                    values = shlex.split(raw, comments=True)
                    if len(values) == 1 and values[0].strip(): return values[0]
                    break
        except (OSError, ValueError, UnicodeError): pass
        raise CredentialError('The local Hermes credential file is unavailable or does not contain a valid API_SERVER_KEY.')
    raise CredentialError('Unknown credential source.')


def save_key(provider, config, key):
    if not isinstance(key, str) or not key.strip() or len(key) > 16384 or any(c in key for c in '\r\n\x00'):
        raise CredentialError('Enter a nonempty API key without line breaks.')
    # Registration creates an immutable entry: canceling a draft must not
    # replace the key referenced by an already active profile.
    reference = account_id(provider, config) + '_' + uuid4().hex
    try:
        store = system_store()
        store.set_password(SERVICE, reference, key.strip())
        if store.get_password(SERVICE, reference) != key.strip(): raise RuntimeError()
    except CredentialError: raise
    except Exception: raise CredentialError('Could not save and verify the API key in OS credential storage.', 'credential-store-error') from None
    return {'state':'ready', 'config':config | {'credential_source':'keyring', 'credential_ref':reference}, 'source':'keyring'}


def runtime_environment(profile, environment):
    """Forward secrets in memory, never in serialized service specs or argv.

    ARDY's isolated interpreter need not install or access the core's OS-store
    dependencies. Each forwarded entry is scoped to the same endpoint/account.
    """
    result = dict(environment)
    for connection in profile.data.get('providers', {}).values():
        config = connection['config']
        if config.get('credential_source') == 'keyring':
            result['VRC_AGENT_CREDENTIAL_' + config['credential_ref']] = resolve_key(connection['provider'], config, environment=environment)
    return result
