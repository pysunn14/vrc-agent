import json
from unittest.mock import patch
import pytest
from vrc_ardy_agent.providers.credentials import prepare, resolve_key, save_key, CredentialError
from vrc_ardy_agent.providers.catalog import validate_connection

class Store:
 def __init__(self): self.values = {}
 def get_password(self, service, account): return self.values.get((service, account))
 def set_password(self, service, account, value): self.values[service, account] = value

def test_local_hermes_reuses_a_reference_without_returning_the_key(tmp_path):
 env = tmp_path / '.env'; env.write_text('API_SERVER_KEY="test-private-value"\n')
 config = validate_connection('hermes', {})
 result = prepare('hermes', config, environment={}, hermes_home=tmp_path)
 assert result['config']['credential_source'] == 'hermes'
 assert 'test-private-value' not in json.dumps(result)
 assert resolve_key('hermes', result['config'], environment={}) == 'test-private-value'

def test_remote_hermes_never_uses_local_credentials(tmp_path):
 (tmp_path / '.env').write_text('API_SERVER_KEY=private\n')
 config = validate_connection('hermes', {'base_url': 'https://remote.example/v1'})
 assert prepare('hermes', config, environment={}, hermes_home=tmp_path)['state'] == 'needs-auth'
 with pytest.raises(CredentialError):
  resolve_key('hermes', config | {'credential_source':'hermes','credential_ref':str(tmp_path/'.env')}, environment={})

def test_stored_keys_survive_new_call_and_are_bound_to_endpoint():
 store = Store(); config = validate_connection('openai', {})
 with patch('vrc_ardy_agent.providers.credentials.system_store', return_value=store):
  saved = save_key('openai', config, 'private-key')
  assert 'private-key' not in json.dumps(saved)
  assert resolve_key('openai', saved['config'], environment={}) == 'private-key'
  with pytest.raises(CredentialError): resolve_key('openai', saved['config'] | {'base_url':'https://other.example/v1'}, environment={})

def test_store_failure_is_redacted_and_does_not_change_config():
 class Broken(Store):
  def set_password(self, *args): raise RuntimeError('private-key')
 config = validate_connection('openai', {})
 with patch('vrc_ardy_agent.providers.credentials.system_store', return_value=Broken()):
  with pytest.raises(CredentialError) as e: save_key('openai', config, 'private-key')
 assert 'private-key' not in str(e.value)
 assert config['credential_source'] == 'environment'

def test_isolated_runtime_uses_forwarded_credentials_without_store_dependency():
 from types import SimpleNamespace
 from vrc_ardy_agent.providers.credentials import runtime_environment
 store = Store()
 with patch('vrc_ardy_agent.providers.credentials.system_store', return_value=store):
  config = save_key('openai', validate_connection('openai', {}), 'runtime-key')['config']
  profile = SimpleNamespace(data={'providers':{'chat':{'provider':'openai','config':config}}})
  environment = runtime_environment(profile, {})
 with patch('vrc_ardy_agent.providers.credentials.system_store', side_effect=AssertionError('runtime must not use keyring')):
  assert resolve_key('openai', config, environment=environment) == 'runtime-key'
 assert 'runtime-key' not in json.dumps(profile.data)

def test_registering_a_new_key_does_not_replace_the_active_profile_key():
 store = Store(); config = validate_connection('openai', {})
 with patch('vrc_ardy_agent.providers.credentials.system_store', return_value=store):
  active = save_key('openai', config, 'old-key')['config']
  draft = save_key('openai', active, 'new-key')['config']
  assert active['credential_ref'] != draft['credential_ref']
  assert resolve_key('openai', active, environment={}) == 'old-key'
  assert resolve_key('openai', draft, environment={}) == 'new-key'
