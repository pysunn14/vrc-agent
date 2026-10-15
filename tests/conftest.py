"""Use an ephemeral credential in tests, never the developer's paired-host key."""
import os
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def transport_test_key(tmp_path, monkeypatch):
    if os.name != 'posix':
        monkeypatch.delenv('VRC_AGENT_AUTH_FILE', raising=False)
        monkeypatch.setattr('vrc_ardy_agent.transport_auth.system_store',
                            lambda: SimpleNamespace(get_password=lambda *_: 'a'*64))
        return
    key_file = tmp_path / 'transport.key'
    key_file.write_text('a' * 64 + '\n')
    key_file.chmod(0o600)
    monkeypatch.setenv('VRC_AGENT_AUTH_FILE', str(key_file))
