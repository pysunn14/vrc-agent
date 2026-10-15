"""Defensive contract checks with fake services and parser boundary injection."""
import io
import json
import os
from email.message import Message
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vrc_ardy_agent.companion_api import CompanionRequestHandler, CompanionHttpServer
from vrc_ardy_agent.follow_protocol import decode_target_observation, FollowProtocolError, _require_number


def test_json_recursion_is_a_protocol_error():
    with patch('vrc_ardy_agent.follow_protocol.json.loads', side_effect=RecursionError):
        with pytest.raises(FollowProtocolError):
            decode_target_observation(b'{}')


def test_float_overflow_is_a_protocol_error():
    with pytest.raises(FollowProtocolError):
        _require_number(10 ** 400, 'confidence')


@pytest.mark.parametrize('method,path', [('GET','/status'),('GET','/health'),('GET','/actions'),('GET','/test/screenshot'),('POST','/stop'),('POST','/calibration')])
def test_http_requires_authentication_before_reading_or_dispatch(method, path):
    handler = object.__new__(CompanionRequestHandler)
    handler.path = path
    handler.headers = Message()
    service = Mock()
    from vrc_ardy_agent.transport_auth import TransportAuth
    handler.server = SimpleNamespace(control_service=service, auth=TransportAuth('a'*64))
    handler._send_json = Mock()
    handler._read_json = Mock(side_effect=AssertionError('must authenticate first'))
    getattr(handler, 'do_'+method)()
    assert handler._send_json.call_args.args[0] == 401
    assert service.mock_calls == []


def test_missing_key_fails_before_binding(monkeypatch, tmp_path):
    monkeypatch.setenv('VRC_AGENT_AUTH_FILE', str(tmp_path/'missing'))
    with pytest.raises(ValueError, match='authentication'):
        CompanionHttpServer(('127.0.0.1', 0), Mock())


def test_websocket_auth_and_origin_gate():
    from vrc_ardy_agent.transport_auth import TransportAuth
    from websockets.datastructures import Headers
    auth = TransportAuth('a'*64)
    connection = SimpleNamespace(respond=lambda code, body: (code, body))
    def check(pairs):
        return auth.websocket_request(connection, SimpleNamespace(headers=Headers(pairs)))
    assert check([])[0] == 401
    assert not auth.accepts(SimpleNamespace(get_all=lambda _: ['Bearer \udcff']))
    assert check([('Authorization','Bearer '+ 'b'*64)])[0] == 401
    assert check([('Authorization',auth.authorization)]) is None
    assert check([('Authorization',auth.authorization),('Authorization',auth.authorization)])[0] == 401
    assert check([('Authorization',auth.authorization),('Origin','https://example.invalid')])[0] == 403


def test_udp_authentication_freshness_and_replay():
    from vrc_ardy_agent.transport_auth import TransportAuth
    from vrc_ardy_agent.follow_auth import FollowPacketAuth
    sender = FollowPacketAuth(TransportAuth('a'*64), clock=lambda: 100.)
    receiver = FollowPacketAuth(TransportAuth('a'*64), clock=lambda: 100.)
    packet = sender.seal(b'{}')
    assert receiver.open(packet) == b'{}'
    for rejected in (packet, b'{}', packet[:-1]+b'x'):
        with pytest.raises(FollowProtocolError): receiver.open(rejected)
    with pytest.raises(FollowProtocolError):
        FollowPacketAuth(TransportAuth('a'*64), clock=lambda: 120.).open(packet)
    with pytest.raises(FollowProtocolError):
        FollowPacketAuth(TransportAuth('b'*64), clock=lambda: 100.).open(packet)


@pytest.mark.parametrize('method,path', [('GET','/status'),('POST','/calibration')])
def test_authenticated_http_dispatches(method, path):
    from vrc_ardy_agent.transport_auth import TransportAuth
    auth = TransportAuth('a'*64)
    handler = object.__new__(CompanionRequestHandler)
    handler.path = path
    handler.headers = Message()
    handler.headers['Authorization'] = auth.authorization
    service = Mock()
    handler.server = SimpleNamespace(control_service=service, auth=auth)
    handler._send_json = Mock()
    handler._read_json = Mock(return_value={})
    getattr(handler, 'do_'+method)()
    assert handler._send_json.call_args.args[0] == 200
    (service.status if method == 'GET' else service.calibrate).assert_called_once()


@pytest.mark.parametrize('code', [301,302,303,307,308])
def test_http_opener_rejects_redirect_without_sending_to_destination(code):
    import urllib.request
    import urllib.error
    import urllib.response
    from vrc_ardy_agent.transport_auth import authenticated_opener, TransportAuth
    seen = []
    class RedirectSource(urllib.request.HTTPHandler):
        handler_order = 100
        def http_open(self, request):
            seen.append(request)
            headers = Message()
            headers['Location'] = 'http://other.invalid/status'
            response = urllib.response.addinfourl(io.BytesIO(b''), headers, request.full_url, code)
            response.msg = 'redirect'
            return response
    opener = authenticated_opener()
    opener.add_handler(RedirectSource())
    request = urllib.request.Request('http://peer.invalid/status', headers={
        'Authorization':TransportAuth('a'*64).authorization})
    with pytest.raises(urllib.error.HTTPError) as exc:
        opener.open(request)
    assert exc.value.code == code
    assert len(seen) == 1
    assert seen[0].full_url == 'http://peer.invalid/status'


def test_health_probe_auth_is_scoped_to_companion_services(monkeypatch):
    from vrc_ardy_agent.runner.core import probe
    from vrc_ardy_agent.runner.config import Service
    from vrc_ardy_agent.transport_auth import TransportAuth
    response = Mock(status=200)
    response.read.return_value = b'{"status":"ok"}'
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    opener = Mock()
    opener.open.return_value = response
    with patch('vrc_ardy_agent.runner.core.authenticated_opener', return_value=opener), patch('vrc_ardy_agent.runner.core.urlopen', return_value=response) as plain:
        assert probe(Service('agent',health_url='http://peer.invalid/health',transport_auth=True))['ok']
        request = opener.open.call_args.args[0]
        assert request.get_header('Authorization') == TransportAuth.load().authorization
        assert probe(Service('provider',health_url='http://provider.invalid/health'))['ok']
        plain.assert_called_once_with('http://provider.invalid/health', timeout=1.0)
        assert opener.open.call_count == 1


@pytest.mark.skipif(os.name != "posix", reason="private transfer export requires POSIX")
def test_credential_setup_and_import_never_print_key(monkeypatch, tmp_path, capsys):
    from vrc_ardy_agent.transport_auth import main, TransportAuth
    saved = {}
    backend = SimpleNamespace(get_password=lambda service,account:saved.get((service,account)),
        set_password=lambda service,account,value:saved.__setitem__((service,account),value))
    monkeypatch.setattr('vrc_ardy_agent.transport_auth.system_store', lambda: backend)
    monkeypatch.delenv('VRC_AGENT_AUTH_FILE')
    transfer = tmp_path/'pair.key'
    main(['init','--export-file',str(transfer)])
    key = transfer.read_text().strip()
    assert len(key) == 64 and transfer.stat().st_mode & 0o077 == 0
    assert TransportAuth.load().authorization.endswith(key)
    saved.clear()
    main(['import','--file',str(transfer)])
    main(['check'])
    assert TransportAuth.load().authorization.endswith(key)
    assert key not in capsys.readouterr().out


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permission contract")
def test_insecure_key_file_is_rejected(monkeypatch, tmp_path):
    from vrc_ardy_agent.transport_auth import TransportAuth
    path=tmp_path/'key';path.write_text('a'*64);path.chmod(0o644)
    monkeypatch.setenv('VRC_AGENT_AUTH_FILE',str(path))
    with pytest.raises(ValueError,match='0600'):TransportAuth.load()


def test_receiver_records_protocol_error_and_continues_without_network():
    import socket
    from vrc_ardy_agent.follow_auth import FollowPacketAuth
    from vrc_ardy_agent.transport_auth import TransportAuth
    from vrc_ardy_agent.follow_receiver import UdpObservationReceiver
    from tests.test_follow_receiver import _observation
    from vrc_ardy_agent.follow_protocol import encode_target_observation
    receiver=UdpObservationReceiver()
    packets=iter([b'{}', encode_target_observation(_observation())])
    auth=FollowPacketAuth(TransportAuth.load())
    def receive(_size):
        try:return auth.seal(next(packets)), ('127.0.0.1',1)
        except StopIteration:
            receiver._stop_event.set()
            raise socket.timeout()
    receiver._socket=SimpleNamespace(recvfrom=receive)
    receiver._run()
    assert receiver.status.invalid_packets == 1
    assert receiver.status.packets_accepted == 1
    assert receiver.status.last_error
    assert receiver.store.snapshot().observation.sequence == 1


def test_websocket_rejections_are_observable_without_opening_a_session():
    from vrc_ardy_agent.mac_bridge_server import MacBridgeServer
    from websockets.datastructures import Headers
    server=MacBridgeServer(output=Mock(), input_transport=Mock())
    connection=SimpleNamespace(respond=lambda code,body:(code,body))
    assert server._authenticate_connection(connection,SimpleNamespace(headers=Headers()))[0] == 401
    status=server.snapshot()
    assert status.connections_rejected == 1
    assert status.connections_opened == 0 and not status.connected
    assert status.last_error == 'connection authentication or origin rejected'
    server.output.open_session.assert_not_called()


def test_websocket_debug_logger_does_not_propagate_secrets(caplog):
    import logging
    from vrc_ardy_agent.transport_auth import websocket_logger, TransportAuth
    caplog.set_level(logging.DEBUG)
    logger=websocket_logger()
    logger.debug('Authorization: %s', TransportAuth.load().authorization)
    assert not logger.isEnabledFor(logging.DEBUG)
    assert caplog.records == []


@pytest.mark.parametrize('constructor', ['server','client','udp-receiver','udp-sender'])
def test_transports_fail_closed_without_credentials(monkeypatch, tmp_path, constructor):
    from vrc_ardy_agent.mac_bridge_server import MacBridgeServer
    from vrc_ardy_agent.windows_bridge_client import WindowsBridgeClient
    from vrc_ardy_agent.follow_receiver import UdpObservationReceiver
    from vrc_ardy_agent.windows_perception import UdpObservationSender
    monkeypatch.setenv('VRC_AGENT_AUTH_FILE', str(tmp_path/'missing'))
    with pytest.raises(ValueError,match='authentication'):
        if constructor == 'server':MacBridgeServer(output=Mock(),input_transport=Mock())
        elif constructor == 'client':WindowsBridgeClient(url='ws://localhost',controller=Mock(),completion_target=Mock(),input_transport=Mock(),client_instance_id='test')
        elif constructor == 'udp-receiver':UdpObservationReceiver()
        else:UdpObservationSender(host='localhost')


def test_http_errors_redact_reflected_credentials():
    import urllib.error
    from vrc_ardy_agent.companion_api import CompanionApiClient, CompanionApiError
    from vrc_ardy_agent.transport_auth import TransportAuth
    auth=TransportAuth.load()
    client=CompanionApiClient('http://peer.invalid')
    body=json.dumps({'error':auth.authorization}).encode()
    client._opener=Mock()
    client._opener.open.side_effect=urllib.error.HTTPError('http://peer.invalid',403,'forbidden',Message(),io.BytesIO(body))
    with pytest.raises(CompanionApiError) as raised:client.status()
    assert auth.authorization not in str(raised.value)
    assert '[redacted]' in str(raised.value)


def test_failed_explicit_file_never_falls_back_to_store(monkeypatch, tmp_path):
    from vrc_ardy_agent.transport_auth import TransportAuth
    monkeypatch.setenv('VRC_AGENT_AUTH_FILE',str(tmp_path/'missing'))
    with patch('vrc_ardy_agent.transport_auth.system_store') as store:
        with pytest.raises(ValueError):TransportAuth.load()
        store.assert_not_called()
