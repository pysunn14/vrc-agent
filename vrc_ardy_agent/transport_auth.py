"""Shared peer credentials; secrets never belong in profiles or process arguments."""
from __future__ import annotations

import argparse
import getpass
import hmac
from http import HTTPStatus
import logging
import os
from pathlib import Path
import re
import secrets
import stat
import sys
import warnings
import urllib.request

from .providers.credentials import system_store

SERVICE = 'vrc-agent-transport'
ACCOUNT = 'paired-hosts'


class TransportAuthError(ValueError):
    """A fixed, safe-to-display credential configuration error."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    # Never forward peer credentials to a redirect destination, even on the same host.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def private_file(path: Path) -> str:
    try:
        with path.open('r', encoding='ascii') as stream:
            info = os.fstat(stream.fileno())
            if info.st_size not in (64, 65):
                raise TransportAuthError('authentication key file must contain one generated key')
            if not stat.S_ISREG(info.st_mode):
                raise TransportAuthError('authentication key must be a regular file')
            if os.name == 'posix' and (info.st_uid != os.getuid() or info.st_mode & 0o077):
                raise TransportAuthError('authentication key file must be owned by this user with mode 0600')
            value = stream.read(66).strip()
    except (OSError, UnicodeError):
        raise TransportAuthError('authentication key file is unavailable') from None
    return value


class TransportAuth:
    def __init__(self, key: str):
        if not isinstance(key, str) or not re.fullmatch('[a-f0-9]{64}', key):
            raise TransportAuthError('transport authentication requires a generated 256-bit key')
        self._key = key

    @classmethod
    def load(cls):
        path = os.environ.get('VRC_AGENT_AUTH_FILE')
        if path is not None:
            if not path:
                raise TransportAuthError('authentication key file is not configured')
            # File override is for unattended accounts and tests. On Windows use
            # Credential Manager so Unix mode bits aren't mistaken for an ACL.
            if os.name != 'posix':
                raise TransportAuthError('use the OS credential store for transport authentication on Windows')
            return cls(private_file(Path(path).expanduser()))
        try:
            value = system_store().get_password(SERVICE, ACCOUNT)
        except Exception:
            raise TransportAuthError('transport authentication credential store is unavailable') from None
        if value is None:
            raise TransportAuthError('transport authentication is not configured; run python -m vrc_ardy_agent.transport_auth init')
        return cls(value)

    @property
    def authorization(self):
        return 'Bearer ' + self._key

    def accepts(self, headers):
        values = headers.get_all('Authorization') or []
        return len(values) == 1 and hmac.compare_digest(
            values[0].encode('ascii', errors='replace'), self.authorization.encode('ascii'))

    def websocket_request(self, connection, request):
        if not self.accepts(request.headers):
            return connection.respond(HTTPStatus.UNAUTHORIZED, 'authentication required\n')
        if request.headers.get_all('Origin'):
            return connection.respond(HTTPStatus.FORBIDDEN, 'browser origins are not supported\n')
        return None

    def redact(self, text: str) -> str:
        return text.replace(self._key, "[redacted]")

    def sign_follow(self, data: bytes) -> bytes:
        # Domain separation keeps the datagram MAC independent of HTTP credentials.
        key = hmac.digest(bytes.fromhex(self._key), b'vrc-agent/follow/v1', 'sha256')
        return hmac.digest(key, data, 'sha256')


def authenticated_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def websocket_logger():
    # websockets DEBUG logs dump request headers. This dedicated logger must not
    # propagate them (including Authorization) into application logging handlers.
    logger = logging.Logger('vrc-agent.transport', level=logging.CRITICAL + 1)
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def main(argv=None):
    parser = argparse.ArgumentParser(description='Pair hosts using the OS credential store; never prints keys.')
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init')
    init.add_argument('--export-file', type=Path, help='Create a new private transfer file on POSIX')
    imp = commands.add_parser('import')
    imp.add_argument('--file', type=Path, help='Read a transferred key instead of a hidden prompt')
    commands.add_parser('check')
    args = parser.parse_args(argv)
    try:
        if args.command == 'check':
            TransportAuth.load()
        else:
            store = system_store()
            if args.command == 'init':
                if store.get_password(SERVICE, ACCOUNT) is not None:
                    raise TransportAuthError('a transport key already exists; use import to explicitly replace it')
                key = secrets.token_hex(32)
                if args.export_file:
                    if os.name != 'posix':
                        raise TransportAuthError('private export files require POSIX; generate the pairing key on the Mac')
                    fd = os.open(args.export_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, 'w', encoding='ascii') as stream:
                        stream.write(key + '\n')
            else:
                if args.file:
                    key = private_file(args.file)
                else:
                    if not sys.stdin.isatty():
                        raise TransportAuthError('use --file or an interactive terminal to import a key')
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', getpass.GetPassWarning)
                        key = getpass.getpass('Pairing key: ').strip()
            TransportAuth(key)
            store.set_password(SERVICE, ACCOUNT, key)
            if store.get_password(SERVICE, ACCOUNT) != key:
                raise TransportAuthError('transport key was not retained by the credential store')
        print('Transport authentication configured.')
    except TransportAuthError as exc:
        parser.exit(1, str(exc) + "\n")
    except Exception:
        # Backend errors can contain credential material. Keep CLI errors fixed.
        parser.exit(1, 'Transport authentication setup failed; check the credential store and private transfer file.\n')


if __name__ == '__main__':
    main()
