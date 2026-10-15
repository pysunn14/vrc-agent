"""Authenticated UDP framing with bounded freshness and replay rejection."""
import hmac
import secrets
import struct
import time
from .follow_protocol import FollowProtocolError, MAX_PACKET_BYTES
from .transport_auth import TransportAuth

OVERHEAD = 60  # magic (4), time (8), nonce (16), MAC (32)
MAX_DATAGRAM_BYTES = MAX_PACKET_BYTES + OVERHEAD


class FollowPacketAuth:
    def __init__(self, auth: TransportAuth, *, clock=time.time):
        self._auth = auth
        self._clock = clock
        self._seen = {}

    def seal(self, payload: bytes) -> bytes:
        message = b'VRF1' + struct.pack('!d', self._clock()) + secrets.token_bytes(16) + payload
        return self._auth.sign_follow(message) + message

    def open(self, packet: bytes) -> bytes:
        if not OVERHEAD < len(packet) <= MAX_DATAGRAM_BYTES:
            raise FollowProtocolError('invalid authenticated packet size')
        mac, message = packet[:32], packet[32:]
        if not hmac.compare_digest(mac, self._auth.sign_follow(message)):
            raise FollowProtocolError('packet authentication failed')
        if message[:4] != b'VRF1':
            raise FollowProtocolError('unsupported authenticated packet version')
        timestamp = struct.unpack('!d', message[4:12])[0]
        now = self._clock()
        if not abs(now - timestamp) <= 5:
            raise FollowProtocolError('packet freshness check failed')
        # Only the receiver thread mutates this cache. Retain each MAC until its
        # entire acceptance window expires, including permitted clock skew.
        self._seen = {key: expires for key, expires in self._seen.items() if expires >= now}
        if mac in self._seen:
            raise FollowProtocolError('packet replay rejected')
        if len(self._seen) >= 8192:
            raise FollowProtocolError('packet replay window is full')
        self._seen[mac] = timestamp + 5
        return message[28:]
