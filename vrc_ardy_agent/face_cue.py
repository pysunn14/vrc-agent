"""A finite face track shares the pose sender's frame clock and returns to zero."""

import math
import socket
import struct

from .vrchat_osc import _osc_string


class FaceCueSink:
    def __init__(self, pose_sink, weights, send_face):
        self.weights = tuple(float(x) for x in weights)
        if not self.weights or self.weights[0] != 0 or self.weights[-1] != 0:
            raise ValueError("face track must begin and end at zero")
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in self.weights):
            raise ValueError("face weights must be finite values in [0, 1]")
        self.pose_sink, self.send_face = pose_sink, send_face
        self.index, self.closed = 0, False

    def send(self, frame):
        if self.closed:
            raise RuntimeError("face cue sink is closed")
        weight = self.weights[self.index] if self.index < len(self.weights) else 0.0
        self.send_face(weight)
        self.pose_sink.send(frame)
        self.index += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.send_face(0.0)
        finally:
            self.pose_sink.close()


class FaceOscOutput:
    def __init__(self, host, port):
        self.address = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def __call__(self, value):
        packet = (
            _osc_string("/avatar/parameters/ArdyYawn")
            + _osc_string(",f")
            + struct.pack(">f", value)
        )
        self.socket.sendto(packet, self.address)

    def close(self):
        self.socket.close()
