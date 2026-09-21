"""Identity-only attachment for the explicit pose calibration mode.

The supervisor still resolves and monitors the selected VRChat process. Audio
and image capture are deliberately absent: neither is needed to adjust a pose.
"""
import time
from .windows_sensor_session import VrchatSensorSnapshot


class CalibrationTargetSession:
    def __init__(self, sensor_id):
        self.sensor_id = sensor_id
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def snapshot(self):
        return VrchatSensorSnapshot(
            running=self._running, sensor_id=self.sensor_id,
            components={'mode': 'calibration_only'},
            heartbeat_monotonic=time.monotonic(), last_errors=(),
        )
