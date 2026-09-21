"""Interpolate already mapped room-space targets, never recalibrate a clip."""
from dataclasses import replace
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .device_payloads import encode_pose_payload
from .opentrack_bridge import OpenTrackFrame, rotation_matrix_to_opentrack_ypr
from .six_point_bridge import SixPointFrame
from .vmt_bridge import VmtFrame

TRACKERS = ('left', 'right', 'hips', 'left_foot', 'right_foot')


def validate_pose(frame: SixPointFrame) -> None:
    encode_pose_payload(frame)
    for name in TRACKERS:
        tracker = getattr(frame, name)
        if not np.isclose(np.linalg.norm(tracker.quaternion_xyzw), 1., atol=1e-5):
            raise ValueError(f'{name} quaternion must be unit length')
        if tracker.fps != frame.fps:
            raise ValueError('tracker fps must match frame fps')
    if frame.head.fps != frame.fps:
        raise ValueError('head fps must match frame fps')


def _head_rotation(frame: SixPointFrame) -> Rotation:
    yaw, pitch, roll = frame.head.ypr_deg
    return Rotation.from_euler('zxy', [roll, pitch, -yaw], degrees=True)


class PoseTransition:
    def __init__(self, source: SixPointFrame, target: SixPointFrame):
        validate_pose(source)
        validate_pose(target)
        if source.fps != target.fps or source.tracker_activation != target.tracker_activation:
            raise ValueError('transition requires matching fps and tracker activation')
        self.source, self.target = source, target
        self._rotations = {
            name: Slerp([0., 1.], Rotation.from_quat([
                getattr(source, name).quaternion_xyzw,
                getattr(target, name).quaternion_xyzw,
            ])) for name in TRACKERS
        }
        self._head = Slerp([0., 1.], Rotation.concatenate([
            _head_rotation(source), _head_rotation(target),
        ]))

    def at(self, progress: float) -> SixPointFrame:
        if not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError('progress must be finite and in [0, 1]')
        if progress == 0:
            return self.source
        if progress == 1:
            return self.target
        # Smoothstep gives zero endpoint speed for this return segment. It does
        # not preserve the incoming clip velocity or guarantee collision avoidance.
        weight = progress * progress * (3 - 2 * progress)
        def lerp(a, b):
            return tuple((1 - weight) * np.asarray(a) + weight * np.asarray(b))
        trackers = {
            name: VmtFrame(
                position=lerp(getattr(self.source, name).position, getattr(self.target, name).position),
                quaternion_xyzw=tuple(self._rotations[name](weight).as_quat()),
                fps=self.target.fps,
            ) for name in TRACKERS
        }
        axes = ('locomotion_x', 'locomotion_y', 'locomotion_turn')
        return replace(
            self.target,
            face=tuple((key, (1 - weight) * dict(self.source.face).get(key, 0.)
                        + weight * dict(self.target.face).get(key, 0.))
                       for key in sorted(dict(self.source.face) | dict(self.target.face))),
            head=OpenTrackFrame(
                lerp(self.source.head.xyz_cm, self.target.head.xyz_cm),
                rotation_matrix_to_opentrack_ypr(self._head(weight).as_matrix()),
                self.target.fps,
            ),
            **trackers,
            **{name: (1 - weight) * getattr(self.source, name) + weight * getattr(self.target, name)
               for name in axes},
        )
