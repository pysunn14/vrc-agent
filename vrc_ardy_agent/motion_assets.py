"""Calibrated base pose and available registered behaviors for one avatar."""
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

from .asset_contract import read_json
from .behavior_assets import prepare_behaviors
from .device_payloads import decode_pose_payload
from .pose_transition import validate_pose
from .six_point_bridge import SixPointFrame


@dataclass(frozen=True)
class MotionClip:
    frames: tuple[SixPointFrame, ...]

    def __post_init__(self):
        if not self.frames: raise ValueError('clip must contain frames')
        for frame in self.frames: validate_pose(frame)

    @property
    def seconds(self): return len(self.frames) / self.frames[0].fps


@dataclass(frozen=True)
class MotionAssets:
    idle: SixPointFrame
    behaviors: object = field(default_factory=dict)
    clips: object = field(default_factory=dict)
    unavailable: object = field(default_factory=dict)

    def __post_init__(self):
        validate_pose(self.idle)
        if any(v for _, v in self.idle.face) or any(getattr(self.idle, n) != 0 for n in
                                    ('locomotion_x', 'locomotion_y', 'locomotion_turn')):
            raise ValueError('base pose must have neutral face and movement')
        if set(self.clips) != {key for key, spec in self.behaviors.items() if spec.finite}:
            raise ValueError('every clip behavior must have exactly one loaded clip')
        for clip in self.clips.values():
            for frame in clip.frames:
                if (frame.fps != self.idle.fps or frame.scale != self.idle.scale
                        or frame.tracker_activation != self.idle.tracker_activation):
                    raise ValueError('motion and base pose must share calibration and activation')
                if any(getattr(frame, n) != 0 for n in ('locomotion_x', 'locomotion_y', 'locomotion_turn')):
                    raise ValueError('clip cannot move the avatar through the world')
        for key in ('behaviors', 'clips', 'unavailable'):
            object.__setattr__(self, key, MappingProxyType(dict(getattr(self, key))))

    def duration(self, name):
        spec = self.behaviors[name]
        return self.clips[name].seconds if spec.finite else spec.duration_seconds

    def snapshot(self):
        return {'base_pose': 'calibrated', 'available': [spec.describe() | {'duration_seconds': self.duration(key)}
                                                               for key, spec in self.behaviors.items()],
                'unavailable': dict(self.unavailable)}

    @classmethod
    def load(cls, idle_path: Path, definitions, *, base_dir, face_channels, avatar_signature=None):
        raw_idle = read_json(idle_path)
        idle = decode_pose_payload(raw_idle)
        cls(idle)
        # A freshly connected Windows sink has no remembered parameters. Include
        # configured neutral channels from its first frame to clear stale faces.
        idle = replace(idle, face=tuple(sorted((parameter, 0.) for parameter in face_channels.values())))
        specs, clips, unavailable = prepare_behaviors(raw_idle, definitions, base_dir=base_dir,
            face_channels=face_channels, avatar_signature=avatar_signature)
        return cls(idle, specs, {key: MotionClip(tuple(decode_pose_payload(f) for f in frames))
                               for key, frames in clips.items()}, unavailable)
