"""Registered actions are data; no action name controls execution semantics."""
from dataclasses import dataclass, field
from types import MappingProxyType

from .asset_contract import fields, name, number, path_value, vector


@dataclass(frozen=True)
class BehaviorSpec:
    name: str
    source: str
    description: str = ''
    avatar_signature: str | None = None
    asset_signature: str | None = None
    frames: str | None = None
    face_tracks: object = field(default_factory=lambda: MappingProxyType({}))
    prompt: str | None = None
    duration_seconds: float | None = None
    velocity: tuple[float, float, float] = (0., 0., 0.)

    @classmethod
    def parse(cls, key, raw):
        name(key)
        fields(raw, ('source', 'description', 'frames', 'face_tracks', 'avatar_signature', 'asset_signature', 'prompt', 'duration_seconds', 'velocity'), ('source',))
        source = raw['source']
        allowed = {
            'clip': ({'source', 'description', 'frames', 'face_tracks', 'avatar_signature', 'asset_signature'}, {'source', 'frames'}),
            'ardy': ({'source', 'description', 'prompt', 'duration_seconds'}, {'source', 'prompt', 'duration_seconds'}),
            'locomotion': ({'source', 'description', 'velocity', 'duration_seconds'}, {'source', 'velocity', 'duration_seconds'}),
        }
        if source not in allowed: raise ValueError('unknown behavior source')
        fields(raw, *allowed[source])
        description = raw.get('description', key)
        if not isinstance(description, str) or not description.strip() or len(description) > 500:
            raise ValueError('behavior description must contain 1..500 characters')
        if source == 'clip':
            tracks = raw.get('face_tracks', {})
            if not isinstance(tracks, dict): raise ValueError('face_tracks must be an object')
            for channel, path in tracks.items(): name(channel); path_value(path)
            for field in ('avatar_signature', 'asset_signature'):
                value = raw.get(field)
                if value is not None and (not isinstance(value, str) or len(value) != 64
                        or any(c not in '0123456789abcdef' for c in value)):
                    raise ValueError(f'invalid clip {field}')
            return cls(key, source, description, avatar_signature=raw.get('avatar_signature'),
                       asset_signature=raw.get('asset_signature'), frames=path_value(raw['frames']),
                       face_tracks=MappingProxyType(dict(tracks)))
        duration = number(raw['duration_seconds'], 1, 10)
        if source == 'ardy':
            prompt = raw['prompt']
            if (not isinstance(prompt, str) or not prompt.isascii() or not prompt.strip()
                    or len(prompt) > 500 or not any(c.isalpha() for c in prompt)
                    or prompt.strip().startswith(('preset:', 'idle:'))):
                raise ValueError('ARDY prompt must be a short English motion description')
            return cls(key, source, description, prompt=prompt.strip(), duration_seconds=duration)
        velocity = vector(raw['velocity'], 3)
        for value in velocity: number(value, -1, 1)
        if not any(velocity): raise ValueError('locomotion velocity must not be zero')
        return cls(key, source, description, duration_seconds=duration, velocity=velocity)

    @property
    def finite(self):
        return self.source == 'clip'

    def describe(self):
        return {'name': self.name, 'source': self.source, 'description': self.description,
                'duration': 'complete_clip' if self.finite else 'optional_override_1_to_10_seconds',
                **({} if self.finite else {'duration_seconds': self.duration_seconds})}
