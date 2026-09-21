"""Read optional behavior assets with identical checks in doctor and playback."""
import hashlib
from .asset_contract import local_path, number, pose_data, read_json
from .behavior_catalog import BehaviorSpec


def clip_signature(spec, base):
    digest = hashlib.sha256()
    inputs = [("frames", spec.frames), *sorted(spec.face_tracks.items())]
    for key, path in inputs:
        data = local_path(base, path).read_bytes()
        digest.update(key.encode()); digest.update(len(data).to_bytes(8, 'big')); digest.update(data)
    return digest.hexdigest()


def prepare_behaviors(idle, definitions, *, base_dir, face_channels, avatar_signature=None):
    specs, clips, unavailable = {}, {}, {}
    for key, raw in definitions.items():
        try:
            spec = BehaviorSpec.parse(key, raw)
            frames = None
            if spec.finite:
                if avatar_signature is not None and spec.avatar_signature != avatar_signature:
                    raise ValueError('clip calibration does not match this avatar configuration')
                if avatar_signature is not None and spec.asset_signature != clip_signature(spec, base_dir):
                    raise ValueError('clip files changed or lack an attestation')
                data = read_json(local_path(base_dir, spec.frames))
                if not isinstance(data, list) or not data: raise ValueError('clip must contain frames')
                frames = [pose_data(frame) for frame in data]
                for frame in frames:
                    for field in ('fps', 'scale', 'tracker_activation'):
                        if frame[field] != idle[field]: raise ValueError('clip and base pose must share calibration and activation')
                    if any(frame['locomotion']): raise ValueError('clip cannot move through the world')
                tracks = {}
                for channel, path in spec.face_tracks.items():
                    if channel not in face_channels: raise ValueError(f'unmapped face channel: {channel}')
                    weights = read_json(local_path(base_dir, path))
                    if not isinstance(weights, list) or len(weights) != len(frames):
                        raise ValueError('face track and clip lengths differ')
                    tracks[face_channels[channel]] = [number(w, 0, 1) for w in weights]
                for index, frame in enumerate(frames):
                    # Retain the concrete archived scalar format at this boundary.
                    values = dict.fromkeys(face_channels.values(), 0.) | dict(frame.get('face', {}))
                    if frame.get('yawn'): values['ArdyYawn'] = frame['yawn']
                    values.update({parameter: weights[index] for parameter, weights in tracks.items()})
                    if set(values) - set(face_channels.values()):
                        raise ValueError('clip uses an unmapped avatar parameter')
                    frame.pop('yawn', None)
                    frame['face'] = values
                if avatar_signature is not None and spec.asset_signature != clip_signature(spec, base_dir):
                    raise ValueError('clip files changed while loading')
            specs[key] = spec
            if frames is not None: clips[key] = frames
        except (OSError, ValueError, TypeError, KeyError) as exc:
            # Optional assets remain visible as unavailable. Requests for them
            # fail validation; they are never silently regenerated or substituted.
            unavailable[key] = f'{type(exc).__name__}: {exc}'
    return specs, clips, unavailable
