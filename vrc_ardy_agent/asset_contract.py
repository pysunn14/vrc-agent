"""Dependency-free asset checks shared by onboarding, doctor and playback."""
import json
import math
from pathlib import Path
import re


def fields(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError('missing, unknown or invalid asset fields')


def number(value, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('asset numbers must be finite')
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f'asset number must be in [{low}, {high}]')
    return float(value)


def vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f'asset vector must contain {size} numbers')
    return tuple(number(v) for v in value)


def name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', value):
        raise ValueError('asset name must use lowercase letters, digits, underscores or hyphens')
    return value


def path_value(value, *, empty=False):
    if not isinstance(value, str) or '\0' in value or (not empty and not value.strip()):
        raise ValueError('asset path must be a non-empty string')
    return value


def local_path(base, value):
    return (Path(base) / Path(path_value(value)).expanduser()).resolve()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def face_values(value):
    if not isinstance(value, dict):
        raise ValueError('face channels must be an object')
    for key, weight in value.items():
        # These are mapped avatar parameter names, not arbitrary OSC addresses.
        if not isinstance(key, str) or not key or '/' in key or '\0' in key or len(key) > 128:
            raise ValueError('invalid avatar parameter name')
        number(weight, 0, 1)
    return value


def pose_data(raw):
    required = {'kind', 'head', 'trackers', 'fps', 'scale', 'locomotion', 'tracker_activation'}
    fields(raw, required | {'face', 'yawn'}, required)
    if raw['kind'] != 'six_point_pose':
        raise ValueError('invalid pose kind')
    for key in ('fps', 'scale'):
        if number(raw[key]) <= 0:
            raise ValueError(f'{key} must be positive')
    fields(raw['head'], ('xyz_cm', 'ypr_deg'), ('xyz_cm', 'ypr_deg'))
    vector(raw['head']['xyz_cm'], 3); vector(raw['head']['ypr_deg'], 3)
    names = ('left', 'right', 'hips', 'left_foot', 'right_foot')
    fields(raw['trackers'], names, names)
    fields(raw['tracker_activation'], names, names)
    for key in names:
        tracker = raw['trackers'][key]
        fields(tracker, ('position', 'quaternion_xyzw'), ('position', 'quaternion_xyzw'))
        vector(tracker['position'], 3)
        q = vector(tracker['quaternion_xyzw'], 4)
        if abs(sum(v*v for v in q) - 1) > 2e-5:
            raise ValueError('tracker quaternion must be unit length')
        if type(raw['tracker_activation'][key]) is not bool:
            raise ValueError('tracker activation must be boolean')
    for value in vector(raw['locomotion'], 3): number(value, -1, 1)
    face_values(raw.get('face', {}))
    # The user's recorded clips predate channel maps. Decode only that concrete
    # on-disk format; newly encoded frames always use the generic face map.
    if 'yawn' in raw:
        number(raw['yawn'], 0, 1)
        if 'face' in raw: raise ValueError('pose cannot mix face formats')
    return raw
