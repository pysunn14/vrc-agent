"""Avatar setup and human calibration evidence, without loading model packages."""
import hashlib
import json
from pathlib import Path

from .avatar_rig_profile import AvatarRigProfile
from .asset_contract import fields, path_value, local_path, vector, name, face_values, read_json, pose_data


def validate_avatar(avatar):
    fields(avatar, ('rig', 'base_pose', 'face_channels', 'hmd_base', 'calibration'),
           ('rig', 'base_pose', 'face_channels', 'hmd_base'))
    for key in ('rig', 'base_pose'): path_value(avatar[key], empty=True)
    vector(avatar['hmd_base'], 3)
    if avatar['hmd_base'][1] <= 0: raise ValueError('HMD height must be positive')
    channels = avatar['face_channels']
    if not isinstance(channels, dict): raise ValueError('face_channels must be an object')
    for key, parameter in channels.items(): name(key); face_values({parameter: 0})
    if len(set(channels.values())) != len(channels): raise ValueError('face channels must map to distinct parameters')
    if 'calibration' in avatar:
        fields(avatar['calibration'], ('signature', 'tracking'), ('signature', 'tracking'))
        signature = avatar['calibration']['signature']
        if not isinstance(signature, str) or len(signature) != 64 or any(c not in '0123456789abcdef' for c in signature):
            raise ValueError('invalid calibration signature')
        if avatar['calibration']['tracking'] != 'active': raise ValueError('calibration requires active tracking')
    return avatar


def rig_path(avatar, base):
    rig = path_value(avatar['rig'])
    bundled = Path(__file__).with_name('rig_profiles') / f'{rig}.avatar-rig.json'
    return bundled if '/' not in rig and '\\' not in rig and bundled.is_file() else local_path(base, rig)


def signature(avatar, base):
    digest = hashlib.sha256()
    for data in (rig_path(avatar, base).read_bytes(), local_path(base, avatar['base_pose']).read_bytes(),
                 json.dumps({'hmd_base': avatar['hmd_base'], 'face_channels': avatar['face_channels']}, sort_keys=True).encode()):
        digest.update(len(data).to_bytes(8, 'big')); digest.update(data)
    return digest.hexdigest()


def inspect_avatar_files(avatar, base):
    try:
        validate_avatar(avatar)
        AvatarRigProfile.load(rig_path(avatar, base))
        pose = pose_data(read_json(local_path(base, avatar['base_pose'])))
        if any(pose['locomotion']) or any(pose.get('face', {}).values()) or pose.get('yawn', 0):
            raise ValueError('base pose must have neutral face and movement')
        if not pose['tracker_activation']['left'] or not pose['tracker_activation']['right']:
            raise ValueError('base pose must keep both hand trackers active')
        return {'ready': True, 'detail': 'avatar files are valid; visual result is not measured'}
    except (OSError, ValueError, TypeError) as exc:
        return {'ready': False, 'detail': str(exc)}


def inspect_avatar(avatar, base):
    result = inspect_avatar_files(avatar, base)
    if not result['ready']: return result
    if avatar.get('calibration') != {'signature': signature(avatar, base), 'tracking': 'active'}:
        return {'ready': False, 'detail': 'active-tracking calibration is missing or assets changed; verify and attest again'}
    return {'ready': True, 'detail': 'files match the active-tracking attestation; visual result is not measured'}


def attest_calibration(avatar, base, *, tracking_active):
    validate_avatar(avatar)
    if tracking_active is not True: raise ValueError('confirm verification with tracking active')
    attestation = {'signature': signature(avatar, base), 'tracking': 'active'}
    result = inspect_avatar(avatar | {'calibration': attestation}, base)
    if not result['ready']: raise ValueError(result['detail'])
    return attestation
