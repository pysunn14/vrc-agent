"""Headless avatar and behavior setup operations shared by CLI and agentctl."""
from pathlib import Path
from copy import deepcopy

from ..asset_contract import fields, local_path, read_json, pose_data
from ..avatar_config import attest_calibration, inspect_avatar, inspect_avatar_files, signature, rig_path
from ..behavior_assets import prepare_behaviors, clip_signature
from ..behavior_catalog import BehaviorSpec


def inspect_assets(profile):
    base = profile.path.parent if profile.path else Path.cwd()
    avatar = profile.data['avatar']
    observed = inspect_avatar(avatar, base)
    if not observed['ready']:
        return {'avatar': observed, 'available': {}, 'unavailable': {
            key: 'avatar calibration is not ready' for key in profile.data['behaviors']}}
    idle = pose_data(read_json(local_path(base, avatar['base_pose'])))
    specs, _, unavailable = prepare_behaviors(idle, profile.data['behaviors'], base_dir=base,
        face_channels=avatar['face_channels'], avatar_signature=signature(avatar, base))
    return {'avatar': observed, 'available': specs, 'unavailable': unavailable}


def asset_files(profile):
    base = profile.path.parent if profile.path else Path.cwd()
    avatar = profile.data['avatar']
    result = {}
    if avatar['rig']: result['avatar.rig'] = rig_path(avatar, base)
    if avatar['base_pose']: result['avatar.base_pose'] = local_path(base, avatar['base_pose'])
    for key, raw in profile.data['behaviors'].items():
        spec = BehaviorSpec.parse(key, raw)
        if spec.frames: result[f'behavior.{key}.frames'] = local_path(base, spec.frames)
        for channel, path in spec.face_tracks.items(): result[f'behavior.{key}.face.{channel}'] = local_path(base, path)
    return result


def setup_action(action, payload, *, base):
    if action == 'avatar.rig.inspect':
        from ..avatar_rig_profile import AvatarRigProfile
        rig = AvatarRigProfile.load(local_path(base, payload['path']))
        return {'name': rig.name, 'height_m': rig.reference_height_m}
    if action == 'avatar.files.inspect': return inspect_avatar_files(payload['avatar'], base)
    if action == 'avatar.attest':
        fields(payload, ('avatar', 'tracking_active'), ('avatar', 'tracking_active'))
        return attest_calibration(payload['avatar'], base, tracking_active=payload['tracking_active'])
    if action == 'avatar.inspect': return inspect_avatar(payload['avatar'], base)
    if action == 'behaviors.inspect':
        fields(payload, ('avatar', 'behaviors'), ('avatar', 'behaviors'))
        from types import SimpleNamespace
        result = inspect_assets(SimpleNamespace(data=payload, path=Path(base) / 'agent.json'))
        return {'avatar': result['avatar'], 'available': [spec.describe() for spec in result['available'].values()],
                'unavailable': result['unavailable']}
    if action == 'behaviors.attest':
        fields(payload, ('avatar', 'behaviors', 'tracking_active'), ('avatar', 'behaviors', 'tracking_active'))
        avatar = payload['avatar']
        attestation = attest_calibration(avatar, base, tracking_active=payload['tracking_active'])
        definitions = deepcopy(payload['behaviors'])
        if not isinstance(definitions, dict): raise ValueError('behaviors must be an object')
        for key, raw in definitions.items():
            spec = BehaviorSpec.parse(key, raw)
            if spec.finite:
                raw['avatar_signature'] = attestation['signature']
                raw['asset_signature'] = clip_signature(spec, base)
        _, _, unavailable = prepare_behaviors(pose_data(read_json(local_path(base, avatar['base_pose']))), definitions,
            base_dir=base, face_channels=avatar['face_channels'], avatar_signature=attestation['signature'])
        if unavailable: raise ValueError(f'cannot attest unavailable behaviors: {unavailable}')
        return definitions
    if action == 'behaviors.import':
        fields(payload, ('path',), ('path',))
        path = local_path(base, payload['path'])
        raw = read_json(path)
        fields(raw, ('version', 'behaviors'), ('version', 'behaviors'))
        if type(raw['version']) is not int or raw['version'] != 1: raise ValueError('unsupported behavior library version')
        if not isinstance(raw['behaviors'], dict) or len(raw['behaviors']) > 100:
            raise ValueError('behavior library must contain at most 100 definitions')
        for key, data in raw['behaviors'].items():
            spec = BehaviorSpec.parse(key, data)
            if spec.frames: data['frames'] = str(local_path(path.parent, spec.frames))
            if spec.face_tracks: data['face_tracks'] = {channel: str(local_path(path.parent, value)) for channel, value in spec.face_tracks.items()}
        return raw['behaviors']
    raise ValueError('unknown avatar setup action')
