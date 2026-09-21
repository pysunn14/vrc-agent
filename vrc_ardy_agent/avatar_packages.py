"""Portable avatar assets, separate from each installation's visual attestation.

Content-addressed directories keep an imported revision immutable: importing an
updated preset must never silently change the assets of a running profile.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from filelock import FileLock

from .asset_contract import fields, pose_data, read_json, local_path
from .avatar_config import validate_avatar, inspect_avatar, rig_path
from .avatar_rig_profile import AvatarRigProfile
from .runner.settings import atomic_json


def validate_package(raw):
    required = ('version', 'name', 'avatar_version', 'conditions', 'rig', 'base_pose', 'hmd_base', 'face_channels')
    fields(raw, required, required)
    if type(raw['version']) is not int or raw['version'] != 1:
        raise ValueError('unsupported avatar package version')
    for key in ('name', 'avatar_version', 'conditions'):
        if not isinstance(raw[key], str) or not raw[key].strip() or len(raw[key]) > 2000:
            raise ValueError(f'{key} must be a non-empty description of at most 2000 characters')
    AvatarRigProfile.from_mapping(raw['rig'])
    pose = pose_data(raw['base_pose'])
    if any(pose['locomotion']) or any(pose.get('face', {}).values()) or pose.get('yawn', 0):
        raise ValueError('base pose must have neutral face and movement')
    if not all(pose['tracker_activation'][side] for side in ('left', 'right')):
        raise ValueError('base pose must keep both hand trackers active')
    validate_avatar(dict(rig='rig.json', base_pose='base.json', hmd_base=raw['hmd_base'], face_channels=raw['face_channels']))
    return deepcopy(raw)


def make_package(avatar, base, *, name, avatar_version, conditions):
    observed = inspect_avatar(avatar, base)
    if not observed['ready']:
        raise ValueError('calibration is not ready: ' + observed['detail'])
    # A creator's attestation is deliberately not distributed as proof that the
    # same settings were checked in the recipient's room / tracking setup.
    return validate_package(dict(version=1, name=name, avatar_version=avatar_version, conditions=conditions,
        rig=read_json(rig_path(avatar, base)), base_pose=read_json(local_path(base, avatar['base_pose'])),
        hmd_base=avatar['hmd_base'], face_channels=avatar['face_channels']))


def digest(raw):
    return hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


class AvatarPackages:
    def __init__(self, directory, bundled=None):
        self.directory = Path(directory)
        self.bundled = Path(bundled) if bundled is not None else Path(__file__).with_name('avatar_presets')

    def _read(self, key):
        if not isinstance(key, str) or not re.fullmatch(r'[a-f0-9]{64}', key):
            raise ValueError('invalid avatar package key')
        folder = self.directory / key
        raw = validate_package(read_json(folder / 'avatar.json'))
        if digest(raw) != key:
            raise ValueError('registered avatar package content changed')
        for filename, field in (('rig.json', 'rig'), ('base.json', 'base_pose')):
            if read_json(folder / filename) != raw[field]:
                raise ValueError(f'registered avatar asset changed: {filename}')
        return raw

    @staticmethod
    def describe(raw, key, source='user'):
        return {k: raw[k] for k in ('name', 'avatar_version', 'conditions')} | {
            'key': key, 'source': source, 'face_channels': list(raw['face_channels']),
            'files_valid': True, 'visual_verification': 'required'}

    def install(self, raw):
        raw = validate_package(raw)
        key = digest(raw)
        self.directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.directory / '.registration.lock'), timeout=10):
            target = self.directory / key
            if target.exists():
                self._read(key)
            else:
                temporary = Path(tempfile.mkdtemp(prefix='.import-', dir=self.directory))
                try:
                    atomic_json(temporary / 'rig.json', raw['rig'])
                    atomic_json(temporary / 'base.json', raw['base_pose'])
                    atomic_json(temporary / 'avatar.json', raw)
                    temporary.rename(target)
                finally:
                    if temporary.exists(): shutil.rmtree(temporary)
        return self.describe(raw, key)

    def list(self):
        items, problems = {}, []
        for path in sorted(self.bundled.glob('*.avatar.json')):
            try:
                raw = validate_package(read_json(path))
                key = digest(raw)
                items[key] = self.describe(raw, key, 'bundled')
            except (OSError, ValueError, TypeError) as exc:
                problems.append({'path': str(path), 'detail': str(exc)})
        for folder in sorted(self.directory.glob('*')):
            if not folder.is_dir() or folder.name.startswith('.'): continue
            try:
                raw = self._read(folder.name)
                items.setdefault(folder.name, self.describe(raw, folder.name))
            except (OSError, ValueError, TypeError) as exc:
                problems.append({'path': str(folder), 'detail': str(exc)})
        return {'items': list(items.values()), 'problems': problems, 'directory': str(self.directory)}

    def export(self, key):
        if (self.directory / key).is_dir(): return self._read(key)
        for path in sorted(self.bundled.glob('*.avatar.json')):
            try: raw = validate_package(read_json(path))
            except (OSError, ValueError, TypeError):
                # Invalid unrelated entries are reported by list(), and must
                # not prevent selecting another valid package.
                continue
            if digest(raw) == key: return raw
        raise ValueError('unknown avatar package')

    def select(self, key):
        raw = self.export(key)
        self.install(raw)
        folder = self.directory / digest(raw)
        return dict(rig=str((folder / 'rig.json').resolve()), base_pose=str((folder / 'base.json').resolve()),
                    hmd_base=raw['hmd_base'], face_channels=raw['face_channels'])
