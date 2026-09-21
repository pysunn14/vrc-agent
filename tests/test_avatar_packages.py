from tests.rig_fixture import RIG_PATH
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.test_motion_replay import pose
from vrc_ardy_agent.device_payloads import encode_pose_payload
from vrc_ardy_agent.avatar_config import attest_calibration, inspect_avatar
from vrc_ardy_agent.avatar_packages import AvatarPackages, make_package


@pytest.fixture
def source(tmp_path):
    (tmp_path / 'base.json').write_text(json.dumps(encode_pose_payload(pose())))
    avatar = dict(rig=RIG_PATH, base_pose='base.json', hmd_base=[0, 1, 0], face_channels={})
    avatar['calibration'] = attest_calibration(avatar, tmp_path, tracking_active=True)
    return avatar


def package(source, tmp_path):
    return make_package(source, tmp_path, name='Example Avatar', avatar_version='1.0',
                        conditions='Default scale; standing tracking')


def test_package_embeds_assets_but_not_machine_paths_or_attestation(source, tmp_path):
    data = package(source, tmp_path)
    serialized = json.dumps(data)
    assert str(tmp_path) not in serialized
    assert 'calibration' not in data
    assert isinstance(data['rig'], dict) and isinstance(data['base_pose'], dict)
    store = AvatarPackages(tmp_path / 'registry')
    entry = store.install(data)
    avatar = store.select(entry['key'])
    assert 'calibration' not in avatar
    assert inspect_avatar(avatar, tmp_path)['ready'] is False
    avatar['calibration'] = attest_calibration(avatar, tmp_path, tracking_active=True)
    assert inspect_avatar(avatar, tmp_path)['ready'] is True


def test_unverified_configuration_cannot_be_published_as_prepared(source, tmp_path):
    del source['calibration']
    with pytest.raises(ValueError, match='calibration'):
        package(source, tmp_path)


def test_concurrent_registration_is_idempotent_and_changed_content_is_distinct(source, tmp_path):
    store = AvatarPackages(tmp_path / 'registry')
    data = package(source, tmp_path)
    with ThreadPoolExecutor() as pool:
        entries = list(pool.map(lambda _: store.install(data), range(8)))
    assert len({entry['key'] for entry in entries}) == 1
    assert len(store.list()['items']) == 1
    changed = dict(data, conditions='Different tracking conditions')
    assert store.install(changed)['key'] != entries[0]['key']
    assert len(store.list()['items']) == 2


def test_bad_or_incomplete_package_is_rejected_without_registration(source, tmp_path):
    store = AvatarPackages(tmp_path / 'registry')
    data = package(source, tmp_path)
    del data['base_pose']
    with pytest.raises(ValueError): store.install(data)
    assert not store.list()['items']


def test_corrupted_registered_asset_is_observable_and_not_selectable(source, tmp_path):
    store = AvatarPackages(tmp_path / 'registry')
    entry = store.install(package(source, tmp_path))
    avatar = store.select(entry['key'])
    from pathlib import Path
    Path(avatar['base_pose']).write_text('{}')
    result = store.list()
    assert not result['items'] and len(result['problems']) == 1
    with pytest.raises(ValueError): store.select(entry['key'])


def test_exported_package_can_move_to_another_machine(source, tmp_path):
    first = AvatarPackages(tmp_path / 'first')
    entry = first.install(package(source, tmp_path))
    second = AvatarPackages(tmp_path / 'second')
    imported = second.install(first.export(entry['key']))
    avatar = second.select(imported['key'])
    assert str(tmp_path / 'second') in avatar['rig']
    assert first.export(entry['key']) == second.export(imported['key'])


def test_headless_registration_import_export_and_stale_preview(source, tmp_path):
    from vrc_ardy_agent.runner.control import dispatch
    cfg = tmp_path / 'agent.json'
    entry = dispatch('avatars.register', {'avatar': source, 'name': 'Example',
        'avatar_version': '1', 'conditions': 'Standing, default scale'}, path=cfg)
    export = tmp_path / 'shared.avatar.json'
    dispatch('avatars.export', {'key': entry['key'], 'path': str(export)}, path=cfg)
    with pytest.raises(FileExistsError):
        dispatch('avatars.export', {'key': entry['key'], 'path': str(export)}, path=cfg)
    second = tmp_path / 'other' / 'agent.json'
    preview = dispatch('avatars.inspect', {'path': str(export)}, path=second)
    imported = dispatch('avatars.import', {'path': str(export), 'expected_key': preview['key']}, path=second)
    assert imported['key'] == entry['key']
    selected = dispatch('avatars.select', {'key': imported['key']}, path=second)
    assert 'calibration' not in selected
    assert dispatch('avatar.files.inspect', {'avatar': selected}, path=second)['ready']
    assert not dispatch('avatar.inspect', {'avatar': selected}, path=second)['ready']
    raw = json.loads(export.read_text())
    raw['conditions'] = 'Changed since preview'
    export.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='changed since preview'):
        dispatch('avatars.import', {'path': str(export), 'expected_key': preview['key']}, path=second)


def test_bundled_package_is_materialized_only_when_selected(source, tmp_path):
    bundled = tmp_path / 'bundled'
    bundled.mkdir()
    (bundled / 'example.avatar.json').write_text(json.dumps(package(source, tmp_path)))
    store = AvatarPackages(tmp_path / 'registry', bundled)
    entry = store.list()['items'][0]
    assert entry['source'] == 'bundled'
    assert not store.directory.exists()
    assert store.select(entry['key'])['base_pose']


def test_invalid_bundled_entry_does_not_block_another_valid_package(source, tmp_path):
    bundled = tmp_path / 'bundled'
    bundled.mkdir()
    (bundled / 'a-invalid.avatar.json').write_text('{}')
    (bundled / 'b-valid.avatar.json').write_text(json.dumps(package(source, tmp_path)))
    store = AvatarPackages(tmp_path / 'registry', bundled)
    observed = store.list()
    assert len(observed['problems']) == 1
    assert len(observed['items']) == 1
    assert store.select(observed['items'][0]['key'])['rig']
