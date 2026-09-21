"""Avatar registration API; usable before a runner profile has been configured."""
from pathlib import Path
from ..asset_contract import fields, read_json, local_path
from ..avatar_packages import AvatarPackages, make_package, validate_package, digest


def package_action(action, payload, *, base):
    store = AvatarPackages(Path(base) / 'avatars')
    if action == 'avatars.list': return store.list()
    if action == 'avatars.inspect':
        raw = validate_package(read_json(local_path(base, payload['path'])))
        return store.describe(raw, digest(raw))
    if action == 'avatars.import':
        fields(payload, ('path', 'expected_key'), ('path',))
        raw = validate_package(read_json(local_path(base, payload['path'])))
        if payload.get('expected_key') and digest(raw) != payload['expected_key']:
            raise ValueError('avatar package changed since preview; inspect it again')
        return store.install(raw)
    if action == 'avatars.select':
        fields(payload, ('key',), ('key',))
        return store.select(payload['key'])
    if action == 'avatars.register':
        fields(payload, ('avatar', 'name', 'avatar_version', 'conditions'), ('avatar', 'name', 'avatar_version', 'conditions'))
        raw = make_package(payload['avatar'], base, **{key: payload[key] for key in ('name', 'avatar_version', 'conditions')})
        return store.install(raw)
    if action == 'avatars.export':
        fields(payload, ('key', 'path'), ('key', 'path'))
        raw = store.export(payload['key'])
        path = local_path(base, payload['path'])
        # Exclusive creation avoids overwriting another package or user file.
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with path.open('x', encoding='utf-8') as stream:
            json.dump(raw, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        return {'path': str(path), 'name': raw['name']}
    raise ValueError('unknown avatar package action')
