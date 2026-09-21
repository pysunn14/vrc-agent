"""Model-free calibration setup, immutable checkpoints and Windows control."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

from ..asset_contract import local_path, read_json, pose_data
from ..avatar_config import rig_path, validate_avatar, attest_calibration
from ..avatar_rig_profile import AvatarRigProfile
from ..calibration_pose import candidate, adjust
from .settings import atomic_json


def rig_hash(avatar,base):
    return hashlib.sha256(rig_path(avatar,base).read_bytes()).hexdigest()


def remote(payload,action,body):
    host=payload['host']; port=payload['port']
    if not isinstance(host,str) or not re.fullmatch(r'[A-Za-z0-9_.:-]+',host): raise ValueError('invalid Windows address')
    if type(port) is not int or not 1<=port<=65535: raise ValueError('invalid Windows control port')
    host=f'[{host}]' if ':' in host else host
    request=urllib.request.Request(f'http://{host}:{port}/calibration',
        data=json.dumps({'action':action,**body},allow_nan=False).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=3) as response: return json.load(response)
    except urllib.error.HTTPError as exc:
        detail=exc.read(8192).decode(errors='replace')
        raise RuntimeError(f'Windows calibration HTTP {exc.code}: {detail}') from exc
    except (OSError,ValueError) as exc:
        raise RuntimeError(f'Windows calibration connection failed: {exc}') from exc


def checkpoint(avatar,pose,base):
    """Every revision gets its own file, including unfinished candidates."""
    pose_data(pose)
    result=deepcopy(avatar); result.pop('calibration',None)
    data=json.dumps(pose,sort_keys=True,allow_nan=False).encode()
    path=Path(base)/'calibration'/'poses'/(hashlib.sha256(data).hexdigest()+'.json')
    atomic_json(path,pose)
    result['base_pose']=str(path.resolve())
    return result


def calibration_action(action,payload,*,base):
    operation=action.removeprefix('calibration.')
    if operation=='prepare':
        avatar=deepcopy(payload['avatar']); validate_avatar(avatar)
        path=rig_path(avatar,base)
        rig=AvatarRigProfile.load(path)
        # Copy the export so editing a Unity-generated file cannot change an
        # in-progress or verified calibration behind the user's back.
        raw=path.read_bytes(); folder=Path(base)/'calibration'/'rigs'/hashlib.sha256(raw).hexdigest()
        atomic_json(folder/'rig.json',json.loads(raw))
        avatar['rig']=str((folder/'rig.json').resolve())
        pose=read_json(local_path(base,avatar['base_pose'])) if avatar['base_pose'] else candidate(rig,avatar['hmd_base'])
        return checkpoint(avatar,pose,base)
    if operation=='adjust':
        avatar=payload['avatar']; validate_avatar(avatar)
        pose=adjust(read_json(local_path(base,avatar['base_pose'])),payload['hand'],payload['axis'],payload['delta'])
        return checkpoint(avatar,pose,base)
    if operation=='status': return remote(payload,'status',{k:payload[k] for k in ('token',) if k in payload})
    if operation=='start':
        avatar=payload['avatar']; validate_avatar(avatar)
        return remote(payload,'start',dict(request_id=payload['request_id'],
            pose=read_json(local_path(base,avatar['base_pose'])),hmd_base=avatar['hmd_base'],rig_hash=rig_hash(avatar,base)))
    if operation=='update':
        return remote(payload,'update',dict(token=payload['token'],revision=payload['revision'],
            pose=read_json(local_path(base,payload['avatar']['base_pose']))))
    if operation=='heartbeat': return remote(payload,'heartbeat',{'token':payload['token']})
    if operation in ('cancel','complete'):
        if operation=='complete':
            avatar=payload['avatar']; validate_avatar(avatar)
            current=remote(payload,'status',{'token':payload['token']})
            if current['rig_hash']!=rig_hash(avatar,base) or current['hmd_base']!=avatar['hmd_base']:
                raise ValueError('avatar geometry changed during calibration')
            if current['pose']!=read_json(local_path(base,avatar['base_pose'])):
                raise ValueError('local pose differs from the pose displayed by Windows')
            if payload.get('verified') is not True: raise ValueError('confirm visual verification first')
        receipt=remote(payload,'finish',dict(token=payload['token'],revision=payload['revision'],verified=operation=='complete'))
        if operation=='cancel': return receipt
        if receipt['state']!='confirmed' or not receipt['restored']:
            raise RuntimeError(f'calibration was not confirmed and restored: {receipt["state"]}: {receipt.get("last_error")}')
        avatar=checkpoint(avatar,receipt['pose'],base)
        avatar['calibration']=attest_calibration(avatar,base,tracking_active=True)
        return {'avatar':avatar,'receipt':{k:v for k,v in receipt.items() if k not in ('token','pose','request_id')}}
    raise ValueError('unknown calibration operation')
