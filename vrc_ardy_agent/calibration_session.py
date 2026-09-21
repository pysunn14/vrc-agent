"""Exclusive, leased calibration on the existing Windows device output path."""
from copy import deepcopy
import re
import threading
import time
import uuid
from .action_contracts import ControlResource, ResourceLease, OutputEnvelope
from .asset_contract import fields, pose_data, vector

RESOURCE=ControlResource.FULL_BODY_POSE


def neutral_pose(raw):
    pose=deepcopy(pose_data(raw))
    if any(pose['locomotion']) or any(pose.get('face',{}).values()) or pose.get('yawn',0):
        raise ValueError('calibration requires neutral movement and expression')
    if not all(pose['tracker_activation'][s] for s in ('left','right')):
        raise ValueError('calibration requires both hand trackers active')
    return pose


class CalibrationSession:
    # A human may pause at any prompt. The client renews this lease every second;
    # five seconds tolerates brief IPC jitter without leaving abandoned output active.
    TTL=5.0

    def __init__(self, gateway, tracker, *, hmd_base, clock=time.monotonic):
        self.gateway=gateway; self.tracker=tracker; self.hmd_base=list(hmd_base); self.clock=clock
        self._lock=threading.RLock(); self._record=None; self._deadline=0.; self._sequence=0
        self._stop=threading.Event(); self._thread=None
        self._closed=False

    def start_watchdog(self):
        with self._lock:
            if self._closed: raise RuntimeError('calibration service is closed')
            if self._thread: return
            self._thread=threading.Thread(target=self._watch,name='avatar-calibration-watchdog',daemon=True)
            self._thread.start()

    def start(self, payload):
        fields(payload,('request_id','pose','hmd_base','rig_hash'),('request_id','pose','hmd_base','rig_hash'))
        pose=neutral_pose(payload['pose']); vector(payload['hmd_base'],3)
        if list(payload['hmd_base'])!=self.hmd_base: raise ValueError('head origin differs from Windows bridge; use its reported hmd_base')
        if not isinstance(payload['request_id'],str) or not 1<=len(payload['request_id'])<=128: raise ValueError('invalid request_id')
        if not isinstance(payload['rig_hash'],str) or not re.fullmatch('[a-f0-9]{64}',payload['rig_hash']): raise ValueError('invalid rig_hash')
        with self._lock:
            self.poll()
            if self._closed: raise RuntimeError('calibration service is closed')
            if self._record and self._record['request_id']==payload['request_id']:
                if self._record['state']!='active': raise RuntimeError('request has ended; start a new request')
                return deepcopy(self._record)
            if self._record and self._record['state']=='active': raise RuntimeError('another calibration is active')
            status=self.tracker.snapshot()
            if not status.running or status.mode!='SAFE': raise RuntimeError('tracker must be running in SAFE mode')
            token=uuid.uuid4().hex
            self.gateway.claim_exclusive_session(token)
            self._record=dict(token=token,request_id=payload['request_id'],state='active',revision=0,
                pose=pose,hmd_base=self.hmd_base,rig_hash=payload['rig_hash'],restored=False,last_error=None)
            self._sequence=0
            self.gateway.authorize(token,ResourceLease(RESOURCE,1,token))
            try: self._apply()
            except Exception as exc:
                self._restore('failed',str(exc)); raise
            return deepcopy(self._record)

    def _active(self, token):
        self.poll()
        if not self._record or self._record['token']!=token: raise ValueError('unknown calibration token')
        if self._record['state']!='active': raise RuntimeError('calibration is no longer active')

    def _apply(self):
        r=self._record
        decision=self.gateway.apply_envelope(OutputEnvelope(r['token'],r['token'],RESOURCE,1,self._sequence,6000,r['pose']))
        if not decision.accepted: raise RuntimeError(f'calibration output rejected: {decision.reason}')
        self._sequence+=1
        r['applied_revision']=self.tracker.flush_target()
        r['heartbeat_monotonic']=self.clock(); self._deadline=self.clock()+self.TTL

    def update(self,payload):
        fields(payload,('token','revision','pose'),('token','revision','pose'))
        pose=neutral_pose(payload['pose'])
        with self._lock:
            self._active(payload['token'])
            if type(payload['revision']) is not int or payload['revision']!=self._record['revision']: raise ValueError('calibration revision changed')
            self._record['pose']=pose; self._record['revision']+=1
            try: self._apply()
            except Exception as exc:
                self._restore('failed',str(exc)); raise
            return deepcopy(self._record)

    def heartbeat(self,payload):
        with self._lock:
            self._active(payload['token'])
            try: self._apply()
            except Exception as exc:
                self._restore('failed',str(exc)); raise
            return deepcopy(self._record)

    def finish(self,payload):
        fields(payload,('token','revision','verified'),('token','revision','verified'))
        if type(payload['verified']) is not bool: raise ValueError('verified must be boolean')
        with self._lock:
            self.poll()
            if not self._record or self._record['token']!=payload['token']: raise ValueError('unknown calibration token')
            if self._record['state']!='active': return deepcopy(self._record)
            if type(payload['revision']) is not int or payload['revision']!=self._record['revision']: raise ValueError('calibration revision changed')
            self._restore('confirmed' if payload['verified'] else 'canceled')
            return deepcopy(self._record)

    def _restore(self,state,error=None):
        r=self._record
        try:
            result=self.gateway.neutralize(r['token'],RESOURCE,lease_token=1)
            if not result.accepted: raise RuntimeError(f'pose restoration rejected: {result.reason}')
            applied=self.tracker.flush_target()
            status=self.tracker.snapshot()
            if status.mode!='SAFE' or status.applied_revision!=applied: raise RuntimeError('SAFE output was not observed')
            self.gateway.release_exclusive_session(r['token'])
            r.update(state=state,restored=True,last_error=error,heartbeat_monotonic=self.clock())
        except Exception as exc:
            # Keep exclusive ownership when restoration fails. Allowing an agent
            # to connect would hide an unresolved device error.
            r.update(state='failed',restored=False,last_error=f'{error or ""}; restore: {exc}',heartbeat_monotonic=self.clock())

    def status(self,payload=None):
        with self._lock:
            self.poll()
            if payload and payload.get('token'):
                if not self._record or payload['token']!=self._record['token']: raise ValueError('unknown calibration token')
                return deepcopy(self._record)
            r={k:deepcopy(v) for k,v in (self._record or {}).items() if k not in ('token','pose','request_id')}
            return {'state':'idle',**r,'hmd_base':self.hmd_base,'available':self.gateway.snapshot().session_id is None}

    def poll(self):
        with self._lock:
            if self._record and self._record['state']=='active' and self.clock()>=self._deadline:
                self._restore('expired','client heartbeat expired')

    def _watch(self):
        while not self._stop.wait(.2): self.poll()

    def close(self):
        with self._lock:
            self._closed=True
            if self._record and self._record['state']=='active': self._restore('canceled','bridge stopping')
            self._stop.set()
        if self._thread and self._thread is not threading.current_thread(): self._thread.join(timeout=2)
