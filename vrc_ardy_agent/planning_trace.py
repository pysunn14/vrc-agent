"""Per-turn planning evidence; worker completion remains attached after cancellation."""
from copy import deepcopy
import threading
import time
import uuid


class PlanningTrace:
    def __init__(self, turn_id, turn_version, *, sink=None, started_ns=None):
        self.turn_id, self.turn_version = turn_id, turn_version
        self.started_ns = time.perf_counter_ns() if started_ns is None else started_ns
        self._sink = sink
        self._lock = threading.RLock()
        self._calls = {}
        self._result = dict(status='running', plan_ms=None, elapsed_ms=None,
                            validation_ms=None, failure_stage=None, plan=None)
        self.event('turn.input')

    def event(self, event, **fields):
        with self._lock:
            row = dict(fields)
            row.update(event=event, turn_id=self.turn_id, turn_version=self.turn_version,
                       perf_counter_ns=time.perf_counter_ns())
            if self._sink is not None:
                self._sink(row)
            return row

    def request_call(self, kind, round_index, *, has_image=False, call_id=None):
        call_id = call_id or uuid.uuid4().hex
        with self._lock:
            self._calls[call_id] = dict(call_id=call_id, kind=kind, round_index=round_index,
                has_image=has_image, status='requested', wait_status=None, late=False,
                started_ns=None, duration_ms=None, usage=None, metadata=None)
            self.event('call.requested', **self._calls[call_id])
        return call_id

    def started(self, call_id):
        with self._lock:
            call = self._calls[call_id]
            call.update(status='running', started_ns=time.perf_counter_ns())
            self.event('call.started', **call)

    def metadata(self, call_id, metadata):
        with self._lock:
            call = self._calls[call_id]
            call.update(metadata=deepcopy(metadata), usage=deepcopy(metadata.get('usage')))
            self.event('call.metadata', call_id=call_id, metadata=call['metadata'], usage=call['usage'])

    def completed(self, call_id, *, error=None):
        with self._lock:
            call = self._calls[call_id]
            call.update(status='failed' if error else 'completed',
                        duration_ms=(time.perf_counter_ns()-call['started_ns'])/1e6,
                        error_type=None if error is None else type(error).__name__,
                        late=call['wait_status'] in ('timeout', 'superseded'))
            self.event('call.finished', **call)

    def abandoned(self, call_id, reason):
        with self._lock:
            call = self._calls[call_id]
            call['wait_status'] = reason
            if call['started_ns'] is None:
                call['status'] = 'not_started'
            elif call['status'] in ('completed', 'failed'):
                call['late'] = True
            self.event('call.wait_ended', **call)

    def validated(self, plan, validation_started_ns):
        now = time.perf_counter_ns()
        with self._lock:
            self._result.update(plan_ms=(now-self.started_ns)/1e6,
                validation_ms=(now-validation_started_ns)/1e6, plan=deepcopy(plan))
            self.event('plan.validated', **self._result)

    def stage(self, stage):
        with self._lock:
            self._result['failure_stage'] = stage

    def finish(self, status, error=None):
        with self._lock:
            self._result.update(status=status, elapsed_ms=(time.perf_counter_ns()-self.started_ns)/1e6,
                                error=error)
            if status in ('PLANNED', 'APPLIED'):
                self._result['failure_stage'] = None
            else:
                self._result['plan_ms'] = None
            self.event('turn.finished', **self._result)

    def snapshot(self):
        with self._lock:
            return deepcopy(dict(turn_id=self.turn_id, turn_version=self.turn_version,
                                 **self._result, calls=list(self._calls.values())))
