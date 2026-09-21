"""Project-owned observation decisions over Hermes' final-JSON interface."""

from dataclasses import dataclass, asdict, replace
import hashlib
import math
import threading
import time
import uuid


class SupersededObservation(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplayFrame:
    """Fixed experimental evidence, explicitly not a newly captured live frame."""
    jpeg: bytes


@dataclass(frozen=True)
class Observation:
    observation_id: str
    query: str
    turn_id: str
    turn_version: int
    requested_monotonic: float
    received_monotonic: float
    # The provider guarantees a frame newer than this request; this is not
    # an invented camera timestamp or a cross-machine monotonic comparison.
    freshness_basis: str = "frame_after_request"
    source: str = "live"
    content_sha256: str | None = None


@dataclass(frozen=True)
class ObservationStatus:
    active_calls: int = 0
    model_calls: int = 0
    image_calls: int = 0
    observations: int = 0
    failures: int = 0
    superseded: int = 0
    limit_hits: int = 0
    last_error: str | None = None
    heartbeat_monotonic: float = 0.0


class ObservationLoop:
    def __init__(
        self,
        provider=None,
        *,
        max_observations=2,
        timeout_seconds=30.0,
        max_frame_age=15.0,
        max_active_calls=4,
        policy="selective",
    ):
        if not isinstance(max_observations, int) or max_observations < 1:
            raise ValueError("max_observations must be positive")
        for value in (timeout_seconds, max_frame_age):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("observation time limits must be positive and finite")
        if not isinstance(max_active_calls, int) or max_active_calls < 1:
            raise ValueError("max_active_calls must be positive")
        if policy not in ("always", "selective"):
            raise ValueError("observation policy must be always or selective")
        self.policy = policy
        self.provider = provider
        self.max_observations = max_observations
        self.timeout_seconds = timeout_seconds
        self.max_frame_age = max_frame_age
        self._slots = threading.BoundedSemaphore(max_active_calls)
        self._lock = threading.Lock()
        self._status = ObservationStatus()

    def snapshot(self):
        with self._lock:
            return asdict(self._status)

    def _record(self, counter=None, **updates):
        with self._lock:
            if counter:
                updates[counter] = getattr(self._status, counter) + 1
            self._status = replace(
                self._status, heartbeat_monotonic=time.monotonic(), **updates
            )

    def _call(self, operation, current, deadline, *, trace=None, kind="model",
              round_index=0, has_image=False, call_id=None):
        call_id = trace.request_call(kind, round_index, has_image=has_image,
                                     call_id=call_id) if trace else None
        def reject(reason, error):
            if trace: trace.abandoned(call_id, reason)
            raise error
        if not current():
            reject("superseded", SupersededObservation())
        if time.monotonic() >= deadline:
            reject("timeout", TimeoutError("observation turn deadline exceeded"))
        if not self._slots.acquire(blocking=False):
            self._record("limit_hits")
            reject("capacity", RuntimeError("observation worker capacity limit reached"))
        done = threading.Event()
        result = []
        self._record("active_calls")

        def work():
            try:
                if trace: trace.started(call_id)
                if kind == "model":
                    self._record("model_calls")
                    if has_image: self._record("image_calls")
                result.append((True, operation()))
            except BaseException as exc:
                result.append((False, exc))
            finally:
                if trace:
                    trace.completed(call_id, error=None if result[0][0] else result[0][1])
                with self._lock:
                    self._status = replace(self._status, active_calls=self._status.active_calls - 1)
                self._slots.release()
                done.set()

        threading.Thread(target=work, name="observation-call", daemon=True).start()
        # External calls cannot be killed: bound workers and fence their results.
        next_heartbeat = time.monotonic()
        while True:
            if time.monotonic() >= next_heartbeat:
                self._record()
                if trace: trace.event("turn.heartbeat", call_id=call_id)
                next_heartbeat = time.monotonic() + 1.0
            if not current():
                reject("superseded", SupersededObservation())
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reject("timeout", TimeoutError("observation turn deadline exceeded"))
            if done.wait(min(0.05, remaining)):
                break
        if not current():
            reject("superseded", SupersededObservation())
        if time.monotonic() >= deadline:
            reject("timeout", TimeoutError("observation turn deadline exceeded"))
        if not result[0][0]:
            raise result[0][1]
        return result[0][1]

    def run(self, *, request_factory, brain, current, event, trace=None):
        deadline = time.monotonic() + self.timeout_seconds
        observations = []
        history = []
        screenshot = None
        observation_count = 0
        def acquire(query, request, round_index):
            nonlocal screenshot, observation_count
            if observation_count >= self.max_observations:
                self._record("limit_hits")
                raise RuntimeError("observation limit reached before a final plan")
            if self.provider is None:
                raise RuntimeError("observation provider is unavailable")
            event("observation.requested", query)
            if trace: trace.stage("image_acquire")
            requested = time.monotonic()
            frame = self._call(self.provider, current, deadline, trace=trace,
                               kind="image", round_index=round_index)
            replay = isinstance(frame, ReplayFrame)
            screenshot = frame.jpeg if replay else frame
            if (not isinstance(screenshot, bytes) or not screenshot.startswith(b"\xff\xd8")
                    or not screenshot.endswith(b"\xff\xd9")):
                raise ValueError("observation provider must return JPEG bytes")
            observation = Observation(uuid.uuid4().hex, query, request.turn_id,
                request.turn_version, requested, time.monotonic(),
                "fixed_fixture" if replay else "frame_after_request",
                "replay" if replay else "live", hashlib.sha256(screenshot).hexdigest())
            observations.append(observation)
            observation_count += 1
            self._record("observations")
            event("observation.received", observation.observation_id)
            if trace: trace.event("image.acquired", round_index=round_index, **asdict(observation))
            return observation

        try:
            if self.policy == "always":
                initial = request_factory()
                acquire(initial.transcript, initial, -1)
            for round_index in range(self.max_observations + 1):
                request = replace(
                    request_factory(),
                    screenshot=screenshot,
                    round_index=round_index,
                    observations=tuple(observations),
                    conversation=tuple(history),
                    deadline_monotonic=deadline,
                )
                call_id = uuid.uuid4().hex
                request = replace(request, call_id=call_id,
                    record_model_metadata=(lambda metadata, cid=call_id: trace.metadata(cid, metadata)) if trace else None)
                if trace: trace.stage("model")
                decision = self._call(lambda: brain.propose(request), current, deadline,
                    trace=trace, kind="model", round_index=round_index,
                    has_image=screenshot is not None, call_id=call_id)
                if trace: trace.event("model.decision", call_id=call_id, round_index=round_index, decision=decision)
                if not isinstance(decision, dict):
                    raise ValueError("brain decision must be an object")
                query = None
                if decision.get("type") == "observe_scene":
                    if set(decision) != {"type", "query"}:
                        raise ValueError(
                            "observe_scene requires exactly type and query"
                        )
                    query = decision["query"]
                    if (
                        not isinstance(query, str)
                        or not query.strip()
                        or len(query) > 1000
                    ):
                        raise ValueError("observation query must be 1..1000 characters")
                    query = query.strip()
                else:
                    requires_visual = decision.get("requires_visual", False)
                    if not isinstance(requires_visual, bool):
                        raise ValueError("requires_visual must be boolean")
                    if requires_visual:
                        fresh = (
                            observations
                            and (observations[-1].source == "replay" or
                                 time.monotonic() - observations[-1].requested_monotonic <= self.max_frame_age)
                        )
                        if not fresh:
                            query = request.transcript
                            event(
                                "observation.required",
                                "visual plan lacks fresh evidence",
                            )
                    if query is None:
                        return {
                            k: v for k, v in decision.items() if k != "requires_visual"
                        }
                observation = acquire(query, request, round_index)
                history.append({"decision": decision, "observation": asdict(observation)})
            raise RuntimeError("observation decision limit reached")
        except SupersededObservation:
            self._record("superseded")
            raise
        except Exception as exc:
            self._record("failures", last_error=str(exc))
            raise
