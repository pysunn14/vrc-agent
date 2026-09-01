from __future__ import annotations

from dataclasses import dataclass, replace
import math
import socket
import threading
import time
from typing import Any, Callable, Iterable, Protocol
from uuid import uuid4

from .follow_protocol import TargetObservation, encode_target_observation


class CaptureSource(Protocol):
    def start(self) -> None: ...

    def read(self, *, timeout_seconds: float) -> Any: ...

    def close(self) -> None: ...


class PersonTracker(Protocol):
    def track(self, frame: Any) -> list["TrackedDetection"]: ...


class ObservationSender(Protocol):
    def send(self, observation: TargetObservation) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int | None
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float

    def __post_init__(self) -> None:
        if self.track_id is not None and (
            isinstance(self.track_id, bool) or not isinstance(self.track_id, int)
        ):
            raise ValueError("track_id must be an integer or None")
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain four values")
        bbox = tuple(float(value) for value in self.bbox_xyxy)
        if not all(math.isfinite(value) for value in bbox):
            raise ValueError("bbox_xyxy must contain finite values")
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox_xyxy must be ordered")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "confidence", confidence)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x2 - x1) * (y2 - y1)


class SingleTargetSelector:
    """Lock the first large person track and avoid jumping on brief tracking loss."""

    def __init__(self, *, reacquire_after_missed_frames: int = 6) -> None:
        if isinstance(reacquire_after_missed_frames, bool) or reacquire_after_missed_frames < 1:
            raise ValueError("reacquire_after_missed_frames must be a positive integer")
        self.reacquire_after_missed_frames = int(reacquire_after_missed_frames)
        self._target_track_id: int | None = None
        self._missed_frames = 0

    @property
    def target_track_id(self) -> int | None:
        return self._target_track_id

    def reset(self) -> None:
        self._target_track_id = None
        self._missed_frames = 0

    def select(self, detections: Iterable[TrackedDetection]) -> TrackedDetection | None:
        candidates = list(detections)
        if self._target_track_id is not None:
            for detection in candidates:
                if detection.track_id == self._target_track_id:
                    self._missed_frames = 0
                    return detection
            self._missed_frames += 1
            if self._missed_frames < self.reacquire_after_missed_frames:
                return None
            self.reset()

        if not candidates:
            return None
        selected = max(candidates, key=lambda detection: detection.area)
        self._target_track_id = selected.track_id
        self._missed_frames = 0
        return selected


class UdpObservationSender:
    def __init__(self, *, host: str, port: int = 9200) -> None:
        if not host:
            raise ValueError("host must not be empty")
        if not 1 <= port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        self._destination = (host, int(port))
        self._socket: socket.socket | None = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, observation: TargetObservation) -> None:
        sock = self._socket
        if sock is None:
            raise RuntimeError("sender is closed")
        sock.sendto(encode_target_observation(observation), self._destination)

    def close(self) -> None:
        sock = self._socket
        self._socket = None
        if sock is not None:
            sock.close()


@dataclass(frozen=True)
class PerceptionStatus:
    running: bool = False
    frames_processed: int = 0
    observations_sent: int = 0
    target_visible: bool = False
    last_inference_seconds: float | None = None
    last_error: str | None = None
    heartbeat_monotonic: float = 0.0


class WindowsPerceptionRunner:
    """Convert captured frames to complete target-state UDP packets."""

    def __init__(
        self,
        *,
        capture: CaptureSource,
        tracker: PersonTracker,
        selector: SingleTargetSelector,
        sender: ObservationSender,
        session_id: str | None = None,
    ) -> None:
        self.capture = capture
        self.tracker = tracker
        self.selector = selector
        self.sender = sender
        self.session_id = session_id or str(uuid4())
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._status = PerceptionStatus()

    @property
    def status(self) -> PerceptionStatus:
        with self._lock:
            return replace(self._status)

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(
        self,
        *,
        max_frames: int | None = None,
        duration_seconds: float | None = None,
        heartbeat_interval_seconds: float = 2.0,
        heartbeat: Callable[[PerceptionStatus], None] | None = None,
    ) -> PerceptionStatus:
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive")
        if duration_seconds is not None and duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")

        self._stop_event.clear()
        started_at = time.monotonic()
        next_heartbeat = started_at + heartbeat_interval_seconds
        sequence = 0
        with self._lock:
            self._status = PerceptionStatus(running=True, heartbeat_monotonic=started_at)

        try:
            self.capture.start()
            while not self._stop_event.is_set():
                if max_frames is not None and sequence >= max_frames:
                    break
                now = time.monotonic()
                if duration_seconds is not None and now - started_at >= duration_seconds:
                    break

                try:
                    frame = self.capture.read(timeout_seconds=0.5)
                except TimeoutError:
                    next_heartbeat = self._emit_heartbeat_if_due(
                        now=time.monotonic(),
                        next_heartbeat=next_heartbeat,
                        interval=heartbeat_interval_seconds,
                        callback=heartbeat,
                    )
                    continue

                inference_started = time.monotonic()
                detections = self.tracker.track(frame)
                inference_seconds = time.monotonic() - inference_started
                selected = self.selector.select(detections)
                observation = self._make_observation(
                    frame=frame,
                    detection=selected,
                    sequence=sequence,
                )
                self.sender.send(observation)
                sequence += 1
                now = time.monotonic()
                with self._lock:
                    self._status = replace(
                        self._status,
                        frames_processed=sequence,
                        observations_sent=self._status.observations_sent + 1,
                        target_visible=observation.visible,
                        last_inference_seconds=inference_seconds,
                        heartbeat_monotonic=now,
                    )
                next_heartbeat = self._emit_heartbeat_if_due(
                    now=now,
                    next_heartbeat=next_heartbeat,
                    interval=heartbeat_interval_seconds,
                    callback=heartbeat,
                )
        except BaseException as exc:
            with self._lock:
                self._status = replace(self._status, last_error=str(exc))
            raise
        finally:
            try:
                self.capture.close()
            finally:
                self.sender.close()
            with self._lock:
                self._status = replace(
                    self._status,
                    running=False,
                    heartbeat_monotonic=time.monotonic(),
                )
        return self.status

    def _emit_heartbeat_if_due(
        self,
        *,
        now: float,
        next_heartbeat: float,
        interval: float,
        callback: Callable[[PerceptionStatus], None] | None,
    ) -> float:
        if now < next_heartbeat:
            return next_heartbeat
        with self._lock:
            self._status = replace(self._status, heartbeat_monotonic=now)
        if callback is not None:
            callback(self.status)
        return now + interval

    def _make_observation(
        self,
        *,
        frame: Any,
        detection: TrackedDetection | None,
        sequence: int,
    ) -> TargetObservation:
        try:
            height, width = int(frame.shape[0]), int(frame.shape[1])
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("captured frame must expose a valid image shape") from exc
        if width <= 0 or height <= 0:
            raise ValueError("captured frame dimensions must be positive")
        if detection is None:
            return TargetObservation(
                session_id=self.session_id,
                sequence=sequence,
                captured_at_ns=time.perf_counter_ns(),
                visible=False,
                bbox=None,
                confidence=0.0,
            )

        x1, y1, x2, y2 = detection.bbox_xyxy
        normalized = (
            max(0.0, min(1.0, x1 / width)),
            max(0.0, min(1.0, y1 / height)),
            max(0.0, min(1.0, x2 / width)),
            max(0.0, min(1.0, y2 / height)),
        )
        if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
            raise ValueError("detected bbox is outside the captured frame")
        return TargetObservation(
            session_id=self.session_id,
            sequence=sequence,
            captured_at_ns=time.perf_counter_ns(),
            visible=True,
            bbox=normalized,
            confidence=detection.confidence,
        )


class UltralyticsPersonTracker:
    """Thin BoT-SORT adapter. Heavy Windows dependencies are imported lazily."""

    def __init__(
        self,
        *,
        model_path: str = "yolov8n.pt",
        tracker_config: str = "botsort.yaml",
        confidence: float = 0.25,
        image_size: int = 640,
        device: str | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is missing; install requirements-windows.txt on Windows"
            ) from exc
        self._model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = device

    def track(self, frame: Any) -> list[TrackedDetection]:
        options: dict[str, Any] = {
            "persist": True,
            "tracker": self.tracker_config,
            "classes": [0],
            "conf": self.confidence,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device is not None:
            options["device"] = self.device
        results = self._model.track(frame, **options)
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
        return [
            TrackedDetection(
                track_id=int(track_id) if track_id is not None else None,
                bbox_xyxy=tuple(float(value) for value in bbox),
                confidence=float(confidence),
            )
            for bbox, confidence, track_id in zip(xyxy, confidences, track_ids)
        ]
