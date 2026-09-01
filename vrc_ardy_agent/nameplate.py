from __future__ import annotations

from dataclasses import dataclass, replace
from difflib import SequenceMatcher
import math
import re
import threading
import time
from typing import Any, Iterable, Protocol


class TextReader(Protocol):
    def read(self, frame: Any) -> list["OcrTextRegion"]: ...


@dataclass(frozen=True)
class OcrTextRegion:
    text: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("OCR text must not be empty")
        bbox = _validated_bbox(self.bbox_xyxy)
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("OCR confidence must be in [0, 1]")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class NameplateMatch:
    target_name: str
    recognized_text: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    match_score: float

    def __post_init__(self) -> None:
        if not self.target_name.strip() or not self.recognized_text.strip():
            raise ValueError("nameplate names must not be empty")
        object.__setattr__(self, "bbox_xyxy", _validated_bbox(self.bbox_xyxy))
        for field in ("confidence", "match_score"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")
            object.__setattr__(self, field, value)


@dataclass(frozen=True)
class ObservedNameplate:
    match: NameplateMatch
    observed_monotonic: float


class NameplateMatcher:
    """Find one configured display name in noisy, possibly split OCR output."""

    def __init__(self, target_name: str, *, minimum_score: float = 0.72) -> None:
        normalized = _normalize_text(target_name)
        if not normalized:
            raise ValueError("target_name must contain letters or digits")
        if not 0.0 < minimum_score <= 1.0:
            raise ValueError("minimum_score must be in (0, 1]")
        self.target_name = target_name.strip()
        self._normalized_target = normalized
        self.minimum_score = float(minimum_score)

    def match(self, regions: Iterable[OcrTextRegion]) -> NameplateMatch | None:
        best: NameplateMatch | None = None
        for candidate in _candidate_regions(regions):
            normalized = _normalize_text(candidate.text)
            coverage = min(len(normalized), len(self._normalized_target)) / len(
                self._normalized_target
            )
            if coverage < 0.65:
                continue
            score = SequenceMatcher(None, normalized, self._normalized_target).ratio()
            if score < self.minimum_score:
                continue
            confidence = candidate.confidence * score
            match = NameplateMatch(
                target_name=self.target_name,
                recognized_text=candidate.text,
                bbox_xyxy=candidate.bbox_xyxy,
                confidence=confidence,
                match_score=score,
            )
            if best is None or (match.confidence, match.match_score) > (
                best.confidence,
                best.match_score,
            ):
                best = match
        return best


@dataclass(frozen=True)
class NameplateTrackerStatus:
    running: bool = False
    scanning: bool = False
    frames_submitted: int = 0
    scans_completed: int = 0
    matches_found: int = 0
    last_scan_seconds: float | None = None
    last_error: str | None = None
    heartbeat_monotonic: float = 0.0


class AsyncNameplateTracker:
    """Run slow OCR off the capture loop while retaining authoritative freshness."""

    def __init__(
        self,
        *,
        reader: TextReader,
        matcher: NameplateMatcher,
        scan_interval_seconds: float = 0.5,
        max_result_age_seconds: float = 1.0,
    ) -> None:
        if scan_interval_seconds <= 0 or max_result_age_seconds <= 0:
            raise ValueError("nameplate timing values must be positive")
        self.reader = reader
        self.matcher = matcher
        self.scan_interval_seconds = float(scan_interval_seconds)
        self.max_result_age_seconds = float(max_result_age_seconds)
        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: tuple[Any, float] | None = None
        self._latest: ObservedNameplate | None = None
        self._next_submit_monotonic = float("-inf")
        self._status = NameplateTrackerStatus()

    @property
    def status(self) -> NameplateTrackerStatus:
        with self._lock:
            return replace(self._status)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("nameplate tracker is already started")
            self._pending = None
            self._latest = None
            self._next_submit_monotonic = float("-inf")
            self._stop_event.clear()
            self._wake_event.clear()
            now = time.monotonic()
            self._status = NameplateTrackerStatus(
                running=True,
                heartbeat_monotonic=now,
            )
            self._thread = threading.Thread(
                target=self._run,
                name="nameplate-ocr",
                daemon=True,
            )
            self._thread.start()

    def submit(self, frame: Any, *, observed_monotonic: float) -> bool:
        observed = float(observed_monotonic)
        with self._lock:
            if self._thread is None or not self._status.running:
                return False
            if observed < self._next_submit_monotonic:
                return False
            self._next_submit_monotonic = observed + self.scan_interval_seconds
            copied = frame.copy()
            self._pending = (copied, observed)
            self._status = replace(
                self._status,
                frames_submitted=self._status.frames_submitted + 1,
                heartbeat_monotonic=time.monotonic(),
            )
        self._wake_event.set()
        return True

    def snapshot(self, *, now_monotonic: float) -> ObservedNameplate | None:
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        if float(now_monotonic) - latest.observed_monotonic > self.max_result_age_seconds:
            return None
        return latest

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        with self._lock:
            self._thread = None
            self._pending = None
            self._status = replace(
                self._status,
                running=False,
                scanning=False,
                heartbeat_monotonic=time.monotonic(),
            )

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._wake_event.wait(0.1)
                self._wake_event.clear()
                if self._stop_event.is_set():
                    break
                with self._lock:
                    pending = self._pending
                    self._pending = None
                    if pending is not None:
                        self._status = replace(
                            self._status,
                            scanning=True,
                            heartbeat_monotonic=time.monotonic(),
                        )
                if pending is None:
                    continue
                frame, observed_monotonic = pending
                started = time.monotonic()
                regions = self.reader.read(frame)
                match = self.matcher.match(regions)
                finished = time.monotonic()
                with self._lock:
                    self._latest = (
                        ObservedNameplate(
                            match=match,
                            observed_monotonic=observed_monotonic,
                        )
                        if match is not None
                        else None
                    )
                    self._status = replace(
                        self._status,
                        scanning=False,
                        scans_completed=self._status.scans_completed + 1,
                        matches_found=self._status.matches_found + int(match is not None),
                        last_scan_seconds=finished - started,
                        heartbeat_monotonic=finished,
                    )
        except BaseException as exc:
            with self._lock:
                self._latest = None
                self._status = replace(
                    self._status,
                    running=False,
                    scanning=False,
                    last_error=str(exc),
                    heartbeat_monotonic=time.monotonic(),
                )
        finally:
            with self._lock:
                self._status = replace(
                    self._status,
                    running=False,
                    scanning=False,
                    heartbeat_monotonic=time.monotonic(),
                )


class EasyOcrTextReader:
    """Lazy CPU EasyOCR adapter so model loading cannot stall video capture."""

    def __init__(
        self,
        *,
        input_width: int = 960,
        model_storage_directory: str | None = None,
    ) -> None:
        if input_width <= 0:
            raise ValueError("input_width must be positive")
        self.input_width = int(input_width)
        self.model_storage_directory = model_storage_directory
        self._reader: Any | None = None

    def read(self, frame: Any) -> list[OcrTextRegion]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is missing; install requirements-windows.txt") from exc
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError(
                    "EasyOCR is missing; install requirements-windows.txt"
                ) from exc
            options: dict[str, Any] = {"gpu": False, "verbose": False}
            if self.model_storage_directory is not None:
                options["model_storage_directory"] = self.model_storage_directory
            self._reader = easyocr.Reader(["en"], **options)

        height, width = int(frame.shape[0]), int(frame.shape[1])
        scale = 1.0
        image = frame
        if width > self.input_width:
            scale = width / self.input_width
            resized_height = max(1, round(height / scale))
            image = cv2.resize(frame, (self.input_width, resized_height))
        raw = self._reader.readtext(
            image,
            detail=1,
            paragraph=False,
            decoder="greedy",
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 _-",
        )
        regions: list[OcrTextRegion] = []
        for quad, text, confidence in raw:
            xs = [float(point[0]) * scale for point in quad]
            ys = [float(point[1]) * scale for point in quad]
            if not str(text).strip():
                continue
            regions.append(
                OcrTextRegion(
                    text=str(text),
                    bbox_xyxy=(min(xs), min(ys), max(xs), max(ys)),
                    confidence=float(confidence),
                )
            )
        return regions


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _candidate_regions(regions: Iterable[OcrTextRegion]) -> list[OcrTextRegion]:
    ordered = sorted(regions, key=lambda region: (region.bbox_xyxy[0], region.bbox_xyxy[1]))
    candidates = list(ordered)
    for first_index, first in enumerate(ordered):
        for second_index in range(first_index + 1, len(ordered)):
            second = ordered[second_index]
            if not _regions_are_adjacent(first, second):
                continue
            candidates.append(_merge_regions([first, second]))
            for third in ordered[second_index + 1 :]:
                if _regions_are_adjacent(second, third):
                    candidates.append(_merge_regions([first, second, third]))
    return candidates


def _regions_are_adjacent(left: OcrTextRegion, right: OcrTextRegion) -> bool:
    lx1, ly1, lx2, ly2 = left.bbox_xyxy
    rx1, ry1, _rx2, ry2 = right.bbox_xyxy
    left_height = ly2 - ly1
    right_height = ry2 - ry1
    center_delta = abs((ly1 + ly2) * 0.5 - (ry1 + ry2) * 0.5)
    gap = rx1 - lx2
    scale = max(left_height, right_height)
    return center_delta <= scale * 0.75 and -scale <= gap <= scale * 3.0


def _merge_regions(regions: list[OcrTextRegion]) -> OcrTextRegion:
    boxes = [region.bbox_xyxy for region in regions]
    weights = [max(1, len(_normalize_text(region.text))) for region in regions]
    confidence = sum(
        region.confidence * weight for region, weight in zip(regions, weights)
    ) / sum(weights)
    return OcrTextRegion(
        text=" ".join(region.text for region in regions),
        bbox_xyxy=(
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ),
        confidence=confidence,
    )


def _validated_bbox(
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox must contain four values")
    bbox = tuple(float(value) for value in bbox_xyxy)
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError("bbox must contain finite values")
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("bbox must be ordered")
    return bbox
