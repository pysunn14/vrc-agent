from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, TYPE_CHECKING

from .follow_protocol import TargetSource
from .nameplate import ObservedNameplate

if TYPE_CHECKING:
    from .windows_perception import TrackedDetection


@dataclass(frozen=True)
class TargetMeasurement:
    center_x: float
    proximity: float
    confidence: float
    source: TargetSource

    def __post_init__(self) -> None:
        for field in ("center_x", "proximity", "confidence"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")
            object.__setattr__(self, field, value)


class TargetFusionSelector:
    """Fuse identity-bearing nameplates with short-term body tracks."""

    def __init__(
        self,
        *,
        require_nameplate_identity: bool = False,
        reacquire_after_missed_frames: int = 6,
        desired_nameplate_width_ratio: float = 0.16,
        hold_proximity: float = 0.45,
    ) -> None:
        if isinstance(reacquire_after_missed_frames, bool) or reacquire_after_missed_frames < 1:
            raise ValueError("reacquire_after_missed_frames must be a positive integer")
        if not 0.0 < desired_nameplate_width_ratio <= 1.0:
            raise ValueError("desired_nameplate_width_ratio must be in (0, 1]")
        if not 0.0 < hold_proximity <= 1.0:
            raise ValueError("hold_proximity must be in (0, 1]")
        self.require_nameplate_identity = bool(require_nameplate_identity)
        self.reacquire_after_missed_frames = int(reacquire_after_missed_frames)
        self.desired_nameplate_width_ratio = float(desired_nameplate_width_ratio)
        self.hold_proximity = float(hold_proximity)
        self._identity_acquired = False
        self._target_track_id: int | None = None
        self._missed_frames = 0

    @property
    def identity_acquired(self) -> bool:
        return self._identity_acquired

    @property
    def target_track_id(self) -> int | None:
        return self._target_track_id

    def reset(self) -> None:
        self._identity_acquired = False
        self._target_track_id = None
        self._missed_frames = 0

    def select(
        self,
        detections: Iterable["TrackedDetection"],
        *,
        nameplate: ObservedNameplate | None,
        frame_shape: tuple[int, ...],
    ) -> TargetMeasurement | None:
        height, width = _frame_dimensions(frame_shape)
        candidates = list(detections)

        if nameplate is not None:
            self._identity_acquired = True
            body = _associated_body(candidates, nameplate)
            if body is not None:
                self._target_track_id = body.track_id
                self._missed_frames = 0
                center_x, proximity = _body_geometry(body, width=width, height=height)
                return TargetMeasurement(
                    center_x=center_x,
                    proximity=proximity,
                    confidence=(body.confidence + nameplate.match.confidence) * 0.5,
                    source=TargetSource.FUSED,
                )
            return self._nameplate_measurement(nameplate, width=width)

        body = self._select_body_without_nameplate(candidates)
        if body is None:
            return None
        center_x, proximity = _body_geometry(body, width=width, height=height)
        return TargetMeasurement(
            center_x=center_x,
            proximity=proximity,
            confidence=body.confidence,
            source=TargetSource.BODY,
        )

    def _select_body_without_nameplate(
        self,
        candidates: list["TrackedDetection"],
    ) -> "TrackedDetection | None":
        if self._target_track_id is not None:
            for detection in candidates:
                if detection.track_id == self._target_track_id:
                    self._missed_frames = 0
                    return detection
            self._missed_frames += 1
            if self._missed_frames < self.reacquire_after_missed_frames:
                return None
            self.reset()
        if self.require_nameplate_identity:
            # A body without a current identity-bearing track can be a different
            # avatar or even decorative world art. Only a new nameplate match may
            # authorize another body track.
            self.reset()
            return None
        if not candidates:
            return None
        selected = max(candidates, key=lambda detection: detection.area)
        self._target_track_id = selected.track_id
        self._missed_frames = 0
        return selected

    def _nameplate_measurement(
        self,
        nameplate: ObservedNameplate,
        *,
        width: int,
    ) -> TargetMeasurement:
        x1, _y1, x2, _y2 = nameplate.match.bbox_xyxy
        width_ratio = max(0.0, min(1.0, (x2 - x1) / width))
        proximity = min(
            1.0,
            self.hold_proximity * width_ratio / self.desired_nameplate_width_ratio,
        )
        if x1 <= 1.0 or x2 >= width - 1.0:
            proximity = max(proximity, self.hold_proximity)
        return TargetMeasurement(
            center_x=max(0.0, min(1.0, (x1 + x2) * 0.5 / width)),
            proximity=proximity,
            confidence=nameplate.match.confidence,
            source=TargetSource.NAMEPLATE,
        )


def _associated_body(
    candidates: list["TrackedDetection"],
    nameplate: ObservedNameplate,
) -> "TrackedDetection | None":
    tx1, ty1, tx2, ty2 = nameplate.match.bbox_xyxy
    tag_center_x = (tx1 + tx2) * 0.5
    tag_center_y = (ty1 + ty2) * 0.5
    tag_width = tx2 - tx1
    ranked: list[tuple[float, "TrackedDetection"]] = []
    for body in candidates:
        bx1, by1, bx2, by2 = body.bbox_xyxy
        body_width = bx2 - bx1
        body_height = by2 - by1
        body_center_x = (bx1 + bx2) * 0.5
        body_center_y = (by1 + by2) * 0.5
        horizontal_delta = abs(body_center_x - tag_center_x)
        max_horizontal_delta = max(tag_width * 1.5, body_width * 0.6)
        vertical_gap = max(0.0, by1 - ty2)
        if body_center_y <= tag_center_y:
            continue
        if horizontal_delta > max_horizontal_delta:
            continue
        if vertical_gap > body_height * 0.5:
            continue
        score = horizontal_delta / max_horizontal_delta + vertical_gap / body_height
        ranked.append((score, body))
    return min(ranked, key=lambda item: item[0])[1] if ranked else None


def _body_geometry(
    body: "TrackedDetection",
    *,
    width: int,
    height: int,
) -> tuple[float, float]:
    x1, y1, x2, y2 = body.bbox_xyxy
    return (
        max(0.0, min(1.0, (x1 + x2) * 0.5 / width)),
        max(0.0, min(1.0, (y2 - y1) / height)),
    )


def _frame_dimensions(frame_shape: tuple[int, ...]) -> tuple[int, int]:
    try:
        height, width = int(frame_shape[0]), int(frame_shape[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("frame_shape must contain positive height and width") from exc
    if height <= 0 or width <= 0:
        raise ValueError("frame_shape must contain positive height and width")
    return height, width
