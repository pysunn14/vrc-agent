from __future__ import annotations

import math
from typing import Any, Protocol


class NameplateVisualLocator(Protocol):
    def reset(self) -> None: ...

    def anchor(
        self,
        frame: Any,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> None: ...

    def locate(
        self,
        frame: Any,
    ) -> tuple[tuple[float, float, float, float], float] | None: ...


class OpenCvTemplateNameplateLocator:
    """Relocate a slow OCR anchor on current frames with fast template matching."""

    def __init__(
        self,
        *,
        input_width: int = 960,
        minimum_score: float = 0.55,
        scale_factors: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.25),
    ) -> None:
        if input_width <= 0:
            raise ValueError("input_width must be positive")
        if not 0.0 < minimum_score <= 1.0:
            raise ValueError("minimum_score must be in (0, 1]")
        if not scale_factors or any(scale <= 0 for scale in scale_factors):
            raise ValueError("scale_factors must contain positive values")
        self.input_width = int(input_width)
        self.minimum_score = float(minimum_score)
        self.scale_factors = tuple(float(scale) for scale in scale_factors)
        self._template: Any | None = None
        self._target_offset: tuple[float, float, float, float] | None = None
        self._last_template_bbox: tuple[int, int, int, int] | None = None

    def reset(self) -> None:
        self._template = None
        self._target_offset = None
        self._last_template_bbox = None

    def anchor(
        self,
        frame: Any,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> None:
        gray, frame_scale = self._gray(frame)
        x1, y1, x2, y2 = (value / frame_scale for value in bbox_xyxy)
        margin = max(2.0, (y2 - y1) * 0.35)
        crop_x1 = max(0, math.floor(x1 - margin))
        crop_y1 = max(0, math.floor(y1 - margin))
        crop_x2 = min(gray.shape[1], math.ceil(x2 + margin))
        crop_y2 = min(gray.shape[0], math.ceil(y2 + margin))
        if crop_x2 - crop_x1 < 8 or crop_y2 - crop_y1 < 6:
            raise RuntimeError("OCR nameplate anchor is too small for visual tracking")
        self._template = gray[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        self._target_offset = (
            x1 - crop_x1,
            y1 - crop_y1,
            x2 - crop_x1,
            y2 - crop_y1,
        )
        self._last_template_bbox = None

    def locate(
        self,
        frame: Any,
    ) -> tuple[tuple[float, float, float, float], float] | None:
        if self._template is None or self._target_offset is None:
            return None
        gray, frame_scale = self._gray(frame)
        best = None
        if self._last_template_bbox is not None:
            search = self._expanded_search_bbox(gray.shape, self._last_template_bbox)
            best = self._best_match(gray, search_bbox=search)
        if best is None or best[0] < self.minimum_score:
            best = self._best_match(gray, search_bbox=None)
        if best is None or best[0] < self.minimum_score:
            self._last_template_bbox = None
            return None
        score, left, top, scale, width, height = best
        self._last_template_bbox = (left, top, left + width, top + height)
        ox1, oy1, ox2, oy2 = self._target_offset
        bbox = (
            (left + ox1 * scale) * frame_scale,
            (top + oy1 * scale) * frame_scale,
            (left + ox2 * scale) * frame_scale,
            (top + oy2 * scale) * frame_scale,
        )
        return _validated_bbox(bbox), min(1.0, max(0.0, score))

    def _gray(self, frame: Any) -> tuple[Any, float]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is missing; install requirements-windows.txt") from exc
        if not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise RuntimeError("nameplate frame must be an image array")
        height, width = int(frame.shape[0]), int(frame.shape[1])
        if width <= 0 or height <= 0:
            raise RuntimeError("nameplate frame must have positive dimensions")
        frame_scale = max(1.0, width / self.input_width)
        image = frame
        if frame_scale > 1.0:
            image = cv2.resize(
                frame,
                (self.input_width, max(1, round(height / frame_scale))),
            )
        if len(image.shape) == 2:
            return image, frame_scale
        conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(image, conversion), frame_scale

    def _expanded_search_bbox(
        self,
        frame_shape: tuple[int, ...],
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return (
            max(0, x1 - width * 3),
            max(0, y1 - height * 4),
            min(frame_shape[1], x2 + width * 3),
            min(frame_shape[0], y2 + height * 4),
        )

    def _best_match(
        self,
        gray: Any,
        *,
        search_bbox: tuple[int, int, int, int] | None,
    ) -> tuple[float, int, int, float, int, int] | None:
        import cv2

        if search_bbox is None:
            sx1, sy1, sx2, sy2 = 0, 0, gray.shape[1], gray.shape[0]
        else:
            sx1, sy1, sx2, sy2 = search_bbox
        search = gray[sy1:sy2, sx1:sx2]
        best: tuple[float, int, int, float, int, int] | None = None
        for scale in self.scale_factors:
            width = max(1, round(self._template.shape[1] * scale))
            height = max(1, round(self._template.shape[0] * scale))
            if width > search.shape[1] or height > search.shape[0]:
                continue
            candidate = cv2.resize(self._template, (width, height))
            response = cv2.matchTemplate(search, candidate, cv2.TM_CCOEFF_NORMED)
            _minimum, score, _minimum_location, location = cv2.minMaxLoc(response)
            if not math.isfinite(score):
                continue
            item = (
                float(score),
                int(location[0] + sx1),
                int(location[1] + sy1),
                scale,
                width,
                height,
            )
            if best is None or item[0] > best[0]:
                best = item
        return best


def _validated_bbox(
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if len(bbox_xyxy) != 4:
        raise RuntimeError("visual nameplate bbox must contain four values")
    bbox = tuple(float(value) for value in bbox_xyxy)
    if not all(math.isfinite(value) for value in bbox):
        raise RuntimeError("visual nameplate bbox must contain finite values")
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise RuntimeError("visual nameplate bbox must be ordered")
    return bbox
