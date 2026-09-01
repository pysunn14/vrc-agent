from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .follow_receiver import ReceivedObservation


class FollowState(str, Enum):
    LOST = "lost"
    SEARCH = "search"
    RELOCATE = "relocate"
    ALIGN = "align"
    FOLLOW = "follow"
    HOLD = "hold"


@dataclass(frozen=True)
class FollowConfig:
    stale_after_seconds: float = 0.3
    active_search: bool = True
    search_delay_seconds: float = 0.75
    search_sweep_seconds: float = 6.0
    search_turn: float = 0.35
    relocate_turn_seconds: float = 1.0
    relocate_forward_seconds: float = 1.0
    relocate_forward: float = 0.2
    align_enter_error: float = 0.2
    align_exit_error: float = 0.1
    resume_follow_below_height: float = 0.35
    hold_above_height: float = 0.45
    turn_gain: float = 1.0
    max_turn: float = 0.6
    forward_gain: float = 1.0
    max_forward: float = 0.5
    smoothing_alpha: float = 0.35

    def __post_init__(self) -> None:
        if self.stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        if self.search_delay_seconds < 0:
            raise ValueError("search_delay_seconds must be non-negative")
        if self.search_sweep_seconds <= 0:
            raise ValueError("search_sweep_seconds must be positive")
        if not 0 < self.search_turn <= 1:
            raise ValueError("search_turn must be in (0, 1]")
        if self.relocate_turn_seconds <= 0 or self.relocate_forward_seconds <= 0:
            raise ValueError("relocation durations must be positive")
        if not 0 < self.relocate_forward <= 1:
            raise ValueError("relocate_forward must be in (0, 1]")
        if not 0 <= self.align_exit_error < self.align_enter_error <= 1:
            raise ValueError("alignment thresholds must satisfy 0 <= exit < enter <= 1")
        if not 0 < self.resume_follow_below_height < self.hold_above_height <= 1:
            raise ValueError("distance thresholds must satisfy 0 < resume < hold <= 1")
        if self.turn_gain <= 0 or self.forward_gain <= 0:
            raise ValueError("controller gains must be positive")
        if not 0 < self.max_turn <= 1 or not 0 < self.max_forward <= 1:
            raise ValueError("maximum control values must be in (0, 1]")
        if not 0 < self.smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha must be in (0, 1]")


@dataclass(frozen=True)
class FollowDecision:
    state: FollowState
    horizontal: float
    vertical: float
    look_horizontal: float
    observation_age_seconds: float | None
    target_center_error: float | None
    target_height: float | None

    def __post_init__(self) -> None:
        for field in ("horizontal", "vertical", "look_horizontal"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{field} must be finite and in [-1, 1]")

    @classmethod
    def neutral(
        cls,
        state: FollowState,
        *,
        observation_age_seconds: float | None = None,
    ) -> "FollowDecision":
        return cls(
            state=state,
            horizontal=0.0,
            vertical=0.0,
            look_horizontal=0.0,
            observation_age_seconds=observation_age_seconds,
            target_center_error=None,
            target_height=None,
        )


class FollowController:
    """Reactive image-space follower with explicit loss and hysteresis states."""

    def __init__(self, config: FollowConfig | None = None) -> None:
        self.config = config or FollowConfig()
        self._state = FollowState.LOST
        self._last_observation_key: tuple[str, int] | None = None
        self._smoothed_center_error: float | None = None
        self._smoothed_height: float | None = None
        self._lost_since_monotonic: float | None = None
        self._search_direction = 1.0

    @property
    def state(self) -> FollowState:
        return self._state

    def step(
        self,
        received: ReceivedObservation | None,
        *,
        now_monotonic: float,
    ) -> FollowDecision:
        if received is None:
            return self._lose_target(now_monotonic=now_monotonic)

        age = max(0.0, float(now_monotonic) - received.received_monotonic)
        observation = received.observation
        if age > self.config.stale_after_seconds or not observation.visible:
            return self._lose_target(
                now_monotonic=now_monotonic,
                observation_age_seconds=age,
            )

        center_x = observation.center_x
        proximity = observation.proximity
        if center_x is None or proximity is None:
            return self._lose_target(
                now_monotonic=now_monotonic,
                observation_age_seconds=age,
            )

        center_error = 2.0 * (center_x - 0.5)
        self._lost_since_monotonic = None
        if center_error != 0.0:
            self._search_direction = math.copysign(1.0, center_error)
        key = (observation.session_id, observation.sequence)
        if key != self._last_observation_key:
            alpha = self.config.smoothing_alpha
            if self._smoothed_center_error is None:
                self._smoothed_center_error = center_error
                self._smoothed_height = proximity
            else:
                self._smoothed_center_error = (
                    alpha * center_error + (1.0 - alpha) * self._smoothed_center_error
                )
                assert self._smoothed_height is not None
                self._smoothed_height = (
                    alpha * proximity + (1.0 - alpha) * self._smoothed_height
                )
            self._last_observation_key = key

        assert self._smoothed_center_error is not None
        assert self._smoothed_height is not None
        center_error = self._smoothed_center_error
        height = self._smoothed_height

        align_threshold = (
            self.config.align_exit_error
            if self._state == FollowState.ALIGN
            else self.config.align_enter_error
        )
        if abs(center_error) > align_threshold:
            self._state = FollowState.ALIGN
            turn = _clamp(
                center_error * self.config.turn_gain,
                -self.config.max_turn,
                self.config.max_turn,
            )
            return FollowDecision(
                state=self._state,
                horizontal=0.0,
                vertical=0.0,
                look_horizontal=turn,
                observation_age_seconds=age,
                target_center_error=center_error,
                target_height=height,
            )

        if self._state == FollowState.FOLLOW:
            should_follow = height < self.config.hold_above_height
        elif self._state == FollowState.HOLD:
            should_follow = height < self.config.resume_follow_below_height
        else:
            should_follow = height < self.config.resume_follow_below_height

        if should_follow:
            self._state = FollowState.FOLLOW
            distance_error = max(0.0, self.config.hold_above_height - height)
            forward = min(self.config.max_forward, distance_error * self.config.forward_gain)
            return FollowDecision(
                state=self._state,
                horizontal=0.0,
                vertical=forward,
                look_horizontal=0.0,
                observation_age_seconds=age,
                target_center_error=center_error,
                target_height=height,
            )

        self._state = FollowState.HOLD
        return FollowDecision(
            state=self._state,
            horizontal=0.0,
            vertical=0.0,
            look_horizontal=0.0,
            observation_age_seconds=age,
            target_center_error=center_error,
            target_height=height,
        )

    def _lose_target(
        self,
        *,
        now_monotonic: float,
        observation_age_seconds: float | None = None,
    ) -> FollowDecision:
        now = float(now_monotonic)
        if self._lost_since_monotonic is None:
            self._lost_since_monotonic = now
        self._last_observation_key = None
        self._smoothed_center_error = None
        self._smoothed_height = None

        elapsed = max(0.0, now - self._lost_since_monotonic)
        if not self.config.active_search or elapsed < self.config.search_delay_seconds:
            self._state = FollowState.LOST
            return FollowDecision.neutral(
                self._state,
                observation_age_seconds=observation_age_seconds,
            )

        active_elapsed = elapsed - self.config.search_delay_seconds
        cycle_seconds = (
            self.config.search_sweep_seconds
            + self.config.relocate_turn_seconds
            + self.config.relocate_forward_seconds
        )
        cycle_index = int(active_elapsed // cycle_seconds)
        phase = active_elapsed % cycle_seconds
        direction = self._search_direction * (1.0 if cycle_index % 2 == 0 else -1.0)

        if phase < self.config.search_sweep_seconds:
            # Keep turning for the whole sweep. Reversing inside one sweep only
            # rocks the view around its starting heading and cannot find a target
            # that is behind the avatar. The next cycle scans the other direction.
            self._state = FollowState.SEARCH
            return FollowDecision(
                state=self._state,
                horizontal=0.0,
                vertical=0.0,
                look_horizontal=direction * self.config.search_turn,
                observation_age_seconds=observation_age_seconds,
                target_center_error=None,
                target_height=None,
            )

        relocation_phase = phase - self.config.search_sweep_seconds
        self._state = FollowState.RELOCATE
        if relocation_phase < self.config.relocate_turn_seconds:
            # Perception has no depth signal yet. Turn before moving so a failed
            # search does not blindly drive straight into the occluder ahead.
            return FollowDecision(
                state=self._state,
                horizontal=0.0,
                vertical=0.0,
                look_horizontal=direction * self.config.search_turn,
                observation_age_seconds=observation_age_seconds,
                target_center_error=None,
                target_height=None,
            )
        return FollowDecision(
            state=self._state,
            horizontal=0.0,
            vertical=self.config.relocate_forward,
            look_horizontal=0.0,
            observation_age_seconds=observation_age_seconds,
            target_center_error=None,
            target_height=None,
        )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
