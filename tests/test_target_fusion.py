from __future__ import annotations

import unittest

from vrc_ardy_agent.follow_protocol import TargetSource
from vrc_ardy_agent.nameplate import NameplateMatch, ObservedNameplate
from vrc_ardy_agent.target_fusion import TargetFusionSelector
from vrc_ardy_agent.windows_perception import TrackedDetection


def _body(
    track_id: int,
    bbox: tuple[float, float, float, float],
    confidence: float = 0.8,
) -> TrackedDetection:
    return TrackedDetection(track_id=track_id, bbox_xyxy=bbox, confidence=confidence)


def _nameplate(
    bbox: tuple[float, float, float, float],
) -> ObservedNameplate:
    return ObservedNameplate(
        match=NameplateMatch(
            target_name="TargetUser 28",
            recognized_text="TargetUser 28",
            bbox_xyxy=bbox,
            confidence=0.9,
            match_score=1.0,
        ),
        observed_monotonic=10.0,
    )


class TargetFusionSelectorTests(unittest.TestCase):
    def test_nameplate_selects_the_body_below_it_instead_of_the_largest_body(self):
        selector = TargetFusionSelector(reacquire_after_missed_frames=2)
        wrong_large_body = _body(1, (0, 100, 500, 950))
        matching_body = _body(2, (700, 150, 920, 850))

        result = selector.select(
            [wrong_large_body, matching_body],
            nameplate=_nameplate((740, 80, 900, 150)),
            frame_shape=(1000, 1000, 3),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.source, TargetSource.FUSED)
        self.assertAlmostEqual(result.center_x, 0.81)
        self.assertAlmostEqual(result.proximity, 0.7)
        self.assertEqual(selector.target_track_id, 2)

    def test_nameplate_without_body_supplies_bearing_and_conservative_proximity(self):
        selector = TargetFusionSelector(
            desired_nameplate_width_ratio=0.16,
            hold_proximity=0.45,
        )

        result = selector.select(
            [],
            nameplate=_nameplate((800, 80, 1000, 150)),
            frame_shape=(1000, 1000, 3),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.source, TargetSource.NAMEPLATE)
        self.assertAlmostEqual(result.center_x, 0.9)
        self.assertGreaterEqual(result.proximity, 0.45)

    def test_body_only_path_preserves_current_single_person_poc(self):
        selector = TargetFusionSelector()

        result = selector.select(
            [_body(4, (250, 100, 750, 900))],
            nameplate=None,
            frame_shape=(1000, 1000, 3),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.source, TargetSource.BODY)
        self.assertAlmostEqual(result.center_x, 0.5)
        self.assertAlmostEqual(result.proximity, 0.8)

    def test_strict_identity_mode_ignores_bodies_until_nameplate_matches(self):
        selector = TargetFusionSelector(require_nameplate_identity=True)

        result = selector.select(
            [_body(4, (250, 100, 750, 900))],
            nameplate=None,
            frame_shape=(1000, 1000, 3),
        )

        self.assertIsNone(result)
        self.assertFalse(selector.identity_acquired)
        self.assertIsNone(selector.target_track_id)

    def test_strict_identity_mode_continues_associated_body_track(self):
        selector = TargetFusionSelector(require_nameplate_identity=True)
        body = _body(4, (250, 100, 750, 900))

        acquired = selector.select(
            [body],
            nameplate=_nameplate((350, 30, 650, 120)),
            frame_shape=(1000, 1000, 3),
        )
        continued = selector.select(
            [body],
            nameplate=None,
            frame_shape=(1000, 1000, 3),
        )

        self.assertIsNotNone(acquired)
        self.assertIsNotNone(continued)
        assert continued is not None
        self.assertEqual(continued.source, TargetSource.BODY)
        self.assertTrue(selector.identity_acquired)
        self.assertEqual(selector.target_track_id, 4)

    def test_strict_identity_mode_does_not_switch_to_an_unverified_body(self):
        selector = TargetFusionSelector(
            require_nameplate_identity=True,
            reacquire_after_missed_frames=2,
        )
        verified = _body(4, (250, 100, 750, 900))
        unverified = _body(9, (0, 0, 900, 1000))
        selector.select(
            [verified],
            nameplate=_nameplate((350, 30, 650, 120)),
            frame_shape=(1000, 1000, 3),
        )

        first_miss = selector.select(
            [unverified],
            nameplate=None,
            frame_shape=(1000, 1000, 3),
        )
        second_miss = selector.select(
            [unverified],
            nameplate=None,
            frame_shape=(1000, 1000, 3),
        )

        self.assertIsNone(first_miss)
        self.assertIsNone(second_miss)
        self.assertFalse(selector.identity_acquired)
        self.assertIsNone(selector.target_track_id)


if __name__ == "__main__":
    unittest.main()
