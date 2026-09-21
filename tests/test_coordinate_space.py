from __future__ import annotations

import unittest

import numpy as np

from vrc_ardy_agent.coordinate_space import (
    ardy_to_unity_position,
    ardy_to_unity_rotation,
)


class CoordinateSpaceTests(unittest.TestCase):
    def test_position_mirrors_only_the_left_positive_axis(self):
        converted = ardy_to_unity_position(np.array([1.0, 2.0, 3.0]))

        np.testing.assert_allclose(converted, (-1.0, 2.0, 3.0), atol=1e-9)

    def test_rotation_conversion_preserves_a_proper_rotation(self):
        angle = np.deg2rad(30.0)
        source = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )

        converted = ardy_to_unity_rotation(source)

        self.assertAlmostEqual(float(np.linalg.det(converted)), 1.0, places=9)
        np.testing.assert_allclose(
            converted,
            np.array(
                [
                    [np.cos(angle), 0.0, -np.sin(angle)],
                    [0.0, 1.0, 0.0],
                    [np.sin(angle), 0.0, np.cos(angle)],
                ]
            ),
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
