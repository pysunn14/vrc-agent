from __future__ import annotations

import numpy as np


_ARDY_TO_UNITY_BASIS = np.diag((-1.0, 1.0, 1.0))


def ardy_to_unity_position(position: np.ndarray) -> np.ndarray:
    """Mirror ARDY's left-positive X axis into Unity's right-positive X axis."""
    value = np.asarray(position, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError(f"position must contain three values, got {value.shape}")
    return _ARDY_TO_UNITY_BASIS @ value


def ardy_to_unity_rotation(rotation: np.ndarray) -> np.ndarray:
    """Change a proper rotation from ARDY coordinates into Unity coordinates."""
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {value.shape}")
    return _ARDY_TO_UNITY_BASIS @ value @ _ARDY_TO_UNITY_BASIS
