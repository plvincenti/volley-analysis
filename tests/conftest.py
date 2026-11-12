import numpy as np
import pytest

from volley_analysis.calibration import CameraPose


@pytest.fixture
def camera() -> CameraPose:
    """Caméra en tribune derrière la ligne de fond, visant le centre du terrain."""
    center = np.array([6.0, -2.5, 4.5])
    forward = np.array([4.5, 9.0, 0.0]) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return CameraPose(950.0, (960, 540), np.vstack([right, down, forward]), center)
