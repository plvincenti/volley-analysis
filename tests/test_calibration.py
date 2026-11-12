import cv2
import numpy as np
import pytest

from volley_analysis.calibration import Calibration
from volley_analysis.court import KEYPOINTS

# Caméra fictive derrière le terrain : le fond opposé apparaît plus étroit et plus haut.
COURT_CORNERS = np.float32([[0, 0], [9, 0], [9, 18], [0, 18]])
IMAGE_CORNERS = np.float32([[300, 1000], [1620, 1000], [1180, 300], [740, 300]])
TRUE_COURT_TO_IMAGE = cv2.getPerspectiveTransform(COURT_CORNERS, IMAGE_CORNERS)


def _project(points):
    return cv2.perspectiveTransform(
        np.float64(points).reshape(-1, 1, 2), TRUE_COURT_TO_IMAGE
    ).reshape(-1, 2)


def _calibration(names=None):
    keypoints = [kp for kp in KEYPOINTS if names is None or kp.name in names]
    pixels = _project([kp.xy for kp in keypoints])
    return Calibration(
        image_points={kp.name: tuple(p) for kp, p in zip(keypoints, pixels, strict=True)},
        frame_size=(1920, 1080),
    )


def test_image_to_court_recovers_court_positions():
    calibration = _calibration()
    court_positions = np.array([[4.5, 9.0], [1.0, 2.0], [8.0, 15.5]])
    recovered = calibration.image_to_court(_project(court_positions))
    np.testing.assert_allclose(recovered, court_positions, atol=1e-6)
    assert calibration.reprojection_error() < 1e-3


def test_four_corners_are_enough():
    calibration = _calibration({"near_left", "near_right", "far_right", "far_left"})
    np.testing.assert_allclose(
        calibration.image_to_court(_project([[3.0, 12.0]])), [[3.0, 12.0]], atol=1e-6
    )


def test_rejects_too_few_points():
    with pytest.raises(ValueError, match="au moins 4"):
        _calibration({"near_left", "near_right", "far_right"})


def test_camera_pose_is_recovered_from_the_floor_homography(camera):
    corners = {"near_left": (0, 0), "near_right": (9, 0), "far_right": (9, 18), "far_left": (0, 18)}
    pixels = camera.project(np.array([[x, y, 0.0] for x, y in corners.values()]))
    calibration = Calibration(
        image_points={name: tuple(p) for name, p in zip(corners, pixels, strict=True)},
        frame_size=(1920, 1080),
    )

    recovered = calibration.camera()
    assert recovered.focal == pytest.approx(camera.focal, rel=1e-6)
    np.testing.assert_allclose(recovered.center, camera.center, atol=1e-6)
    ball = np.array([[3.0, 12.0, 2.5]])
    depth = (ball - recovered.center) @ recovered.rotation[2]
    np.testing.assert_allclose(
        recovered.backproject(recovered.project(ball), depth), ball, atol=1e-6
    )


def test_save_and_load_roundtrip(tmp_path):
    calibration = _calibration()
    path = tmp_path / "calib.json"
    calibration.save(path)
    loaded = Calibration.load(path)
    assert loaded.image_points.keys() == calibration.image_points.keys()
    np.testing.assert_allclose(loaded.homography, calibration.homography)
