import cv2
import numpy as np

from volley_analysis.calibration_drawing import (
    assign_court_lines,
    calibration_from_drawing,
    fit_lines,
)

# Terrain vu depuis l'arrière, coins côté caméra hors de l'image (comme dans le gymnase).
CORNERS = {
    "near_left": (-226.0, 996.0),
    "near_right": (1724.0, 1186.0),
    "far_left": (766.0, 215.0),
    "far_right": (1219.0, 203.0),
}
SEGMENTS = {
    "far_end": ("far_left", "far_right"),
    "near_end": ("near_left", "near_right"),
    "left": ("near_left", "far_left"),
    "right": ("near_right", "far_right"),
}


def _shaky_stroke(start, end, rng, n=400, jitter=2.5):
    t = np.linspace(0, 1, n)[:, None]
    points = np.asarray(start) * (1 - t) + np.asarray(end) * t
    return points + rng.normal(0, jitter, points.shape)


def _inside(points, width=1920, height=1080):
    keep = (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0)
    return points[keep & (points[:, 1] < height)]


def test_fit_and_assign_lines_from_shaky_strokes():
    rng = np.random.default_rng(1)
    strokes = [_inside(_shaky_stroke(CORNERS[a], CORNERS[b], rng)) for a, b in SEGMENTS.values()]
    lines = assign_court_lines(fit_lines(np.vstack(strokes)))

    for name, (a, b) in SEGMENTS.items():
        line = lines[name]
        assert line.distances(np.array([CORNERS[a], CORNERS[b]])).max() < 3.0, name
    near_left = lines["left"].intersect(lines["near_end"])
    np.testing.assert_allclose(near_left, CORNERS["near_left"], atol=8.0)


def test_calibration_from_scaled_screenshot():
    rng = np.random.default_rng(2)
    frame = cv2.GaussianBlur(rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8), (5, 5), 0)
    scale, offset = 1.35, np.array([4.0, -30.0])
    screenshot = cv2.warpAffine(
        frame,
        np.array([[scale, 0, offset[0]], [0, scale, offset[1]]]),
        (int(1920 * scale), int(1080 * scale)),
    )
    for a, b in SEGMENTS.values():
        stroke = _shaky_stroke(CORNERS[a], CORNERS[b], rng, jitter=1.5) * scale + offset
        cv2.polylines(screenshot, [stroke.round().astype(np.int32)], False, (0, 0, 255), 4)

    calibration, _ = calibration_from_drawing(screenshot, frame)

    for name, expected in CORNERS.items():
        np.testing.assert_allclose(calibration.image_points[name], expected, atol=10.0)
    np.testing.assert_allclose(
        calibration.image_to_court([CORNERS["far_right"]]), [[9.0, 18.0]], atol=0.1
    )
