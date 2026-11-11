import numpy as np

from volley_analysis.court import COURT_LENGTH, NEAR
from volley_analysis.render import CourtCanvas


def test_canvas_puts_camera_side_at_the_bottom():
    canvas = CourtCanvas(pixels_per_meter=10, margin_m=1.0)
    near, far = canvas.to_pixels([[0.0, 0.0], [0.0, COURT_LENGTH]])
    assert near[1] > far[1]
    np.testing.assert_allclose(far, [10, 10])


def test_heatmap_and_positions_have_canvas_shape():
    canvas = CourtCanvas()
    positions = np.array([[4.5, 3.0], [2.0, 7.0]])
    assert canvas.heatmap(positions).shape == (canvas.height, canvas.width, 3)
    assert canvas.draw_positions(positions, np.array([NEAR, NEAR])).shape == (
        canvas.height,
        canvas.width,
        3,
    )
    assert canvas.heatmap(np.empty((0, 2))).shape == (canvas.height, canvas.width, 3)
