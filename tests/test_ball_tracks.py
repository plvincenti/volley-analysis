import numpy as np
import pandas as pd

from volley_analysis.ball_tracks import (
    BALL_RADIUS_M,
    RADIUS_BIAS_PX,
    ball_positions_3d,
    clean_ball_track,
)

FPS = 60


def _detections(frames, xs, ys):
    return pd.DataFrame(
        {
            "frame": frames,
            "time_s": np.asarray(frames) / FPS,
            "visible": ~np.isnan(xs),
            "x": xs,
            "y": ys,
            "radius": 10.0,
            "conf": 0.7,
        }
    )


def _parabola(frames):
    t = (np.asarray(frames) - frames[0]) / FPS
    return 400 + 600 * t, 700 - 900 * t + 900 * t**2


def test_isolated_and_distant_detections_are_rejected():
    frames = np.arange(0, 120, 2)
    xs, ys = _parabola(frames)
    xs[20], ys[20] = 1500, 340  # ballon du terrain voisin, sur une seule image
    xs[21:23], ys[21:23] = 1505, 345
    track = clean_ball_track(_detections(frames, xs, ys), FPS).set_index("frame")

    assert track.loc[frames[20:23], "source"].ne("detected").all()
    assert (track.loc[frames[20:23], "x"] < 1000).all()
    assert track.loc[frames[:20], "source"].eq("detected").all()


def test_ball_position_in_3d_from_apparent_radius(camera):
    balls = np.array([[2.0, 16.0, 3.0], [7.0, 1.0, 1.0]])
    depths = (balls - camera.center) @ camera.rotation[2]
    pixels = camera.project(balls)
    track = pd.DataFrame(
        {
            "x": pixels[:, 0],
            "y": pixels[:, 1],
            "radius": camera.focal * BALL_RADIUS_M / depths + RADIUS_BIAS_PX,
        }
    )

    positions = ball_positions_3d(track, camera)
    np.testing.assert_allclose(positions[["court_x", "court_y", "height"]], balls, atol=1e-6)


def test_skipped_frames_and_short_gaps_are_interpolated_but_not_long_ones():
    frames = np.arange(0, 240, 2)
    xs, ys = _parabola(frames)
    xs[30:34] = ys[30:34] = np.nan  # ~0.13 s sans ballon : interpolé
    xs[60:80] = ys[60:80] = np.nan  # ~0.67 s hors cadre : laissé vide
    track = clean_ball_track(_detections(frames, xs, ys), FPS).set_index("frame")

    assert track.loc[1, "source"] == "interpolated"
    expected_x, _ = _parabola(np.arange(0, 240))
    np.testing.assert_allclose(track.loc[61:67, "x"], expected_x[61:68], atol=2.0)
    assert track.loc[125:155, "x"].isna().all()
