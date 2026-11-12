import numpy as np
import pandas as pd

from volley_analysis.court import FAR, NEAR
from volley_analysis.referee import SAMPLE_S, referee_call, referee_gestures

SHOULDER = np.array([100.0, 100.0])
TORSO_PX = 50.0
RESTING = ((110.0, 130.0), (95.0, 125.0))  # mains jointes devant la poitrine, coude plié


def _pose_row(time_s, left_arm=RESTING, right_arm=RESTING):
    row = {"time_s": time_s}
    points = {5: SHOULDER, 6: SHOULDER, 11: SHOULDER + [0, TORSO_PX], 12: SHOULDER + [0, TORSO_PX]}
    for (elbow, wrist), (k_elbow, k_wrist) in ((left_arm, (7, 9)), (right_arm, (8, 10))):
        points[k_elbow], points[k_wrist] = np.array(elbow), np.array(wrist)
    for k in range(17):
        x, y = points.get(k, SHOULDER)
        row[f"k{k}x"], row[f"k{k}y"], row[f"k{k}c"] = x, y, 0.9
    return row


def _extended(dx, dy):
    """Bras tendu, poignet à (dx, dy) torses de l'épaule."""
    wrist = SHOULDER + np.array([dx, dy]) * TORSO_PX
    return tuple((SHOULDER + wrist) / 2), tuple(wrist)


def _pose(*segments):
    """Segments (début, fin, bras gauche) échantillonnés comme `track_referee`."""
    rows = [
        _pose_row(t, left_arm=arm) for t0, t1, arm in segments for t in np.arange(t0, t1, SAMPLE_S)
    ]
    return pd.DataFrame(rows)


def test_brief_extended_arms_point_to_a_side_and_resting_postures_do_not():
    pose = _pose(
        (0.0, 2.0, RESTING),
        (2.0, 2.6, _extended(0.6, 0.1)),  # vers la caméra
        (2.6, 6.0, RESTING),
        (6.0, 6.6, _extended(-0.5, -0.25)),  # vers le fond
        (6.6, 9.0, RESTING),
        (9.0, 13.0, _extended(-0.5, -0.25)),  # main posée sur le poteau
        (13.0, 14.0, RESTING),
    )
    gestures = referee_gestures(pose)

    assert list(gestures["side"]) == [NEAR, FAR]
    assert referee_call(gestures, 1.0, 5.0) == NEAR
    assert referee_call(gestures, 5.0, 14.0) == FAR
    assert referee_call(gestures, 8.0, 14.0) is None
