import numpy as np

from volley_analysis.court import FAR, NEAR, is_in_play_area, side_of_net


def test_in_play_area_uses_larger_margin_behind_end_lines():
    positions = np.array(
        [
            [4.5, 9.0],  # centre du terrain
            [4.5, -3.0],  # serveur derrière la ligne de fond
            [-2.0, 9.0],  # arbitre sur le côté, au niveau du filet
            [4.5, 23.0],  # trop loin derrière le fond opposé
        ]
    )
    assert is_in_play_area(positions).tolist() == [True, True, False, False]


def test_side_of_net():
    assert side_of_net(np.array([0.0, 8.9, 9.0, 17.0])).tolist() == [NEAR, NEAR, FAR, FAR]
