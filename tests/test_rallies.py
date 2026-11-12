import numpy as np
import pandas as pd
import pytest

from volley_analysis.court import FAR, NEAR
from volley_analysis.rallies import detect_rallies

RATE = 30


def _ball(*segments):
    """Détections à 30 img/s : chaque segment va de (t0, y0) à (t1, y1) à une hauteur fixe."""
    parts = []
    for t0, t1, y0, y1, height in segments:
        t = np.arange(t0, t1, 1 / RATE)
        parts.append(
            pd.DataFrame({"time_s": t, "court_y": np.linspace(y0, y1, len(t)), "height": height})
        )
    return pd.concat(parts, ignore_index=True).assign(source="detected", court_x=4.5)


def test_balls_on_the_neighbouring_court_are_ignored():
    rally = _ball(
        (0.0, 0.5, 2.0, 2.0, 1.0), (20.0, 21.0, 2.0, 16.0, 3.0), (21.0, 23.0, 16.0, 2.0, 3.0)
    )
    neighbour = _ball((32.0, 33.0, 16.0, 2.0, 3.0)).assign(court_x=18.0)
    rallies = detect_rallies(pd.concat([rally, neighbour], ignore_index=True))

    assert len(rallies) == 1
    assert rallies.loc[0, "crossings"] == 2


FAR_SERVE_RALLY = [
    (0.0, 4.0, 19.5, 19.5, 1.0),  # le serveur fait rebondir le ballon derrière le fond opposé
    (4.0, 5.5, 19.5, 2.0, 3.0),  # service
    (5.5, 7.5, 2.0, 6.0, 3.0),  # réception, passe
    (7.5, 8.5, 6.0, 14.0, 3.0),  # attaque
    (8.5, 9.0, 14.0, 14.0, 0.5),  # ballon au sol
]


def test_serves_start_rallies_and_give_the_serving_side():
    ball = _ball(
        *FAR_SERVE_RALLY,
        (20.0, 21.5, 1.0, 15.0, 3.0),  # service côté caméra après une longue pause
        (21.5, 22.0, 15.0, 15.0, 0.5),
    )
    rallies = detect_rallies(ball)

    assert list(rallies["serving_side"]) == [FAR, NEAR]
    assert list(rallies["crossings"]) == [2, 1]
    assert rallies.loc[0, "server_hold_s"] > 1.0
    assert 2.5 < rallies.loc[0, "serve_s"] < 4.8


@pytest.mark.parametrize(
    "ball_return",
    [(12.0, 13.0, 4.0, 15.0, 0.3), (12.0, 13.0, 4.0, 15.0, 3.0)],
    ids=["roule-sous-le-filet", "lance-par-dessus"],
)
def test_balls_handed_back_to_the_server_are_not_rallies(ball_return):
    ball = _ball(
        (0.0, 4.0, 19.5, 19.5, 1.0),
        (4.0, 5.5, 19.5, 2.0, 3.0),  # service adverse, point direct
        (5.5, 6.0, 2.0, 2.0, 0.3),
        ball_return,
        (13.0, 17.0, 19.5, 19.5, 1.0),
        (17.0, 18.5, 19.5, 2.0, 3.0),  # nouveau service adverse
    )
    rallies = detect_rallies(ball)

    assert list(rallies["serving_side"]) == [FAR, FAR]
    assert list(rallies["crossings"]) == [1, 1]


def test_video_starting_mid_rally_does_not_count_a_serve():
    ball = _ball(
        (0.0, 1.5, 3.0, 14.0, 3.0),
        (1.5, 3.0, 14.0, 3.0, 3.0),
        (3.0, 3.5, 3.0, 3.0, 0.5),
        *[(t0 + 15, t1 + 15, y0, y1, h) for t0, t1, y0, y1, h in FAR_SERVE_RALLY],
    )
    rallies = detect_rallies(ball)

    assert list(rallies["serving_side"]) == [FAR]
    assert rallies.loc[0, "serve_s"] > 15


def test_quick_serve_needs_a_visible_server():
    ball = _ball(
        *FAR_SERVE_RALLY,
        (13.0, 14.5, 1.0, 15.0, 3.0),  # service côté caméra 4,5 s après l'attaque précédente
    )
    assert len(detect_rallies(ball)) == 1

    rallies = detect_rallies(ball, server_visible=lambda side, time_s: time_s > 12)
    assert list(rallies["serving_side"]) == [FAR, NEAR]


def test_manual_corrections_drop_and_add_serves():
    ball = _ball(
        *FAR_SERVE_RALLY,
        (20.0, 21.5, 1.0, 15.0, 3.0),
        (21.5, 40.0, 15.0, 15.0, 0.5),
    )
    rallies = detect_rallies(ball, ignored_s=[13.0], added_serves=[(30.0, FAR)])

    assert list(rallies["serving_side"]) == [NEAR, FAR]
    assert list(rallies["manual"]) == [False, True]
    assert list(rallies["rally"]) == [0, 1]
    assert rallies.loc[0, "end_s"] == 30.0
