import numpy as np
import pandas as pd

from volley_analysis.rallies import Crossing
from volley_analysis.touches import assign_players, find_touches

RATE = 30


def _ball(xs, ys):
    t = np.arange(len(xs)) / RATE
    return pd.DataFrame(
        {"frame": np.arange(len(xs)), "time_s": t, "x": xs, "y": ys, "source": "detected"}
    )


def test_ball_going_back_up_was_touched():
    ys = np.concatenate([np.linspace(100, 400, 15), np.linspace(400, 100, 15)])
    touches = find_touches(_ball(np.full(30, 500.0), ys))

    assert touches["kind"].tolist() == ["remontee"]
    assert abs(touches.loc[0, "frame"] - 14) <= 1


def test_top_of_a_lob_is_not_a_touch():
    t = np.arange(40) / RATE
    ys = 400 - 900 * t + 700 * t**2
    assert find_touches(_ball(500 + 200 * t, ys)).empty


def test_sideways_turn_is_a_touch():
    xs = np.concatenate([np.linspace(800, 500, 15), np.linspace(500, 800, 15)])
    touches = find_touches(_ball(xs, np.full(30, 300.0)))

    assert touches["kind"].tolist() == ["demi_tour"]


def test_touch_goes_to_the_nearest_player_within_reach():
    touches = pd.DataFrame(
        {"time_s": [0.0, 1.0], "frame": [0, 30], "x": [520.0, 1500.0], "y": [250.0, 250.0]}
    ).assign(kind="remontee")
    players = pd.DataFrame(
        {
            "frame": [0, 0, 30],
            "x1": [480.0, 900.0, 480.0],
            "y1": [300.0, 300.0, 300.0],
            "x2": [560.0, 980.0, 560.0],
            "y2": [500.0, 500.0, 500.0],
            "track_id": [7, 8, 7],
            "side": ["near", "far", "near"],
            "court_y": [3.0, 12.0, 3.0],
            "is_player": True,
        }
    )
    assigned = assign_players(touches, players)

    assert assigned["track_id"].tolist() == [7]
    assert assigned.loc[0, "side"] == "near"


def _overlapping_players():
    """Un joueur proche, bras levés vers le ballon, devant un joueur du fond qui le touche."""
    return pd.DataFrame(
        {
            "frame": [0, 0],
            "x1": [480.0, 500.0],
            "y1": [300.0, 240.0],
            "x2": [560.0, 540.0],
            "y2": [500.0, 320.0],
            "track_id": [7, 8],
            "side": ["near", "far"],
            "court_y": [8.0, 11.0],
            "is_player": True,
        }
    )


def test_touch_in_reach_of_two_players_goes_to_the_one_on_the_ball_side():
    touch = pd.DataFrame({"time_s": [2.0], "frame": [0], "x": [520.0], "y": [250.0]})
    touch = touch.assign(kind="remontee")
    players = _overlapping_players()

    assert assign_players(touch, players)["track_id"].tolist() == [8]
    to_near = [Crossing(1.0, "far", "near", 3.0)]
    assert assign_players(touch, players, to_near)["track_id"].tolist() == [7]
