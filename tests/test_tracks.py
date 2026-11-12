import numpy as np
import pandas as pd

from volley_analysis.court import FAR, NEAR
from volley_analysis.tracks import clean_tracks

FPS = 30


def _track(track_id, xs, ys):
    n = len(xs)
    return pd.DataFrame(
        {
            "frame": np.arange(n),
            "track_id": track_id,
            "court_x": xs,
            "court_y": ys,
            "in_play": True,
        }
    )


def test_jump_at_the_net_does_not_switch_side():
    ys = np.full(120, 8.0)
    ys[50:70] = 9.6  # pieds décollés pendant un contre : projetés de l'autre côté du filet
    df = clean_tracks(_track(1, np.full(120, 4.5), ys), FPS)
    assert (df["side"] == NEAR).all()


def test_static_person_outside_the_lines_is_not_a_player():
    rng = np.random.default_rng(0)
    spectator = _track(1, 9.7 + rng.normal(0, 0.05, 300), 19.4 + rng.normal(0, 0.05, 300))
    server = _track(2, np.full(300, 6.0), np.r_[np.full(60, 19.5), np.linspace(19.5, 14, 240)])
    df = clean_tracks(pd.concat([spectator, server]), FPS)

    assert not df.loc[df["track_id"] == 1, "is_player"].any()
    assert df.loc[df["track_id"] == 2, "is_player"].all()
    assert (df.loc[df["track_id"] == 2, "side"] == FAR).all()


def test_person_walking_along_the_sideline_is_not_a_player():
    walker = _track(1, np.full(300, 9.6), np.linspace(0.5, 7.5, 300))
    df = clean_tracks(walker, FPS)
    assert not df["is_player"].any()


def test_at_most_six_players_per_side():
    players = [_track(i, np.full(100, 1.0 + i), np.full(100, 4.0)) for i in range(6)]
    # 7e personne côté caméra, qui ne passe qu'une partie de son temps dans le terrain
    intruder = _track(9, np.full(100, 4.5), np.r_[np.full(50, -1.5), np.full(50, 2.0)])
    df = clean_tracks(pd.concat(players + [intruder]), FPS)

    counts = df[df["is_player"]].groupby("frame").size()
    assert counts.max() == 6
    assert not df.loc[(df["track_id"] == 9) & (df["frame"] >= 50), "is_player"].any()
