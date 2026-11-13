import pandas as pd

from volley_analysis.actions import (
    ATTACK,
    BLOCK,
    FAILED,
    RECEPTION,
    SERVE,
    SET,
    SUCCESS,
    WINNING,
    ActionTimeline,
    action_label,
    action_summary,
    team_actions,
)
from volley_analysis.court import FAR, NEAR
from volley_analysis.rallies import Crossing
from volley_analysis.teams import CORBAS, OPPONENT


def _rally(serving_team, winner):
    return {
        "video": "v",
        "serve_s": 0.0,
        "end_s": 20.0,
        "set": 1,
        "corbas_side": NEAR,
        "serving_team": serving_team,
        "winner": winner,
    }


def _touches(*touches):
    """`(instant, côté du joueur)` ou `(instant, côté, profondeur de ses pieds)`."""
    rows = [(t[0], t[1], t[2] if len(t) > 2 else 4.0, 1) for t in touches]
    return pd.DataFrame(rows, columns=["time_s", "side", "court_y", "track_id"])


def _pairs(rally, touches, crossings=()):
    actions = team_actions(pd.DataFrame([rally]), {"v": touches}, CORBAS, {"v": list(crossings)})
    return list(zip(actions["action"], actions["result"], strict=True))


def test_possession_is_reception_sets_and_attack():
    touches = _touches((2, FAR), (3, FAR), (4, FAR), (6, NEAR), (7, NEAR), (8, NEAR))

    assert _pairs(_rally(CORBAS, CORBAS), touches) == [
        (SERVE, SUCCESS),
        (RECEPTION, SUCCESS),
        (SET, SUCCESS),
        (ATTACK, WINNING),
    ]


def test_server_touching_the_ball_is_the_serve():
    touches = _touches((0.5, NEAR), (2, FAR), (3, FAR))

    assert _pairs(_rally(CORBAS, CORBAS), touches) == [(SERVE, SUCCESS)]


def test_quick_touch_at_the_net_after_an_attack_is_a_block():
    touches = _touches((2, NEAR), (3, NEAR), (5, FAR), (6, FAR), (6.5, NEAR, 8.5))

    assert _pairs(_rally(OPPONENT, CORBAS), touches) == [
        (RECEPTION, SUCCESS),
        (ATTACK, SUCCESS),
        (BLOCK, WINNING),
    ]


def test_last_touch_fails_with_the_point():
    touches = _touches((2, NEAR), (3, NEAR), (4, NEAR))

    assert _pairs(_rally(OPPONENT, OPPONENT), touches)[-1] == (ATTACK, FAILED)


def test_untouched_serve_or_attack_is_a_failed_reception():
    assert _pairs(_rally(OPPONENT, OPPONENT), _touches()) == [(RECEPTION, FAILED)]
    assert _pairs(_rally(CORBAS, CORBAS), _touches()) == [(SERVE, WINNING)]
    touches = _touches((2, NEAR), (3, NEAR), (5, FAR), (6, FAR))
    assert _pairs(_rally(OPPONENT, OPPONENT), touches) == [
        (RECEPTION, SUCCESS),
        (ATTACK, SUCCESS),
        (RECEPTION, FAILED),
    ]


def test_ball_sent_back_over_the_net_means_the_rally_went_on():
    touches = _touches((2, NEAR), (3, NEAR), (8, NEAR))
    crossings = [Crossing(3.5, NEAR, FAR, 3.0), Crossing(7.0, FAR, NEAR, 3.0)]

    assert _pairs(_rally(OPPONENT, OPPONENT), touches, crossings)[-2:] == [
        (ATTACK, SUCCESS),
        (RECEPTION, FAILED),
    ]


def test_ball_handed_back_after_the_point_is_not_a_gesture():
    touches = _touches((2, NEAR), (3, NEAR), (9, NEAR))
    crossings = [Crossing(3.5, NEAR, FAR, 3.0), Crossing(7.0, FAR, NEAR, 3.0)]

    assert _pairs(_rally(OPPONENT, CORBAS), touches, crossings) == [
        (RECEPTION, SUCCESS),
        (ATTACK, WINNING),
    ]


def test_ball_crossing_without_a_seen_touch_is_an_unseen_attack():
    touches = _touches((2, NEAR))
    crossings = [
        Crossing(0.8, FAR, NEAR, 3.0),
        Crossing(5.0, NEAR, FAR, 3.0),
        Crossing(6.5, FAR, NEAR, 3.0),
    ]

    assert _pairs(_rally(OPPONENT, CORBAS), touches, crossings) == [
        (RECEPTION, SUCCESS),
        (ATTACK, WINNING),
    ]


def test_timeline_shows_the_last_gesture_for_a_while():
    actions = pd.DataFrame(
        {"time_s": [1.0, 2.0], "action": [SERVE, ATTACK], "result": [SUCCESS, WINNING]}
    )
    timeline = ActionTimeline(actions, display_s=2.0)

    assert timeline.at(0.5) is None
    assert timeline.at(1.5) == (SERVE, SUCCESS)
    assert timeline.at(3.0) == (ATTACK, WINNING)
    assert timeline.at(4.1) is None
    assert action_label(ATTACK, WINNING, CORBAS) == "Attaque Corbas : gagnante"
    assert action_label(BLOCK, FAILED, CORBAS) == "Contre Corbas : rate"


def test_summary_counts_winning_gestures_as_successful():
    actions = pd.DataFrame({"action": [SERVE, SERVE, SERVE], "result": [SUCCESS, WINNING, FAILED]})
    summary = action_summary(actions)

    assert summary.loc[SERVE, "total"] == 3
    assert summary.loc[SERVE, "reussite_pct"] == 67
    assert summary.loc[BLOCK, "total"] == 0


def test_touches_after_a_long_silence_follow_the_point():
    touches = _touches((2, NEAR), (3, NEAR), (9, FAR), (10, FAR))

    assert _pairs(_rally(OPPONENT, OPPONENT), touches) == [
        (RECEPTION, SUCCESS),
        (ATTACK, FAILED),
    ]
