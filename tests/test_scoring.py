import pandas as pd

from volley_analysis.court import FAR, NEAR
from volley_analysis.scoring import ScoreTimeline, SetRules, score_rallies, set_summary
from volley_analysis.teams import CORBAS, OPPONENT


def _rallies(serving_sides, corbas_sides=None):
    corbas_sides = corbas_sides or [NEAR] * len(serving_sides)
    return pd.DataFrame({"serving_side": serving_sides, "corbas_side": corbas_sides})


def test_the_winner_of_a_rally_serves_the_next_one():
    scored = score_rallies(_rallies([NEAR, NEAR, FAR, FAR, NEAR]))

    assert list(scored["serving_team"]) == [CORBAS, CORBAS, OPPONENT, OPPONENT, CORBAS]
    assert list(scored["winner"][:4]) == [CORBAS, OPPONENT, OPPONENT, CORBAS]
    assert pd.isna(scored["winner"].iloc[-1])
    assert list(zip(scored["corbas"], scored["adversaire"], strict=True)) == [
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 2),
    ]


def test_set_is_won_with_two_points_lead_then_teams_switch_sides():
    # 24-24 : Corbas marque deux fois, gagne le set 26-24 et passe de l'autre côté.
    scored = score_rallies(
        _rallies([NEAR, NEAR, NEAR, FAR], [NEAR, NEAR, None, None]), start_score=(24, 24)
    )

    assert list(scored["set"]) == [1, 1, 2, 2]
    assert (scored.loc[1, "corbas"], scored.loc[1, "adversaire"]) == (26, 24)
    assert "fin du set 1" in scored.loc[1, "note"]
    assert list(scored["corbas_side"]) == [NEAR, NEAR, FAR, FAR]
    # Set 2 : Corbas est au fond, le service côté caméra est adverse, et Corbas le gagne.
    assert scored.loc[2, "serving_team"] == OPPONENT
    assert (scored.loc[2, "corbas"], scored.loc[2, "adversaire"]) == (1, 0)
    assert list(set_summary(scored).itertuples(index=False, name=None)) == [(1, 26, 24), (2, 1, 0)]


def test_side_switch_gives_the_set_point_to_the_team_that_can_win_the_set():
    # 24-20 et changement de camp observé : le dernier point est celui du set pour Corbas,
    # même si c'est l'adversaire qui sert en premier au set suivant.
    scored = score_rallies(_rallies([FAR, NEAR], [NEAR, FAR]), start_score=(24, 20))

    assert scored.loc[0, "winner"] == CORBAS
    assert (scored.loc[0, "corbas"], scored.loc[0, "adversaire"]) == (25, 20)
    assert list(scored["set"]) == [1, 2]
    assert scored.loc[1, "serving_team"] == OPPONENT


def test_side_switch_without_set_point_is_flagged():
    scored = score_rallies(_rallies([NEAR, NEAR], [NEAR, FAR]), start_score=(10, 8))

    assert "score douteux" in scored.loc[0, "note"]
    assert scored.loc[1, "set"] == 2


def test_deciding_set_is_played_to_15():
    rules = SetRules()
    assert rules.target(5) == 15
    assert rules.is_won(15, 13, 5)
    assert not rules.is_won(15, 13, 4)


def test_timeline_shows_score_before_rally_then_point_at_rally_end():
    rallies = _rallies([NEAR, FAR, FAR])
    rallies["serve_s"] = [10.0, 30.0, 50.0]
    rallies["end_s"] = [20.0, 40.0, 60.0]
    timeline = ScoreTimeline(score_rallies(rallies), point_display_s=4.0)

    start = timeline.at(5.0)
    assert (start.rally, start.corbas, start.opponent, start.serving_team) == (None, 0, 0, None)
    serving = timeline.at(15.0)
    assert (serving.rally, serving.corbas, serving.serving_team) == (0, 0, CORBAS)
    point = timeline.at(21.0)
    assert (point.corbas, point.opponent, point.point_to) == (0, 1, OPPONENT)
    assert timeline.at(25.0).point_to is None
    last = timeline.at(61.0)
    assert (last.corbas, last.opponent, last.point_to) == (0, 2, "?")


def test_referee_disagreement_is_noted_without_changing_the_winner():
    rallies = _rallies([NEAR, FAR, FAR])
    rallies["referee_side"] = [NEAR, FAR, None]
    scored = score_rallies(rallies)

    assert list(scored["winner"][:2]) == [OPPONENT, OPPONENT]
    assert list(scored["referee"][:2]) == [CORBAS, OPPONENT]
    assert "arbitre" in scored.loc[0, "note"]
    assert scored.loc[1, "note"] == ""


def test_manual_point_sets_the_winner_and_the_next_server():
    rallies = _rallies([FAR, FAR, FAR])
    rallies["winner_override"] = [CORBAS, None, None]
    scored = score_rallies(rallies)

    assert list(scored["winner"][:2]) == [CORBAS, OPPONENT]
    assert list(scored["serving_team"]) == [OPPONENT, CORBAS, OPPONENT]
    assert scored.loc[1, "serving_side"] == NEAR
    assert list(zip(scored["corbas"], scored["adversaire"], strict=True))[:2] == [(1, 0), (1, 1)]


def test_manual_set_ends_replace_automatic_ones():
    # 4 échanges : Corbas sert, perd, l'adversaire sert deux fois ; fin du set saisie au 3e.
    rallies = _rallies([NEAR, FAR, FAR, FAR], [NEAR, NEAR, NEAR, FAR])
    rallies["set_end"] = [False, False, True, False]
    scored = score_rallies(rallies, SetRules(points=2, min_lead=1))

    assert list(scored["set"]) == [1, 1, 1, 2]
    assert scored.loc[2, "winner"] == OPPONENT
    assert "fin du set 1" in scored.loc[2, "note"]
    assert "impossible" not in scored.loc[2, "note"]
    assert (scored.loc[2, "corbas"], scored.loc[2, "adversaire"]) == (0, 3)


def test_impossible_manual_set_end_score_is_reported():
    rallies = _rallies([NEAR, FAR, NEAR])
    rallies["set_end"] = [False, True, False]
    scored = score_rallies(rallies)

    assert "score de fin de set impossible : 0-2" in scored.loc[1, "note"]
    assert list(scored["set"]) == [1, 1, 2]
