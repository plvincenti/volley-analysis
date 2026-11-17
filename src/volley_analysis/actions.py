"""Gestes et leur résultat, d'après les touches du ballon attribuées aux joueurs (`touches`).

Dans un échange, après le service, les touches successives d'une même équipe
forment une possession : la première est une réception, la dernière une attaque,
celles du milieu des passes. Une touche près du filet juste après une attaque
adverse est un contre ; la touche suivante de la même équipe est une réception.

Un geste est réussi si l'échange continue après lui, raté si c'est la dernière
touche de son équipe et qu'elle perd le point sans que l'adversaire ait attaqué
entre-temps. Un service, une attaque ou un contre est gagnant si son équipe
marque sans retoucher le ballon. Un service ou une attaque qui tombe sans être
touché donne à l'équipe adverse une réception ratée.
"""

import numpy as np
import pandas as pd

from volley_analysis.court import NET_Y
from volley_analysis.rallies import Crossing
from volley_analysis.scoring import other, team_on

SERVE = "service"
RECEPTION = "reception"
SET = "passe"
ATTACK = "attaque"
BLOCK = "contre"
ACTIONS = [SERVE, RECEPTION, SET, ATTACK, BLOCK]
SENDS = {SERVE, ATTACK, BLOCK}

SUCCESS = "reussi"
FAILED = "rate"
WINNING = "gagnant"
UNKNOWN = "inconnu"
RESULTS = [SUCCESS, FAILED, WINNING, UNKNOWN]

# Origine d'un geste : touche vue, service repéré, passage du filet sans touche vue,
# réception ratée d'un ballon tombé sans être touché, ou saisie à la main.
TOUCH, SERVED, CROSSING, UNTOUCHED, MANUAL = "touche", "service", "filet", "sans_touche", "manuel"

ACTION_COLUMNS = [
    "video",
    "rally",
    "set",
    "time_s",
    "team",
    "action",
    "result",
    "track_id",
    "origin",
]

# Le serveur ne peut retoucher le ballon qu'une fois le service passé et renvoyé.
SERVER_SILENCE_S = 1.5
BLOCK_MAX_DELAY_S = 0.8
BLOCK_MAX_NET_DISTANCE_M = 2.0
# Une réception ratée sans touche est datée un peu après l'envoi adverse (ballon au sol).
UNTOUCHED_DELAY_S = 1.0
# Le ballon envoyé passe le filet dans ce délai : au-delà, la touche d'envoi n'a pas été vue.
SEND_CROSSING_MAX_S = 1.5
HIT_BEFORE_CROSSING_S = 0.3
# Un renvoi adverse (réception, passe, attaque) ne repasse pas le filet plus tôt.
MIN_RETURN_S = 2.0
MIN_CROSSING_GAP_S = 1.0
# Pendant le jeu, le ballon est touché ou passe le filet au moins toutes les quelques
# secondes ; après un plus long silence, le point est fini.
MAX_GESTURE_GAP_S = 4.5
ACTION_DISPLAY_S = 2.0


def rally_gestures(
    rally: pd.Series, touches: pd.DataFrame, crossings: list[Crossing] = ()
) -> list[dict]:
    """Gestes des deux équipes dans un échange, dans l'ordre, sans leur résultat.

    `crossings` (`crossing_starts`) complète les touches : un passage du filet
    sans touche de l'équipe juste avant est une attaque qui n'a pas été vue.
    """
    serve = {
        "time_s": rally["serve_s"],
        "team": rally["serving_team"],
        "action": SERVE,
        "track_id": -1,
        "origin": SERVED,
        "unseen": False,
        "sent": True,
    }
    gestures = [serve]
    stream = []
    for touch in touches[touches["time_s"].between(rally["serve_s"], rally["end_s"])].itertuples():
        team = team_on(touch.side, rally["corbas_side"])
        if team == rally["serving_team"] and touch.time_s < rally["serve_s"] + SERVER_SILENCE_S:
            continue
        stream.append(
            {
                "time_s": touch.time_s,
                "team": team,
                "track_id": touch.track_id,
                "court_y": touch.court_y,
                "origin": TOUCH,
                "unseen": False,
                "sent": False,
            }
        )
    sends = _send_crossings(crossings, rally)
    stream = _add_unseen_sends(stream, sends, rally)

    possession: list[dict] = []
    for current in stream:
        if possession and (
            current["team"] != possession[-1]["team"]
            or current["unseen"]
            and any(
                possession[-1]["time_s"] < c.time_s < current["time_s"]
                and team_on(c.from_side, rally["corbas_side"]) == current["team"]
                for c in sends
            )
        ):
            gestures += _label_possession(possession)
            possession = []
        previous = gestures[-1]
        is_block = (
            not possession
            and current["team"] != previous["team"]
            and previous["action"] == ATTACK
            and current["time_s"] - previous["time_s"] <= BLOCK_MAX_DELAY_S
            and abs(current["court_y"] - NET_Y) <= BLOCK_MAX_NET_DISTANCE_M
        )
        if is_block:
            gestures.append({**current, "action": BLOCK})
        else:
            possession.append(current)
    gestures += _label_possession(possession)
    for i in range(1, len(gestures)):
        if gestures[i]["time_s"] - gestures[i - 1]["time_s"] > MAX_GESTURE_GAP_S:
            gestures = gestures[:i]
            break
    winner = rally.get("winner")
    return [_public(g) for g in _without_post_point(gestures, winner)]


def _send_crossings(crossings: list[Crossing], rally: pd.Series) -> list[Crossing]:
    """Passages du filet après le service, sans les retours trop rapides pour être un
    vrai renvoi (profondeur du ballon bruitée)."""
    sends, previous = [], None
    for c in crossings:
        if previous is None and team_on(c.from_side, rally["corbas_side"]) == rally["serving_team"]:
            previous = c
            continue
        if (
            previous
            and c.from_side != previous.from_side
            and c.time_s - previous.time_s < MIN_RETURN_S
        ):
            continue
        sends.append(c)
        previous = c
    return sends


def _add_unseen_sends(stream: list[dict], sends: list[Crossing], rally: pd.Series) -> list[dict]:
    added = []
    for c in sends:
        team = team_on(c.from_side, rally["corbas_side"])
        before = [t for t in stream if t["time_s"] <= c.time_s]
        after = [t for t in stream if c.time_s < t["time_s"] <= c.time_s + SEND_CROSSING_MAX_S]
        sent_by_touch = (
            bool(before)
            and before[-1]["team"] == team
            and c.time_s - before[-1]["time_s"] <= SEND_CROSSING_MAX_S
        )
        if sent_by_touch:
            before[-1]["sent"] = True
        elif not (after and after[0]["team"] == team):
            added.append(
                {
                    "time_s": c.time_s - HIT_BEFORE_CROSSING_S,
                    "team": team,
                    "track_id": -1,
                    "court_y": np.nan,
                    "origin": CROSSING,
                    "unseen": True,
                    "sent": True,
                }
            )
    return sorted(stream + added, key=lambda t: t["time_s"])


def _label_possession(touches: list[dict]) -> list[dict]:
    if not touches:
        return []
    labels = [RECEPTION] + [SET] * (len(touches) - 2) + [ATTACK] * (len(touches) > 1)
    if touches[-1]["unseen"]:
        labels[-1] = ATTACK
    return [{**touch, "action": label} for touch, label in zip(touches, labels, strict=True)]


def _without_post_point(gestures: list[dict], winner: str | None) -> list[dict]:
    """Retire la fin d'échange qui suit le point : ballon renvoyé au gagnant sans touche
    vue, ou rattrapé par le gagnant sans être renvoyé."""
    if not isinstance(winner, str):
        return gestures
    while len(gestures) > 2:
        last, previous = gestures[-1], gestures[-2]
        returned = last["team"] != winner and last["unseen"]
        caught = (
            last["team"] == winner
            and previous["team"] != winner
            and last["action"] == RECEPTION
            and not last["sent"]
        )
        if not (returned or caught):
            break
        gestures = gestures[:-1]
    return gestures


def _public(gesture: dict) -> dict:
    return {key: gesture[key] for key in ("time_s", "team", "action", "track_id", "origin")}


def crossing_starts(crossings: list[Crossing], start_s: float, end_s: float) -> list[Crossing]:
    """Passages du filet d'un échange, en ne gardant que le premier de chaque rafale (la
    profondeur du ballon oscille quand il reste près du filet)."""
    starts, previous_s = [], -np.inf
    for c in crossings:
        if start_s <= c.time_s <= end_s and not c.under_net:
            if c.time_s - previous_s >= MIN_CROSSING_GAP_S:
                starts.append(c)
            previous_s = c.time_s
    return starts


def gesture_results(gestures: list[dict], winner: str | None) -> list[dict]:
    """Résultat de chaque geste ; ajoute les réceptions ratées sans touche (`UNTOUCHED`)."""
    results = []
    for i, gesture in enumerate(gestures):
        team = gesture["team"]
        later = gestures[i + 1 :]
        team_again = any(g["team"] == team for g in later)
        opponent_attacked = any(g["team"] != team and g["action"] in (SERVE, ATTACK) for g in later)
        if team_again or (later and opponent_attacked):
            result = SUCCESS
        elif winner is None:
            result = UNKNOWN
        elif winner == team:
            result = WINNING if gesture["action"] in SENDS else SUCCESS
        else:
            result = FAILED
        results.append({**gesture, "result": result})
    last = gestures[-1]
    if winner == last["team"] and last["action"] in (SERVE, ATTACK):
        results.append(
            {
                "time_s": last["time_s"] + UNTOUCHED_DELAY_S,
                "team": other(winner),
                "action": RECEPTION,
                "track_id": -1,
                "origin": UNTOUCHED,
                "result": FAILED,
            }
        )
    return results


def team_actions(
    scored: pd.DataFrame,
    touches: dict[str, pd.DataFrame],
    team: str | None,
    crossings: dict[str, list[Crossing]] | None = None,
) -> pd.DataFrame:
    """Gestes de `team` (des deux équipes si None), un par ligne, avec leur résultat.

    `scored` vient de `scoring.score_rallies` ; `touches` donne, par vidéo, les
    touches attribuées aux joueurs (`touches.assign_players`), et `crossings` les
    passages du filet (`rallies.net_crossings`). Sans touches pour une vidéo,
    seuls les services sont listés, sans résultat.
    """
    crossings = crossings or {}
    rows = []
    for index, rally in scored.iterrows():
        winner = rally["winner"] if isinstance(rally["winner"], str) else None
        video_touches = touches.get(rally["video"])
        if video_touches is None:
            serve = {"time_s": rally["serve_s"], "team": rally["serving_team"], "action": SERVE}
            results = [{**serve, "track_id": -1, "origin": SERVED, "result": UNKNOWN}]
        else:
            kept = crossing_starts(
                crossings.get(rally["video"], []), rally["serve_s"], rally["end_s"]
            )
            results = gesture_results(rally_gestures(rally, video_touches, kept), winner)
        for gesture in results:
            if team is None or gesture["team"] == team:
                rows.append(
                    [
                        rally["video"],
                        index,
                        rally["set"],
                        gesture["time_s"],
                        gesture["team"],
                        gesture["action"],
                        gesture["result"],
                        gesture["track_id"],
                        gesture["origin"],
                    ]
                )
    return pd.DataFrame(rows, columns=ACTION_COLUMNS).sort_values(["rally", "time_s"])


class ActionTimeline:
    """Geste affiché à chaque instant d'une vidéo : le dernier, pendant `display_s`."""

    def __init__(self, actions: pd.DataFrame, display_s: float = ACTION_DISPLAY_S):
        self.actions = actions.sort_values("time_s").reset_index(drop=True)
        self.times = self.actions["time_s"].to_numpy()
        self.display_s = display_s

    def at(self, time_s: float) -> tuple[str, str] | None:
        """`(geste, résultat)` en cours, ou None."""
        i = int(np.searchsorted(self.times, time_s, side="right")) - 1
        if i < 0 or time_s >= self.times[i] + self.display_s:
            return None
        return self.actions.at[i, "action"], self.actions.at[i, "result"]


def action_label(action: str, result: str, team: str) -> str:
    """Libellé sans accents (police OpenCV) : « Attaque Corbas : gagnante »."""
    names = {
        SERVE: "Service",
        RECEPTION: "Reception",
        SET: "Passe",
        ATTACK: "Attaque",
        BLOCK: "Contre",
    }
    word = {SUCCESS: "reussi", FAILED: "rate", WINNING: "gagnant", UNKNOWN: "inconnu"}[result]
    if action in (RECEPTION, SET, ATTACK) and result != UNKNOWN:
        word += "e"
    return f"{names[action]} {team} : {word}"


def action_summary(actions: pd.DataFrame) -> pd.DataFrame:
    """Nombre de gestes par résultat, et part réussie (gagnants compris) parmi les connus."""
    counts = (
        pd.crosstab(actions["action"], actions["result"])
        .reindex(index=ACTIONS, columns=RESULTS, fill_value=0)
        .fillna(0)
        .astype(int)
    )
    known = counts[SUCCESS] + counts[FAILED] + counts[WINNING]
    counts.insert(0, "total", counts.sum(axis=1))
    counts["reussite_pct"] = (100 * (known - counts[FAILED]) / known.where(known > 0)).round()
    return counts
