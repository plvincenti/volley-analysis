"""Comptage des points à partir de la suite des services.

L'équipe qui gagne un échange sert le suivant : le vainqueur d'un échange est
donc l'équipe au service de l'échange d'après. Ce n'est plus vrai au
changement de set (le premier service ne dépend pas du dernier point) ; le
changement de camp, observé sur les maillots, signale ce cas et le dernier
point revient à l'équipe à qui il donne le set.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volley_analysis.court import FAR, NEAR
from volley_analysis.teams import CORBAS, OPPONENT

DECIDING_SET_SWITCH_POINTS = 8


@dataclass(frozen=True)
class SetRules:
    points: int = 25
    deciding_set_points: int = 15
    min_lead: int = 2
    sets_to_win: int = 3

    def target(self, set_number: int) -> int:
        deciding = set_number == 2 * self.sets_to_win - 1
        return self.deciding_set_points if deciding else self.points

    def is_won(self, points: int, opponent_points: int, set_number: int) -> bool:
        return points >= self.target(set_number) and points - opponent_points >= self.min_lead


STANDARD_RULES = SetRules()


def other(team: str) -> str:
    return OPPONENT if team == CORBAS else CORBAS


def flip(side: str) -> str:
    return FAR if side == NEAR else NEAR


def team_on(side: str, corbas_side: str) -> str:
    return CORBAS if side == corbas_side else OPPONENT


def score_rallies(
    rallies: pd.DataFrame,
    rules: SetRules = STANDARD_RULES,
    start_score: tuple[int, int] = (0, 0),
    start_set: int = 1,
    first_corbas_side: str = NEAR,
) -> pd.DataFrame:
    """Score échange par échange.

    `rallies` contient au moins `serving_side` et `corbas_side` (côté de Corbas
    observé, vide si les maillots n'ont pas permis de trancher), dans l'ordre
    du match. Les colonnes ajoutées décrivent le set, l'équipe au service, le
    vainqueur de l'échange et le score (Corbas, adversaire) après celui-ci.

    `winner_override`, facultatif, impose le vainqueur d'un échange (point saisi
    à la main) ; l'échange suivant est alors servi par cette équipe.

    `set_end`, facultatif, marque le dernier échange de chaque set. S'il est
    renseigné, les sets ne se terminent qu'à ces échanges, et un score final
    impossible (pas 25 points avec 2 d'écart) est signalé dans `note`.

    `referee_side`, facultatif, est le camp désigné par l'arbitre à la fin de
    l'échange : il ne change pas le vainqueur, mais un désaccord est signalé
    dans `note` (souvent un service manqué ou détecté à tort).
    """
    observed = [s if s in (NEAR, FAR) else None for s in rallies["corbas_side"]]
    serving_sides = list(rallies["serving_side"])
    referee_sides = [
        s if s in (NEAR, FAR) else None
        for s in rallies.get("referee_side", pd.Series([None] * len(rallies)))
    ]
    overrides = [
        t if t in (CORBAS, OPPONENT) else None
        for t in rallies.get("winner_override", pd.Series([None] * len(rallies)))
    ]
    set_ends = [bool(e) for e in rallies.get("set_end", pd.Series([False] * len(rallies)))]
    manual_sets = any(set_ends)
    forced_server: str | None = None
    points = {CORBAS: start_score[0], OPPONENT: start_score[1]}
    set_number = start_set
    corbas_side = next((s for s in observed if s), first_corbas_side)
    rows = []
    for i in range(len(serving_sides)):
        corbas_side = observed[i] or corbas_side
        notes = []
        if forced_server:
            side = corbas_side if forced_server == CORBAS else flip(corbas_side)
            if side != serving_sides[i]:
                notes.append("côté du service corrigé d'après le point précédent")
                serving_sides[i] = side
        row = {
            "set": set_number,
            "corbas_side": corbas_side,
            "serving_team": team_on(serving_sides[i], corbas_side),
            "winner": None,
            "note": "",
        }
        next_observed = observed[i + 1] if i + 1 < len(serving_sides) else None
        switched = next_observed is not None and next_observed != corbas_side
        set_end = set_ends[i] or (switched and not manual_sets)
        if overrides[i]:
            winner = overrides[i]
            notes.append("point saisi à la main")
        elif i + 1 == len(serving_sides):
            notes.append("dernier échange : vainqueur inconnu")
            winner = None
        elif set_end:
            winner = _set_point_winner(points, set_number, rules)
            if winner is None:
                winner = max(points, key=points.get)
                notes.append("fin de set sans balle de set : score douteux")
        else:
            winner = team_on(serving_sides[i + 1], next_observed or corbas_side)

        row["referee"] = team_on(referee_sides[i], row["corbas_side"]) if referee_sides[i] else None
        if row["referee"] and winner and row["referee"] != winner:
            notes.append(f"l'arbitre désigne {row['referee']}")

        if winner:
            points[winner] += 1
            row["winner"] = winner
        row["corbas"], row["adversaire"] = points[CORBAS], points[OPPONENT]

        set_won = winner is not None and rules.is_won(
            points[winner], points[other(winner)], set_number
        )
        if manual_sets:
            if set_ends[i]:
                notes.append(f"fin du set {set_number}")
                if not set_won:
                    score = f"{points[CORBAS]}-{points[OPPONENT]}"
                    notes.append(f"score de fin de set impossible : {score}")
                set_number += 1
                points = {CORBAS: 0, OPPONENT: 0}
                corbas_side = flip(corbas_side)
            elif switched:
                notes.append("changement de camp observé hors fin de set")
        elif set_won or switched:
            deciding_switch = (
                switched
                and not set_won
                and set_number == 2 * rules.sets_to_win - 1
                and max(points.values()) == DECIDING_SET_SWITCH_POINTS
            )
            if deciding_switch:
                notes.append("changement de camp du set décisif")
                corbas_side = flip(corbas_side)
            else:
                notes.append(f"fin du set {set_number}")
                if set_won and not switched and next_observed is not None:
                    notes.append("pas de changement de camp observé")
                set_number += 1
                points = {CORBAS: 0, OPPONENT: 0}
                corbas_side = flip(corbas_side)
        row["note"] = " ; ".join(notes)
        rows.append(row)
        forced_server = overrides[i]
    result = rallies.drop(columns=["corbas_side"]).reset_index(drop=True)
    result["serving_side"] = serving_sides
    return pd.concat([result, pd.DataFrame(rows)], axis=1)


def _set_point_winner(points: dict[str, int], set_number: int, rules: SetRules) -> str | None:
    """L'équipe à qui un point de plus donnerait le set, s'il y en a une seule."""
    winners = [
        team for team in points if rules.is_won(points[team] + 1, points[other(team)], set_number)
    ]
    return winners[0] if len(winners) == 1 else None


@dataclass
class ScoreState:
    rally: int | None
    set_number: int
    corbas: int
    opponent: int
    serving_team: str | None
    point_to: str | None
    referee: str | None = None


class ScoreTimeline:
    """Score affiché à chaque instant d'une vidéo, d'après les échanges comptés (`score_rallies`).

    Pendant un échange : score avant l'échange et équipe au service. À la fin de
    l'échange, le score passe au score après, et le gagnant du point est annoncé
    pendant `point_display_s`, avec l'équipe désignée par l'arbitre si elle est
    connue. Les numéros d'échange sont ceux de l'index de `scored`.
    """

    def __init__(self, scored: pd.DataFrame, point_display_s: float = 4.0):
        self.scored = scored.sort_values("serve_s")
        self.serves = self.scored["serve_s"].to_numpy()
        self.point_display_s = point_display_s

    def at(self, time_s: float) -> ScoreState | None:
        if len(self.scored) == 0:
            return None
        i = int(np.searchsorted(self.serves, time_s, side="right")) - 1
        rally = self.scored.iloc[max(i, 0)]
        winner = rally["winner"] if isinstance(rally["winner"], str) else None
        corbas_before = int(rally["corbas"]) - (winner == CORBAS)
        opponent_before = int(rally["adversaire"]) - (winner == OPPONENT)
        if i < 0:
            return ScoreState(None, int(rally["set"]), corbas_before, opponent_before, None, None)
        if time_s < rally["end_s"]:
            return ScoreState(
                int(self.scored.index[i]),
                int(rally["set"]),
                corbas_before,
                opponent_before,
                rally["serving_team"],
                None,
            )
        winner = winner or "?"
        recent = time_s < rally["end_s"] + self.point_display_s
        referee = rally.get("referee")
        return ScoreState(
            int(self.scored.index[i]),
            int(rally["set"]),
            int(rally["corbas"]),
            int(rally["adversaire"]),
            None,
            winner if recent else None,
            referee if recent and isinstance(referee, str) else None,
        )


def set_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Score final de chaque set (ou score courant du dernier set)."""
    last = scored.groupby("set").tail(1)
    return last[["set", "corbas", "adversaire"]].reset_index(drop=True)
