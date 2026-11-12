"""Découpage d'une vidéo en échanges à partir de la trajectoire 3D du ballon.

Le côté du filet où se trouve le ballon se lit sur sa profondeur `court_y`,
reconstruite à partir de son rayon apparent (`ball_tracks.ball_positions_3d`).

Un échange commence par un service : un passage du filet par-dessus, après une
pause sans jeu (le temps de récupérer le ballon et de se replacer), ou juste
après que le serveur a fait rebondir le ballon derrière sa ligne de fond. Le
côté d'où part ce passage est celui de l'équipe au service. Les ballons roulés
ou lancés sous le filet pour le rendre au serveur ne sont pas du jeu.
"""

from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from volley_analysis.court import COURT_LENGTH, FAR, NEAR, NET_Y

# La profondeur est d'autant plus imprécise que le ballon est loin : marge plus large côté opposé.
NEAR_MAX_Y_M = NET_Y - 1.0
FAR_MIN_Y_M = NET_Y + 1.5
# Notre poteau de droite est à ~10 m du bord gauche, celui du terrain voisin à ~14 m.
MAX_COURT_X_M = 12.0
DEPTH_SMOOTHING_DETECTIONS = 3
SIDE_CONFIRM_DETECTIONS = 3
MAX_UNSEEN_S = 3.0
# Un ballon roulé sous le filet reste au ras du sol avant et après le passage.
UNDER_NET_MAX_HEIGHT_M = 1.0
CROSSING_WINDOW_S = 0.5

SERVE_PAUSE_S = 6.5
SERVE_AFTER_HOLD_PAUSE_S = 3.0
BALL_RETURN_MAX_S = 8.0
HOLD_WINDOW_S = 5.0
HOLD_MIN_S = 0.5
HOLD_MAX_HEIGHT_M = 3.0
HOLD_BASELINE_MARGIN_M = 0.3
SERVE_LOOKBACK_S = 2.0
TRACK_GAP_S = 0.5

RALLY_COLUMNS = [
    "rally",
    "serve_s",
    "end_s",
    "serving_side",
    "crossings",
    "pause_s",
    "server_hold_s",
    "manual",
]


@dataclass
class Crossing:
    time_s: float
    from_side: str
    to_side: str
    max_height_m: float

    @property
    def under_net(self) -> bool:
        return self.max_height_m < UNDER_NET_MAX_HEIGHT_M


def ball_sides(ball: pd.DataFrame) -> pd.DataFrame:
    """Détections du ballon avec leur côté (`near`, `far`, ou vide près du filet).

    Les ballons au-delà de `MAX_COURT_X_M` sont écartés : ils sont sur le terrain
    voisin, visible en haut à droite de l'image, dont le filet prolonge le nôtre.
    """
    detected = ball[(ball["source"] == "detected") & (ball["court_x"] <= MAX_COURT_X_M)]
    detected = detected.sort_values("time_s")
    depth = (
        detected["court_y"]
        .rolling(DEPTH_SMOOTHING_DETECTIONS, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    side = np.where(depth <= NEAR_MAX_Y_M, NEAR, np.where(depth >= FAR_MIN_Y_M, FAR, ""))
    return pd.DataFrame(
        {
            "time_s": detected["time_s"].to_numpy(),
            "depth": depth,
            "height": detected["height"].to_numpy(),
            "side": side,
        }
    )


def net_crossings(ball: pd.DataFrame) -> list[Crossing]:
    """Passages du filet, à partir des détections (pas des positions interpolées).

    Le côté change quand `SIDE_CONFIRM_DETECTIONS` détections d'affilée sont
    franchement de l'autre côté. Après `MAX_UNSEEN_S` sans ballon, le côté est
    oublié : on ne sait pas comment le ballon a circulé entre-temps.
    """
    sides = ball_sides(ball)
    times = sides["time_s"].to_numpy()
    labels = sides["side"].to_numpy()

    crossings: list[Crossing] = []
    side: str | None = None
    last_on_side = 0
    candidate: str | None = None
    candidate_start = 0
    candidate_count = 0
    for i, observed in enumerate(labels):
        if i > 0 and times[i] - times[i - 1] > MAX_UNSEEN_S:
            side, candidate, candidate_count = None, None, 0
        if not observed:
            continue
        if observed == side:
            last_on_side = i
            candidate, candidate_count = None, 0
            continue
        if observed != candidate:
            candidate, candidate_start, candidate_count = observed, i, 0
        candidate_count += 1
        if candidate_count < SIDE_CONFIRM_DETECTIONS:
            continue
        if side is not None:
            crossings.append(_crossing(sides, last_on_side, candidate_start, side, observed))
        side, last_on_side = observed, i
        candidate, candidate_count = None, 0
    return crossings


def _crossing(sides: pd.DataFrame, left: int, right: int, from_side: str, to_side: str) -> Crossing:
    """Passage entre la dernière détection d'un côté (`left`) et la 1re de l'autre (`right`)."""
    times = sides["time_s"].to_numpy()
    window = (times >= times[left] - CROSSING_WINDOW_S) & (
        times <= times[right] + CROSSING_WINDOW_S
    )
    max_height = float(sides["height"].to_numpy()[window].max())
    return Crossing((times[left] + times[right]) / 2, from_side, to_side, max_height)


def server_hold_s(sides: pd.DataFrame, side: str, before_s: float) -> float:
    """Durée pendant laquelle le ballon est resté bas derrière la ligne de fond de `side`.

    C'est le serveur qui fait rebondir le ballon avant de servir. Côté caméra, il
    est souvent hors champ : cet indice sert surtout pour le côté opposé.
    """
    window = sides[(sides["time_s"] >= before_s - HOLD_WINDOW_S) & (sides["time_s"] < before_s)]
    if side == FAR:
        behind = window["depth"] >= COURT_LENGTH + HOLD_BASELINE_MARGIN_M
    else:
        behind = window["depth"] <= -HOLD_BASELINE_MARGIN_M
    held = window[behind & (window["height"] <= HOLD_MAX_HEIGHT_M)]
    if len(held) == 0:
        return 0.0
    period = float(np.median(np.diff(sides["time_s"].to_numpy()))) if len(sides) > 1 else 0.0
    return len(held) * period


def detect_rallies(
    ball: pd.DataFrame,
    server_visible: Callable[[str, float], bool] | None = None,
    ignored_s: Sequence[float] = (),
    added_serves: Sequence[tuple[float, str]] = (),
    ignored_periods: Sequence[tuple[float, float]] = (),
) -> pd.DataFrame:
    """Échanges de la vidéo, un par service détecté.

    `ball` doit contenir la position 3D (`ball_tracks.ball_positions_3d`).
    `pause_s` est le temps sans jeu avant le service, compté depuis le début de
    la vidéo pour le premier : une vidéo qui démarre en plein échange ne doit
    pas faire croire à un service.

    `server_visible(side, time_s)`, si fourni, dit si un joueur est en position
    de service de ce côté juste avant `time_s`. Il départage les passages qui
    suivent une pause courte sans rebonds visibles du ballon (service enchaîné
    vite après l'échange précédent).

    Corrections manuelles : chaque instant de `ignored_s` retire l'échange en
    cours à ce moment (dernier service avant lui) ; `ignored_periods` retire tous
    les services de chaque période `(début, fin)` ; `added_serves` ajoute des
    services `(instant, côté du serveur)` que la détection a manqués.
    """
    sides = ball_sides(ball)
    detected_times = sides["time_s"].to_numpy()
    rallies: list[dict] = []
    last_play_s = float(ball["time_s"].min())
    for crossing in net_crossings(ball):
        if crossing.under_net:
            continue
        pause = crossing.time_s - last_play_s
        last_play_s = crossing.time_s
        serve_s = _flight_start(detected_times, crossing.time_s)
        hold = server_hold_s(sides, crossing.from_side, crossing.time_s)
        is_serve = pause >= SERVE_PAUSE_S or (
            pause >= SERVE_AFTER_HOLD_PAUSE_S
            and (
                hold >= HOLD_MIN_S
                or (server_visible is not None and server_visible(crossing.from_side, serve_s))
            )
        )
        if is_serve:
            rallies.append(
                {
                    "serve_s": serve_s,
                    "serving_side": crossing.from_side,
                    "crossings": 0,
                    "pause_s": pause,
                    "server_hold_s": hold,
                    "manual": False,
                }
            )
        if rallies:
            rallies[-1]["crossings"] += 1
            rallies[-1]["last_crossing_s"] = crossing.time_s

    rallies = _apply_corrections(
        _drop_ball_returns(rallies), ignored_s, added_serves, ignored_periods
    )
    for index, (rally, following) in enumerate(zip(rallies, [*rallies[1:], None], strict=False)):
        rally["rally"] = index
        end = _flight_end(detected_times, rally.pop("last_crossing_s"))
        rally["end_s"] = min(end, following["serve_s"]) if following else end
    return pd.DataFrame(rallies, columns=RALLY_COLUMNS)


def _drop_ball_returns(rallies: list[dict]) -> list[dict]:
    """Retire les ballons renvoyés par-dessus le filet au serveur adverse.

    Un tel renvoi ressemble à un service sans retour, mais il est suivi de peu
    par le vrai service, depuis le côté où le ballon est arrivé. On remonte
    depuis la fin pour comparer chaque candidat au service suivant retenu : un
    service gagnant suivi du renvoi du ballon au serveur reste un échange.
    """
    kept: list[dict] = []
    for rally in reversed(rallies):
        following = kept[-1] if kept else None
        is_return = (
            following is not None
            and rally["crossings"] == 1
            and following["serving_side"] != rally["serving_side"]
            and following["serve_s"] - rally["last_crossing_s"] <= BALL_RETURN_MAX_S
        )
        if not is_return:
            kept.append(rally)
    return kept[::-1]


def _apply_corrections(
    rallies: list[dict],
    ignored_s: Sequence[float],
    added_serves: Sequence[tuple[float, str]],
    ignored_periods: Sequence[tuple[float, float]],
) -> list[dict]:
    serves = [rally["serve_s"] for rally in rallies]
    dropped = {bisect_right(serves, time_s) - 1 for time_s in ignored_s} - {-1}
    dropped |= {
        i
        for i, serve_s in enumerate(serves)
        for start_s, end_s in ignored_periods
        if start_s <= serve_s <= end_s
    }
    kept = [rally for i, rally in enumerate(rallies) if i not in dropped]
    added = [
        {
            "serve_s": time_s,
            "serving_side": side,
            "crossings": 0,
            "pause_s": np.nan,
            "server_hold_s": np.nan,
            "manual": True,
            "last_crossing_s": time_s,
        }
        for time_s, side in added_serves
    ]
    return sorted(kept + added, key=lambda rally: rally["serve_s"])


def _flight_start(times: np.ndarray, crossing_s: float) -> float:
    """Début de la trajectoire continue qui mène au passage du filet (la frappe de service)."""
    i = np.searchsorted(times, crossing_s) - 1
    if i < 0:
        return crossing_s
    while (
        i > 0
        and times[i] - times[i - 1] <= TRACK_GAP_S
        and crossing_s - times[i - 1] <= SERVE_LOOKBACK_S
    ):
        i -= 1
    return float(times[i])


def _flight_end(times: np.ndarray, crossing_s: float) -> float:
    """Dernière détection avant que le ballon disparaisse plus de `MAX_UNSEEN_S`."""
    i = np.searchsorted(times, crossing_s)
    if i >= len(times):
        return crossing_s
    while i + 1 < len(times) and times[i + 1] - times[i] <= MAX_UNSEEN_S:
        i += 1
    return float(times[i])
