"""Touches du ballon, repérées sur sa trajectoire dans l'image, attribuées au joueur le plus proche.

Entre deux touches, seule la gravité agit sur le ballon : dans l'image, il peut
monter puis redescendre, mais pas repartir vers le haut, ni changer de sens
horizontalement, ni accélérer brusquement. Chacun de ces changements est une
touche (ou un rebond au sol). Les positions de l'image sont fiables, contrairement
à la profondeur déduite du rayon du ballon.
"""

import numpy as np
import pandas as pd

from volley_analysis.rallies import Crossing

TOUCH_COLUMNS = ["time_s", "frame", "x", "y", "kind"]

# Vitesses estimées sur `TOUCH_WINDOW` détections de part et d'autre (30 img/s).
TOUCH_WINDOW = 2
MAX_DETECTION_GAP_S = 0.12
MIN_VERTICAL_SPEED_PX_S = 60
MIN_HORIZONTAL_SPEED_PX_S = 120
KICK_SPEED_RATIO = 1.8
MIN_KICK_SPEED_PX_S = 400
KICK_MAX_COS = 0.8
# Deux candidats plus proches sont la même touche.
MIN_TOUCH_GAP_S = 0.25
# Distance ballon-joueur en tailles de boîte ; la zone monte au-dessus de la tête (bras levés).
MAX_PLAYER_DISTANCE = 0.8
REACH_ABOVE_HEAD = 0.6


def find_touches(ball: pd.DataFrame) -> pd.DataFrame:
    """Touches candidates : `kind` vaut `remontee`, `demi_tour` ou `frappe`."""
    detected = ball[ball["source"] == "detected"].sort_values("frame")
    t = detected["time_s"].to_numpy()
    x = detected["x"].to_numpy()
    y = detected["y"].to_numpy()
    frames = detected["frame"].to_numpy()
    k = TOUCH_WINDOW
    rows: list[list] = []
    for i in range(k, len(t) - k):
        if np.any(np.diff(t[i - k : i + k + 1]) > MAX_DETECTION_GAP_S):
            continue
        v_in = np.array([x[i] - x[i - k], y[i] - y[i - k]]) / (t[i] - t[i - k])
        v_out = np.array([x[i + k] - x[i], y[i + k] - y[i]]) / (t[i + k] - t[i])
        kind = _change_kind(v_in, v_out)
        if kind is None:
            continue
        if rows and t[i] - rows[-1][0] < MIN_TOUCH_GAP_S:
            continue
        rows.append([float(t[i]), int(frames[i]), float(x[i]), float(y[i]), kind])
    return pd.DataFrame(rows, columns=TOUCH_COLUMNS)


def _change_kind(v_in: np.ndarray, v_out: np.ndarray) -> str | None:
    if v_in[1] > MIN_VERTICAL_SPEED_PX_S and v_out[1] < -MIN_VERTICAL_SPEED_PX_S:
        return "remontee"
    if v_in[0] * v_out[0] < 0 and min(abs(v_in[0]), abs(v_out[0])) > MIN_HORIZONTAL_SPEED_PX_S:
        return "demi_tour"
    speed_in, speed_out = np.hypot(*v_in), np.hypot(*v_out)
    cos = v_in @ v_out / (speed_in * speed_out + 1e-9)
    if speed_out > KICK_SPEED_RATIO * speed_in and speed_out > MIN_KICK_SPEED_PX_S:
        if cos < KICK_MAX_COS:
            return "frappe"
    return None


def assign_players(
    touches: pd.DataFrame, players: pd.DataFrame, crossings: list[Crossing] = ()
) -> pd.DataFrame:
    """Ajoute le joueur le plus proche de chaque touche (`track_id`, `side`, `court_y` de
    ses pieds) ; les touches sans joueur assez proche sont retirées.

    Près du filet, les joueurs des deux camps se superposent dans l'image : à égalité,
    le joueur du camp où se trouve le ballon d'après `crossings`
    (`rallies.net_crossings`) l'emporte, puis celui qui l'atteint sans lever les bras.
    """
    players = players[players["is_player"]]
    over = [c for c in crossings if not c.under_net]
    crossing_s = np.array([c.time_s for c in over])
    by_frame = players.groupby("frame").indices
    xyxy = players[["x1", "y1", "x2", "y2"]].to_numpy()
    columns = ["track_id", "side", "court_y"]
    info = players[columns].to_numpy()
    rows = []
    for touch in touches.itertuples():
        candidates = by_frame.get(touch.frame)
        if candidates is None:
            continue
        x1, y1, x2, y2 = xyxy[candidates].T
        width, height = x2 - x1, y2 - y1
        dx = np.maximum(0, np.maximum(x1 - touch.x, touch.x - x2)) / width
        dy = np.maximum(0, np.maximum(y1 - REACH_ABOVE_HEAD * height - touch.y, touch.y - y2))
        distance = np.hypot(dx, dy / height)
        last = np.searchsorted(crossing_s, touch.time_s) - 1
        ball_side = over[last].to_side if last >= 0 else None
        other_side = (info[candidates, 1] != ball_side) if ball_side else np.zeros(len(candidates))
        above_head = np.maximum(0, y1 - touch.y) / height
        best = int(np.lexsort((above_head, other_side, distance))[0])
        if distance[best] <= MAX_PLAYER_DISTANCE:
            rows.append([*touch[1:], *info[candidates[best]], float(distance[best])])
    return pd.DataFrame(rows, columns=[*TOUCH_COLUMNS, *columns, "distance"])
