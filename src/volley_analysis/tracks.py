"""Nettoyage des pistes de joueurs sur toute la vidéo, après le suivi image par image."""

import numpy as np
import pandas as pd

from volley_analysis.court import COURT_LENGTH, COURT_WIDTH, side_of_net

SIDE_WINDOW_S = 2.0
BYSTANDER_MIN_OUTSIDE_FRACTION = 0.9
BYSTANDER_MAX_SPREAD_M = 1.0
SIDELINE_TOLERANCE_M = 0.5
MAX_BEYOND_SIDELINES_FRACTION = 0.25
MAX_PLAYERS_PER_SIDE = 6


def clean_tracks(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Ajoute le côté du filet lissé (`side`) et le statut de joueur (`is_player`).

    - Un joueur ne change pas de côté pendant un set, mais ses pieds décollent quand il
      saute au filet et la projection le fait alors passer de l'autre côté : le côté est
      la médiane glissante de la profondeur sur `SIDE_WINDOW_S`.
    - Les spectateurs, remplaçants, arbitre et marqueur restent hors du terrain :
      quasi immobiles hors des lignes, ou souvent au-delà d'une ligne de côté (le serveur,
      lui, sort par la ligne de fond).
    - Il y a au plus 6 joueurs par côté : au-delà, on garde les pistes qui passent le plus
      de temps dans le terrain.
    """
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    if df.empty:
        return df.assign(side=pd.Series(dtype=str), is_player=pd.Series(dtype=bool))
    window = max(int(round(SIDE_WINDOW_S * fps)), 1)
    smoothed_y = df.groupby("track_id")["court_y"].transform(
        lambda y: y.rolling(window, center=True, min_periods=1).median()
    )
    df["side"] = side_of_net(smoothed_y.to_numpy())

    stats = track_stats(df[df["in_play"]])
    non_players = stats.index[
        (
            (stats["outside"] >= BYSTANDER_MIN_OUTSIDE_FRACTION)
            & (stats["spread"] < BYSTANDER_MAX_SPREAD_M)
        )
        | (stats["beyond_sidelines"] > MAX_BEYOND_SIDELINES_FRACTION)
    ]
    df["is_player"] = df["in_play"] & ~df["track_id"].isin(non_players)
    df["is_player"] &= _within_side_capacity(df, 1 - stats["outside"])
    return df.sort_values(["frame", "track_id"]).reset_index(drop=True)


def track_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Par piste : part du temps hors des lignes, au-delà des côtés, et dispersion (m)."""
    x, y = df["court_x"], df["court_y"]
    flags = df.assign(
        outside=~(x.between(0, COURT_WIDTH) & y.between(0, COURT_LENGTH)),
        beyond_sidelines=(x < -SIDELINE_TOLERANCE_M) | (x > COURT_WIDTH + SIDELINE_TOLERANCE_M),
    )
    stats = flags.groupby("track_id")[["outside", "beyond_sidelines"]].mean()
    stats["spread"] = flags.groupby("track_id")[["court_x", "court_y"]].apply(_spread)
    return stats


def _spread(xy: pd.DataFrame) -> float:
    points = xy.to_numpy()
    return float(np.percentile(np.linalg.norm(points - np.median(points, axis=0), axis=1), 90))


def _within_side_capacity(df: pd.DataFrame, track_score: pd.Series) -> pd.Series:
    candidates = df[df["is_player"]].assign(score=df["track_id"].map(track_score))
    rank = candidates.groupby(["frame", "side"])["score"].rank(method="first", ascending=False)
    keep = pd.Series(False, index=df.index)
    keep.loc[rank.index] = rank <= MAX_PLAYERS_PER_SIDE
    return keep
