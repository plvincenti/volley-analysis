"""Nettoyage des pistes de joueurs sur toute la vidéo, après le suivi image par image."""

import numpy as np
import pandas as pd

from volley_analysis.court import COURT_LENGTH, COURT_WIDTH, side_of_net

SIDE_WINDOW_S = 2.0
BYSTANDER_MIN_OUTSIDE_FRACTION = 0.9
BYSTANDER_MAX_SPREAD_M = 1.0


def clean_tracks(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Ajoute le côté du filet lissé (`side`) et le statut de joueur (`is_player`).

    - Un joueur ne change pas de côté pendant un set, mais ses pieds décollent quand il
      saute au filet et la projection le fait alors passer de l'autre côté : le côté est
      la médiane glissante de la profondeur sur `SIDE_WINDOW_S`.
    - Les spectateurs, remplaçants ou arbitres restent quasi immobiles hors des lignes,
      contrairement au serveur qui ne sort que quelques secondes.
    """
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    if df.empty:
        return df.assign(side=pd.Series(dtype=str), is_player=pd.Series(dtype=bool))
    window = max(int(round(SIDE_WINDOW_S * fps)), 1)
    smoothed_y = df.groupby("track_id")["court_y"].transform(
        lambda y: y.rolling(window, center=True, min_periods=1).median()
    )
    df["side"] = side_of_net(smoothed_y.to_numpy())
    df["is_player"] = df["in_play"] & ~df["track_id"].isin(bystander_tracks(df[df["in_play"]]))
    return df.sort_values(["frame", "track_id"]).reset_index(drop=True)


def bystander_tracks(df: pd.DataFrame) -> set[int]:
    inside = df["court_x"].between(0, COURT_WIDTH) & df["court_y"].between(0, COURT_LENGTH)
    bystanders = set()
    for track_id, group in df.assign(inside=inside).groupby("track_id"):
        xy = group[["court_x", "court_y"]].to_numpy()
        spread = np.percentile(np.linalg.norm(xy - np.median(xy, axis=0), axis=1), 90)
        outside = 1 - group["inside"].mean()
        if outside >= BYSTANDER_MIN_OUTSIDE_FRACTION and spread < BYSTANDER_MAX_SPREAD_M:
            bystanders.add(int(track_id))
    return bystanders
