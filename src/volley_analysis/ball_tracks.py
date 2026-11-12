"""Nettoyage de la trajectoire du ballon : tronçons cohérents, rejet des fausses détections,
interpolation des petits trous."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volley_analysis.calibration import CameraPose

BALL_RADIUS_M = 0.105
RADIUS_BIAS_PX = 1.2
MIN_RADIUS_PX = 1.0

# Smash ou service rapide : ~30 m/s, soit ~4000 px/s près de la caméra.
MAX_SPEED_PX_S = 5000.0
GATE_MARGIN_PX = 25.0
MAX_JUMP_PX = 250.0
MAX_TRACKLET_GAP_S = 0.12
MIN_TRACKLET_DETECTIONS = 5
MAX_INTERPOLATION_GAP_S = 0.25

TRACK_COLUMNS = ["frame", "time_s", "x", "y", "radius", "conf", "source", "tracklet"]


@dataclass
class _Tracklet:
    id: int
    frames: list[int]
    points: list[np.ndarray]

    def predict(self, frame: int) -> np.ndarray:
        if len(self.points) < 2:
            return self.points[-1]
        velocity = (self.points[-1] - self.points[-2]) / (self.frames[-1] - self.frames[-2])
        return self.points[-1] + velocity * (frame - self.frames[-1])


def build_tracklets(detections: pd.DataFrame, fps: float) -> pd.Series:
    """Attribue un numéro de tronçon à chaque détection visible.

    Une détection prolonge le tronçon ouvert dont la position prédite (vitesse constante)
    est la plus proche, si elle est atteignable à la vitesse maximale du ballon.
    """
    visible = detections[detections["visible"]].sort_values("frame")
    max_gap = MAX_TRACKLET_GAP_S * fps
    open_tracklets: list[_Tracklet] = []
    labels = {}
    next_id = 0
    for index, row in visible.iterrows():
        frame, point = int(row["frame"]), np.array([row["x"], row["y"]])
        open_tracklets = [t for t in open_tracklets if frame - t.frames[-1] <= max_gap]
        best, best_distance = None, np.inf
        for tracklet in open_tracklets:
            dt = (frame - tracklet.frames[-1]) / fps
            distance = np.linalg.norm(point - tracklet.predict(frame))
            gate = min(MAX_SPEED_PX_S * dt, MAX_JUMP_PX) + GATE_MARGIN_PX
            if distance <= gate and distance < best_distance:
                best, best_distance = tracklet, distance
        if best is None:
            best = _Tracklet(next_id, [], [])
            next_id += 1
            open_tracklets.append(best)
        best.frames.append(frame)
        best.points.append(point)
        labels[index] = best.id
    return pd.Series(labels, dtype="Int64")


def clean_ball_track(detections: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Trajectoire image par image : `source` vaut `detected`, `interpolated` ou est vide."""
    detections = detections.copy()
    detections["tracklet"] = build_tracklets(detections, fps)
    sizes = detections["tracklet"].value_counts()
    kept = sizes[sizes >= MIN_TRACKLET_DETECTIONS].index
    good = detections[detections["tracklet"].isin(kept)].set_index("frame")

    frames = np.arange(int(detections["frame"].min()), int(detections["frame"].max()) + 1)
    track = pd.DataFrame(index=pd.Index(frames, name="frame"))
    track["time_s"] = frames / fps
    for column in ("x", "y", "radius", "conf", "tracklet"):
        track[column] = good[column]
    track["source"] = pd.Series("detected", index=good.index)

    gaps = _short_gaps(track["x"].notna().to_numpy(), int(MAX_INTERPOLATION_GAP_S * fps))
    interpolated = track[["x", "y", "radius"]].interpolate(method="index", limit_area="inside")
    track.loc[gaps, ["x", "y", "radius"]] = interpolated.loc[gaps]
    track.loc[gaps, "source"] = "interpolated"
    return track.reset_index()[TRACK_COLUMNS]


def ball_positions_3d(track: pd.DataFrame, camera: CameraPose) -> pd.DataFrame:
    """Position du ballon sur le terrain (`court_x`, `court_y`) et sa hauteur (`height`), en m.

    La distance à la caméra se déduit du rayon apparent : un ballon de 10,5 cm de
    rayon vu à `r` pixels est à `focale * 0.105 / r` mètres le long de l'axe optique.
    VballNet estimant le rayon sur une image réduite, les petits rayons sont
    surestimés d'environ `RADIUS_BIAS_PX`. La profondeur reste imprécise pour un
    ballon lointain (quelques mètres), ce qui suffit pour savoir de quel côté du
    filet il se trouve.
    """
    radius = np.maximum(track["radius"].to_numpy() - RADIUS_BIAS_PX, MIN_RADIUS_PX)
    depths = camera.focal * BALL_RADIUS_M / radius
    points = camera.backproject(track[["x", "y"]].to_numpy(), depths)
    return track.assign(court_x=points[:, 0], court_y=points[:, 1], height=points[:, 2])


def _short_gaps(present: np.ndarray, max_length: int) -> np.ndarray:
    """Masque des trous (entre deux positions connues) d'au plus `max_length` images."""
    mask = np.zeros_like(present)
    known = np.flatnonzero(present)
    for start, end in zip(known[:-1], known[1:], strict=True):
        if 1 < end - start <= max_length + 1:
            mask[start + 1 : end] = True
    return mask
