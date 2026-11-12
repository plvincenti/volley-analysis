"""Geste de l'arbitre en fin d'échange : bras tendu vers le camp qui marque le point.

L'arbitre est debout contre le poteau droit du filet. Caméra derrière le fond,
un bras tendu vers l'un des camps est très raccourci à l'image :

- vers le camp proche, le poignet part à droite de l'épaule, à sa hauteur ;
- vers le camp du fond, il part à gauche et au-dessus de l'épaule.

Les mains jointes devant la poitrine (il tient son sifflet) ont le coude plié,
et la main posée sur le poteau, bras tendu vers la gauche, reste immobile
plusieurs secondes : seuls les bras tendus, le temps d'un geste, comptent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from volley_analysis.court import FAR, NEAR
from volley_analysis.players import resolve_device
from volley_analysis.video import iter_frames, probe

POSE_MODEL = "yolo11m-pose.pt"
SAMPLE_S = 0.2
# Zone de l'image (x1, y1, x2, y2) où se trouvent les pieds de l'arbitre, propre au
# cadrage de la caméra. Les remplaçants assis ou accroupis à côté de lui font moins
# de 140 px de haut ; les joueurs devant lui ont les pieds plus bas dans l'image.
REFEREE_FEET_ZONE = (1380, 340, 1560, 420)
STANDING_MIN_HEIGHT_PX = 140
CROP = (1280, 70, 1660, 500)

SHOULDERS, ELBOWS, WRISTS, HIPS = (5, 6), (7, 8), (9, 10), (11, 12)
MIN_KEYPOINT_CONF = 0.5
MIN_EXTENSION = 0.85
MIN_REACH_TORSO = 0.4
SIDEWAYS_TORSO = 0.25
NEAR_MAX_RISE_TORSO = 0.1
NEAR_MAX_DROP_TORSO = 0.4
FAR_MIN_RISE_TORSO = 0.12
GESTURE_GAP_S = 0.45
MAX_GESTURE_S = 1.2


def track_referee(video_path: Path, model_path: str = POSE_MODEL) -> pd.DataFrame:
    """Points clés (COCO, 17 points) de l'arbitre toutes les `SAMPLE_S` secondes.

    L'arbitre est la seule personne debout dont les pieds sont dans
    `REFEREE_FEET_ZONE`. `standing` compte ces personnes : sans personne (arbitre
    parti) ou à plusieurs (temps mort, fin de set), la pose est laissée vide.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    device = resolve_device(None)
    info = probe(video_path)
    step = max(round(SAMPLE_S * info.fps), 1)
    cx1, cy1, cx2, cy2 = CROP
    fx1, fy1, fx2, fy2 = REFEREE_FEET_ZONE
    rows = []
    frames = iter_frames(video_path, step=step)
    for index, frame in tqdm(frames, total=info.frame_count // step, unit="img", desc="Arbitre"):
        result = model.predict(frame[cy1:cy2, cx1:cx2], conf=0.25, device=device, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy() + [cx1, cy1, cx1, cy1]
        feet_x = (boxes[:, 0] + boxes[:, 2]) / 2
        standing = np.flatnonzero(
            (boxes[:, 3] - boxes[:, 1] >= STANDING_MIN_HEIGHT_PX)
            & (feet_x >= fx1)
            & (feet_x <= fx2)
            & (boxes[:, 3] >= fy1)
            & (boxes[:, 3] <= fy2)
        )
        row = {"frame": index, "time_s": index / info.fps, "standing": len(standing)}
        if len(standing) == 1:
            keypoints = result.keypoints.data.cpu().numpy()[standing[0]]
            for k, (x, y, conf) in enumerate(keypoints):
                row[f"k{k}x"], row[f"k{k}y"], row[f"k{k}c"] = x + cx1, y + cy1, conf
        rows.append(row)
    return pd.DataFrame(rows)


def pointed_sides(pose: pd.DataFrame) -> pd.Series:
    """Camp désigné par un bras tendu à chaque instant (`near`, `far` ou vide)."""
    if "k5x" not in pose:
        return pd.Series(dtype=object)
    pose = pose.dropna(subset=["k5x"])

    def point(k: int) -> np.ndarray:
        return pose[[f"k{k}x", f"k{k}y"]].to_numpy()

    shoulders = (point(SHOULDERS[0]) + point(SHOULDERS[1])) / 2
    hips = (point(HIPS[0]) + point(HIPS[1])) / 2
    torso = np.linalg.norm(shoulders - hips, axis=1)[:, None]
    sides = np.full(len(pose), "", dtype=object)
    for shoulder, elbow, wrist in zip(SHOULDERS, ELBOWS, WRISTS, strict=True):
        s, e, w = point(shoulder), point(elbow), point(wrist)
        conf = np.minimum(pose[f"k{elbow}c"], pose[f"k{wrist}c"]).to_numpy()
        arm_length = np.linalg.norm(e - s, axis=1) + np.linalg.norm(w - e, axis=1)
        extension = np.linalg.norm(w - s, axis=1) / np.maximum(arm_length, 1e-6)
        dx, dy = ((w - s) / torso).T
        extended = (
            (conf > MIN_KEYPOINT_CONF)
            & (extension > MIN_EXTENSION)
            & (np.hypot(dx, dy) > MIN_REACH_TORSO)
        )
        near = (
            extended
            & (dx > SIDEWAYS_TORSO)
            & (dy > -NEAR_MAX_RISE_TORSO)
            & (dy < NEAR_MAX_DROP_TORSO)
        )
        far = extended & (dx < -SIDEWAYS_TORSO) & (dy < -FAR_MIN_RISE_TORSO)
        sides[near & (sides == "")] = NEAR
        sides[far & (sides == "")] = FAR
    return pd.Series(sides, index=pose["time_s"].to_numpy())


def referee_gestures(pose: pd.DataFrame) -> pd.DataFrame:
    """Gestes de l'arbitre : `start_s`, `end_s`, `side` du camp désigné."""
    pointed = pointed_sides(pose)
    pointed = pointed[pointed != ""]
    runs: list[list] = []
    for time_s, side in pointed.items():
        if runs and runs[-1][2] == side and time_s - runs[-1][1] <= GESTURE_GAP_S:
            runs[-1][1] = time_s
        else:
            runs.append([time_s, time_s, side])
    gestures = pd.DataFrame(runs, columns=["start_s", "end_s", "side"])
    return gestures[gestures["end_s"] - gestures["start_s"] <= MAX_GESTURE_S].reset_index(drop=True)


def referee_call(gestures: pd.DataFrame, start_s: float, end_s: float) -> str | None:
    """Camp désigné par le premier geste entre `start_s` et `end_s`."""
    window = gestures[(gestures["start_s"] >= start_s) & (gestures["start_s"] < end_s)]
    return window["side"].iloc[0] if len(window) else None
