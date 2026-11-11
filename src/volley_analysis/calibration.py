"""Calibration caméra -> terrain par homographie.

Une caméra fixe permet de calculer une seule homographie par vidéo, à partir
de points du terrain cliqués sur une image (voir `calibration_ui`).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from volley_analysis.court import KEYPOINTS_BY_NAME

MIN_POINTS = 4


@dataclass
class Calibration:
    image_points: dict[str, tuple[float, float]]
    frame_size: tuple[int, int]
    frame_index: int = 0
    video: str | None = None
    homography: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        unknown = set(self.image_points) - set(KEYPOINTS_BY_NAME)
        if unknown:
            raise ValueError(f"Points de terrain inconnus : {sorted(unknown)}")
        if len(self.image_points) < MIN_POINTS:
            raise ValueError(f"Il faut au moins {MIN_POINTS} points, reçu {len(self.image_points)}")

        src, dst = self._point_pairs()
        homography, _ = cv2.findHomography(src, dst, method=0)
        if homography is None:
            raise ValueError("Homographie impossible : points alignés ou dégénérés")
        self.homography = homography

    def _point_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        names = list(self.image_points)
        src = np.array([self.image_points[n] for n in names], dtype=np.float64)
        dst = np.array([KEYPOINTS_BY_NAME[n].xy for n in names], dtype=np.float64)
        return src, dst

    def image_to_court(self, pixels: np.ndarray) -> np.ndarray:
        """Projette des pixels (N, 2) en positions terrain (N, 2), en mètres."""
        return _apply(self.homography, pixels)

    def court_to_image(self, court_xy: np.ndarray) -> np.ndarray:
        """Projette des positions terrain (N, 2) en pixels (N, 2)."""
        return _apply(np.linalg.inv(self.homography), court_xy)

    def reprojection_error(self) -> float:
        """Erreur moyenne (pixels) entre les points cliqués et leur reprojection."""
        src, dst = self._point_pairs()
        return float(np.linalg.norm(self.court_to_image(dst) - src, axis=1).mean())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "video": self.video,
            "frame_index": self.frame_index,
            "frame_size": list(self.frame_size),
            "image_points": {k: list(v) for k, v in self.image_points.items()},
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        payload = json.loads(Path(path).read_text())
        return cls(
            image_points={
                k: (float(v[0]), float(v[1])) for k, v in payload["image_points"].items()
            },
            frame_size=(int(payload["frame_size"][0]), int(payload["frame_size"][1])),
            frame_index=int(payload.get("frame_index", 0)),
            video=payload.get("video"),
        )


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
