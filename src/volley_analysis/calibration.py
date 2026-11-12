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

    def camera(self) -> "CameraPose":
        """Pose de la caméra déduite de l'homographie du sol.

        Suppose des pixels carrés et le centre optique au centre de l'image,
        ce qui laisse la focale comme seule inconnue interne : elle est fixée
        par l'orthogonalité des axes du terrain.
        """
        court_to_image = np.linalg.inv(self.homography)
        cx, cy = self.frame_size[0] / 2, self.frame_size[1] / 2
        h1, h2, h3 = court_to_image.T
        a1 = np.array([h1[0] - cx * h1[2], h1[1] - cy * h1[2]])
        a2 = np.array([h2[0] - cx * h2[2], h2[1] - cy * h2[2]])
        focal_sq = -(a1 @ a2) / (h1[2] * h2[2])
        if focal_sq <= 0:
            raise ValueError("Calibration incompatible avec une caméra perspective standard")
        focal = float(np.sqrt(focal_sq))

        k_inv = np.linalg.inv(np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]]))
        r1, r2, t = k_inv @ h1, k_inv @ h2, k_inv @ h3
        # L'homographie est définie au signe près : le terrain doit être devant la caméra.
        scale = np.sign(t[2]) / np.sqrt(np.linalg.norm(r1) * np.linalg.norm(r2))
        r1, r2, t = r1 * scale, r2 * scale, t * scale
        u, _, vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
        rotation = u @ vt
        return CameraPose(focal, (cx, cy), rotation, -rotation.T @ t)

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


@dataclass
class CameraPose:
    """Caméra perspective dans le repère du terrain (z vers le haut, en mètres)."""

    focal: float
    principal_point: tuple[float, float]
    rotation: np.ndarray
    center: np.ndarray

    def backproject(self, pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """Points 3D (N, 3) vus aux pixels (N, 2), à la profondeur `depths` le long de l'axe."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        cx, cy = self.principal_point
        rays = np.column_stack(
            [
                (pixels[:, 0] - cx) / self.focal,
                (pixels[:, 1] - cy) / self.focal,
                np.ones(len(pixels)),
            ]
        )
        return self.center + (rays * np.asarray(depths)[:, None]) @ self.rotation

    def project(self, points: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) des points 3D (N, 3)."""
        cam = (np.asarray(points, dtype=np.float64) - self.center) @ self.rotation.T
        cx, cy = self.principal_point
        return np.column_stack(
            [self.focal * cam[:, 0] / cam[:, 2] + cx, self.focal * cam[:, 1] / cam[:, 2] + cy]
        )


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
