"""Calibration à partir des lignes du terrain dessinées à main levée sur une capture d'écran.

Le dessin (trait rouge vif) suit les 4 lignes délimitant le terrain : les deux lignes
de fond et les deux lignes de côté. Comme le tracé tremble, on ajuste une droite sur
chaque trait, puis les coins sont les intersections des droites, même hors de l'image.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from volley_analysis.calibration import Calibration

MIN_REGISTRATION_INLIERS = 50


@dataclass(frozen=True)
class FittedLine:
    """Droite a*x + b*y + c = 0 avec (a, b) unitaire, et les points du trait associé."""

    abc: np.ndarray
    points: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return self.points.mean(axis=0)

    @property
    def tilt(self) -> float:
        """Angle (degrés, 0 à 90) entre la droite et l'horizontale de l'image."""
        a, b, _ = self.abc
        return float(np.degrees(np.arctan2(abs(a), abs(b))))

    def distances(self, points: np.ndarray) -> np.ndarray:
        return np.abs(points @ self.abc[:2] + self.abc[2])

    def intersect(self, other: "FittedLine") -> tuple[float, float]:
        x, y, w = np.cross(self.abc, other.abc)
        if abs(w) < 1e-12:
            raise ValueError("Droites parallèles")
        return float(x / w), float(y / w)


def register_screenshot(screenshot: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Transformation (2x3) des pixels de la capture vers ceux de l'image vidéo.

    La capture est une version mise à l'échelle et recadrée de la vidéo : une
    similitude suffit, estimée sur les éléments fixes du gymnase.
    """
    sift = cv2.SIFT_create(8000)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    transform, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0
    )
    if transform is None or int(inliers.sum()) < MIN_REGISTRATION_INLIERS:
        raise ValueError("La capture ne correspond pas à l'image de la vidéo")
    return transform


def red_stroke_points(image: np.ndarray) -> np.ndarray:
    """Pixels (N, 2) d'un tracé rouge vif, plus saturé que les maillots et lignes peintes."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((h <= 4) | (h >= 176)) & (s >= 200) & (v >= 200)
    ys, xs = np.nonzero(mask)
    return np.column_stack([xs, ys]).astype(np.float64)


def fit_lines(
    points: np.ndarray,
    count: int = 4,
    threshold: float = 6.0,
    iterations: int = 2000,
    seed: int = 0,
) -> list[FittedLine]:
    """Extrait `count` droites d'un nuage de points par RANSAC successifs."""
    rng = np.random.default_rng(seed)
    remaining = points
    lines = []
    for _ in range(count):
        if len(remaining) < 2:
            raise ValueError(f"Seulement {len(lines)} traits trouvés, {count} attendus")
        best = None
        for _ in range(iterations):
            p, q = remaining[rng.choice(len(remaining), 2, replace=False)]
            abc = _line_through(p, q)
            if abc is None:
                continue
            n = int((np.abs(remaining @ abc[:2] + abc[2]) < threshold).sum())
            if best is None or n > best[0]:
                best = (n, abc)
        abc = best[1]
        for _ in range(3):
            inliers = remaining[np.abs(remaining @ abc[:2] + abc[2]) < threshold]
            abc = _total_least_squares(inliers)
        line = FittedLine(abc=abc, points=inliers)
        lines.append(line)
        remaining = remaining[line.distances(remaining) >= 2 * threshold]
    return lines


def assign_court_lines(lines: list[FittedLine]) -> dict[str, FittedLine]:
    """Identifie les lignes : fond opposé (en haut), fond côté caméra, côté gauche, côté droit.

    Caméra derrière une ligne de fond : les lignes de fond sont les deux droites les plus
    proches de l'horizontale, les lignes de côté fuient vers le haut de l'image.
    """
    if len(lines) != 4:
        raise ValueError(f"4 lignes attendues, reçu {len(lines)}")
    by_tilt = sorted(lines, key=lambda ln: ln.tilt)
    horizontal = sorted(by_tilt[:2], key=lambda ln: ln.center[1])
    vertical = sorted(by_tilt[2:], key=lambda ln: ln.center[0])
    return {
        "far_end": horizontal[0],
        "near_end": horizontal[1],
        "left": vertical[0],
        "right": vertical[1],
    }


def calibration_from_drawing(
    screenshot: np.ndarray, frame: np.ndarray, frame_index: int = 0, video: str | None = None
) -> tuple[Calibration, dict[str, FittedLine]]:
    transform = register_screenshot(screenshot, frame)
    points = red_stroke_points(screenshot)
    if len(points) == 0:
        raise ValueError("Aucun tracé rouge trouvé sur la capture")
    points = cv2.transform(points.reshape(-1, 1, 2), transform).reshape(-1, 2)
    court_lines = assign_court_lines(fit_lines(points))

    corners = {
        "near_left": ("left", "near_end"),
        "near_right": ("right", "near_end"),
        "far_left": ("left", "far_end"),
        "far_right": ("right", "far_end"),
    }
    image_points = {
        name: court_lines[a].intersect(court_lines[b]) for name, (a, b) in corners.items()
    }
    calibration = Calibration(
        image_points=image_points,
        frame_size=(frame.shape[1], frame.shape[0]),
        frame_index=frame_index,
        video=video,
    )
    return calibration, court_lines


def _line_through(p: np.ndarray, q: np.ndarray) -> np.ndarray | None:
    direction = q - p
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return None
    a, b = -direction[1] / norm, direction[0] / norm
    return np.array([a, b, -(a * p[0] + b * p[1])])


def _total_least_squares(points: np.ndarray) -> np.ndarray:
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    a, b = -vy, vx
    return np.array([a, b, -(a * x0 + b * y0)], dtype=np.float64)
