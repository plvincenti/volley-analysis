"""Outil interactif (fenêtre OpenCV) pour cliquer les points du terrain."""

import cv2
import numpy as np

from volley_analysis.calibration import MIN_POINTS, Calibration
from volley_analysis.court import KEYPOINTS
from volley_analysis.render import draw_court_overlay

WINDOW = "Calibration terrain"
KEY_ENTER = (10, 13)
KEY_ESCAPE = 27
MAGNIFIER_RADIUS = 30
MAGNIFIER_ZOOM = 5


class CalibrationAborted(Exception):
    pass


def run_calibration_ui(
    frame: np.ndarray, max_display_width: int = 1600
) -> dict[str, tuple[float, float]]:
    """Fait cliquer les points du terrain puis valider la reprojection.

    Renvoie les points image (en pixels de la vidéo d'origine) par nom de point.
    """
    scale = min(1.0, max_display_width / frame.shape[1])
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            points = _collect_points(frame, scale)
            calibration = Calibration(
                image_points=points, frame_size=(frame.shape[1], frame.shape[0])
            )
            if _confirm(frame, calibration, scale):
                return points
    finally:
        cv2.destroyWindow(WINDOW)
        cv2.waitKey(1)


def _collect_points(frame: np.ndarray, scale: float) -> dict[str, tuple[float, float]]:
    mouse = {"pos": None, "click": None}

    def on_mouse(event, x, y, _flags, _param):
        mouse["pos"] = (x / scale, y / scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["click"] = mouse["pos"]

    cv2.setMouseCallback(WINDOW, on_mouse)
    points: dict[str, tuple[float, float]] = {}
    history: list[str | None] = []
    step = 0

    while step < len(KEYPOINTS):
        keypoint = KEYPOINTS[step]
        canvas = _resize(_draw_points(frame, points), scale)
        if mouse["pos"] is not None:
            _draw_magnifier(canvas, frame, mouse["pos"], scale)
        _draw_banner(
            canvas,
            [
                f"[{step + 1}/{len(KEYPOINTS)}] Cliquez : {keypoint.label}",
                f"s: point invisible   u: annuler   Entree: terminer (>= {MIN_POINTS} points)"
                "   q: quitter",
            ],
        )
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(20) & 0xFF

        if mouse["click"] is not None:
            points[keypoint.name] = mouse["click"]
            mouse["click"] = None
            history.append(keypoint.name)
            step += 1
        elif key == ord("s"):
            history.append(None)
            step += 1
        elif key == ord("u") and history:
            last = history.pop()
            if last is not None:
                points.pop(last)
            step -= 1
        elif key in KEY_ENTER and len(points) >= MIN_POINTS:
            break
        elif key in (ord("q"), KEY_ESCAPE):
            raise CalibrationAborted

    if len(points) < MIN_POINTS:
        raise CalibrationAborted(f"Seulement {len(points)} points saisis, il en faut {MIN_POINTS}")
    return points


def _confirm(frame: np.ndarray, calibration: Calibration, scale: float) -> bool:
    canvas = _resize(draw_court_overlay(frame, calibration), scale)
    _draw_banner(
        canvas,
        [
            f"Erreur de reprojection : {calibration.reprojection_error():.1f} px",
            "Les lignes jaunes suivent-elles le terrain ?  Entree: valider   r: recommencer"
            "   q: quitter",
        ],
    )
    while True:
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(50) & 0xFF
        if key in KEY_ENTER:
            return True
        if key == ord("r"):
            return False
        if key in (ord("q"), KEY_ESCAPE):
            raise CalibrationAborted


def _draw_points(frame: np.ndarray, points: dict[str, tuple[float, float]]) -> np.ndarray:
    canvas = frame.copy()
    for name, (u, v) in points.items():
        cv2.circle(canvas, (int(u), int(v)), 5, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            name,
            (int(u) + 8, int(v) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _draw_magnifier(
    canvas: np.ndarray, frame: np.ndarray, pos: tuple[float, float], scale: float
) -> None:
    """Loupe à côté du curseur, pour viser précisément les intersections de lignes."""
    r = MAGNIFIER_RADIUS
    padded = cv2.copyMakeBorder(frame, r, r, r, r, cv2.BORDER_CONSTANT)
    u, v = int(pos[0]) + r, int(pos[1]) + r
    patch = padded[v - r : v + r, u - r : u + r]
    if patch.shape[:2] != (2 * r, 2 * r):
        return
    size = 2 * r * MAGNIFIER_ZOOM
    zoom = cv2.resize(patch, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.line(zoom, (size // 2, 0), (size // 2, size), (0, 255, 0), 1)
    cv2.line(zoom, (0, size // 2), (size, size // 2), (0, 255, 0), 1)
    cv2.rectangle(zoom, (0, 0), (size - 1, size - 1), (255, 255, 255), 2)

    h, w = canvas.shape[:2]
    cu, cv_ = int(pos[0] * scale), int(pos[1] * scale)
    offset = 30
    x0 = cu + offset if cu + offset + size <= w else cu - offset - size
    y0 = min(max(cv_ - size // 2, 0), h - size)
    if x0 < 0 or y0 < 0:
        return
    canvas[y0 : y0 + size, x0 : x0 + size] = zoom


def _resize(image: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _draw_banner(image: np.ndarray, lines: list[str]) -> None:
    height = 28 * len(lines) + 12
    cv2.rectangle(image, (0, 0), (image.shape[1], height), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (10, 28 * (i + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
