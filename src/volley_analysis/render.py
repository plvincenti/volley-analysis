"""Rendus visuels : surimpression du terrain, joueurs, mini-carte et heatmaps.

Les textes passent par `cv2.putText`, qui ne gère pas les accents.
"""

import cv2
import numpy as np

from volley_analysis.calibration import Calibration
from volley_analysis.court import COURT_LENGTH, COURT_WIDTH, FAR, LINES, NEAR, NET_Y

# Couleurs BGR
TEAM_COLORS = {NEAR: (255, 140, 30), FAR: (40, 60, 230)}
LINE_COLOR = (0, 255, 255)
BALL_COLOR = (0, 230, 255)
FLOOR_COLOR = (60, 110, 170)
COURT_LINE_COLOR = (255, 255, 255)
SCORE_TEXT_COLOR = (255, 255, 255)
SCORE_POINT_COLOR = (0, 230, 255)
REFEREE_AGREE_COLOR = (80, 220, 80)
REFEREE_DISAGREE_COLOR = (60, 60, 255)


def draw_court_overlay(
    frame: np.ndarray, calibration: Calibration, color=LINE_COLOR, thickness: int = 2
) -> np.ndarray:
    """Dessine les lignes du terrain reprojetées sur l'image (contrôle de calibration)."""
    out = frame.copy()
    for start, end in LINES:
        p1, p2 = calibration.court_to_image([start, end]).round().astype(int)
        cv2.line(out, tuple(p1), tuple(p2), color, thickness, cv2.LINE_AA)
    for name, (u, v) in calibration.image_points.items():
        cv2.circle(out, (int(u), int(v)), 6, (0, 0, 255), -1)
        cv2.putText(
            out,
            name,
            (int(u) + 8, int(v) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def draw_players(
    frame: np.ndarray, xyxy: np.ndarray, track_ids: np.ndarray, sides: np.ndarray
) -> np.ndarray:
    out = frame
    for box, track_id, side in zip(xyxy, track_ids, sides, strict=True):
        color = TEAM_COLORS[side]
        x1, y1, x2, y2 = box.round().astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.ellipse(out, ((x1 + x2) // 2, y2), (max((x2 - x1) // 2, 4), 6), 0, 0, 360, color, 2)
        cv2.putText(
            out,
            f"#{track_id}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def draw_ball(frame: np.ndarray, trail_xy: np.ndarray, radius: float, detected: bool) -> np.ndarray:
    """Traînée des dernières positions, puis cercle plein si détecté, fin si interpolé.

    `trail_xy` se termine par la position courante ; les NaN coupent la traînée.
    """
    for i in range(1, len(trail_xy)):
        p, q = trail_xy[i - 1], trail_xy[i]
        if np.isnan(p).any() or np.isnan(q).any():
            continue
        thickness = 1 + 4 * i // len(trail_xy)
        cv2.line(
            frame, tuple(p.astype(int)), tuple(q.astype(int)), BALL_COLOR, thickness, cv2.LINE_AA
        )
    if len(trail_xy) and not np.isnan(trail_xy[-1]).any():
        center = tuple(trail_xy[-1].astype(int))
        r = int(radius) + 4 if not np.isnan(radius) else 12
        cv2.circle(frame, center, r, BALL_COLOR, 3 if detected else 1, cv2.LINE_AA)
    return frame


class CourtCanvas:
    """Vue de dessus du terrain, côté caméra en bas (comme dans la vidéo)."""

    def __init__(self, pixels_per_meter: int = 20, margin_m: float = 3.0):
        self.ppm = pixels_per_meter
        self.margin = margin_m
        self.width = int((COURT_WIDTH + 2 * margin_m) * pixels_per_meter)
        self.height = int((COURT_LENGTH + 2 * margin_m) * pixels_per_meter)
        self._background = self._draw_background()

    def to_pixels(self, court_xy: np.ndarray) -> np.ndarray:
        court_xy = np.asarray(court_xy, dtype=float).reshape(-1, 2)
        u = (court_xy[:, 0] + self.margin) * self.ppm
        v = (COURT_LENGTH + self.margin - court_xy[:, 1]) * self.ppm
        return np.column_stack([u, v])

    def _draw_background(self) -> np.ndarray:
        image = np.full((self.height, self.width, 3), 35, dtype=np.uint8)
        corners = self.to_pixels(
            [(0, 0), (COURT_WIDTH, 0), (COURT_WIDTH, COURT_LENGTH), (0, COURT_LENGTH)]
        ).astype(np.int32)
        cv2.fillPoly(image, [corners], FLOOR_COLOR)
        for start, end in LINES:
            p1, p2 = self.to_pixels([start, end]).astype(int)
            thickness = 3 if start[1] == end[1] == NET_Y else 1
            cv2.line(image, tuple(p1), tuple(p2), COURT_LINE_COLOR, thickness, cv2.LINE_AA)
        return image

    def background(self) -> np.ndarray:
        return self._background.copy()

    def draw_positions(
        self, court_xy: np.ndarray, sides: np.ndarray, labels: np.ndarray | None = None
    ) -> np.ndarray:
        image = self.background()
        for i, ((u, v), side) in enumerate(zip(self.to_pixels(court_xy), sides, strict=True)):
            center = (int(u), int(v))
            cv2.circle(image, center, max(self.ppm // 3, 3), TEAM_COLORS[side], -1, cv2.LINE_AA)
            if labels is not None:
                cv2.putText(
                    image,
                    str(labels[i]),
                    (center[0] + 6, center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        return image

    def heatmap(
        self, court_xy: np.ndarray, sigma_m: float = 0.5, alpha: float = 0.65
    ) -> np.ndarray:
        """Carte de densité de présence superposée au terrain."""
        image = self.background()
        pixels = self.to_pixels(court_xy).round().astype(int)
        valid = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < self.width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < self.height)
        )
        pixels = pixels[valid]
        if len(pixels) == 0:
            return image

        density = np.zeros((self.height, self.width), dtype=np.float32)
        np.add.at(density, (pixels[:, 1], pixels[:, 0]), 1.0)
        density = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma_m * self.ppm)
        density /= density.max()

        colored = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_JET)
        weight = (np.clip(density * 3, 0, 1) * alpha)[..., None]
        blended = image * (1 - weight) + colored * weight
        return blended.astype(np.uint8)


def draw_caption(image: np.ndarray, lines: list[str], scale: float = 0.8) -> np.ndarray:
    """Lignes de texte blanc sur un bandeau noir, en haut à gauche."""
    line_height = int(32 * scale)
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] for line in lines)
    cv2.rectangle(image, (0, 0), (width + 16, line_height * len(lines) + 10), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        position = (8, line_height * (i + 1))
        cv2.putText(
            image, line, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA
        )
    return image


def draw_scoreboard(
    frame: np.ndarray,
    set_number: int,
    corbas: int,
    opponent: int,
    rally: int | None,
    serving_team: str | None,
    point_to: str | None,
    referee: str | None = None,
) -> np.ndarray:
    """Tableau de score en haut à gauche : set, score, équipe au service ou point marqué.

    `referee` est l'équipe désignée par l'arbitre, en rouge si elle contredit `point_to`.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    header = f"SET {set_number}" + (f"   echange #{rally}" if rally is not None else "")
    score = f"Corbas {corbas} - {opponent} Adversaire"
    if point_to:
        status, color = f"POINT {point_to}", SCORE_POINT_COLOR
    elif serving_team:
        status, color = f"Service : {serving_team}", SCORE_TEXT_COLOR
    else:
        status, color = "", SCORE_TEXT_COLOR

    width = max(cv2.getTextSize(score, font, 1.3, 3)[0][0], 360) + 40
    bottom = 212 if referee else 172
    cv2.rectangle(frame, (12, 12), (12 + width, bottom), (0, 0, 0), -1)
    cv2.putText(frame, header, (30, 52), font, 0.9, SCORE_TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(frame, score, (30, 104), font, 1.3, SCORE_TEXT_COLOR, 3, cv2.LINE_AA)
    cv2.putText(frame, status, (30, 152), font, 1.0, color, 2, cv2.LINE_AA)
    if referee:
        agrees = referee == point_to
        referee_color = REFEREE_AGREE_COLOR if agrees else REFEREE_DISAGREE_COLOR
        text = f"Arbitre : {referee}" + ("" if agrees else " (a verifier)")
        cv2.putText(frame, text, (30, 196), font, 0.9, referee_color, 2, cv2.LINE_AA)
    return frame


def contact_sheet(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    """Assemble des vignettes de même taille en grille, ligne par ligne."""
    blank = np.zeros_like(tiles[0])
    tiles = tiles + [blank] * (-len(tiles) % columns)
    rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows)


def paste_inset(
    frame: np.ndarray, inset: np.ndarray, max_height_ratio: float = 0.45, margin: int = 12
) -> np.ndarray:
    """Incruste une image (mini-carte) en haut à droite de `frame`."""
    target_h = int(frame.shape[0] * max_height_ratio)
    scale = target_h / inset.shape[0]
    resized = cv2.resize(
        inset, (int(inset.shape[1] * scale), target_h), interpolation=cv2.INTER_AREA
    )
    h, w = resized.shape[:2]
    x0 = frame.shape[1] - w - margin
    frame[margin : margin + h, x0 : x0 + w] = resized
    return frame
