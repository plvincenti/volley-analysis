"""Côté du terrain occupé par chaque équipe, d'après la couleur des maillots.

Corbas joue en rouge. Seules les personnes dont les pieds sont sur le terrain
comptent : les remplaçants, l'arbitre et le marqueur, en rouge eux aussi, sont
sur les côtés.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from volley_analysis.calibration import Calibration
from volley_analysis.court import COURT_LENGTH, COURT_WIDTH, FAR, NEAR, NET_Y
from volley_analysis.players import PERSON_CLASS, resolve_device

CORBAS = "Corbas"
OPPONENT = "Adversaire"

# Rouge vif des maillots (OpenCV : teinte sur 0-180).
RED_MAX_HUE = 8
RED_MIN_HUE = 172
RED_MIN_SATURATION = 120
RED_MIN_VALUE = 70
# Torse : sous la tête, au-dessus du short, sans les bras.
TORSO_TOP, TORSO_BOTTOM = 0.18, 0.5
TORSO_SIDE_MARGIN = 0.2
RED_JERSEY_MIN_FRACTION = 0.25
COURT_MARGIN_M = 0.5
MIN_RED_SHARE_GAP = 0.3


def red_fraction(frame: np.ndarray, box: np.ndarray) -> float:
    """Part de pixels rouge vif sur le torse de la personne encadrée par `box` (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    top, bottom = int(y1 + TORSO_TOP * height), int(y1 + TORSO_BOTTOM * height)
    left, right = int(x1 + TORSO_SIDE_MARGIN * width), int(x2 - TORSO_SIDE_MARGIN * width)
    torso = frame[max(top, 0) : max(bottom, 0), max(left, 0) : max(right, 0)]
    if torso.size == 0:
        return 0.0
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (
        ((hue <= RED_MAX_HUE) | (hue >= RED_MIN_HUE))
        & (saturation >= RED_MIN_SATURATION)
        & (value >= RED_MIN_VALUE)
    )
    return float(red.mean())


def court_side(court_xy: np.ndarray) -> np.ndarray:
    """Côté (`near`, `far`) des positions sur le terrain, vide en dehors."""
    x, y = court_xy[:, 0], court_xy[:, 1]
    inside = (
        (x >= -COURT_MARGIN_M)
        & (x <= COURT_WIDTH + COURT_MARGIN_M)
        & (y >= -COURT_MARGIN_M)
        & (y <= COURT_LENGTH + COURT_MARGIN_M)
    )
    return np.where(inside, np.where(y < NET_Y, NEAR, FAR), "")


@dataclass
class SideColors:
    """Joueurs en rouge et joueurs comptés, de chaque côté du filet."""

    red: dict[str, int]
    players: dict[str, int]

    def red_share(self, side: str) -> float:
        return self.red[side] / self.players[side] if self.players[side] else 0.0

    def corbas_side(self) -> str | None:
        """Le côté nettement plus rouge que l'autre, ou None si on ne peut pas trancher."""
        near, far = self.red_share(NEAR), self.red_share(FAR)
        if abs(near - far) < MIN_RED_SHARE_GAP:
            return None
        return NEAR if near > far else FAR


def side_colors(frame: np.ndarray, boxes: np.ndarray, calibration: Calibration) -> SideColors:
    feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
    sides = court_side(calibration.image_to_court(feet))
    is_red = np.array([red_fraction(frame, box) >= RED_JERSEY_MIN_FRACTION for box in boxes])
    return SideColors(
        red={side: int((is_red & (sides == side)).sum()) for side in (NEAR, FAR)},
        players={side: int((sides == side).sum()) for side in (NEAR, FAR)},
    )


class PersonDetector:
    """Détection de personnes image par image (sans suivi), pour quelques images isolées."""

    def __init__(self, model: str = "yolo11m.pt", imgsz: int = 1280, conf: float = 0.25):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.imgsz = imgsz
        self.conf = conf
        self.device = resolve_device(None)

    def boxes(self, frame: np.ndarray) -> np.ndarray:
        result = self.model.predict(
            frame,
            classes=[PERSON_CLASS],
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 4), dtype=np.float32)
        return result.boxes.xyxy.cpu().numpy()
