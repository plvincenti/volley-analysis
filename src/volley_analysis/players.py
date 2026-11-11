"""Détection et suivi des joueurs (YOLO + ByteTrack via Ultralytics)."""

from dataclasses import dataclass

import numpy as np

from volley_analysis.calibration import Calibration
from volley_analysis.court import is_in_play_area

PERSON_CLASS = 0


@dataclass
class TrackerConfig:
    model: str = "yolo11m.pt"
    imgsz: int = 1280
    conf: float = 0.25
    device: str | None = None
    tracker: str = "bytetrack.yaml"
    side_margin: float = 1.0
    end_margin: float = 4.0


@dataclass
class FrameTracks:
    """Personnes suivies sur une image. Les tableaux ont tous N lignes."""

    track_ids: np.ndarray
    confidences: np.ndarray
    xyxy: np.ndarray
    court_xy: np.ndarray
    in_play: np.ndarray

    @classmethod
    def empty(cls) -> "FrameTracks":
        return cls(
            track_ids=np.empty(0, dtype=np.int64),
            confidences=np.empty(0, dtype=np.float32),
            xyxy=np.empty((0, 4), dtype=np.float32),
            court_xy=np.empty((0, 2), dtype=np.float64),
            in_play=np.empty(0, dtype=bool),
        )

    def __len__(self) -> int:
        return len(self.track_ids)


def resolve_device(device: str | None) -> str:
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PlayerTracker:
    """Suit les personnes image par image et les place sur le terrain.

    La position d'un joueur est le milieu du bas de sa boîte (ses pieds),
    seul point de la personne qui est sur le plan du sol, donc valide pour
    l'homographie.
    """

    def __init__(self, config: TrackerConfig, calibration: Calibration):
        from ultralytics import YOLO

        self.config = config
        self.calibration = calibration
        self.device = resolve_device(config.device)
        self.model = YOLO(config.model)

    def update(self, frame: np.ndarray) -> FrameTracks:
        result = self.model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS],
            conf=self.config.conf,
            imgsz=self.config.imgsz,
            device=self.device,
            tracker=self.config.tracker,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return FrameTracks.empty()

        xyxy = boxes.xyxy.cpu().numpy()
        feet = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]])
        court_xy = self.calibration.image_to_court(feet)
        return FrameTracks(
            track_ids=boxes.id.int().cpu().numpy().astype(np.int64),
            confidences=boxes.conf.cpu().numpy(),
            xyxy=xyxy,
            court_xy=court_xy,
            in_play=is_in_play_area(court_xy, self.config.side_margin, self.config.end_margin),
        )
