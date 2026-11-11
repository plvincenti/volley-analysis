"""Lecture des vidéos (métadonnées, image isolée, itération sur les images)."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def _open(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {path}")
    return capture


def probe(path: Path) -> VideoInfo:
    capture = _open(path)
    try:
        return VideoInfo(
            path=Path(path),
            fps=capture.get(cv2.CAP_PROP_FPS),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def read_frame(path: Path, index: int) -> np.ndarray:
    capture = _open(path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Image {index} illisible dans {path}")
        return frame
    finally:
        capture.release()


def iter_frames(
    path: Path, start_frame: int = 0, end_frame: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Itère sur (index, image BGR) de `start_frame` inclus à `end_frame` exclu."""
    capture = _open(path)
    try:
        if start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        index = start_frame
        while end_frame is None or index < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        capture.release()
