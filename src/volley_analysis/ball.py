"""Détection du ballon avec VballNet (TrackNetV4 entraîné sur le volley), au format ONNX.

Modèles : https://github.com/asigatchov/fast-volleyball-tracking-inference (licence MIT),
clonés dans `vendor/`. Le réseau prend une séquence d'images consécutives en niveaux de
gris et produit une carte de chaleur par image : c'est le mouvement sur la séquence qui
permet de distinguer le ballon des objets ronds immobiles (tirets au sol, etc.).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from volley_analysis.video import iter_frames, probe

VENDOR_MODELS = Path("vendor/fast-volleyball-tracking-inference/models")
DEFAULT_MODEL = VENDOR_MODELS / "VballNetV4c_seq9_grayscale_20260908_213829.onnx"
BALL_COLUMNS = ["frame", "time_s", "visible", "x", "y", "radius", "conf"]


@dataclass
class BallDetectorConfig:
    model_path: Path = DEFAULT_MODEL
    threshold: float = 0.5
    # Le modèle a été entraîné sur des vidéos à ~30 img/s : pour une vidéo à 60 img/s,
    # ne garder qu'une image sur deux redonne au ballon le déplacement attendu.
    frame_step: int | None = None
    use_coreml: bool = False


@dataclass(frozen=True)
class BallDetection:
    x: float
    y: float
    radius: float
    conf: float


class BallDetector:
    def __init__(self, config: BallDetectorConfig):
        import onnxruntime as ort

        if not Path(config.model_path).exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {config.model_path}. Clonez "
                "https://github.com/asigatchov/fast-volleyball-tracking-inference dans vendor/"
            )
        providers = ["CPUExecutionProvider"]
        if config.use_coreml:
            providers.insert(0, "CoreMLExecutionProvider")
        self.config = config
        self.session = ort.InferenceSession(str(config.model_path), providers=providers)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        _, self.seq, self.height, self.width = model_input.shape
        output_channels = self.session.get_outputs()[0].shape[1]
        self.has_radius = output_channels == 2 * self.seq

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (self.width, self.height)).astype(np.float32) / 255.0

    def detect(self, inputs: list[np.ndarray], frame_size: tuple[int, int]) -> list:
        """Détecte le ballon sur une séquence d'images prétraitées (au plus `seq`).

        Renvoie une `BallDetection` (en pixels de la vidéo) ou None pour chaque image.
        """
        count = len(inputs)
        padded = list(inputs) + [inputs[-1]] * (self.seq - count)
        output = self.session.run(None, {self.input_name: np.stack(padded)[None]})[0][0]
        scale_x, scale_y = frame_size[0] / self.width, frame_size[1] / self.height
        detections = []
        for i in range(count):
            radius_map = output[self.seq + i] if self.has_radius else None
            detections.append(
                _decode_heatmap(
                    output[i], radius_map, self.config.threshold, scale_x, scale_y, frame_size[0]
                )
            )
        return detections


def detect_ball(
    video_path: Path,
    config: BallDetectorConfig,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> pd.DataFrame:
    """Une ligne par image analysée : position du ballon en pixels, ou `visible=False`."""
    info = probe(video_path)
    step = config.frame_step or max(int(round(info.fps / 30)), 1)
    detector = BallDetector(config)
    start_frame = int(start_s * info.fps)
    end_frame = min(int(end_s * info.fps), info.frame_count) if end_s else info.frame_count

    rows = []
    frames = iter_frames(video_path, start_frame, end_frame)
    total = (end_frame - start_frame) // step
    for chunk in tqdm(
        _chunks(frames, step, detector.seq), total=total // detector.seq, desc="Ballon"
    ):
        indices = [index for index, _ in chunk]
        inputs = [detector.preprocess(frame) for _, frame in chunk]
        for index, det in zip(
            indices, detector.detect(inputs, (info.width, info.height)), strict=True
        ):
            rows.append(
                (index, index / info.fps, det is not None)
                + ((det.x, det.y, det.radius, det.conf) if det else (np.nan,) * 4)
            )
    return pd.DataFrame(rows, columns=BALL_COLUMNS)


def _chunks(
    frames: Iterator[tuple[int, np.ndarray]], step: int, size: int
) -> Iterator[list[tuple[int, np.ndarray]]]:
    chunk = []
    for position, item in enumerate(frames):
        if position % step:
            continue
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _decode_heatmap(
    heatmap: np.ndarray,
    radius_map: np.ndarray | None,
    threshold: float,
    scale_x: float,
    scale_y: float,
    frame_width: int,
) -> BallDetection | None:
    binary = (heatmap > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        x, y = contour[0, 0].astype(float)
    else:
        x, y = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    col, row = int(round(x)), int(round(y))
    row, col = min(row, heatmap.shape[0] - 1), min(col, heatmap.shape[1] - 1)
    radius = float(radius_map[row, col]) * frame_width if radius_map is not None else np.nan
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [contour], -1, 1, -1)
    return BallDetection(
        x=x * scale_x,
        y=y * scale_y,
        radius=radius,
        conf=float(heatmap[mask.astype(bool)].max()),
    )
