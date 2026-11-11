"""Chaîne de traitement : suivi des joueurs (étape 1), du ballon (étape 2) et rendus."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from volley_analysis.ball import BallDetectorConfig, detect_ball
from volley_analysis.ball_tracks import clean_ball_track
from volley_analysis.calibration import Calibration
from volley_analysis.court import FAR, NEAR
from volley_analysis.players import FrameTracks, PlayerTracker, TrackerConfig
from volley_analysis.render import CourtCanvas, draw_ball, draw_players, paste_inset
from volley_analysis.tracks import clean_tracks
from volley_analysis.video import iter_frames, probe

TRACKS_FILE = "players.parquet"
BALL_RAW_FILE = "ball_raw.parquet"
BALL_FILE = "ball.parquet"
VIDEO_FILE = "annotated.mp4"
BALL_TRAIL_S = 0.4
RAW_COLUMNS = [
    "frame",
    "time_s",
    "track_id",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "court_x",
    "court_y",
    "in_play",
]


@dataclass
class TrackingSummary:
    frames: int
    tracks_path: Path
    video_path: Path | None
    heatmap_paths: dict[str, Path]
    mean_players: dict[str, float]


@dataclass
class BallSummary:
    frames: int
    ball_path: Path
    video_path: Path | None
    visible_fraction: float
    known_fraction: float


def track_players(
    video_path: Path,
    calibration: Calibration,
    output_dir: Path,
    config: TrackerConfig,
    start_s: float = 0.0,
    end_s: float | None = None,
    render_video: bool = True,
) -> TrackingSummary:
    info = probe(video_path)
    if (info.width, info.height) != tuple(calibration.frame_size):
        raise ValueError(
            f"La calibration a été faite en {calibration.frame_size[0]}x{calibration.frame_size[1]}"
            f" mais la vidéo est en {info.width}x{info.height}"
        )

    start_frame = int(start_s * info.fps)
    end_frame = min(int(end_s * info.fps), info.frame_count) if end_s else info.frame_count
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = PlayerTracker(config, calibration)
    rows = _TrackAccumulator()
    frames = iter_frames(video_path, start_frame, end_frame)
    for index, frame in tqdm(frames, total=end_frame - start_frame, unit="img", desc="Suivi"):
        rows.add(index, index / info.fps, tracker.update(frame))

    df = clean_tracks(rows.to_dataframe(), info.fps)
    df.to_parquet(output_dir / TRACKS_FILE, index=False)
    return _build_outputs(video_path, df, output_dir, start_frame, end_frame, render_video)


def rebuild_outputs(
    video_path: Path, output_dir: Path, render_video: bool = True
) -> TrackingSummary:
    """Ré-applique les nettoyages et régénère heatmaps et vidéo sans relancer les détections.

    Le ballon est inclus dans la vidéo s'il a été suivi (`volley ball`).
    """
    fps = probe(video_path).fps
    df = pd.read_parquet(output_dir / TRACKS_FILE)[RAW_COLUMNS]
    df = clean_tracks(df, fps)
    df.to_parquet(output_dir / TRACKS_FILE, index=False)
    ball = None
    if (output_dir / BALL_RAW_FILE).exists():
        ball = clean_ball_track(pd.read_parquet(output_dir / BALL_RAW_FILE), fps)
        ball.to_parquet(output_dir / BALL_FILE, index=False)
    start_frame, end_frame = int(df["frame"].min()), int(df["frame"].max()) + 1
    return _build_outputs(video_path, df, output_dir, start_frame, end_frame, render_video, ball)


def _build_outputs(
    video_path: Path,
    df: pd.DataFrame,
    output_dir: Path,
    start_frame: int,
    end_frame: int,
    render_video: bool,
    ball: pd.DataFrame | None = None,
) -> TrackingSummary:
    video_out = output_dir / VIDEO_FILE if render_video else None
    if video_out:
        render_annotated_video(video_path, video_out, start_frame, end_frame, df, ball)

    frame_count = max(end_frame - start_frame, 1)
    players = df[df["is_player"]]
    return TrackingSummary(
        frames=end_frame - start_frame,
        tracks_path=output_dir / TRACKS_FILE,
        video_path=video_out,
        heatmap_paths=save_heatmaps(df, output_dir),
        mean_players={
            side: float((players["side"] == side).sum() / frame_count) for side in (NEAR, FAR)
        },
    )


def save_heatmaps(df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    canvas = CourtCanvas()
    players = df[df["is_player"]]
    paths = {}
    for name, subset in (
        ("all", players),
        (NEAR, players[players["side"] == NEAR]),
        (FAR, players[players["side"] == FAR]),
    ):
        path = output_dir / f"heatmap_{name}.png"
        cv2.imwrite(str(path), canvas.heatmap(subset[["court_x", "court_y"]].to_numpy()))
        paths[name] = path
    return paths


def track_ball(
    video_path: Path,
    output_dir: Path,
    config: BallDetectorConfig,
    start_s: float = 0.0,
    end_s: float | None = None,
    render_video: bool = True,
) -> BallSummary:
    info = probe(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = detect_ball(video_path, config, start_s, end_s)
    raw.to_parquet(output_dir / BALL_RAW_FILE, index=False)
    track = clean_ball_track(raw, info.fps)
    track.to_parquet(output_dir / BALL_FILE, index=False)

    video_out = output_dir / VIDEO_FILE if render_video else None
    if video_out:
        start_frame, end_frame = int(track["frame"].min()), int(track["frame"].max()) + 1
        players_path = output_dir / TRACKS_FILE
        players = pd.read_parquet(players_path) if players_path.exists() else None
        render_annotated_video(video_path, video_out, start_frame, end_frame, players, track)
    return BallSummary(
        frames=len(track),
        ball_path=output_dir / BALL_FILE,
        video_path=video_out,
        visible_fraction=float(raw["visible"].mean()),
        known_fraction=float(track["x"].notna().mean()),
    )


def render_annotated_video(
    video_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    players: pd.DataFrame | None = None,
    ball: pd.DataFrame | None = None,
) -> None:
    """Vidéo annotée : joueurs (boîtes + mini-carte) et/ou ballon (position + traînée)."""
    info = probe(video_path)
    canvas = CourtCanvas()
    if players is not None:
        players = players[players["is_player"]]
        rows_by_frame = players.groupby("frame").indices
        xyxy = players[["x1", "y1", "x2", "y2"]].to_numpy()
        court_xy = players[["court_x", "court_y"]].to_numpy()
        track_ids = players["track_id"].to_numpy()
        sides = players["side"].to_numpy()
    no_rows = np.empty(0, dtype=np.int64)
    if ball is not None:
        ball = ball.set_index("frame").reindex(range(start_frame, end_frame))
        ball_xy = ball[["x", "y"]].to_numpy()
        ball_radius = ball["radius"].to_numpy()
        ball_detected = (ball["source"] == "detected").to_numpy()
        trail = max(int(BALL_TRAIL_S * info.fps), 1)

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height)
    )
    try:
        frames = iter_frames(video_path, start_frame, end_frame)
        for index, frame in tqdm(frames, total=end_frame - start_frame, unit="img", desc="Rendu"):
            out = frame
            if players is not None:
                rows = rows_by_frame.get(index, no_rows)
                out = draw_players(out, xyxy[rows], track_ids[rows], sides[rows])
            if ball is not None:
                i = index - start_frame
                out = draw_ball(
                    out, ball_xy[max(i - trail, 0) : i + 1], ball_radius[i], ball_detected[i]
                )
            if players is not None:
                minimap = canvas.draw_positions(court_xy[rows], sides[rows], track_ids[rows])
                out = paste_inset(out, minimap)
            writer.write(out)
    finally:
        writer.release()


class _TrackAccumulator:
    """Concatène les détections de toutes les images sans créer un DataFrame par image."""

    def __init__(self) -> None:
        self._chunks: list[dict[str, np.ndarray]] = []

    def add(self, frame_index: int, time_s: float, tracks: FrameTracks) -> None:
        n = len(tracks)
        if n == 0:
            return
        self._chunks.append(
            {
                "frame": np.full(n, frame_index, dtype=np.int64),
                "time_s": np.full(n, time_s, dtype=np.float64),
                "track_id": tracks.track_ids,
                "conf": tracks.confidences,
                "x1": tracks.xyxy[:, 0],
                "y1": tracks.xyxy[:, 1],
                "x2": tracks.xyxy[:, 2],
                "y2": tracks.xyxy[:, 3],
                "court_x": tracks.court_xy[:, 0],
                "court_y": tracks.court_xy[:, 1],
                "in_play": tracks.in_play,
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        if not self._chunks:
            return pd.DataFrame({c: pd.Series(dtype=float) for c in RAW_COLUMNS}).astype(
                {"frame": np.int64, "track_id": np.int64, "in_play": bool}
            )
        return pd.DataFrame(
            {c: np.concatenate([chunk[c] for chunk in self._chunks]) for c in RAW_COLUMNS}
        )
