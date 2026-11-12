"""Chaîne de traitement : suivi des joueurs (étape 1), du ballon (étape 2) et rendus."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from volley_analysis.ball import BallDetectorConfig, detect_ball
from volley_analysis.ball_tracks import ball_positions_3d, clean_ball_track
from volley_analysis.calibration import Calibration
from volley_analysis.corrections import Corrections, load_corrections
from volley_analysis.court import FAR, NEAR
from volley_analysis.players import (
    FrameTracks,
    PlayerTracker,
    TrackerConfig,
    near_server_in_view,
)
from volley_analysis.rallies import detect_rallies
from volley_analysis.referee import referee_call, referee_gestures, track_referee
from volley_analysis.render import (
    BALL_COLOR,
    CourtCanvas,
    contact_sheet,
    draw_ball,
    draw_caption,
    draw_players,
    draw_scoreboard,
    paste_inset,
)
from volley_analysis.scoring import (
    STANDARD_RULES,
    ScoreTimeline,
    SetRules,
    flip,
    score_rallies,
)
from volley_analysis.teams import CORBAS, PersonDetector, side_colors
from volley_analysis.tracks import clean_tracks
from volley_analysis.video import iter_frames, probe, read_frame

TRACKS_FILE = "players.parquet"
BALL_RAW_FILE = "ball_raw.parquet"
BALL_FILE = "ball.parquet"
VIDEO_FILE = "annotated.mp4"
POINTS_FILE = "points.csv"
BALL_TRAIL_S = 0.4
BALL_COVERAGE_MARGIN_S = 2.0
# Les joueurs sont en place une seconde avant le service ; le serveur, juste avant la frappe.
TEAM_CHECK_OFFSET_S = 1.0
SERVER_CHECK_OFFSETS_S = (1.0, 0.4)
REVIEW_BEFORE_S = 0.4
REVIEW_AFTER_S = 1.2
REVIEW_TILE_SIZE = (640, 360)
REVIEW_COLUMNS = 3
REFEREE_FILE = "referee_pose.parquet"
# La fin détectée d'un échange est souvent en retard (ballon encore suivi après le point).
REFEREE_AFTER_SERVE_S = 2.0
REFEREE_BEFORE_END_S = 2.0
REFEREE_BEFORE_NEXT_SERVE_S = 0.5
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
    tracks_path: Path | None
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
    video_path: Path,
    output_dir: Path,
    render_video: bool = True,
    score: ScoreTimeline | None = None,
) -> TrackingSummary:
    """Ré-applique les nettoyages et régénère heatmaps et vidéo sans relancer les détections.

    Les joueurs sont inclus s'ils ont été suivis (`volley track`), le ballon
    (`volley ball`) et le score (`volley score`) de même.
    """
    info = probe(video_path)
    df = None
    start_frame, end_frame = 0, info.frame_count
    if (output_dir / TRACKS_FILE).exists():
        df = clean_tracks(pd.read_parquet(output_dir / TRACKS_FILE)[RAW_COLUMNS], info.fps)
        df.to_parquet(output_dir / TRACKS_FILE, index=False)
        start_frame, end_frame = int(df["frame"].min()), int(df["frame"].max()) + 1
    ball = None
    if (output_dir / BALL_RAW_FILE).exists():
        ball = clean_ball_track(pd.read_parquet(output_dir / BALL_RAW_FILE), info.fps)
        ball.to_parquet(output_dir / BALL_FILE, index=False)
    if df is None and ball is None:
        raise FileNotFoundError(f"Ni joueurs ni ballon suivis dans {output_dir}")
    return _build_outputs(
        video_path, df, output_dir, start_frame, end_frame, render_video, ball, score
    )


def _build_outputs(
    video_path: Path,
    df: pd.DataFrame | None,
    output_dir: Path,
    start_frame: int,
    end_frame: int,
    render_video: bool,
    ball: pd.DataFrame | None = None,
    score: ScoreTimeline | None = None,
) -> TrackingSummary:
    video_out = output_dir / VIDEO_FILE if render_video else None
    if video_out:
        render_annotated_video(video_path, video_out, start_frame, end_frame, df, ball, score)
    if df is None:
        return TrackingSummary(end_frame - start_frame, None, video_out, {}, {})

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


def ensure_ball_track(
    video_path: Path, output_dir: Path, config: BallDetectorConfig
) -> pd.DataFrame:
    """Trajectoire du ballon sur toute la vidéo.

    Les détections brutes ne sont calculées que si elles manquent ; le nettoyage,
    rapide, est refait pour profiter des dernières règles.
    """
    raw_path = output_dir / BALL_RAW_FILE
    info = probe(video_path)
    if raw_path.exists():
        raw = pd.read_parquet(raw_path)
        margin = BALL_COVERAGE_MARGIN_S * info.fps
        if raw["frame"].min() <= margin and raw["frame"].max() >= info.frame_count - margin:
            track = clean_ball_track(raw, info.fps)
            track.to_parquet(output_dir / BALL_FILE, index=False)
            return track
    track_ball(video_path, output_dir, config, render_video=False)
    return pd.read_parquet(output_dir / BALL_FILE)


def ensure_referee_gestures(video_path: Path, output_dir: Path) -> pd.DataFrame:
    """Gestes de l'arbitre ; sa pose, longue à calculer, est gardée dans `output_dir`."""
    pose_path = output_dir / REFEREE_FILE
    if pose_path.exists():
        pose = pd.read_parquet(pose_path)
    else:
        pose = track_referee(video_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        pose.to_parquet(pose_path, index=False)
    return referee_gestures(pose)


def detect_video_rallies(
    video_path: Path,
    calibration: Calibration,
    ball: pd.DataFrame,
    persons: PersonDetector,
    corrections: Corrections | None = None,
    gestures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Échanges d'une vidéo, avec le côté de Corbas observé juste avant chaque service.

    Avec `gestures` (`referee.referee_gestures`), `referee_side` donne le camp
    désigné par l'arbitre à la fin de chaque échange, quand il a fait un geste net.
    """
    corrections = corrections or Corrections()
    info = probe(video_path)

    def frame_at(time_s: float) -> np.ndarray:
        return read_frame(video_path, min(max(int(time_s * info.fps), 0), info.frame_count - 1))

    def server_visible(side: str, serve_s: float) -> bool:
        if side != NEAR:
            return False
        return any(
            near_server_in_view(persons.boxes(frame_at(serve_s - offset)), info.height)
            for offset in SERVER_CHECK_OFFSETS_S
        )

    rallies = detect_rallies(
        ball_positions_3d(ball, calibration.camera()),
        server_visible,
        corrections.ignored_s,
        [(time_s, None) for time_s, _ in corrections.serves],
        corrections.ignored_periods,
    )
    corbas_sides = []
    for serve_s in rallies["serve_s"]:
        frame = frame_at(serve_s - TEAM_CHECK_OFFSET_S)
        corbas_sides.append(side_colors(frame, persons.boxes(frame), calibration).corbas_side())
    rallies["corbas_side"] = corbas_sides

    serving_teams = dict(corrections.serves)
    nearby_corbas_side = rallies["corbas_side"].ffill().bfill().fillna(NEAR)
    for i in rallies.index[rallies["manual"]]:
        side = nearby_corbas_side[i]
        team = serving_teams[rallies.at[i, "serve_s"]]
        rallies.at[i, "serving_side"] = side if team == CORBAS else flip(side)

    def rally_at(time_s: float) -> int:
        return int(np.searchsorted(rallies["serve_s"].to_numpy(), time_s, side="right")) - 1

    rallies["winner_override"] = None
    for time_s, team in corrections.points:
        if rally_at(time_s) >= 0:
            rallies.at[rally_at(time_s), "winner_override"] = team
    rallies["set_end"] = False
    for time_s in corrections.set_ends_s:
        if rally_at(time_s) >= 0:
            rallies.at[rally_at(time_s), "set_end"] = True

    if gestures is not None:
        next_serves = [*rallies["serve_s"].iloc[1:], info.duration_s]
        rallies["referee_side"] = [
            referee_call(
                gestures,
                max(rally.serve_s + REFEREE_AFTER_SERVE_S, rally.end_s - REFEREE_BEFORE_END_S),
                next_serve - REFEREE_BEFORE_NEXT_SERVE_S,
            )
            for rally, next_serve in zip(rallies.itertuples(), next_serves, strict=True)
        ]
    return rallies


@dataclass
class ScoreSummary:
    points_path: Path
    review_paths: list[Path]
    scored: pd.DataFrame


def score_match(
    videos: list[Path],
    calibration: Calibration,
    output_dir: Path,
    ball_config: BallDetectorConfig,
    rules: SetRules = STANDARD_RULES,
    start_score: tuple[int, int] = (0, 0),
    videos_root: Path = Path("outputs"),
    corrections_path: Path | None = None,
) -> ScoreSummary:
    """Compte les points sur des vidéos consécutives d'un même match, dans l'ordre donné.

    La trajectoire du ballon de chaque vidéo est rangée dans `videos_root/<video>/`.
    Les corrections manuelles (`corrections.load_corrections`) sont appliquées aux
    services détectés.
    """
    persons = PersonDetector()
    corrections = load_corrections(corrections_path)
    balls, per_video = {}, []
    for video in tqdm(videos, unit="vidéo", desc="Échanges"):
        balls[video] = ensure_ball_track(video, videos_root / video.stem, ball_config)
        video_corrections = corrections.get(video.stem, Corrections())
        gestures = (
            None
            if video_corrections.ignore_referee
            else ensure_referee_gestures(video, videos_root / video.stem)
        )
        rallies = detect_video_rallies(
            video, calibration, balls[video], persons, video_corrections, gestures
        )
        rallies.insert(0, "video", video.stem)
        per_video.append(rallies)

    scored = score_rallies(pd.concat(per_video, ignore_index=True), rules, start_score)
    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = output_dir / POINTS_FILE
    scored.to_csv(points_path, index=False, float_format="%.2f")

    review_paths = []
    for video in videos:
        rows = scored[scored["video"] == video.stem]
        if len(rows) == 0:
            continue
        path = output_dir / f"services_{video.stem}.jpg"
        cv2.imwrite(str(path), serve_review_sheet(video, rows, balls[video]))
        review_paths.append(path)
    return ScoreSummary(points_path, review_paths, scored)


def serve_review_sheet(video_path: Path, scored: pd.DataFrame, ball: pd.DataFrame) -> np.ndarray:
    """Planche de contrôle : une vignette par service, avec la trajectoire du ballon."""
    info = probe(video_path)
    known = ball[ball["x"].notna()]
    tiles = []
    for row in scored.itertuples():
        frame = read_frame(video_path, max(int((row.serve_s - REVIEW_BEFORE_S) * info.fps), 0))
        flight = known[
            (known["time_s"] >= row.serve_s - REVIEW_BEFORE_S)
            & (known["time_s"] <= row.serve_s + REVIEW_AFTER_S)
        ]
        for x, y in flight[["x", "y"]].to_numpy():
            cv2.circle(frame, (int(x), int(y)), 7, BALL_COLOR, -1, cv2.LINE_AA)
        winner = row.winner if isinstance(row.winner, str) else "?"
        minutes, seconds = divmod(row.serve_s, 60)
        tile = cv2.resize(frame, REVIEW_TILE_SIZE, interpolation=cv2.INTER_AREA)
        draw_caption(
            tile,
            [
                f"#{row.Index} {int(minutes)}:{seconds:04.1f}  service {row.serving_team}"
                f" ({row.serving_side})" + (" ajoute a la main" if row.manual else ""),
                f"point {winner}  set {row.set}  {row.corbas}-{row.adversaire}",
            ],
            scale=0.6,
        )
        tiles.append(tile)
    return contact_sheet(tiles, REVIEW_COLUMNS)


def render_annotated_video(
    video_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    players: pd.DataFrame | None = None,
    ball: pd.DataFrame | None = None,
    score: ScoreTimeline | None = None,
) -> None:
    """Vidéo annotée : joueurs (boîtes + mini-carte), ballon (position + traînée), score."""
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
            state = score.at(index / info.fps) if score is not None else None
            if state is not None:
                out = draw_scoreboard(
                    out,
                    state.set_number,
                    state.corbas,
                    state.opponent,
                    state.rally,
                    state.serving_team,
                    state.point_to,
                    state.referee,
                )
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
