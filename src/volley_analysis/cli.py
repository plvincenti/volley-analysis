"""Interface en ligne de commande : `volley --help`."""

from pathlib import Path
from typing import Annotated

import cv2
import typer

from volley_analysis.calibration import Calibration
from volley_analysis.video import probe, read_frame

app = typer.Typer(help="Analyse vidéo de matchs de volley-ball.", no_args_is_help=True)

CALIBRATIONS_DIR = Path("data/calibrations")
SHARED_CALIBRATION = CALIBRATIONS_DIR / "default.json"
OUTPUTS_DIR = Path("outputs")

VideoArg = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Fichier vidéo")]
CalibrationOpt = Annotated[
    Path | None,
    typer.Option(
        "--calibration",
        "-c",
        help="Fichier de calibration (défaut : data/calibrations/<video>.json, "
        "sinon data/calibrations/default.json)",
    ),
]


def _video_calibration(video: Path) -> Path:
    return CALIBRATIONS_DIR / f"{video.stem}.json"


def _load_calibration(video: Path, path: Path | None) -> Calibration:
    if path is None:
        path = _video_calibration(video)
        if not path.exists():
            path = SHARED_CALIBRATION
    if not path.exists():
        typer.secho(
            f"Calibration introuvable : {path}. Lancez d'abord `volley calibrate {video}`"
            " ou `volley calibrate-drawing`.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    return Calibration.load(path)


@app.command()
def info(video: VideoArg) -> None:
    """Affiche les caractéristiques de la vidéo."""
    meta = probe(video)
    minutes, seconds = divmod(int(meta.duration_s), 60)
    typer.echo(
        f"{video.name} : {meta.width}x{meta.height}, {meta.fps:.2f} img/s, "
        f"{meta.frame_count} images, {minutes} min {seconds:02d} s"
    )


@app.command()
def calibrate(
    video: VideoArg,
    at: Annotated[float, typer.Option(help="Instant (s) de l'image utilisée pour calibrer")] = 5.0,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Calibre le terrain en cliquant ses lignes sur une image de la vidéo."""
    from volley_analysis.calibration_ui import CalibrationAborted, run_calibration_ui

    meta = probe(video)
    frame_index = min(int(at * meta.fps), meta.frame_count - 1)
    frame = read_frame(video, frame_index)
    try:
        points = run_calibration_ui(frame)
    except CalibrationAborted as exc:
        typer.secho(f"Calibration abandonnée. {exc}", fg=typer.colors.YELLOW)
        raise typer.Exit(1) from exc

    calibration = Calibration(
        image_points=points,
        frame_size=(meta.width, meta.height),
        frame_index=frame_index,
        video=str(video),
    )
    output = output or _video_calibration(video)
    calibration.save(output)
    typer.secho(
        f"Calibration enregistrée : {output} ({len(points)} points, erreur de "
        f"reprojection {calibration.reprojection_error():.1f} px)",
        fg=typer.colors.GREEN,
    )


@app.command("calibrate-drawing")
def calibrate_drawing(
    screenshot: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, help="Capture avec le cadre du terrain tracé en rouge"
        ),
    ],
    video: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Vidéo dont provient la capture")
    ],
    at: Annotated[float, typer.Option(help="Instant (s) de l'image servant au recalage")] = 5.0,
    output: Annotated[Path, typer.Option("--output", "-o")] = SHARED_CALIBRATION,
) -> None:
    """Calibre à partir des 4 lignes du terrain (fonds et côtés) dessinées sur une capture."""
    from volley_analysis.calibration_drawing import calibration_from_drawing
    from volley_analysis.render import draw_court_overlay

    meta = probe(video)
    frame_index = min(int(at * meta.fps), meta.frame_count - 1)
    frame = read_frame(video, frame_index)
    image = cv2.imread(str(screenshot))
    calibration, lines = calibration_from_drawing(image, frame, frame_index, str(video))
    calibration.save(output)

    preview_path = output.with_suffix(".jpg")
    cv2.imwrite(str(preview_path), draw_court_overlay(frame, calibration))
    typer.secho(f"Calibration enregistrée : {output}", fg=typer.colors.GREEN)
    for name, line in lines.items():
        residual = line.distances(line.points)
        typer.echo(f"  {name:9s} écart moyen du tracé à la droite : {residual.mean():.1f} px")
    typer.echo(f"Image de contrôle : {preview_path}")


@app.command()
def preview(
    video: VideoArg,
    calibration: CalibrationOpt = None,
    at: Annotated[float, typer.Option(help="Instant (s) de l'image à contrôler")] = 60.0,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Enregistre une image avec les lignes du terrain reprojetées, pour vérifier la calibration."""
    from volley_analysis.render import draw_court_overlay

    calib = _load_calibration(video, calibration)
    meta = probe(video)
    frame = read_frame(video, min(int(at * meta.fps), meta.frame_count - 1))
    output = output or OUTPUTS_DIR / video.stem / "calibration_preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), draw_court_overlay(frame, calib))
    typer.echo(f"Image de contrôle : {output}")


@app.command()
def track(
    video: VideoArg,
    calibration: CalibrationOpt = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", "-o")] = None,
    start: Annotated[float, typer.Option(help="Début (s)")] = 0.0,
    end: Annotated[float | None, typer.Option(help="Fin (s), défaut : fin de la vidéo")] = None,
    model: Annotated[
        str, typer.Option(help="Poids YOLO (téléchargés automatiquement)")
    ] = "yolo11m.pt",
    imgsz: Annotated[int, typer.Option(help="Résolution d'inférence")] = 1280,
    conf: Annotated[float, typer.Option(help="Seuil de confiance")] = 0.25,
    device: Annotated[str | None, typer.Option(help="mps, cpu, cuda:0 (défaut : auto)")] = None,
    no_video: Annotated[
        bool, typer.Option("--no-video", help="Ne pas générer la vidéo annotée")
    ] = False,
) -> None:
    """Étape 1 : détecte et suit les joueurs, exporte leurs positions et des heatmaps."""
    from volley_analysis.pipeline import track_players
    from volley_analysis.players import TrackerConfig

    calib = _load_calibration(video, calibration)
    config = TrackerConfig(model=model, imgsz=imgsz, conf=conf, device=device)
    summary = track_players(
        video,
        calib,
        output_dir or OUTPUTS_DIR / video.stem,
        config,
        start_s=start,
        end_s=end,
        render_video=not no_video,
    )

    _print_summary(summary)


@app.command()
def ball(
    video: VideoArg,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", "-o")] = None,
    start: Annotated[float, typer.Option(help="Début (s)")] = 0.0,
    end: Annotated[float | None, typer.Option(help="Fin (s), défaut : fin de la vidéo")] = None,
    model: Annotated[Path | None, typer.Option(help="Modèle VballNet ONNX")] = None,
    step: Annotated[
        int | None,
        typer.Option(help="Analyser une image sur N (défaut : ramène la vidéo à ~30 img/s)"),
    ] = None,
    threshold: Annotated[float, typer.Option(help="Seuil de la carte de chaleur")] = 0.5,
    no_video: Annotated[
        bool, typer.Option("--no-video", help="Ne pas générer la vidéo annotée")
    ] = False,
) -> None:
    """Étape 2 : détecte et suit le ballon (VballNet). Inclut les joueurs s'ils sont suivis."""
    from volley_analysis.ball import DEFAULT_MODEL, BallDetectorConfig
    from volley_analysis.pipeline import track_ball

    config = BallDetectorConfig(
        model_path=model or DEFAULT_MODEL, threshold=threshold, frame_step=step
    )
    summary = track_ball(
        video,
        output_dir or OUTPUTS_DIR / video.stem,
        config,
        start_s=start,
        end_s=end,
        render_video=not no_video,
    )
    typer.secho(f"{summary.frames} images couvertes", fg=typer.colors.GREEN)
    typer.echo(
        f"Ballon détecté sur {summary.visible_fraction:.0%} des images analysées, "
        f"position connue (avec interpolation) sur {summary.known_fraction:.0%}"
    )
    typer.echo(f"Trajectoire : {summary.ball_path}")
    if summary.video_path:
        typer.echo(f"Vidéo annotée : {summary.video_path}")


@app.command()
def render(
    video: VideoArg,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", "-o")] = None,
    no_video: Annotated[
        bool, typer.Option("--no-video", help="Ne pas générer la vidéo annotée")
    ] = False,
) -> None:
    """Ré-applique le nettoyage des pistes et régénère les rendus, sans relancer la détection."""
    from volley_analysis.pipeline import rebuild_outputs

    _print_summary(
        rebuild_outputs(video, output_dir or OUTPUTS_DIR / video.stem, render_video=not no_video)
    )


def _print_summary(summary) -> None:
    typer.secho(f"{summary.frames} images traitées", fg=typer.colors.GREEN)
    typer.echo(f"Positions : {summary.tracks_path}")
    if summary.video_path:
        typer.echo(f"Vidéo annotée : {summary.video_path}")
    for name, path in summary.heatmap_paths.items():
        typer.echo(f"Heatmap {name} : {path}")
    typer.echo(
        "Joueurs détectés en moyenne par image : "
        + ", ".join(f"{side} {n:.1f}" for side, n in summary.mean_players.items())
    )


if __name__ == "__main__":
    app()
