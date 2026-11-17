"""API de revue d'un match, qui sert aussi les vidéos et l'interface (`web/dist`)."""

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from volley_analysis.actions import ACTIONS, RESULTS, action_summary
from volley_analysis.teams import CORBAS, OPPONENT
from volley_analysis.web.store import STATUSES, Store

Team = Literal[CORBAS, OPPONENT]
Action = Literal[tuple(ACTIONS)]
Result = Literal[tuple(RESULTS)]
Status = Literal[tuple(STATUSES)]


class RallyChanges(BaseModel, extra="forbid"):
    winner: Team | None = None
    status: Status | None = None
    note: str | None = None
    serve_s: float | None = None
    end_s: float | None = None
    serving_team: Team | None = None


class Split(BaseModel, extra="forbid"):
    time_s: float


class NewGesture(BaseModel, extra="forbid"):
    time_s: float
    team: Team = CORBAS
    action: Action
    result: Result


class GestureChanges(BaseModel, extra="forbid"):
    time_s: float | None = None
    team: Team | None = None
    action: Action | None = None
    result: Result | None = None


def create_app(store: Store, static_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Volley analysis")

    def found(value):
        if value is None:
            raise HTTPException(404)
        return value

    @app.get("/api/match")
    def match():
        return {
            "videos": store.videos(),
            "rallies": store.rallies(),
            "actions": ACTIONS,
            "results": RESULTS,
            "teams": [CORBAS, OPPONENT],
        }

    @app.get("/api/rallies/{rally_id}")
    def rally(rally_id: int):
        return found(store.rally(rally_id))

    def checked(edit):
        try:
            return found(edit())
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.patch("/api/rallies/{rally_id}")
    def update_rally(rally_id: int, changes: RallyChanges):
        return checked(lambda: store.update_rally(rally_id, changes.model_dump(exclude_unset=True)))

    @app.post("/api/rallies/{rally_id}/split")
    def split_rally(rally_id: int, split: Split):
        return checked(lambda: store.split_rally(rally_id, split.time_s))

    @app.post("/api/rallies/{rally_id}/gestures")
    def add_gesture(rally_id: int, gesture: NewGesture):
        return found(store.add_gesture(rally_id, gesture.model_dump()))

    @app.patch("/api/gestures/{gesture_id}")
    def update_gesture(gesture_id: int, changes: GestureChanges):
        return found(store.update_gesture(gesture_id, changes.model_dump(exclude_unset=True)))

    @app.delete("/api/gestures/{gesture_id}")
    def delete_gesture(gesture_id: int):
        return found(store.delete_gesture(gesture_id))

    @app.get("/api/stats")
    def stats():
        summary = action_summary(store.team_actions(CORBAS)).reset_index()
        summary["reussite_pct"] = summary["reussite_pct"].astype(object)
        summary = summary.where(summary.notna(), None)
        return summary.to_dict(orient="records")

    @app.get("/media/{name}")
    def media(name: str):
        path = found(store.video_path(name))
        if not path.exists():
            raise HTTPException(404, f"Vidéo introuvable : {path}")
        return FileResponse(path, media_type="video/mp4")

    if static_dir is not None and static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="interface")
    return app
