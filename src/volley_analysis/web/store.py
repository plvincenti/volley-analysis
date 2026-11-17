"""Match en cours de revue : vidéos, échanges et gestes, dans une base SQLite.

L'analyse automatique (`volley score`) est importée une fois ; ensuite l'utilisateur
corrige le vainqueur de chaque point et les gestes, puis valide l'échange. Changer le
vainqueur recalcule le résultat des gestes qu'il n'a pas modifiés.
"""

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from string import ascii_lowercase

import cv2
import pandas as pd

from volley_analysis.actions import MANUAL, SERVE, UNKNOWN, UNTOUCHED, gesture_results
from volley_analysis.teams import CORBAS, OPPONENT

TEAMS = (CORBAS, OPPONENT)
TO_REVIEW, VALIDATED, DELETED = "a_revoir", "valide", "supprime"
STATUSES = (TO_REVIEW, VALIDATED, DELETED)

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    name TEXT PRIMARY KEY,
    position INTEGER NOT NULL,
    path TEXT NOT NULL,
    fps REAL NOT NULL,
    duration_s REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rallies (
    id INTEGER PRIMARY KEY,
    label TEXT NOT NULL DEFAULT '',
    video TEXT NOT NULL REFERENCES videos(name),
    set_number INTEGER NOT NULL,
    serve_s REAL NOT NULL,
    end_s REAL NOT NULL,
    serving_team TEXT NOT NULL,
    auto_winner TEXT,
    winner TEXT,
    status TEXT NOT NULL DEFAULT 'a_revoir',
    note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS gestures (
    id INTEGER PRIMARY KEY,
    rally_id INTEGER NOT NULL REFERENCES rallies(id) ON DELETE CASCADE,
    time_s REAL NOT NULL,
    team TEXT NOT NULL,
    action TEXT NOT NULL,
    result TEXT NOT NULL,
    track_id INTEGER NOT NULL DEFAULT -1,
    origin TEXT NOT NULL,
    edited INTEGER NOT NULL DEFAULT 0
);
"""


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript(SCHEMA)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(rallies)")}
            if "label" not in columns:
                db.execute("ALTER TABLE rallies ADD COLUMN label TEXT NOT NULL DEFAULT ''")
                db.execute("UPDATE rallies SET label = CAST(id AS TEXT)")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def is_empty(self) -> bool:
        with closing(self._connect()) as db:
            return db.execute("SELECT COUNT(*) FROM rallies").fetchone()[0] == 0

    def reviewed_count(self) -> int:
        """Échanges validés ou supprimés, et échanges dont un geste a été corrigé."""
        with closing(self._connect()) as db:
            return db.execute(
                "SELECT COUNT(*) FROM rallies WHERE status != ? OR id IN"
                " (SELECT rally_id FROM gestures WHERE edited = 1)",
                (TO_REVIEW,),
            ).fetchone()[0]

    def backup(self, directory: Path, keep: int = 30) -> Path:
        """Copie la base dans `directory` (sûre même pendant une écriture), en ne gardant
        que les `keep` copies les plus récentes."""
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{self.path.stem}-{datetime.now():%Y%m%d-%H%M%S}.db"
        with closing(self._connect()) as source, closing(sqlite3.connect(target)) as copy:
            source.backup(copy)
        for old in sorted(directory.glob(f"{self.path.stem}-*.db"))[:-keep]:
            old.unlink()
        return target

    def import_analysis(self, points: pd.DataFrame, actions: pd.DataFrame, videos_dir: Path):
        """Remplace le match par l'analyse automatique : `points.csv` et `actions.csv`
        de `volley score`. Le numéro d'un échange (`label`) est sa ligne dans `points.csv`."""
        names = list(dict.fromkeys(points["video"]))
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM gestures")
            db.execute("DELETE FROM rallies")
            db.execute("DELETE FROM videos")
            for position, name in enumerate(names):
                path = (videos_dir / f"{name}.mp4").resolve()
                fps, duration_s = _probe(path)
                db.execute(
                    "INSERT INTO videos VALUES (?, ?, ?, ?, ?)",
                    (name, position, str(path), fps, duration_s),
                )
            for number, rally in points.iterrows():
                winner = rally["winner"] if isinstance(rally["winner"], str) else None
                db.execute(
                    "INSERT INTO rallies (id, label, video, set_number, serve_s, end_s,"
                    " serving_team, auto_winner, winner) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        int(number),
                        str(number),
                        rally["video"],
                        int(rally["set"]),
                        float(rally["serve_s"]),
                        float(rally["end_s"]),
                        rally["serving_team"],
                        winner,
                        winner,
                    ),
                )
            db.executemany(
                "INSERT INTO gestures (rally_id, time_s, team, action, result, track_id, origin)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        int(g.rally),
                        float(g.time_s),
                        g.team,
                        g.action,
                        g.result,
                        int(g.track_id),
                        g.origin,
                    )
                    for g in actions.itertuples()
                ],
            )

    def videos(self) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM videos ORDER BY position").fetchall()
        return [dict(row) for row in rows]

    def video_path(self, name: str) -> Path | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT path FROM videos WHERE name = ?", (name,)).fetchone()
        return Path(row["path"]) if row else None

    def rallies(self) -> list[dict]:
        """Échanges dans l'ordre du match, avec le score après chacun (sans les supprimés)."""
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT r.*, (SELECT COUNT(*) FROM gestures g WHERE g.rally_id = r.id"
                " AND g.team = ?) AS corbas_gestures FROM rallies r"
                " JOIN videos v ON v.name = r.video ORDER BY v.position, r.serve_s",
                (CORBAS,),
            ).fetchall()
        rallies, score, current_set = [], {CORBAS: 0, OPPONENT: 0}, None
        for row in rows:
            rally = dict(row)
            if rally["set_number"] != current_set:
                score, current_set = {CORBAS: 0, OPPONENT: 0}, rally["set_number"]
            if rally["status"] != DELETED and rally["winner"] in TEAMS:
                score[rally["winner"]] += 1
            rally["score"] = {"corbas": score[CORBAS], "adversaire": score[OPPONENT]}
            rallies.append(rally)
        return rallies

    def rally(self, rally_id: int) -> dict | None:
        rally = next((r for r in self.rallies() if r["id"] == rally_id), None)
        if rally is None:
            return None
        rally["gestures"] = self._gestures(rally_id)
        return rally

    def _gestures(self, rally_id: int) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT * FROM gestures WHERE rally_id = ? ORDER BY time_s, id", (rally_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def update_rally(self, rally_id: int, changes: dict) -> dict | None:
        """`changes` peut contenir `winner`, `status`, `note`, les bornes `serve_s` et
        `end_s`, et `serving_team` (qui change aussi l'équipe du geste de service)."""
        before = self.rally(rally_id)
        if before is None:
            return None
        editable = ("winner", "status", "note", "serve_s", "end_s", "serving_team")
        fields = {key: changes[key] for key in editable if key in changes}
        if fields.get("serve_s", before["serve_s"]) >= fields.get("end_s", before["end_s"]):
            raise ValueError("Le début de l'échange doit précéder sa fin")
        serve = next((g for g in before["gestures"] if g["action"] == SERVE), None)
        if fields:
            assignments = ", ".join(f"{key} = ?" for key in fields)
            with closing(self._connect()) as db, db:
                db.execute(
                    f"UPDATE rallies SET {assignments} WHERE id = ?", (*fields.values(), rally_id)
                )
                if serve and "serve_s" in fields:
                    db.execute(
                        "UPDATE gestures SET time_s = ? WHERE id = ?",
                        (fields["serve_s"], serve["id"]),
                    )
                if serve and "serving_team" in fields:
                    db.execute(
                        "UPDATE gestures SET team = ? WHERE id = ?",
                        (fields["serving_team"], serve["id"]),
                    )
        winner = fields.get("winner", before["winner"])
        if (
            winner != before["winner"]
            or fields.get("serving_team", before["serving_team"]) != (before["serving_team"])
        ):
            self._recompute_results(rally_id, winner)
        return self.rally(rally_id)

    def split_rally(self, rally_id: int, time_s: float) -> dict | None:
        """Coupe en deux un échange qui en contenait deux : la seconde partie, à partir de
        `time_s`, devient un nouvel échange (« 10b ») avec les gestes qui suivent et le
        vainqueur d'origine ; celui de la première partie reste à saisir."""
        rally = self.rally(rally_id)
        if rally is None:
            return None
        if not rally["serve_s"] < time_s < rally["end_s"]:
            raise ValueError("La coupure doit tomber pendant l'échange")
        root = rally["label"].rstrip(ascii_lowercase)
        with closing(self._connect()) as db, db:
            labels = {row["label"] for row in db.execute("SELECT label FROM rallies")}
            suffix = next(c for c in ascii_lowercase[1:] if f"{root}{c}" not in labels)
            new_id = db.execute("SELECT MAX(id) + 1 FROM rallies").fetchone()[0]
            db.execute(
                "INSERT INTO rallies (id, label, video, set_number, serve_s, end_s, serving_team,"
                " auto_winner, winner) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id,
                    f"{root}{suffix}",
                    rally["video"],
                    rally["set_number"],
                    time_s,
                    rally["end_s"],
                    rally["serving_team"],
                    rally["auto_winner"],
                    rally["winner"],
                ),
            )
            db.execute(
                "UPDATE gestures SET rally_id = ? WHERE rally_id = ? AND time_s >= ?",
                (new_id, rally_id, time_s),
            )
            db.execute(
                "INSERT INTO gestures (rally_id, time_s, team, action, result, origin)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (new_id, time_s, rally["serving_team"], SERVE, UNKNOWN, MANUAL),
            )
            db.execute(
                "UPDATE rallies SET end_s = ?, winner = NULL, auto_winner = NULL, status = ?"
                " WHERE id = ?",
                (time_s, TO_REVIEW, rally_id),
            )
        self._recompute_results(rally_id, None)
        self._recompute_results(new_id, rally["winner"])
        return self.rally(rally_id)

    def _recompute_results(self, rally_id: int, winner: str | None):
        """Résultats des gestes non modifiés à la main, et réceptions ratées sans touche,
        d'après le nouveau vainqueur."""
        gestures = self._gestures(rally_id)
        played = [g for g in gestures if g["origin"] != UNTOUCHED]
        kept_untouched = [g for g in gestures if g["origin"] == UNTOUCHED and g["edited"]]
        with closing(self._connect()) as db, db:
            db.execute(
                "DELETE FROM gestures WHERE rally_id = ? AND origin = ? AND edited = 0",
                (rally_id, UNTOUCHED),
            )
            if not played:
                return
            for gesture in gesture_results(played, winner):
                if "id" not in gesture:
                    if not kept_untouched:
                        db.execute(
                            "INSERT INTO gestures (rally_id, time_s, team, action, result,"
                            " track_id, origin) VALUES (?, ?, ?, ?, ?, -1, ?)",
                            (
                                rally_id,
                                gesture["time_s"],
                                gesture["team"],
                                gesture["action"],
                                gesture["result"],
                                UNTOUCHED,
                            ),
                        )
                elif not gesture["edited"]:
                    db.execute(
                        "UPDATE gestures SET result = ? WHERE id = ?",
                        (gesture["result"], gesture["id"]),
                    )

    def add_gesture(self, rally_id: int, gesture: dict) -> dict | None:
        if self.rally(rally_id) is None:
            return None
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO gestures (rally_id, time_s, team, action, result, origin, edited)"
                " VALUES (?, ?, ?, ?, ?, ?, 1)",
                (
                    rally_id,
                    gesture["time_s"],
                    gesture["team"],
                    gesture["action"],
                    gesture["result"],
                    MANUAL,
                ),
            )
        return self.rally(rally_id)

    def update_gesture(self, gesture_id: int, changes: dict) -> dict | None:
        """`changes` peut contenir `time_s`, `team`, `action` et `result`."""
        fields = {k: changes[k] for k in ("time_s", "team", "action", "result") if k in changes}
        with closing(self._connect()) as db, db:
            row = db.execute("SELECT rally_id FROM gestures WHERE id = ?", (gesture_id,)).fetchone()
            if row is None:
                return None
            if fields:
                assignments = ", ".join(f"{key} = ?" for key in fields)
                db.execute(
                    f"UPDATE gestures SET {assignments}, edited = 1 WHERE id = ?",
                    (*fields.values(), gesture_id),
                )
        return self.rally(row["rally_id"])

    def delete_gesture(self, gesture_id: int) -> dict | None:
        with closing(self._connect()) as db, db:
            row = db.execute("SELECT rally_id FROM gestures WHERE id = ?", (gesture_id,)).fetchone()
            if row is None:
                return None
            db.execute("DELETE FROM gestures WHERE id = ?", (gesture_id,))
        return self.rally(row["rally_id"])

    def team_actions(self, team: str = CORBAS) -> pd.DataFrame:
        """Gestes de `team` dans les échanges non supprimés."""
        with closing(self._connect()) as db:
            return pd.read_sql_query(
                "SELECT g.action, g.result FROM gestures g JOIN rallies r ON r.id = g.rally_id"
                " WHERE g.team = ? AND r.status != ?",
                db,
                params=(team, DELETED),
            )


def _probe(path: Path) -> tuple[float, float]:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    duration_s = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    capture.release()
    return float(fps), float(duration_s)
