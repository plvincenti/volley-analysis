import pandas as pd
from fastapi.testclient import TestClient

from volley_analysis.actions import ATTACK, FAILED, RECEPTION, SERVE, SUCCESS, UNKNOWN, WINNING
from volley_analysis.teams import CORBAS, OPPONENT
from volley_analysis.web.app import create_app
from volley_analysis.web.store import DELETED, Store


def _store(tmp_path):
    points = pd.DataFrame(
        {
            "video": ["v", "v", "v"],
            "set": [1, 1, 2],
            "serve_s": [10.0, 30.0, 50.0],
            "end_s": [20.0, 40.0, 60.0],
            "serving_team": [CORBAS, OPPONENT, OPPONENT],
            "winner": [OPPONENT, OPPONENT, CORBAS],
        }
    )
    actions = pd.DataFrame(
        {
            "rally": [0, 0, 0],
            "time_s": [10.0, 12.0, 14.0],
            "team": [CORBAS, OPPONENT, CORBAS],
            "action": [SERVE, RECEPTION, ATTACK],
            "result": [SUCCESS, SUCCESS, FAILED],
            "track_id": [-1, 3, 5],
            "origin": ["service", "touche", "touche"],
        }
    )
    store = Store(tmp_path / "match.db")
    store.import_analysis(points, actions, tmp_path)
    return store


def _results(rally):
    return [(g["team"], g["action"], g["result"]) for g in rally["gestures"]]


def test_score_follows_the_corrected_winners_and_restarts_each_set(tmp_path):
    store = _store(tmp_path)
    store.update_rally(1, {"status": DELETED})

    scores = [r["score"] for r in store.rallies()]

    assert scores == [
        {"corbas": 0, "adversaire": 1},
        {"corbas": 0, "adversaire": 1},
        {"corbas": 1, "adversaire": 0},
    ]


def test_changing_the_winner_updates_the_results_of_unedited_gestures(tmp_path):
    store = _store(tmp_path)

    rally = store.update_rally(0, {"winner": CORBAS})

    assert _results(rally) == [
        (CORBAS, SERVE, SUCCESS),
        (OPPONENT, RECEPTION, SUCCESS),
        (CORBAS, ATTACK, WINNING),
        (OPPONENT, RECEPTION, FAILED),
    ]
    attack = rally["gestures"][2]
    store.update_gesture(attack["id"], {"result": SUCCESS})
    rally = store.update_rally(0, {"winner": OPPONENT})
    assert _results(rally)[2:] == [(CORBAS, ATTACK, SUCCESS)]


def test_splitting_a_rally_moves_the_later_gestures_to_a_new_rally(tmp_path):
    store = _store(tmp_path)

    first = store.split_rally(0, 13.0)

    assert (first["label"], first["end_s"], first["winner"]) == ("0", 13.0, None)
    assert _results(first) == [(CORBAS, SERVE, UNKNOWN), (OPPONENT, RECEPTION, UNKNOWN)]
    second = next(r for r in store.rallies() if r["label"] == "0b")
    assert (second["serve_s"], second["end_s"], second["winner"]) == (13.0, 20.0, OPPONENT)
    assert _results(store.rally(second["id"])) == [
        (CORBAS, SERVE, SUCCESS),
        (CORBAS, ATTACK, FAILED),
    ]
    assert [r["label"] for r in store.rallies()][:2] == ["0", "0b"]
    store.split_rally(second["id"], 15.0)
    assert [r["label"] for r in store.rallies()][:3] == ["0", "0b", "0c"]


def test_backup_copies_the_review_and_keeps_the_latest_copies(tmp_path):
    store = _store(tmp_path)
    store.update_rally(0, {"status": "valide"})
    assert store.reviewed_count() == 1
    for i in range(3):
        (tmp_path / "backups").mkdir(exist_ok=True)
        (tmp_path / "backups" / f"match-2000010{i}-000000.db").touch()

    copy = Store(store.backup(tmp_path / "backups", keep=2))

    assert copy.reviewed_count() == 1
    assert len(list((tmp_path / "backups").glob("*.db"))) == 2


def test_api_edits_gestures_and_counts_corbas_stats(tmp_path):
    client = TestClient(create_app(_store(tmp_path)))

    assert client.patch("/api/rallies/0", json={"winner": "personne"}).status_code == 422
    rally = client.post(
        "/api/rallies/1/gestures", json={"time_s": 31.0, "action": RECEPTION, "result": FAILED}
    ).json()
    assert _results(rally) == [(CORBAS, RECEPTION, FAILED)]
    assert rally["gestures"][0]["origin"] == "manuel"

    stats = {row["action"]: row for row in client.get("/api/stats").json()}
    assert stats[RECEPTION]["total"] == 1
    assert stats[ATTACK]["rate"] == 1
    assert client.get("/api/rallies/99").status_code == 404
    assert client.post("/api/rallies/0/split", json={"time_s": 25.0}).status_code == 400
    assert client.patch("/api/rallies/0", json={"end_s": 5.0}).status_code == 400
