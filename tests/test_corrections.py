import pytest

from volley_analysis.corrections import load_corrections, parse_time
from volley_analysis.teams import CORBAS, OPPONENT


def test_times_in_seconds_or_minutes():
    assert parse_time("75.5") == 75.5
    assert parse_time("1:15.5") == 75.5
    assert parse_time("0:01:15") == 75.0


def test_corrections_are_grouped_by_video(tmp_path):
    path = tmp_path / "corrections.csv"
    path.write_text(
        "video,temps,action,commentaire\n"
        "video1,0:13,ignorer,passe avant le match #1\n"
        "video1.mp4,2:05,service_corbas,\n"
        "video2,10,SERVICE_ADVERSAIRE,\n"
        "video3,,ignorer_arbitre,\n"
        "video3,1:04.5-3:27,ignorer,pause\n"
        "video3,1:04,fin_set,\n"
    )
    corrections = load_corrections(path)

    assert corrections["video1"].ignored_s == [13.0]
    assert corrections["video1"].serves == [(125.0, CORBAS)]
    assert corrections["video2"].serves == [(10.0, OPPONENT)]
    assert corrections["video3"].ignore_referee
    assert corrections["video3"].ignored_periods == [(64.5, 207.0)]
    assert corrections["video3"].set_ends_s == [64.0]
    assert not corrections["video1"].ignore_referee
    assert load_corrections(tmp_path / "absent.csv") == {}


def test_unknown_action_is_rejected(tmp_path):
    path = tmp_path / "corrections.csv"
    path.write_text("video,temps,action\nvideo1,12,supprimer\n")
    with pytest.raises(ValueError, match="supprimer"):
        load_corrections(path)
