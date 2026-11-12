"""Corrections manuelles des services détectés, saisies en regardant la vidéo annotée.

Fichier CSV `video,temps,action,commentaire`, une ligne par correction :

- `ignorer` retire l'échange affiché dans la vidéo à `temps` (celui du dernier
  service détecté avant cet instant) : ballon passé avant le match, faux service...
  Avec une période (`1:04-3:27`), retire tous les services de la période (pause).
- `fin_set` termine le set avec l'échange affiché à `temps`. Dès qu'une fin de
  set est saisie, les sets ne se terminent plus d'eux-mêmes à 25 points.
- `service_corbas` / `service_adversaire` ajoute un service manqué à `temps`.
- `point_corbas` / `point_adversaire` donne le point de l'échange affiché à
  `temps` ; l'échange suivant est alors servi par cette équipe.
- `ignorer_arbitre` (sans `temps`) n'utilise pas les gestes de l'arbitre sur la
  vidéo, quand il ne désigne pas les points.

`temps` s'écrit en secondes (`75.5`) ou en minutes:secondes (`1:15.5`).
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from volley_analysis.teams import CORBAS, OPPONENT

IGNORE = "ignorer"
IGNORE_REFEREE = "ignorer_arbitre"
SET_END = "fin_set"
SERVE_ACTIONS = {"service_corbas": CORBAS, "service_adversaire": OPPONENT}
POINT_ACTIONS = {"point_corbas": CORBAS, "point_adversaire": OPPONENT}
CORRECTION_COLUMNS = ["video", "temps", "action", "commentaire"]


@dataclass
class Corrections:
    ignored_s: list[float] = field(default_factory=list)
    ignored_periods: list[tuple[float, float]] = field(default_factory=list)
    serves: list[tuple[float, str]] = field(default_factory=list)
    points: list[tuple[float, str]] = field(default_factory=list)
    set_ends_s: list[float] = field(default_factory=list)
    ignore_referee: bool = False


def parse_time(text: str | float) -> float:
    """`75.5`, `1:15.5` ou `0:01:15` → secondes."""
    seconds = 0.0
    for part in str(text).strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def load_corrections(path: Path | None) -> dict[str, Corrections]:
    """Corrections par vidéo (nom sans extension). Fichier absent : aucune correction."""
    if path is None or not path.exists():
        return {}
    table = pd.read_csv(path, dtype=str).fillna("")
    missing = set(CORRECTION_COLUMNS[:3]) - set(table.columns)
    if missing:
        raise ValueError(f"{path} : colonnes manquantes {sorted(missing)}")
    corrections: dict[str, Corrections] = {}
    for row in table.itertuples():
        video = Path(row.video.strip()).stem
        action = row.action.strip().lower()
        entry = corrections.setdefault(video, Corrections())
        if action == IGNORE_REFEREE:
            entry.ignore_referee = True
        elif action == IGNORE and "-" in row.temps:
            start, end = row.temps.split("-")
            entry.ignored_periods.append((parse_time(start), parse_time(end)))
        elif action == IGNORE:
            entry.ignored_s.append(parse_time(row.temps))
        elif action == SET_END:
            entry.set_ends_s.append(parse_time(row.temps))
        elif action in SERVE_ACTIONS:
            entry.serves.append((parse_time(row.temps), SERVE_ACTIONS[action]))
        elif action in POINT_ACTIONS:
            entry.points.append((parse_time(row.temps), POINT_ACTIONS[action]))
        else:
            actions = [IGNORE, *SERVE_ACTIONS, *POINT_ACTIONS, SET_END, IGNORE_REFEREE]
            expected = ", ".join(actions)
            raise ValueError(f"{path} : action inconnue « {row.action} » (attendu : {expected})")
    return corrections
