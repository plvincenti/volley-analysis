"""Géométrie du terrain de volley, dans un repère vu de dessus en mètres.

Repère utilisé dans tout le projet (caméra placée derrière une ligne de fond) :
- origine : coin gauche de la ligne de fond côté caméra ;
- x : vers la droite de l'image, de 0 à 9 m (largeur) ;
- y : en s'éloignant de la caméra, de 0 à 18 m (longueur), filet à y = 9 m.
"""

from dataclasses import dataclass

import numpy as np

COURT_WIDTH = 9.0
COURT_LENGTH = 18.0
NET_Y = COURT_LENGTH / 2
ATTACK_LINE_DISTANCE = 3.0

NEAR = "near"
FAR = "far"


@dataclass(frozen=True)
class CourtKeypoint:
    name: str
    label: str
    xy: tuple[float, float]


# Ordre de saisie lors de la calibration : les 4 coins d'abord (les plus
# importants pour une homographie stable), puis les intersections intermédiaires.
KEYPOINTS: tuple[CourtKeypoint, ...] = (
    CourtKeypoint("near_left", "Coin gauche, ligne de fond cote camera", (0.0, 0.0)),
    CourtKeypoint("near_right", "Coin droit, ligne de fond cote camera", (COURT_WIDTH, 0.0)),
    CourtKeypoint("far_right", "Coin droit, ligne de fond opposee", (COURT_WIDTH, COURT_LENGTH)),
    CourtKeypoint("far_left", "Coin gauche, ligne de fond opposee", (0.0, COURT_LENGTH)),
    CourtKeypoint(
        "near_attack_left", "Ligne d'attaque cote camera x ligne gauche", (0.0, NET_Y - 3.0)
    ),
    CourtKeypoint(
        "near_attack_right",
        "Ligne d'attaque cote camera x ligne droite",
        (COURT_WIDTH, NET_Y - 3.0),
    ),
    CourtKeypoint("center_left", "Ligne centrale (sous le filet) x ligne gauche", (0.0, NET_Y)),
    CourtKeypoint(
        "center_right", "Ligne centrale (sous le filet) x ligne droite", (COURT_WIDTH, NET_Y)
    ),
    CourtKeypoint("far_attack_left", "Ligne d'attaque opposee x ligne gauche", (0.0, NET_Y + 3.0)),
    CourtKeypoint(
        "far_attack_right", "Ligne d'attaque opposee x ligne droite", (COURT_WIDTH, NET_Y + 3.0)
    ),
)

KEYPOINTS_BY_NAME = {kp.name: kp for kp in KEYPOINTS}

# Segments des lignes du terrain, pour le dessin (en mètres).
LINES: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = (
    ((0.0, 0.0), (COURT_WIDTH, 0.0)),
    ((0.0, COURT_LENGTH), (COURT_WIDTH, COURT_LENGTH)),
    ((0.0, 0.0), (0.0, COURT_LENGTH)),
    ((COURT_WIDTH, 0.0), (COURT_WIDTH, COURT_LENGTH)),
    ((0.0, NET_Y), (COURT_WIDTH, NET_Y)),
    ((0.0, NET_Y - ATTACK_LINE_DISTANCE), (COURT_WIDTH, NET_Y - ATTACK_LINE_DISTANCE)),
    ((0.0, NET_Y + ATTACK_LINE_DISTANCE), (COURT_WIDTH, NET_Y + ATTACK_LINE_DISTANCE)),
)


def is_in_play_area(
    court_xy: np.ndarray, side_margin: float = 1.0, end_margin: float = 4.0
) -> np.ndarray:
    """Indique si des positions (N, 2) sont dans la zone de jeu élargie.

    La marge derrière les lignes de fond est plus grande que sur les côtés :
    les joueurs servent et défendent souvent loin derrière le fond, alors que
    le long des lignes de côté se trouvent arbitres, bancs et coachs.
    """
    court_xy = np.asarray(court_xy, dtype=float).reshape(-1, 2)
    x, y = court_xy[:, 0], court_xy[:, 1]
    return (
        (x >= -side_margin)
        & (x <= COURT_WIDTH + side_margin)
        & (y >= -end_margin)
        & (y <= COURT_LENGTH + end_margin)
    )


def side_of_net(court_y: np.ndarray) -> np.ndarray:
    """Renvoie NEAR (côté caméra) ou FAR (côté opposé) pour chaque ordonnée."""
    return np.where(np.asarray(court_y, dtype=float) < NET_Y, NEAR, FAR)
