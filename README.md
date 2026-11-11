# volley-analysis

Analyse vidéo (computer vision) de matchs de volley-ball amateur, pour produire
des statistiques d'équipe : réceptions, passes, attaques, contres, services, et
leur qualité. L'objectif est d'identifier ce que l'équipe doit travailler.

Le fonctionnement visé est semi-automatique : l'outil pré-remplit les actions,
on corrige rapidement, et les corrections servent à améliorer les modèles.

## Feuille de route

| Étape | Contenu | État |
|---|---|---|
| 1 | Calibration du terrain (homographie), détection et suivi des joueurs, équipe par côté du filet, heatmaps | fait |
| 2 | Détection et suivi du ballon (VballNet), nettoyage et interpolation des trajectoires | en cours |
| 3 | Détection des touches, découpage en rallyes, classification des actions par les règles du jeu | à faire |
| 4 | Qualité des actions (notation type DataVolley), statistiques, rapport et clips | à faire |
| 5 | Classifieur sur la pose, action spotting, outil de correction | à faire |

## Installation

Python 3.11 à 3.13 recommandé.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Sur Mac Apple Silicon, l'inférence utilise automatiquement le GPU (MPS).

## Utilisation

Placer les vidéos dans `data/videos/` (ignoré par git).

```bash
# Caractéristiques de la vidéo
volley info data/videos/match.mp4

# 1a. Calibrer à partir d'une capture où les 4 lignes du terrain (fonds et côtés)
#     sont tracées en rouge vif, même à main levée et même si des coins sont hors champ.
#     Produit data/calibrations/default.json, utilisée par défaut pour toutes les vidéos.
volley calibrate-drawing capture.png --video data/videos/match.mp4

# 1b. Ou calibrer en cliquant les coins et intersections de lignes demandés
#     (s = point invisible, u = annuler, Entrée = terminer, une loupe aide à viser).
#     Produit data/calibrations/<video>.json, prioritaire sur default.json.
volley calibrate data/videos/match.mp4 --at 5

# 2. Vérifier la calibration sur une autre image de la vidéo
volley preview data/videos/match.mp4 --at 600

# 3. Suivre les joueurs (tester d'abord sur un extrait de 60 s)
volley track data/videos/match.mp4 --start 120 --end 180

# 4. Suivre le ballon (rapide : ~130 img/s) ; la vidéo annotée inclut les joueurs
volley ball data/videos/match.mp4 --start 120 --end 180

# Régénérer nettoyages, heatmaps et vidéo sans relancer les détections (rapide)
volley render data/videos/match.mp4
```

Le ballon est détecté par [VballNet](https://github.com/asigatchov/fast-volleyball-tracking-inference)
(licence MIT), un dérivé de TrackNetV4 entraîné sur le volley, qui exploite le mouvement
sur 9 images consécutives. Le dépôt doit être cloné dans `vendor/` :

```bash
git clone --depth 1 https://github.com/asigatchov/fast-volleyball-tracking-inference.git \
  vendor/fast-volleyball-tracking-inference
```

Les détections sont ensuite regroupées en tronçons de trajectoire cohérents (ce qui écarte
les fausses détections isolées et le ballon du terrain voisin), puis les trous de moins de
0,25 s sont interpolés.

Après le suivi image par image, les pistes sont nettoyées sur toute la vidéo :

- le côté du filet est lissé sur 2 s, car un joueur qui saute au filet a les pieds
  en l'air et serait sinon projeté de l'autre côté ;
- les personnes quasi immobiles hors des lignes (spectateurs, remplaçants, arbitres)
  sont écartées, alors que le serveur, qui ne sort que quelques secondes, est conservé.

Les sorties sont dans `outputs/<nom de la vidéo>/` :

- `players.parquet` : une ligne par personne détectée et par image (boîte, identifiant
  de suivi, position terrain en mètres, côté du filet, `is_player`) ;
- `ball_raw.parquet` (détections brutes) et `ball.parquet` (trajectoire nettoyée, une ligne
  par image, `source` = `detected`, `interpolated` ou vide) ;
- `annotated.mp4` : vidéo annotée (joueurs, ballon, mini-carte vue de dessus) ;
- `heatmap_all.png`, `heatmap_near.png`, `heatmap_far.png` : cartes de présence.

## Conventions

Repère terrain en mètres, vu de dessus, caméra derrière une ligne de fond :

- origine au coin gauche de la ligne de fond côté caméra ;
- `x` vers la droite de l'image (0 à 9 m), `y` en s'éloignant de la caméra (0 à 18 m) ;
- filet à `y = 9` ; côté `near` = côté caméra, côté `far` = côté opposé.

La position d'un joueur est le milieu du bas de sa boîte (ses pieds), seul point
au sol, donc compatible avec l'homographie.

## Développement

```bash
pytest
ruff check src tests && ruff format src tests
```
