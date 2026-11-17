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
| 3a | Découpage en échanges, équipe au service, comptage des points et des sets | fait |
| 3b | Détection des touches, classification des actions par les règles du jeu | fait |
| 4 | Qualité des actions (notation type DataVolley), statistiques, rapport et clips | à faire |
| 5 | Application web de revue : correction des points et des gestes | en cours |
| 6 | Classifieur sur la pose, action spotting, entraînés sur les corrections | à faire |

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

# 5. Compter les points sur les vidéos consécutives d'un match, dans l'ordre
#    (calcule la trajectoire du ballon des vidéos qui n'en ont pas)
volley score data/videos/video1.mp4 data/videos/video2.mp4 data/videos/video3.mp4
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

### Comptage des points

L'équipe qui gagne un échange sert le suivant : il suffit donc de savoir qui sert.

1. **Position 3D du ballon.** La pose de la caméra se déduit de la calibration du sol
   (focale et position) ; la distance du ballon à la caméra se déduit de son rayon
   apparent (21 cm de diamètre). On en tire sa profondeur sur le terrain et sa hauteur,
   donc le côté du filet où il se trouve, même haut en l'air.
2. **Passages du filet** : changements de côté confirmés sur plusieurs détections.
   Ceux à moins d'un mètre de haut sont des ballons roulés sous le filet, pas du jeu.
3. **Services** : passage du filet après une pause sans jeu (6,5 s), ou après une pause
   plus courte si le serveur était visible (ballon rebondi derrière la ligne de fond
   opposée, ou joueur coupé par le bas de l'image côté caméra). Un passage isolé
   suivi de peu d'un service depuis le côté où le ballon est arrivé est un renvoi du
   ballon au serveur, pas un échange.
4. **Équipes** : avant chaque service, le côté de Corbas est celui où les joueurs sur le
   terrain portent le plus de rouge (remplaçants, arbitre et marqueur sont hors terrain).
5. **Score** : 0-0 au premier échange (ou `--start-score`), vainqueur = équipe au service
   de l'échange suivant ; set gagné à 25 points (15 au 5e) avec 2 points d'écart, puis
   changement de camp. Quand le changement de camp est observé, le dernier point du set
   revient à l'équipe à qui il donne le set (le service suivant ne dit rien dans ce cas).

Sorties dans `outputs/score/` :

- `points.csv` : un échange par ligne (vidéo, instant du service, côté et équipe au
  service, vainqueur, score après l'échange, set, remarques sur les incohérences) ;
- `services_<video>.jpg` : planche de contrôle, une vignette par service avec la
  trajectoire du ballon et le score, pour vérifier rapidement ;
- `actions_corbas.csv` : services, réceptions et attaques de Corbas avec leur résultat.

Les gestes viennent des touches du ballon (`touches.py`) : un ballon qui remonte dans
l'image, change de sens horizontalement ou accélère brusquement a été touché, par le
joueur suivi (`volley track`) le plus proche. Après le service, les touches d'une même
équipe forment une possession : réception, passe(s), attaque ; une touche au filet
juste après une attaque adverse est un contre. Un passage du filet sans touche vue de
l'équipe juste avant est une attaque non vue ; un retour à moins de 2 s du passage
précédent est du bruit. Après le point, le ballon renvoyé au gagnant ou rattrapé par lui
n'est pas un geste, ni rien de ce qui suit un silence de plus de 4,5 s. Près du filet,
où les joueurs des deux camps se superposent dans l'image, une touche va au joueur du
camp où se trouve le ballon d'après le dernier passage du filet. Un geste est réussi si l'échange
continue, raté si c'est la dernière touche de l'équipe et qu'elle perd le point ; un
service, une attaque ou un contre est gagnant si l'équipe marque sans retoucher le
ballon. Sans suivi des joueurs pour une vidéo, seuls ses services sont listés, sans résultat.

`volley render` affiche ensuite ce score dans `annotated.mp4` (set, score, équipe au
service, puis « POINT … » à la fin de l'échange).

L'arbitre, debout contre le poteau droit du filet, désigne du bras le camp qui marque.
Sa pose (YOLO11-pose, gardée dans `outputs/<video>/referee_pose.parquet`) sert de
contrôle : un bras tendu vers la droite de l'image désigne le camp proche, vers le
haut à gauche le camp du fond. Quand son geste contredit le vainqueur déduit du
service suivant, `points.csv` le signale et la vidéo l'affiche en rouge ; le
vainqueur n'est pas modifié, l'erreur venant en général d'un service à corriger.

Les erreurs repérées en regardant la vidéo se corrigent dans `data/corrections.csv`,
pris en compte au prochain `volley score` :

```csv
video,temps,action,commentaire
video1,0:13,ignorer,passe entre joueurs avant le début du match
video1,2:41.5,service_corbas,service manqué
```

- `ignorer` retire l'échange affiché dans la vidéo à cet instant ; avec une période
  (`1:04.5-3:27`), retire tous les échanges servis pendant celle-ci (pause entre sets…) ;
- `service_corbas` / `service_adversaire` ajoute un service que la détection a manqué ;
- `point_corbas` / `point_adversaire` donne le point de l'échange affiché à cet instant,
  et fait servir cette équipe à l'échange suivant ;
- `fin_echange` arrête à cet instant l'échange en cours : les ballons échangés ensuite,
  entre deux points, ne comptent plus comme gestes ;
- `fin_set` termine le set avec l'échange affiché à cet instant. Dès qu'une fin de set
  est saisie, les sets ne se terminent plus d'eux-mêmes (25 points, changement de camp) ;
  un score final impossible est signalé, ce qui révèle des échanges en trop ou manqués ;
- `ignorer_arbitre` (sans temps) n'utilise pas les gestes de l'arbitre sur la vidéo.

`volley render` fonctionne aussi sans suivi des joueurs : la vidéo n'affiche alors que
le ballon et le score.

### Revue du match dans le navigateur

L'application web présente chaque échange analysé par `volley score` : la vidéo démarre
juste avant le service, les gestes sont placés sur une frise, et l'on corrige le
vainqueur du point, les gestes (geste, résultat, ajout, suppression) puis on valide.
Changer le vainqueur recalcule le résultat des gestes non modifiés à la main ; le score
et les statistiques de Corbas suivent les corrections.

```bash
pip install -e ".[web]"
cd web && npm install && npm run build && cd ..

# Importe outputs/score (points.csv, actions.csv) dans data/match.db si elle est vide,
# puis sert l'application sur http://127.0.0.1:8000
volley web
# Repartir d'une nouvelle analyse (efface les corrections de la revue, après confirmation)
volley web --reimport
```

La revue est enregistrée à chaque correction dans `data/match.db` (fichier SQLite, hors
git) : elle survit à l'arrêt du serveur et au redémarrage de la machine. Chaque
lancement de `volley web` en fait aussi une copie dans `data/backups/` (30 dernières) ;
pour revenir à une copie, arrêter le serveur et la recopier sur `data/match.db`.

Pour modifier l'interface (`web/src`, React + TypeScript), lancer `volley web` puis
`npm run dev` dans `web/` : Vite recharge la page à chaque modification et transmet
l'API et les vidéos au serveur Python.

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
