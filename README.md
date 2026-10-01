# Football Analysis

Un projet perso de computer vision : on donne une vidéo de match en entrée, et on récupère
une vidéo annotée avec les joueurs suivis, leur équipe, la possession de balle, ainsi que la
vitesse et la distance parcourue de chaque joueur. Les stats sont aussi exportées en CSV.

Projet réalisé par Alae-eddine Naggaz.

## Ce que fait le programme

- détecte les joueurs, les arbitres et le ballon avec YOLOv8
- suit chaque joueur d'une frame à l'autre avec ByteTrack (chaque joueur garde son numéro)
- sépare les deux équipes à partir de la couleur des maillots (K-Means)
- calcule quelle équipe a le ballon et affiche le pourcentage de possession
- compense les mouvements de la caméra (flux optique)
- convertit les positions en mètres sur le terrain grâce à une homographie
- en déduit la vitesse (km/h) et la distance parcourue (m) de chaque joueur

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # sous Linux/Mac : source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

1. Mettre la vidéo dans `input_videos/match.mp4` (un plan large, type caméra TV).

2. Calibrer le terrain, une fois par vidéo :

   ```bash
   python tools/pick_points.py input_videos/match.mp4
   ```

   On clique sur 4 points du terrain (bas-gauche, haut-gauche, haut-droit, bas-droit), puis on
   copie le résultat dans `PIXEL_VERTICES` dans `config.py`. Il faut aussi indiquer les
   dimensions réelles de la zone choisie (`FIELD_LENGTH_M` et `FIELD_WIDTH_M`).

3. Lancer l'analyse :

   ```bash
   python main.py
   ```

   Le premier lancement est long parce que YOLO doit traiter toute la vidéo. Les détections
   sont ensuite gardées en cache, donc pour les lancements suivants :

   ```bash
   python main.py --use-stubs
   ```

   Pour un test rapide sur le début de la vidéo :

   ```bash
   python main.py --max-frames 300 --show-camera
   ```

   Attention : si on change de vidéo ou de nombre de frames, il faut vider le dossier `stubs/`.

4. Les résultats se trouvent dans `output_videos/` : `match_annotated.mp4` et `player_stats.csv`.

Pour lancer les tests : `pytest -q`

## Le modèle YOLO

Si `models/best.pt` n'existe pas, le programme utilise le modèle générique `yolov8x.pt`. Ça
marche, mais il ne fait pas la différence entre un arbitre et un joueur, et il rate souvent le
ballon. Pour de meilleurs résultats, il vaut mieux entraîner le modèle sur un dataset de foot
(par exemple *football-players-detection* sur Roboflow), de préférence sur GPU (Google Colab) :

```bash
python training/train_yolo.py --data datasets/football/data.yaml --epochs 100
```

Il suffit ensuite de copier `runs/detect/train/weights/best.pt` dans `models/best.pt`.

## Organisation du code

```
main.py                          lance tout le pipeline
config.py                        les paramètres (chemins, seuils, calibration...)
trackers/                        détection YOLO, tracking et dessin sur la vidéo
team_assigner/                   séparation des équipes
player_ball_assigner/            qui a le ballon
camera_movement_estimator/       mouvement de la caméra
view_transformer/                passage des pixels aux mètres
speed_and_distance_estimator/    vitesse, distance et export CSV
utils/                           lecture/écriture vidéo et petites fonctions de calcul
tools/pick_points.py             outil de calibration
training/train_yolo.py           entraînement de YOLO
tests/                           tests unitaires
```

## Quelques détails sur le fonctionnement

**Les équipes.** Pour chaque joueur, je prends le haut de sa bounding box (le maillot) et je
sépare les pixels en deux groupes. Le groupe qu'on retrouve dans les coins correspond à la
pelouse, l'autre au maillot. Ensuite, je regroupe les couleurs de maillot de tous les joueurs
en deux équipes. Pour éviter qu'un joueur change d'équipe en cours de match à cause d'une
occlusion, son équipe est fixée après 10 observations.

**Le ballon.** Il est petit et souvent flou, donc YOLO le perd régulièrement. Je garde la
meilleure détection de chaque frame et je comble les trous par interpolation.

**La caméra.** Comme la caméra bouge, une position en pixels ne veut rien dire toute seule.
Je suis des points fixes du décor (les joueurs sont masqués) pour estimer le déplacement de la
caméra, puis je corrige la position de chaque joueur.

**Les vitesses.** Je les calcule sur des fenêtres de 5 frames plutôt que d'une frame à l'autre,
sinon les petits tremblements des bounding box donnent des vitesses absurdes. Au-delà de
40 km/h, je considère que c'est une erreur et la valeur est ignorée.

## Limites et idées pour la suite

- La calibration est manuelle et ne suit pas les zooms de la caméra. L'idéal serait de
  détecter automatiquement les lignes du terrain.
- Toute la vidéo est chargée en mémoire : ça va pour un extrait, pas pour un match entier.
- Les gardiens peuvent être mal classés puisque leur maillot est différent.
- Quand un joueur est caché longtemps, il peut changer de numéro.
- La possession est basée uniquement sur la distance au ballon.
- Ce que j'aimerais ajouter : des heatmaps par joueur, une minimap vue de dessus, un
  dashboard Streamlit.
