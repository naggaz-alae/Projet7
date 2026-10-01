# ⚽ Football Analysis — Analyse vidéo de match par Computer Vision

Pipeline Python qui transforme une vidéo de match en données tactiques : détection et suivi
des joueurs, arbitres et ballon, séparation automatique des deux équipes, possession de balle,
compensation du mouvement de caméra, projection sur le terrain réel (105 m × 68 m), vitesse et
distance parcourue par joueur — le tout restitué dans une vidéo annotée et un fichier CSV.

> Projet personnel réalisé par **Alae-eddine Naggaz** dans le cadre de ma recherche d'alternance
> en Data / Machine Learning / Computer Vision.

---

## 🎯 Fonctionnalités

| # | Fonctionnalité | Technique |
|---|---|---|
| 1 | Détection joueurs / arbitres / ballon | YOLOv8 (Ultralytics), fine-tunable |
| 2 | Suivi avec identifiants persistants | ByteTrack (`supervision`) |
| 3 | Séparation des équipes par couleur de maillot | K-Means (scikit-learn), non supervisé |
| 4 | Possession de balle dynamique | Distance pied–ballon + historique |
| 5 | Compensation du mouvement de caméra | Flux optique Lucas-Kanade (OpenCV) |
| 6 | Pixels → mètres | Homographie (géométrie projective) |
| 7 | Vitesse (km/h) & distance (m) par joueur | Fenêtre glissante + filtrage des aberrations |
| 8 | Vidéo annotée + export statistiques | OpenCV + pandas |

---

## 🏗️ Architecture

```
vidéo ─► [Tracker] YOLO + ByteTrack ─► tracks {players, referees, ball}
                │  (cache pickle : stubs/track_stubs.pkl)
                ▼
     position (pieds / centre du ballon)
                ▼
     [CameraMovementEstimator] flux optique ─► position_adjusted
                ▼
     [ViewTransformer] homographie ─► position_transformed (mètres)
                ▼
     [Tracker] interpolation du ballon (pandas)
                ▼
     [SpeedAndDistanceEstimator] ─► speed (km/h), distance (m)
                ▼
     [TeamAssigner] K-Means ─► team, team_color
                ▼
     [PlayerBallAssigner] ─► has_ball, possession par équipe
                ▼
     Rendu : ellipses, triangles, HUD possession, vitesses ─► vidéo .mp4 + CSV
```

```
football_analysis/
├── main.py                          # orchestration du pipeline (CLI)
├── config.py                        # tous les paramètres ajustables
├── trackers/                        # YOLO + ByteTrack + dessin des overlays
├── team_assigner/                   # K-Means couleurs de maillot
├── player_ball_assigner/            # attribution du ballon
├── camera_movement_estimator/       # flux optique
├── view_transformer/                # homographie pixels -> mètres
├── speed_and_distance_estimator/    # vitesse, distance, export CSV
├── utils/                           # I/O vidéo, géométrie des bbox
├── tools/pick_points.py             # calibration du terrain (clic sur 4 points)
├── training/train_yolo.py           # fine-tuning YOLO sur dataset football
└── tests/test_core.py               # tests unitaires (pytest)
```

Chaque module a **une seule responsabilité** et communique via la même structure `tracks`,
ce qui permet de tester, remplacer ou améliorer une brique sans toucher aux autres.

---

## 🚀 Installation & utilisation

```bash
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

1. Placer une vidéo dans `input_videos/match.mp4` (plan large type caméra TV).
2. **Calibrer le terrain** (une fois par vidéo) :
   ```bash
   python tools/pick_points.py input_videos/match.mp4
   ```
   Cliquer 4 repères du terrain, copier `PIXEL_VERTICES` dans `config.py`, et renseigner
   `FIELD_LENGTH_M` / `FIELD_WIDTH_M` = dimensions réelles de la zone délimitée.
3. Lancer le pipeline :
   ```bash
   python main.py                                  # 1er run : YOLO tourne et crée le cache
   python main.py --use-stubs                      # runs suivants : quasi instantané
   python main.py --max-frames 300 --show-camera   # test rapide + debug caméra
   ```
4. Résultats : `output_videos/match_annotated.mp4` et `output_videos/player_stats.csv`.

Tests : `pytest -q`

### Modèle YOLO
Sans `models/best.pt`, le pipeline utilise `yolov8x.pt` (COCO) : ça fonctionne mais les
arbitres sont confondus avec les joueurs et le ballon est souvent raté. Pour de vrais
résultats, fine-tuner sur un dataset football (ex. *football-players-detection* sur Roboflow
Universe) — idéalement sur GPU (Google Colab) :
```bash
python training/train_yolo.py --data datasets/football/data.yaml --epochs 100
cp runs/detect/train/weights/best.pt models/best.pt
```

---

## 🧠 Choix techniques (et pourquoi)

**Seuil de confiance bas (0.1) + ByteTrack.** ByteTrack exploite aussi les détections à faible
score pour maintenir les identités lors d'occlusions ; filtrer trop tôt ferait perdre des pistes.

**Le ballon n'est pas tracké mais interpolé.** Il n'y a qu'un ballon : on garde la détection
la plus confiante par frame, puis `pandas.interpolate()` comble les frames où il est invisible.

**K-Means en deux niveaux.** (1) Par joueur, 2 clusters sur la moitié haute de la bbox : le
cluster dominant dans les coins est la pelouse, l'autre est le maillot. (2) Global, 2 clusters
sur tous les maillots = 2 équipes. Un **vote majoritaire** sur les 10 premières observations
d'un ID évite qu'un joueur change d'équipe à cause d'une occlusion.

**Flux optique sur le décor uniquement.** Les joueurs sont masqués pour ne suivre que des
points statiques. On prend la **médiane** des vecteurs (robuste aux outliers) et on **cumule**
le déplacement : toutes les positions sont ramenées dans le repère de la première frame, qui est
aussi celle utilisée pour la calibration de l'homographie — les deux étapes sont cohérentes.

**Homographie à 4 points.** Un plan (le terrain) vu en perspective est relié à sa vue de dessus
par une matrice 3×3 ; `cv2.getPerspectiveTransform` la calcule à partir de 4 correspondances.
Les points hors de la zone calibrée sont ignorés (projection non fiable).

**Vitesse sur fenêtre de 5 frames + plafond à 40 km/h.** Mesurer frame par frame amplifie le
« tremblement » des bbox. Tout ce qui dépasse une vitesse humaine plausible est traité comme
une erreur (changement d'ID, mauvaise projection) et exclu.

**Cache pickle (stubs).** L'inférence YOLO est l'étape la plus lente ; le cache permet
d'itérer sur le reste du pipeline en quelques secondes.

---

## ⚠️ Limites connues (et pistes d'amélioration)

- **Calibration manuelle et fixe** : l'homographie ne suit pas les zooms. → Détection
  automatique des lignes/points clés du terrain (modèle de keypoints) pour recalculer
  l'homographie à chaque frame.
- **Frames chargées en mémoire** : adapté à des clips, pas à un match entier. → Traitement
  en streaming (générateur de frames).
- **Gardiens et arbitres** : leur maillot diffère ; un gardien peut être classé dans la mauvaise
  équipe. → Règle géographique (côté du terrain) ou classe dédiée.
- **Changements d'ID** lors d'occlusions longues. → Ré-identification par apparence
  (embeddings type OSNet / BoT-SORT).
- **Possession** = proximité du ballon, pas contrôle réel. → Ajouter la vitesse relative
  joueur/ballon, détecter passes et tirs.
- Pistes bonus : heatmaps par joueur, minimap 2D temps réel, dashboard Streamlit, API FastAPI.

---

## 📚 Ce que ce projet m'a permis de travailler

- Détection d'objets et fine-tuning (YOLO), métriques mAP
- Multi-object tracking et gestion des identités
- Apprentissage non supervisé (clustering) appliqué à l'image
- Vision géométrique : flux optique, homographie, changements de repère
- Manipulation de données temporelles (pandas : interpolation, agrégations)
- Architecture modulaire, configuration centralisée, cache, tests unitaires
