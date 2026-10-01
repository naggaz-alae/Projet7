"""
Configuration centrale du pipeline.
Tous les "nombres magiques" du projet sont regroupés ici pour pouvoir
les ajuster sans toucher au code métier.
"""
import numpy as np

# --- Chemins -----------------------------------------------------------------
INPUT_VIDEO = "input_videos/match.mp4"
OUTPUT_VIDEO = "output_videos/match_annotated.mp4"
OUTPUT_STATS_CSV = "output_videos/player_stats.csv"
MODEL_PATH = "models/best.pt"          # YOLO fine-tuné (voir training/train_yolo.py)
FALLBACK_MODEL = "yolov8x.pt"          # modèle COCO générique si best.pt absent

TRACKS_STUB = "stubs/track_stubs.pkl"
CAMERA_STUB = "stubs/camera_movement_stub.pkl"

# --- Détection ---------------------------------------------------------------
DETECTION_CONF = 0.1        # seuil bas : on laisse ByteTrack filtrer ensuite
BATCH_SIZE = 20

# Mapping des noms de classes YOLO vers nos 3 catégories.
# Couvre le modèle fine-tuné (player/goalkeeper/referee/ball)
# ET le modèle COCO générique (person/sports ball).
PLAYER_CLASSES = {"player", "goalkeeper", "person"}
REFEREE_CLASSES = {"referee"}
BALL_CLASSES = {"ball", "sports ball"}

# --- Possession --------------------------------------------------------------
MAX_PLAYER_BALL_DISTANCE = 70   # pixels entre pied du joueur et ballon

# --- Mouvement caméra --------------------------------------------------------
CAMERA_MIN_DISTANCE = 5         # px : en dessous, on considère la caméra fixe

# --- Transformation de perspective -------------------------------------------
# 4 points du terrain dans l'image (en pixels), dans l'ordre :
# bas-gauche, haut-gauche, haut-droit, bas-droit.
# ⚠️ À recalibrer pour CHAQUE vidéo avec tools/pick_points.py
PIXEL_VERTICES = np.array([
    [110, 1035],
    [265, 275],
    [910, 260],
    [1640, 915],
], dtype=np.float32)

# Dimensions réelles (en mètres) de la zone délimitée par les 4 points.
# Terrain entier = 105 x 68. Si seule une portion est visible, mettre
# la longueur réelle de cette portion (ex. distance entre deux lignes connues).
FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0

# --- Vitesse / distance ------------------------------------------------------
SPEED_FRAME_WINDOW = 5          # calcul de la vitesse sur N frames (lissage)
MAX_HUMAN_SPEED_KMH = 40.0      # au-delà : artefact de tracking, ignoré

# --- Couleurs (BGR) ----------------------------------------------------------
REFEREE_COLOR = (0, 255, 255)
BALL_COLOR = (0, 255, 0)
BALL_HOLDER_COLOR = (0, 0, 255)
