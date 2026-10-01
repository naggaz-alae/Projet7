"""
Classification non supervisée des joueurs en 2 équipes via K-Means.

Étape 1 (par joueur) : K-Means à 2 clusters sur les pixels de la moitié haute
    de la bbox (le maillot). Le cluster majoritaire dans les coins = fond (pelouse),
    l'autre = couleur du maillot.
Étape 2 (global) : K-Means à 2 clusters sur les couleurs de maillot de tous les
    joueurs de la frame de référence -> 2 centroïdes = 2 équipes.
Stabilisation : vote majoritaire sur les premières observations de chaque ID
    pour éviter qu'un joueur change d'équipe à cause d'une occlusion.
"""
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import KMeans

VOTES_BEFORE_LOCK = 10


class TeamAssigner:
    def __init__(self):
        self.team_colors: dict[int, np.ndarray] = {}
        self.kmeans: KMeans | None = None
        self._votes: dict[int, Counter] = defaultdict(Counter)
        self._locked: dict[int, int] = {}

    @staticmethod
    def _crop_top_half(frame, bbox):
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        return crop[: max(1, crop.shape[0] // 2), :]

    def get_player_color(self, frame, bbox) -> np.ndarray:
        top = self._crop_top_half(frame, bbox)
        if top.size == 0 or top.shape[0] < 2 or top.shape[1] < 2:
            return np.zeros(3)

        pixels = top.reshape(-1, 3).astype(np.float64)
        km = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=0).fit(pixels)
        labels = km.labels_.reshape(top.shape[:2])

        corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        background = max(set(corners), key=corners.count)
        return km.cluster_centers_[1 - background].astype(np.float64)

    def assign_team_color(self, frame, player_detections: dict) -> None:
        colors = [self.get_player_color(frame, d["bbox"]) for d in player_detections.values()]
        if len(colors) < 2:
            raise ValueError("Pas assez de joueurs sur la frame de référence pour séparer 2 équipes.")
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0).fit(
            np.asarray(colors, dtype=np.float64))
        self.team_colors = {1: self.kmeans.cluster_centers_[0], 2: self.kmeans.cluster_centers_[1]}

    def get_player_team(self, frame, bbox, player_id: int) -> int:
        if player_id in self._locked:
            return self._locked[player_id]

        color = self.get_player_color(frame, bbox)
        team = int(self.kmeans.predict(color.reshape(1, -1))[0]) + 1
        self._votes[player_id][team] += 1

        if sum(self._votes[player_id].values()) >= VOTES_BEFORE_LOCK:
            self._locked[player_id] = self._votes[player_id].most_common(1)[0][0]
        return self._votes[player_id].most_common(1)[0][0]
