"""
Transformation de perspective (homographie) : pixels -> mètres sur le terrain.

On donne 4 points du terrain dans l'image (trapèze à cause de la perspective)
et leurs coordonnées réelles (rectangle en mètres). OpenCV calcule la matrice
3x3 H telle que  [x', y', w]^T = H · [u, v, 1]^T  puis  (X, Y) = (x'/w, y'/w).
"""
import cv2
import numpy as np

import config


class ViewTransformer:
    def __init__(self,
                 pixel_vertices: np.ndarray = config.PIXEL_VERTICES,
                 field_length: float = config.FIELD_LENGTH_M,
                 field_width: float = config.FIELD_WIDTH_M):
        self.pixel_vertices = np.asarray(pixel_vertices, dtype=np.float32)
        # même ordre que pixel_vertices : bas-gauche, haut-gauche, haut-droit, bas-droit
        self.target_vertices = np.array([
            [0, field_width],
            [0, 0],
            [field_length, 0],
            [field_length, field_width],
        ], dtype=np.float32)
        self.matrix = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def is_inside(self, point) -> bool:
        return cv2.pointPolygonTest(self.pixel_vertices, (float(point[0]), float(point[1])), False) >= 0

    def transform_point(self, point):
        """Retourne (X, Y) en mètres, ou None si le point est hors de la zone calibrée."""
        if not self.is_inside(point):
            return None
        p = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.matrix).reshape(2).tolist()

    def add_transformed_position_to_tracks(self, tracks):
        for obj_tracks in tracks.values():
            for frame_tracks in obj_tracks:
                for info in frame_tracks.values():
                    info["position_transformed"] = self.transform_point(info["position_adjusted"])
