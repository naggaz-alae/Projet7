"""
Estimation du mouvement de caméra par flux optique (Lucas-Kanade).

Idée : les points d'intérêt du décor (lignes, pelouse, panneaux) ne bougent pas
dans le monde réel. Leur déplacement dans l'image = mouvement de la caméra.
On masque les joueurs/arbitres/ballon pour ne suivre QUE le décor.

On calcule un déplacement par frame (médiane des vecteurs de flux, robuste aux
outliers), puis on le CUMULE : chaque position est ramenée dans le repère de la
première frame, qui est aussi celle utilisée pour calibrer la perspective.
"""
import os
import pickle

import cv2
import numpy as np

import config


class CameraMovementEstimator:
    def __init__(self, first_frame):
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=3, blockSize=7)
        self.frame_shape = first_frame.shape[:2]

    def _static_mask(self, frame_tracks: dict | None):
        mask = np.full(self.frame_shape, 255, dtype=np.uint8)
        if frame_tracks:
            for obj_tracks in frame_tracks.values():
                for info in obj_tracks.values():
                    x1, y1, x2, y2 = (int(v) for v in info["bbox"])
                    mask[max(0, y1 - 5):y2 + 5, max(0, x1 - 5):x2 + 5] = 0
        return mask

    def get_camera_movement(self, frames, tracks=None, read_from_stub=False, stub_path=None):
        """Retourne une liste [dx, dy] CUMULÉE par frame (repère = frame 0)."""
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                print(f"  Mouvement caméra chargé depuis le cache {stub_path}")
                return pickle.load(f)

        def tracks_at(i):
            return None if tracks is None else {k: v[i] for k, v in tracks.items()}

        movement = [[0.0, 0.0]]
        cum_x = cum_y = 0.0
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frames)):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            old_pts = cv2.goodFeaturesToTrack(prev_gray, mask=self._static_mask(tracks_at(i - 1)),
                                              **self.feature_params)
            dx = dy = 0.0
            if old_pts is not None and len(old_pts) > 0:
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, old_pts, None,
                                                               **self.lk_params)
                good = status.reshape(-1) == 1
                if good.any():
                    flow = (new_pts[good] - old_pts[good]).reshape(-1, 2)
                    mdx, mdy = np.median(flow, axis=0)
                    if np.hypot(mdx, mdy) > config.CAMERA_MIN_DISTANCE / 5:  # filtre le bruit
                        dx, dy = float(mdx), float(mdy)
            cum_x += dx
            cum_y += dy
            movement.append([cum_x, cum_y])
            prev_gray = gray

        if stub_path:
            os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(movement, f)
        return movement

    @staticmethod
    def add_adjust_positions_to_tracks(tracks, camera_movement):
        """position_adjusted = position image - mouvement caméra cumulé."""
        for obj_tracks in tracks.values():
            for frame_num, frame_tracks in enumerate(obj_tracks):
                cx, cy = camera_movement[frame_num]
                for info in frame_tracks.values():
                    x, y = info["position"]
                    info["position_adjusted"] = (x - cx, y - cy)

    @staticmethod
    def draw_camera_movement(frames, camera_movement):
        output = []
        for frame, (cx, cy) in zip(frames, camera_movement):
            frame = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (430, 100), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"Camera X : {cx:.1f} px", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(frame, f"Camera Y : {cy:.1f} px", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            output.append(frame)
        return output
