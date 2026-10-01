"""
Détection (YOLO) + suivi multi-objets (ByteTrack via supervision) + dessin des overlays.

Structure des tracks produite :
    tracks = {
        "players":  [ {track_id: {"bbox": [x1,y1,x2,y2], ...}, ...}, ... ],  # une entrée par frame
        "referees": [ ... ],
        "ball":     [ {1: {"bbox": [...]}}, ... ],
    }
"""
import os
import pickle

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

import config
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    # ------------------------------------------------------------------ détection
    def detect_frames(self, frames):
        detections = []
        for i in range(0, len(frames), config.BATCH_SIZE):
            batch = frames[i:i + config.BATCH_SIZE]
            detections += self.model.predict(batch, conf=config.DETECTION_CONF, verbose=False)
            print(f"  YOLO : {min(i + config.BATCH_SIZE, len(frames))}/{len(frames)} frames", end="\r")
        print()
        return detections

    # ------------------------------------------------------------------ tracking
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                print(f"  Tracks chargés depuis le cache {stub_path}")
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            names = detection.names  # {id: "player", ...}
            det_sv = sv.Detections.from_ultralytics(detection)

            # Catégorie de chaque détection selon son nom de classe
            categories = np.array([self._category(names[int(c)]) for c in det_sv.class_id], dtype="<U10")

            # Joueurs + arbitres passent dans ByteTrack (identités persistantes).
            # Le ballon n'est pas tracké : un seul ballon, on garde la détection la plus sûre.
            people = det_sv[np.isin(categories, ["player", "referee"])]
            tracked = self.tracker.update_with_detections(people)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for xyxy, _, _, class_id, track_id, _ in tracked:
                if track_id is None:
                    continue
                cat = self._category(names[int(class_id)])
                key = "players" if cat == "player" else "referees"
                tracks[key][frame_num][int(track_id)] = {"bbox": xyxy.tolist()}

            balls = det_sv[categories == "ball"]
            if len(balls) > 0:
                best = int(np.argmax(balls.confidence))
                tracks["ball"][frame_num][1] = {"bbox": balls.xyxy[best].tolist()}

        if stub_path:
            os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
        return tracks

    @staticmethod
    def _category(class_name: str) -> str:
        if class_name in config.PLAYER_CLASSES:
            return "player"
        if class_name in config.REFEREE_CLASSES:
            return "referee"
        if class_name in config.BALL_CLASSES:
            return "ball"
        return "other"

    # ------------------------------------------------------------------ post-traitement
    @staticmethod
    def add_position_to_tracks(tracks):
        """Position de référence : pieds pour les humains, centre pour le ballon."""
        for obj, obj_tracks in tracks.items():
            for frame_tracks in obj_tracks:
                for info in frame_tracks.values():
                    bbox = info["bbox"]
                    info["position"] = get_center_of_bbox(bbox) if obj == "ball" else get_foot_position(bbox)

    @staticmethod
    def interpolate_ball_positions(ball_tracks):
        """Le ballon est souvent perdu (petit, flou, masqué) : on interpole linéairement les trous."""
        rows = [frame.get(1, {}).get("bbox", [np.nan] * 4) for frame in ball_tracks]
        df = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2"])
        df = df.interpolate().bfill().ffill()
        if df.isna().any().any():  # aucun ballon détecté dans toute la vidéo
            return ball_tracks
        return [{1: {"bbox": row}} for row in df.to_numpy().tolist()]

    # ------------------------------------------------------------------ dessin
    @staticmethod
    def draw_ellipse(frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = int(get_bbox_width(bbox))

        cv2.ellipse(frame, (x_center, y2), (width, int(0.35 * width)), 0.0, -45, 235,
                    color, 2, cv2.LINE_4)

        if track_id is not None:
            rw, rh = 40, 20
            x1, x2 = x_center - rw // 2, x_center + rw // 2
            y1, y2r = y2 - rh // 2 + 15, y2 + rh // 2 + 15
            cv2.rectangle(frame, (x1, y1), (x2, y2r), color, cv2.FILLED)
            tx = x1 + 12 - (10 if track_id > 99 else 0)
            cv2.putText(frame, str(track_id), (tx, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

    @staticmethod
    def draw_triangle(frame, bbox, color):
        """Petit triangle au-dessus de l'objet (ballon ou porteur du ballon)."""
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        pts = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]], dtype=np.int32)
        cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)
        return frame

    @staticmethod
    def draw_team_ball_control(frame, frame_num, team_ball_control, team_colors):
        """HUD semi-transparent avec le % de possession cumulé jusqu'à la frame courante."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = w - 560, h - 150, w - 20, h - 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        history = team_ball_control[:frame_num + 1]
        t1 = int(np.sum(history == 1))
        t2 = int(np.sum(history == 2))
        total = max(t1 + t2, 1)

        for i, (team, count) in enumerate(((1, t1), (2, t2))):
            y = y1 + 45 + i * 50
            color = tuple(int(c) for c in team_colors.get(team, (0, 0, 0)))
            cv2.circle(frame, (x1 + 25, y - 8), 12, color, -1)
            cv2.putText(frame, f"Equipe {team} - Possession : {count / total * 100:.1f}%",
                        (x1 + 50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return frame

    def draw_annotations(self, frames, tracks, team_ball_control, team_colors):
        output = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            for track_id, player in tracks["players"][frame_num].items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], config.BALL_HOLDER_COLOR)

            for referee in tracks["referees"][frame_num].values():
                frame = self.draw_ellipse(frame, referee["bbox"], config.REFEREE_COLOR)

            for ball in tracks["ball"][frame_num].values():
                frame = self.draw_triangle(frame, ball["bbox"], config.BALL_COLOR)

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, team_colors)
            output.append(frame)
        return output
