"""
Vitesse instantanée (km/h) et distance cumulée (m) par joueur.

On travaille sur des fenêtres de N frames plutôt que frame par frame :
le bruit du tracking (bbox qui "tremble" de quelques pixels) serait sinon
amplifié en vitesses absurdes.
"""
import os

import cv2
import pandas as pd

import config
from utils import measure_distance, get_foot_position


class SpeedAndDistanceEstimator:
    def __init__(self, fps: float, frame_window: int = config.SPEED_FRAME_WINDOW):
        self.fps = fps
        self.frame_window = frame_window

    def add_speed_and_distance_to_tracks(self, tracks, objects=("players",)):
        for obj in objects:
            obj_tracks = tracks[obj]
            n = len(obj_tracks)
            total_distance: dict[int, float] = {}

            for start in range(0, n, self.frame_window):
                last = min(start + self.frame_window, n - 1)
                if last <= start:
                    break
                for track_id, info in obj_tracks[start].items():
                    if track_id not in obj_tracks[last]:
                        continue
                    p0 = info.get("position_transformed")
                    p1 = obj_tracks[last][track_id].get("position_transformed")
                    if p0 is None or p1 is None:
                        continue

                    dist = measure_distance(p0, p1)
                    elapsed = (last - start) / self.fps
                    speed_kmh = dist / elapsed * 3.6
                    if speed_kmh > config.MAX_HUMAN_SPEED_KMH:
                        continue  # saut d'ID ou erreur de projection

                    total_distance[track_id] = total_distance.get(track_id, 0.0) + dist
                    for f in range(start, last):
                        if track_id in obj_tracks[f]:
                            obj_tracks[f][track_id]["speed"] = speed_kmh
                            obj_tracks[f][track_id]["distance"] = total_distance[track_id]

    @staticmethod
    def draw_speed_and_distance(frames, tracks):
        output = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            for obj, obj_tracks in tracks.items():
                if obj in ("ball", "referees"):
                    continue
                for info in obj_tracks[frame_num].values():
                    if "speed" not in info:
                        continue
                    x, y = get_foot_position(info["bbox"])
                    y += 40
                    for dy, text in ((0, f"{info['speed']:.1f} km/h"), (20, f"{info['distance']:.1f} m")):
                        cv2.putText(frame, text, (x - 40, y + dy), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 0), 2)
            output.append(frame)
        return output

    @staticmethod
    def export_player_stats(tracks, csv_path: str) -> pd.DataFrame:
        """Tableau récapitulatif par joueur : utile pour une analyse hors vidéo (pandas, BI...)."""
        rows = []
        for frame_num, frame_tracks in enumerate(tracks["players"]):
            for track_id, info in frame_tracks.items():
                rows.append({
                    "frame": frame_num,
                    "player_id": track_id,
                    "team": info.get("team"),
                    "speed_kmh": info.get("speed"),
                    "distance_m": info.get("distance"),
                    "has_ball": info.get("has_ball", False),
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        summary = (df.groupby("player_id")
                     .agg(team=("team", lambda s: s.mode().iat[0] if not s.mode().empty else None),
                          frames_visible=("frame", "count"),
                          total_distance_m=("distance_m", "max"),
                          max_speed_kmh=("speed_kmh", "max"),
                          avg_speed_kmh=("speed_kmh", "mean"),
                          frames_with_ball=("has_ball", "sum"))
                     .round(2)
                     .sort_values("total_distance_m", ascending=False))
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        summary.to_csv(csv_path)
        return summary
