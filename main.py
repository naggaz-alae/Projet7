"""
Point d'entrée du pipeline d'analyse vidéo de football.

Usage :
    python main.py --input input_videos/match.mp4
    python main.py --input input_videos/match.mp4 --use-stubs      # réutilise le cache YOLO
    python main.py --max-frames 300 --show-camera                  # test rapide
"""
import argparse
import os
import time

import numpy as np

import config
from camera_movement_estimator import CameraMovementEstimator
from player_ball_assigner import PlayerBallAssigner
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from team_assigner import TeamAssigner
from trackers import Tracker
from utils import read_video, save_video, get_video_fps
from view_transformer import ViewTransformer


def parse_args():
    p = argparse.ArgumentParser(description="Analyse vidéo de football (YOLO + ByteTrack + K-Means + homographie)")
    p.add_argument("--input", default=config.INPUT_VIDEO)
    p.add_argument("--output", default=config.OUTPUT_VIDEO)
    p.add_argument("--model", default=None, help="Poids YOLO (défaut : models/best.pt sinon yolov8x.pt)")
    p.add_argument("--use-stubs", action="store_true", help="Charger tracks / caméra depuis le cache pickle")
    p.add_argument("--max-frames", type=int, default=None, help="Limiter le nombre de frames (debug)")
    p.add_argument("--show-camera", action="store_true", help="Afficher le mouvement caméra estimé")
    return p.parse_args()


def assign_teams(frames, tracks):
    """Fit K-Means sur une frame de référence puis attribue une équipe à chaque joueur."""
    ref = next((i for i, f in enumerate(tracks["players"]) if len(f) >= 6), 0)
    assigner = TeamAssigner()
    assigner.assign_team_color(frames[ref], tracks["players"][ref])

    for frame_num, players in enumerate(tracks["players"]):
        for player_id, info in players.items():
            team = assigner.get_player_team(frames[frame_num], info["bbox"], player_id)
            info["team"] = team
            info["team_color"] = tuple(int(c) for c in assigner.team_colors[team])
    return assigner.team_colors


def compute_possession(tracks):
    """Possession : équipe du joueur le plus proche du ballon. Si personne, on garde la dernière."""
    assigner = PlayerBallAssigner()
    control, last_team = [], 0
    for frame_num, players in enumerate(tracks["players"]):
        ball = tracks["ball"][frame_num].get(1)
        holder = assigner.assign_ball_to_player(players, ball["bbox"]) if ball else -1
        if holder != -1:
            players[holder]["has_ball"] = True
            last_team = players[holder]["team"]
        control.append(last_team)
    return np.array(control)


def main():
    args = parse_args()
    t0 = time.time()

    model_path = args.model or (config.MODEL_PATH if os.path.exists(config.MODEL_PATH) else config.FALLBACK_MODEL)

    print("[1/8] Lecture de la vidéo")
    frames = read_video(args.input, max_frames=args.max_frames)
    fps = get_video_fps(args.input)
    print(f"  {len(frames)} frames @ {fps:.1f} fps")

    print(f"[2/8] Détection + tracking ({model_path})")
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(frames, read_from_stub=args.use_stubs, stub_path=config.TRACKS_STUB)
    tracker.add_position_to_tracks(tracks)

    print("[3/8] Mouvement de caméra (flux optique)")
    cam = CameraMovementEstimator(frames[0])
    camera_movement = cam.get_camera_movement(frames, tracks, read_from_stub=args.use_stubs,
                                              stub_path=config.CAMERA_STUB)
    cam.add_adjust_positions_to_tracks(tracks, camera_movement)

    print("[4/8] Transformation de perspective (pixels -> mètres)")
    ViewTransformer().add_transformed_position_to_tracks(tracks)

    print("[5/8] Interpolation du ballon")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    tracker.add_position_to_tracks({"ball": tracks["ball"]})

    print("[6/8] Vitesse et distance")
    sde = SpeedAndDistanceEstimator(fps=fps)
    sde.add_speed_and_distance_to_tracks(tracks)

    print("[7/8] Équipes (K-Means) + possession")
    team_colors = assign_teams(frames, tracks)
    team_ball_control = compute_possession(tracks)

    print("[8/8] Rendu vidéo + export des stats")
    out = tracker.draw_annotations(frames, tracks, team_ball_control, team_colors)
    if args.show_camera:
        out = cam.draw_camera_movement(out, camera_movement)
    out = sde.draw_speed_and_distance(out, tracks)
    save_video(out, args.output, fps=fps)

    stats = sde.export_player_stats(tracks, config.OUTPUT_STATS_CSV)
    print(f"\nVidéo : {args.output}\nStats : {config.OUTPUT_STATS_CSV}")
    if not stats.empty:
        print(stats.head(10).to_string())
    print(f"\nTerminé en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
