"""
Tests unitaires des briques "pures" (sans YOLO ni vidéo).
Lancer : pytest -q
"""
import numpy as np
import pytest

from player_ball_assigner import PlayerBallAssigner
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from team_assigner import TeamAssigner
from utils import get_center_of_bbox, get_foot_position, measure_distance
from view_transformer import ViewTransformer


def test_bbox_helpers():
    bbox = [10, 20, 30, 60]
    assert get_center_of_bbox(bbox) == (20, 40)
    assert get_foot_position(bbox) == (20, 60)
    assert measure_distance((0, 0), (3, 4)) == 5


def test_view_transformer_maps_corners_to_meters():
    pixels = np.array([[0, 100], [0, 0], [200, 0], [200, 100]], dtype=np.float32)
    vt = ViewTransformer(pixels, field_length=105, field_width=68)
    assert vt.transform_point((0, 0)) == pytest.approx([0, 0], abs=1e-3)
    assert vt.transform_point((200, 100)) == pytest.approx([105, 68], abs=1e-3)
    assert vt.transform_point((100, 50)) == pytest.approx([52.5, 34], abs=1e-3)
    assert vt.transform_point((500, 500)) is None  # hors zone calibrée


def test_ball_assigned_to_closest_player():
    players = {1: {"bbox": [0, 0, 20, 100]}, 2: {"bbox": [300, 0, 320, 100]}}
    ball = [15, 95, 25, 105]
    assert PlayerBallAssigner(max_distance=70).assign_ball_to_player(players, ball) == 1
    far_ball = [150, 95, 160, 105]
    assert PlayerBallAssigner(max_distance=70).assign_ball_to_player(players, far_ball) == -1


def test_speed_of_player_running_at_known_speed():
    # joueur qui avance de 1 m par frame à 25 fps -> 25 m/s = 90 km/h (filtré, > 40 km/h)
    # on prend 0.2 m/frame -> 5 m/s = 18 km/h
    fps, n = 25, 26
    tracks = {"players": [{7: {"position_transformed": [i * 0.2, 0.0]}} for i in range(n)]}
    SpeedAndDistanceEstimator(fps=fps, frame_window=5).add_speed_and_distance_to_tracks(tracks)
    assert tracks["players"][0][7]["speed"] == pytest.approx(18.0, rel=1e-3)
    assert tracks["players"][20][7]["distance"] == pytest.approx(5.0, rel=1e-3)


def test_team_assigner_separates_two_jersey_colors():
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[:] = (0, 128, 0)  # pelouse verte
    bboxes = {}
    for i in range(4):
        x = 20 + i * 90
        color = (0, 0, 255) if i % 2 == 0 else (255, 255, 255)  # rouge / blanc
        frame[20:60, x + 5:x + 35] = color  # maillot au centre, fond aux coins
        bboxes[i] = {"bbox": [x, 10, x + 40, 110]}

    ta = TeamAssigner()
    ta.assign_team_color(frame, bboxes)
    teams = [ta.get_player_team(frame, b["bbox"], pid) for pid, b in bboxes.items()]
    assert teams[0] == teams[2] and teams[1] == teams[3] and teams[0] != teams[1]
