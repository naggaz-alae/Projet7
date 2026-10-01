"""Attribue le ballon au joueur dont un pied est le plus proche (sous un seuil)."""
import config
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    def __init__(self, max_distance: float = config.MAX_PLAYER_BALL_DISTANCE):
        self.max_distance = max_distance

    def assign_ball_to_player(self, players: dict, ball_bbox) -> int:
        """Retourne l'ID du joueur en possession, ou -1 si personne n'est assez proche."""
        ball_pos = get_center_of_bbox(ball_bbox)
        best_id, best_dist = -1, float("inf")

        for player_id, player in players.items():
            x1, _, x2, y2 = player["bbox"]
            # distance minimale entre le ballon et les deux "pieds" (coins bas de la bbox)
            dist = min(measure_distance((x1, y2), ball_pos),
                       measure_distance((x2, y2), ball_pos))
            if dist < self.max_distance and dist < best_dist:
                best_id, best_dist = player_id, dist
        return best_id
