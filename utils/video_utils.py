"""Lecture / écriture vidéo avec OpenCV."""
import os
import cv2


def get_video_fps(video_path: str, default: float = 24.0) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else default


def read_video(video_path: str, max_frames: int | None = None) -> list:
    """Charge toutes les frames en mémoire (simple et suffisant pour des clips courts).

    Pour des matchs entiers, il faudrait passer à un traitement en streaming.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Vidéo introuvable : {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def save_video(frames: list, output_path: str, fps: float = 24.0) -> None:
    if not frames:
        raise ValueError("Aucune frame à sauvegarder.")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
