"""
Outil de calibration : cliquer 4 points du terrain sur la 1re frame de la vidéo
pour obtenir PIXEL_VERTICES (à copier dans config.py).

Ordre des clics : bas-gauche, haut-gauche, haut-droit, bas-droit.
Choisir des repères dont on connaît la distance réelle (coins de surface,
ligne médiane, lignes de touche...).

Usage : python tools/pick_points.py input_videos/match.mp4
"""
import sys

import cv2

LABELS = ["bas-gauche", "haut-gauche", "haut-droit", "bas-droit"]


def main(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Impossible de lire {video_path}")

    points = []
    display = frame.copy()

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(display, LABELS[len(points) - 1], (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.namedWindow("calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibration", on_click)
    print("Cliquez 4 points :", " -> ".join(LABELS), "| 'q' pour quitter")
    while len(points) < 4:
        cv2.imshow("calibration", display)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

    if len(points) == 4:
        print("\nÀ copier dans config.py :\nPIXEL_VERTICES = np.array([")
        for p in points:
            print(f"    [{p[0]}, {p[1]}],")
        print("], dtype=np.float32)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "input_videos/match.mp4")
