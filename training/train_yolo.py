"""
Fine-tuning de YOLO sur un dataset football (player / goalkeeper / referee / ball).

Pourquoi fine-tuner ? Le modèle COCO générique détecte des "person" (dont le public
et les ramasseurs de balles) et rate souvent le ballon, petit et flou.
Un modèle spécialisé distingue arbitres/joueurs et détecte bien mieux le ballon.

Dataset : n'importe quel dataset au format YOLO, par ex. le dataset public
"football-players-detection" sur Roboflow Universe (export "YOLOv8").

Usage (idéalement sur GPU, ex. Google Colab) :
    python training/train_yolo.py --data datasets/football/data.yaml --epochs 100
Puis copier runs/detect/train/weights/best.pt dans models/best.pt
"""
import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Chemin du data.yaml")
    p.add_argument("--base", default="yolov8x.pt", help="Poids de départ (yolov8n/s/m/l/x)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    model = YOLO(args.base)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=20)
    metrics = model.val()
    print(f"mAP50 : {metrics.box.map50:.3f} | mAP50-95 : {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
