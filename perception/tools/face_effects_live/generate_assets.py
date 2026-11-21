"""Generate original demo PNGs; no external art or network dependency."""
from pathlib import Path
import json
import cv2
import numpy as np


def generate(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    crown = np.zeros((100, 240, 4), np.uint8)
    cv2.fillPoly(crown, [np.array([[15, 85], [5, 15], [65, 45], [120, 5], [175, 45], [235, 15], [225, 85]])], (30, 190, 255, 255))
    cv2.rectangle(crown, (15, 75), (225, 95), (80, 130, 245, 255), -1)
    cv2.imwrite(str(root / "crown.png"), crown)
    glasses = np.zeros((90, 240, 4), np.uint8)
    cv2.ellipse(glasses, (62, 44), (49, 31), 0, 0, 360, (90, 35, 25, 245), 8, cv2.LINE_AA)
    cv2.ellipse(glasses, (178, 44), (49, 31), 0, 0, 360, (90, 35, 25, 245), 8, cv2.LINE_AA)
    cv2.line(glasses, (110, 39), (130, 39), (90, 35, 25, 255), 7, cv2.LINE_AA)
    cv2.line(glasses, (4, 35), (14, 38), (90, 35, 25, 255), 7)
    cv2.line(glasses, (226, 38), (236, 35), (90, 35, 25, 255), 7)
    cv2.imwrite(str(root / "glasses.png"), glasses)
    star = np.zeros((64, 64, 4), np.uint8)
    points = np.array([[32, 1], [40, 23], [63, 24], [46, 38], [51, 61], [32, 48], [13, 61], [18, 38], [1, 24], [24, 23]])
    cv2.fillPoly(star, [points], (140, 245, 255, 255), cv2.LINE_AA)
    cv2.imwrite(str(root / "star.png"), star)
    config = {"version": 1, "effects": [
        {"id": "crown", "plugin": "sticker", "asset": "crown.png", "width": 2.7, "offset": -1.05},
        {"id": "glasses", "plugin": "glasses", "asset": "glasses.png", "width": 2.1, "offset": .05},
        {"id": "orbit", "plugin": "orbit", "asset": "star.png"}]}
    (root / "effects.json").write_text(json.dumps(config, indent=2)+"\n")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[2] / "plugins/face_effects_live/assets")
