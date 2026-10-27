"""Small helpers shared by several scripts: reading images and drawing labels."""
import cv2
import numpy as np


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def label(img, text, org, color=(255, 255, 255), scale=0.5, thick=1, box=True):
    """Text on a dark box, kept inside the image so it is never cut off."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = int(min(max(org[0], 3), img.shape[1] - tw - 3))
    y = int(min(max(org[1], th + 3), img.shape[0] - base - 3))
    if box:
        cv2.rectangle(img, (x - 3, y - th - 3), (x + tw + 3, y + base + 2), (25, 25, 25), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def dashed(img, p, q, color, thick=2, dash=10):
    """Dashed line from p to q."""
    p, q = np.array(p, float), np.array(q, float)
    n = max(1, int(np.linalg.norm(q - p) / dash))
    for i in range(0, n, 2):
        a, b = p + (q - p) * i / n, p + (q - p) * min(i + 1, n) / n
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thick, cv2.LINE_AA)
