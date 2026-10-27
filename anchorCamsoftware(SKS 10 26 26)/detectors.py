"""Interchangeable on-demand recognizers. The default is explicitly a MOCK."""
from pathlib import Path
import cv2
import numpy as np
from data import Detection

# BGR colour ranges the mock recognizer looks for. "A"/"B" keep the video demo working.
MOCK_COLORS = {
    "B": ((0, 0, 180), (75, 75, 255)),
    "red": ((0, 0, 180), (75, 75, 255)),
    "A": ((180, 0, 0), (255, 75, 75)),
    "blue": ((180, 0, 0), (255, 75, 75)),
    "Target-Center": ((0, 0, 140), (110, 110, 255)),   # red dots in tests/scope_sim.py
}


class MockDetector:
    """Finds our generated red/blue symbols. Scores are INVENTED, not YOLO output."""
    name = "MOCK detector (synthetic color symbols; invented confidence)"

    def detect(self, image, requested_label):
        if requested_label not in MOCK_COLORS:
            return None
        lower, upper = MOCK_COLORS[requested_label]
        mask = cv2.inRange(image, np.array(lower, np.uint8), np.array(upper, np.uint8))
        moments = cv2.moments(mask, binaryImage=True)
        if moments["m00"] < 25:
            return None
        position = (moments["m10"] / moments["m00"],
                    moments["m01"] / moments["m00"])
        x, y, w, h = cv2.boundingRect(mask)
        return Detection(requested_label, 0.95, position, (x, y, x + w, y + h))

    def detect_all(self, image, requested_label):
        """Every separate blob of the colour, one Detection each (largest first)."""
        if requested_label not in MOCK_COLORS:
            return []
        lower, upper = MOCK_COLORS[requested_label]
        mask = cv2.inRange(image, np.array(lower, np.uint8), np.array(upper, np.uint8))
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        found = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area >= 25:
                found.append((area, Detection(requested_label, 0.95, tuple(map(float, centroids[i])),
                                              (x, y, x + w, y + h))))
        return [d for _, d in sorted(found, key=lambda t: -t[0])]


class YoloDetector:
    name = "Ultralytics YOLO (your supplied model weights)"

    def __init__(self, weights):
        # No implicit downloads: a real model file must be supplied.
        if not weights or not Path(weights).exists():      # a .pt/.onnx file, or an NCNN model folder
            raise FileNotFoundError(f"Model weights not found: {weights!r}. "
                                    "Supply a local trained model, e.g. models/best.pt")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Install requirements-yolo.txt to use the real YOLO adapter") from error
        self.model = YOLO(str(weights), task="detect")  # loaded once; .pt, .onnx or NCNN folder
        self.name = f"Ultralytics YOLO ({Path(weights).name})"
        self._last = (None, None)        # (image, result): one inference per image

    def _predict(self, image):
        """Run the model once per image, even when asked for several classes.
        With ZOOM > 1 the model looks at a crop around the eyepiece view, enlarged,
        so small things (target dots, impacts) are big enough for it; boxes are
        mapped back to full-frame pixels."""
        if self._last[0] is not image:
            x0, y0, k, src = 0, 0, 1.0, image
            if ZOOM > 1:
                import scope
                circle = scope.eyepiece_circle(image)
                h, w = image.shape[:2]
                cx, cy, r = circle if circle is not None else (w / 2, h / 2, min(w, h) / 3)
                half = int(1.25 * r)
                x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
                crop = image[y0:int(min(h, cy + half)), x0:int(min(w, cx + half))]
                k = ZOOM
                src = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
            result = self.model.predict(source=src, conf=0.01, verbose=False)[0]
            self._last = (image, (result, x0, y0, k))
        return self._last[1]

    def detect(self, image, requested_label=None):
        """Highest-confidence box of `requested_label` (or of any class if None)."""
        return next(iter(self.detect_all(image, requested_label)), None)

    def detect_all(self, image, requested_label=None):
        """Every box of `requested_label` (any class if None), most confident first."""
        if requested_label is not None and requested_label not in self.model.names.values():
            raise ValueError(f"Class {requested_label!r} absent from model. Available: {self.model.names}")
        result, x0, y0, k = self._predict(image)
        candidates = []
        for box in result.boxes:
            label = result.names[int(box.cls.item())]
            if requested_label is not None and label != requested_label:
                continue
            x1, y1, x2, y2 = (v / k for v in box.xyxy[0].tolist())
            x1, x2, y1, y2 = x1 + x0, x2 + x0, y1 + y0, y2 + y0
            candidates.append(Detection(label, float(box.conf.item()),
                                        ((x1 + x2) / 2, (y1 + y2) / 2), (x1, y1, x2, y2)))
        return sorted(candidates, key=lambda item: -item.confidence)


_LOADED = {}
ZOOM = 1.0      # set from --yolo-zoom: crop around the eyepiece and enlarge before detecting


def load_detector(spec):
    """'mock' -> MockDetector; an object with detect_all() is used as-is (tests);
    anything else is a path to YOLO weights. The same path given twice returns
    the same loaded model (loaded once)."""
    if hasattr(spec, "detect_all"):
        return spec
    key = spec if spec == "mock" else str(Path(spec).resolve())
    if key not in _LOADED:
        _LOADED[key] = MockDetector() if spec == "mock" else YoloDetector(spec)
    return _LOADED[key]
