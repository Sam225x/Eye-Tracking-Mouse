"""Nine-point gaze calibration using normalized iris position features."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

CALIBRATION_PATH = Path.home() / ".eye_tracking_mouse_calibration.json"

# left iris x ratio, right iris x ratio, average vertical ratio
FEATURE_COUNT = 3


class GazeCalibration:
    def __init__(self) -> None:
        self.coefficients: Optional[np.ndarray] = None

    @property
    def calibrated(self) -> bool:
        return self.coefficients is not None

    def fit(self, features: list[list[float]], targets: list[list[float]]) -> None:
        if len(features) < 5 or len(features) != len(targets):
            raise ValueError("At least five matching calibration samples are required.")
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != FEATURE_COUNT or y.shape != (len(x), 2):
            raise ValueError("Invalid calibration sample dimensions.")
        design = np.column_stack([x, np.ones(len(x))])
        self.coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)

    def predict(self, features: list[float]) -> tuple[float, float]:
        if self.coefficients is None:
            raise RuntimeError("Calibration has not been completed.")
        if len(features) != FEATURE_COUNT:
            raise ValueError(f"Expected {FEATURE_COUNT} gaze features.")
        row = np.asarray([*features, 1.0], dtype=np.float64)
        result = row @ self.coefficients
        return float(result[0]), float(result[1])

    def save(self) -> None:
        if self.coefficients is not None:
            CALIBRATION_PATH.write_text(
                json.dumps(self.coefficients.tolist()),
                encoding="utf-8",
            )

    def load(self) -> bool:
        try:
            values = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
            arr = np.asarray(values, dtype=np.float64)
            if arr.shape != (FEATURE_COUNT + 1, 2):
                return False
            self.coefficients = arr
            return True
        except (OSError, ValueError, TypeError):
            return False

    def reset(self) -> None:
        self.coefficients = None
        try:
            CALIBRATION_PATH.unlink(missing_ok=True)
        except OSError:
            pass


def nine_screen_targets(
    width: int,
    height: int,
    margin: float = 0.12,
) -> list[tuple[int, int]]:
    """Return nine targets arranged in a 3 x 3 grid."""
    xs = [int(width * margin), width // 2, int(width * (1 - margin))]
    ys = [int(height * margin), height // 2, int(height * (1 - margin))]
    return [(x, y) for y in ys for x in xs]
