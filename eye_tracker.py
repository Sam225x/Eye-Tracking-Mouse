"""Webcam eye/iris landmark tracking with MediaPipe Face Mesh."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp


@dataclass
class TrackingResult:
    frame: object
    features: Optional[list[float]]
    eyes_closed: bool
    face_detected: bool
    message: str


class EyeTracker:
    def __init__(self, camera_index: int = 0, width: int = 960, height: int = 540) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.capture = None
        self.face_mesh = None
        self._create_face_mesh()

    def _create_face_mesh(self) -> None:
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def start(self) -> None:
        if self.capture is not None and self.capture.isOpened():
            return
        # FaceMesh is closed by stop(); recreate it when tracking is restarted.
        if self.face_mesh is None:
            self._create_face_mesh()
        self.capture = cv2.VideoCapture(self.camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(
                f"Could not open camera index {self.camera_index}. "
                "Check camera permissions or try another index."
            )

    @staticmethod
    def _ratio(value: float, start: float, end: float) -> float:
        denominator = end - start
        if abs(denominator) < 1e-6:
            return 0.5
        return float((value - start) / denominator)

    def read(self) -> TrackingResult:
        if self.capture is None:
            raise RuntimeError("Camera has not been started.")
        if self.face_mesh is None:
            raise RuntimeError("Face landmark detector is not initialized.")

        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Webcam frame could not be read.")

        # Mirror the preview; measurements use the same mirrored frame.
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return TrackingResult(frame, None, False, False, "Face not detected")

        lm = result.multi_face_landmarks[0].landmark
        left_iris, right_iris = lm[468], lm[473]
        left_outer, left_inner = lm[33], lm[133]
        right_inner, right_outer = lm[362], lm[263]

        left_x = self._ratio(left_iris.x, left_outer.x, left_inner.x)
        right_x = self._ratio(right_iris.x, right_inner.x, right_outer.x)

        left_top, left_bottom = lm[159], lm[145]
        right_top, right_bottom = lm[386], lm[374]
        left_y = self._ratio(left_iris.y, left_top.y, left_bottom.y)
        right_y = self._ratio(right_iris.y, right_top.y, right_bottom.y)
        features = [left_x, right_x, (left_y + right_y) / 2.0]

        # Eye aspect ratios: low values indicate eye closure.
        def ear(a: int, b: int, c: int, d: int, e: int, f: int) -> float:
            vertical = (
                abs(lm[b].y - lm[f].y) + abs(lm[c].y - lm[e].y)
            ) / 2.0
            horizontal = abs(lm[a].x - lm[d].x)
            return vertical / max(horizontal, 1e-6)

        left_ear = ear(33, 160, 158, 133, 153, 144)
        right_ear = ear(362, 385, 387, 263, 373, 380)
        eyes_closed = left_ear < 0.19 and right_ear < 0.19

        height, width = frame.shape[:2]
        for index in (468, 473):
            point = (int(lm[index].x * width), int(lm[index].y * height))
            cv2.circle(frame, point, 4, (40, 220, 80), -1)
        cv2.putText(
            frame,
            "Tracking active",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (50, 220, 80),
            2,
        )
        return TrackingResult(frame, features, eyes_closed, True, "Tracking active")

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None
