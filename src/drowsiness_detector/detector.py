"""Wraps the MediaPipe Face Landmarker and turns a video frame into eye measurements."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import TracebackType

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from . import landmarks as lm
from .geometry import mean_ratio, openness_ratio


@dataclass(frozen=True)
class EyeMetrics:
    """Eye measurements for one frame. ``points`` holds all landmarks in pixel coordinates."""

    points: np.ndarray  # shape (478, 2)
    left_ratio: float | None
    right_ratio: float | None

    @property
    def ratio(self) -> float | None:
        """Average of both eyes, so one hidden or squinting eye does not trigger an alert alone."""
        return mean_ratio([self.left_ratio, self.right_ratio])


def metrics_from_points(points: np.ndarray) -> EyeMetrics:
    """Compute both eyes' openness ratios from an ``(N, 2)`` landmark array."""
    return EyeMetrics(
        points=points,
        left_ratio=openness_ratio(points[list(lm.LEFT_EYE)], points[list(lm.LEFT_IRIS)]),
        right_ratio=openness_ratio(points[list(lm.RIGHT_EYE)], points[list(lm.RIGHT_IRIS)]),
    )


class FaceDetector:
    """Runs the Face Landmarker in VIDEO mode, which tracks the face between frames.

    Use as a context manager so the underlying graph is released::

        with FaceDetector(model_path) as detector:
            metrics = detector.process(frame_bgr, timestamp_ms)
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> EyeMetrics | None:
        """Return eye metrics for the most prominent face, or ``None`` if no face is found."""
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(int(timestamp_ms), self._last_ts_ms + 1)
        self._last_ts_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        face = result.face_landmarks[0]
        if len(face) < lm.NUM_LANDMARKS:
            return None

        height, width = frame_bgr.shape[:2]
        points = np.array([(p.x * width, p.y * height) for p in face], dtype=float)
        return metrics_from_points(points)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceDetector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
