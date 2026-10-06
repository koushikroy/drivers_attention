"""Integration tests that run the real MediaPipe model (downloaded on first run)."""

import numpy as np
import pytest

from drowsiness_detector import landmarks as lm
from drowsiness_detector.detector import metrics_from_points


def synthetic_landmarks(eye_height: float) -> np.ndarray:
    """Landmarks with ellipse-shaped eyes of the given height and unit-radius irises."""
    points = np.zeros((lm.NUM_LANDMARKS, 2))
    for eye, iris, cx in ((lm.LEFT_EYE, lm.LEFT_IRIS, 60), (lm.RIGHT_EYE, lm.RIGHT_IRIS, 20)):
        t = np.linspace(0, 2 * np.pi, len(eye), endpoint=False)
        points[list(eye)] = np.column_stack([cx + 6 * np.cos(t), 50 + eye_height * np.sin(t)])
        t = np.linspace(0, 2 * np.pi, len(iris), endpoint=False)
        points[list(iris)] = np.column_stack([cx + np.cos(t), 50 + np.sin(t)])
    return points


def test_metrics_from_points_open_vs_closed():
    open_eyes = metrics_from_points(synthetic_landmarks(eye_height=3))
    closed_eyes = metrics_from_points(synthetic_landmarks(eye_height=0.3))
    assert open_eyes.left_ratio == pytest.approx(open_eyes.right_ratio)
    assert open_eyes.ratio > 5 * closed_eyes.ratio


@pytest.mark.integration
def test_blank_frame_has_no_face():
    from drowsiness_detector.detector import FaceDetector
    from drowsiness_detector.model import ensure_model

    try:
        model_path = ensure_model()
    except OSError as exc:  # offline
        pytest.skip(f"model unavailable: {exc}")

    with FaceDetector(model_path) as detector:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert detector.process(frame, 0) is None
        assert detector.process(frame, 0) is None  # repeated timestamps are tolerated
