"""Draws eye contours and the status readout onto a frame."""

from __future__ import annotations

import cv2
import numpy as np

from . import landmarks as lm
from .detector import EyeMetrics
from .monitor import State, Status

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colours
GREEN = (80, 200, 80)
AMBER = (0, 190, 255)
RED = (50, 50, 230)
GREY = (170, 170, 170)
WHITE = (255, 255, 255)
CYAN = (230, 200, 0)

_STATE_STYLE = {
    State.NO_FACE: ("NO FACE DETECTED", GREY),
    State.AWAKE: ("AWAKE", GREEN),
    State.EYES_CLOSED: ("EYES CLOSED", AMBER),
    State.DROWSY: ("DROWSY - WAKE UP!", RED),
}


def _polyline(frame: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> None:
    cv2.polylines(frame, [np.round(points).astype(np.int32)], True, color, 1, cv2.LINE_AA)


def draw_eyes(frame: np.ndarray, metrics: EyeMetrics, color: tuple[int, int, int]) -> None:
    pts = metrics.points
    for eye, iris in ((lm.LEFT_EYE, lm.LEFT_IRIS), (lm.RIGHT_EYE, lm.RIGHT_IRIS)):
        _polyline(frame, pts[list(eye)], color)
        _polyline(frame, pts[list(iris)], CYAN)


def draw_mesh(frame: np.ndarray, metrics: EyeMetrics) -> None:
    for x, y in np.round(metrics.points).astype(np.int32):
        cv2.circle(frame, (int(x), int(y)), 1, GREY, -1, cv2.LINE_AA)


def _text(frame: np.ndarray, text: str, org: tuple[int, int], scale: float, color) -> None:
    """Draw text on a darkened backdrop so it stays readable on any background."""
    thickness = 1 if scale < 0.8 else 2
    (w, h), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    pad = 4
    y0, y1 = max(0, y - h - pad), min(frame.shape[0], y + baseline + pad)
    x0, x1 = max(0, x - pad), min(frame.shape[1], x + w + pad)
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi * 0.4).astype(frame.dtype)
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_status(
    frame: np.ndarray, status: Status, threshold: float, fps: float | None = None
) -> None:
    label, color = _STATE_STYLE[status.state]
    height, width = frame.shape[:2]

    if status.alarm:
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), RED, 12)

    _text(frame, label, (16, 40), 1.0, color)
    ratio = "--" if status.ratio is None else f"{status.ratio:.2f}"
    lines = [
        f"Eye/iris ratio: {ratio}  (threshold {threshold:.2f})",
        f"Closed for: {status.closed_for:.1f}s   Blinks: {status.blinks}",
    ]
    if fps is not None:
        lines.append(f"FPS: {fps:.0f}")
    for i, line in enumerate(lines):
        _text(frame, line, (16, 72 + 24 * i), 0.6, WHITE)
    _text(frame, "q / Esc: quit", (16, height - 16), 0.5, GREY)


def draw_calibration(frame: np.ndarray, remaining: float) -> None:
    _text(frame, "CALIBRATING", (16, 40), 1.0, AMBER)
    _text(frame, f"Look at the camera with eyes open... {remaining:.1f}s", (16, 72), 0.6, WHITE)


def draw_frame(
    frame: np.ndarray,
    metrics: EyeMetrics | None,
    status: Status,
    threshold: float,
    fps: float | None = None,
    show_mesh: bool = False,
) -> None:
    """Draw the full overlay in place."""
    if metrics is not None:
        if show_mesh:
            draw_mesh(frame, metrics)
        draw_eyes(frame, metrics, _STATE_STYLE[status.state][1])
    draw_status(frame, status, threshold, fps)
