"""The capture → detect → monitor → display loop."""

from __future__ import annotations

import csv
import logging
import time
from contextlib import ExitStack
from dataclasses import dataclass

import cv2

from .alert import Alarm
from .calibration import Calibrator
from .config import Config
from .detector import FaceDetector
from .model import ensure_model
from .monitor import DrowsinessMonitor, State, Status
from .overlay import draw_calibration, draw_frame

log = logging.getLogger(__name__)

WINDOW_NAME = "Driver Drowsiness Detection"
QUIT_KEYS = {ord("q"), 27}  # q or Esc
MAX_CONSECUTIVE_READ_FAILURES = 100
FALLBACK_FPS = 30.0


@dataclass
class RunSummary:
    frames: int = 0
    frames_with_face: int = 0
    drowsy_episodes: int = 0
    blinks: int = 0
    threshold: float = 0.0


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _open_capture(config: Config) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(config.source)
    if not cap.isOpened():
        kind = "camera" if config.is_camera else "video file"
        raise RuntimeError(f"could not open {kind}: {config.source}")
    return cap


def run(config: Config) -> RunSummary:
    """Process frames until the video ends or the user quits. Returns run statistics."""
    model_path = ensure_model(config.model_path)
    summary = RunSummary(threshold=config.threshold)

    with ExitStack() as stack:
        cap = _open_capture(config)
        stack.callback(cap.release)
        detector = stack.enter_context(FaceDetector(model_path))
        alarm = Alarm(enabled=config.sound and config.display)
        stack.callback(alarm.close)

        source_fps = cap.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS
        if source_fps <= 1 or source_fps > 240:
            source_fps = FALLBACK_FPS

        writer: cv2.VideoWriter | None = None
        csv_writer = None
        if config.log_csv is not None:
            csv_file = stack.enter_context(config.log_csv.open("w", newline=""))
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                ["time_s", "state", "ratio", "left_ratio", "right_ratio", "closed_for_s", "blinks"]
            )
        if config.display:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            stack.callback(cv2.destroyAllWindows)

        calibrator = (
            Calibrator(config.calibration_seconds, config.calibration_factor)
            if config.calibrate
            else None
        )
        monitor = DrowsinessMonitor(config.threshold, config.closed_seconds, config.hysteresis)

        start = time.monotonic()
        last_tick = start
        fps: float | None = None
        failures = 0
        previous_state: State | None = None

        while True:
            ok, frame = cap.read()
            if not ok:
                if not config.is_camera:
                    break  # end of file
                failures += 1
                if failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    raise RuntimeError("camera stopped returning frames")
                continue
            failures = 0

            # Video files are timed by frame position so results do not depend on
            # processing speed; webcams are timed by the wall clock.
            now = time.monotonic() - start if config.is_camera else summary.frames / source_fps
            summary.frames += 1

            # Flip before detecting and drawing so the overlay text is never mirrored.
            if config.should_mirror:
                frame = cv2.flip(frame, 1)

            metrics = detector.process(frame, int(now * 1000))
            ratio = metrics.ratio if metrics is not None else None
            if metrics is not None:
                summary.frames_with_face += 1

            tick = time.monotonic()
            instant = 1.0 / max(tick - last_tick, 1e-6)
            fps = instant if fps is None else 0.9 * fps + 0.1 * instant
            last_tick = tick

            if calibrator is not None:
                calibrator.add(ratio, now)
                if calibrator.done(now):
                    monitor = DrowsinessMonitor(
                        calibrator.threshold(), config.closed_seconds, config.hysteresis
                    )
                    summary.threshold = monitor.threshold
                    log.info("Calibrated threshold: %.3f", monitor.threshold)
                    calibrator = None
                status = Status(State.AWAKE if metrics else State.NO_FACE, ratio, 0.0, 0)
            else:
                status = monitor.update(ratio, now)
                if status.state is State.DROWSY and previous_state is not State.DROWSY:
                    summary.drowsy_episodes += 1
                    log.warning("Drowsiness detected at %.1fs", now)
                previous_state = status.state
                alarm.set(status.alarm)

            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        f"{now:.3f}",
                        status.state.value,
                        _fmt(ratio),
                        _fmt(metrics.left_ratio if metrics else None),
                        _fmt(metrics.right_ratio if metrics else None),
                        f"{status.closed_for:.3f}",
                        status.blinks,
                    ]
                )

            if config.display or config.output is not None:
                draw_frame(frame, metrics, status, monitor.threshold, fps, config.show_mesh)
                if calibrator is not None:
                    remaining = max(0.0, config.calibration_seconds - calibrator.elapsed(now))
                    draw_calibration(frame, remaining)

            if config.output is not None:
                if writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(config.output), fourcc, source_fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"could not write video: {config.output}")
                    stack.callback(writer.release)
                writer.write(frame)

            if config.display:
                cv2.imshow(WINDOW_NAME, frame)
                if (cv2.waitKey(1) & 0xFF) in QUIT_KEYS:
                    break

        if calibrator is not None:
            log.warning("Calibration did not finish; used the default threshold.")
        summary.blinks = monitor.blinks
        summary.threshold = monitor.threshold
    return summary
