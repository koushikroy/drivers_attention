"""Command-line entry point: ``drowsiness-detector`` or ``python -m drowsiness_detector``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from . import config as defaults
from .config import Config


def _source(value: str) -> int | str:
    """A bare integer selects a webcam; anything else is treated as a video path."""
    return int(value) if value.isdigit() else value


def _positive(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drowsiness-detector",
        description="Real-time driver drowsiness detection from a webcam or video file.",
    )
    parser.add_argument(
        "--source",
        type=_source,
        default=0,
        help="webcam index (default: 0) or path to a video file",
    )
    parser.add_argument(
        "--threshold",
        type=_positive,
        default=defaults.DEFAULT_THRESHOLD,
        help="eye/iris ratio below which eyes count as closed (default: %(default)s)",
    )
    parser.add_argument(
        "--closed-seconds",
        type=_positive,
        default=defaults.DEFAULT_CLOSED_SECONDS,
        help="seconds of closed eyes before the alarm (default: %(default)s)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="measure your open-eye ratio first and set the threshold from it",
    )
    parser.add_argument(
        "--calibration-seconds",
        type=_positive,
        default=defaults.DEFAULT_CALIBRATION_SECONDS,
        help="length of the calibration phase (default: %(default)s)",
    )
    parser.add_argument(
        "--calibration-factor",
        type=_positive,
        default=defaults.DEFAULT_CALIBRATION_FACTOR,
        help="threshold = factor x median open-eye ratio (default: %(default)s)",
    )
    mirror = parser.add_mutually_exclusive_group()
    mirror.add_argument(
        "--mirror", dest="mirror", action="store_true", default=None, help="flip frames"
    )
    mirror.add_argument(
        "--no-mirror", dest="mirror", action="store_false", help="do not flip frames"
    )
    parser.add_argument("--show-mesh", action="store_true", help="draw all face landmarks")
    parser.add_argument("--no-display", action="store_true", help="run without a window")
    parser.add_argument("--no-sound", action="store_true", help="disable the audible alarm")
    parser.add_argument("--output", type=Path, help="save the annotated video (e.g. demo.mp4)")
    parser.add_argument("--log-csv", type=Path, help="save per-frame measurements to a CSV")
    parser.add_argument("--model", type=Path, help="path to face_landmarker.task")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def parse_config(argv: Sequence[str] | None = None) -> tuple[Config, argparse.Namespace]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.calibrate and not 0 < args.calibration_factor < 1:
        parser.error("--calibration-factor must be between 0 and 1")
    config = Config(
        source=args.source,
        model_path=args.model,
        threshold=args.threshold,
        closed_seconds=args.closed_seconds,
        calibrate=args.calibrate,
        calibration_seconds=args.calibration_seconds,
        calibration_factor=args.calibration_factor,
        mirror=args.mirror,
        show_mesh=args.show_mesh,
        display=not args.no_display,
        sound=not args.no_sound,
        output=args.output,
        log_csv=args.log_csv,
    )
    return config, args


def main(argv: Sequence[str] | None = None) -> int:
    config, args = parse_config(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    from .app import run  # deferred so --help works without loading MediaPipe

    try:
        summary = run(config)
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        logging.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 130

    print(
        f"Processed {summary.frames} frames ({summary.frames_with_face} with a face). "
        f"Drowsy episodes: {summary.drowsy_episodes}. Blinks: {summary.blinks}. "
        f"Threshold: {summary.threshold:.2f}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
