"""Runtime settings, with defaults in one place."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_THRESHOLD = 0.9
DEFAULT_CLOSED_SECONDS = 1.5
DEFAULT_HYSTERESIS = 0.1
DEFAULT_CALIBRATION_SECONDS = 3.0
DEFAULT_CALIBRATION_FACTOR = 0.6


@dataclass
class Config:
    source: int | str = 0
    """Webcam index, or a path to a video file."""

    model_path: Path | None = None
    """Face Landmarker ``.task`` file. ``None`` downloads it to the user cache."""

    threshold: float = DEFAULT_THRESHOLD
    """Eye/iris area ratio below which an eye counts as closed."""

    closed_seconds: float = DEFAULT_CLOSED_SECONDS
    """How long the eyes must stay closed before raising the alarm."""

    hysteresis: float = DEFAULT_HYSTERESIS
    """Eyes count as open again only above ``threshold * (1 + hysteresis)``."""

    calibrate: bool = False
    calibration_seconds: float = DEFAULT_CALIBRATION_SECONDS
    calibration_factor: float = DEFAULT_CALIBRATION_FACTOR

    mirror: bool | None = None
    """Flip frames horizontally (selfie view). ``None`` means on for webcams, off for files."""

    show_mesh: bool = False
    display: bool = True
    sound: bool = True
    output: Path | None = None
    """Write the annotated video to this file (e.g. ``demo.mp4``)."""

    log_csv: Path | None = None
    """Write per-frame measurements to this CSV file."""

    @property
    def is_camera(self) -> bool:
        return isinstance(self.source, int)

    @property
    def should_mirror(self) -> bool:
        return self.is_camera if self.mirror is None else self.mirror
