"""Per-user threshold calibration from a few seconds of open-eye samples."""

from __future__ import annotations

import statistics


class Calibrator:
    """Collects open-eye ratios for ``duration`` seconds and derives a threshold.

    Eye shape varies between people, so a fixed threshold that works for one
    driver can miss closures, or fire constantly, for another. The threshold is
    ``factor`` times the median open-eye ratio; the median ignores the odd blink
    during calibration.
    """

    def __init__(self, duration: float, factor: float, min_samples: int = 10) -> None:
        if duration <= 0:
            raise ValueError("duration must be positive")
        if not 0 < factor < 1:
            raise ValueError("factor must be between 0 and 1")
        self.duration = duration
        self.factor = factor
        self.min_samples = min_samples
        self._samples: list[float] = []
        self._started_at: float | None = None

    def add(self, ratio: float | None, timestamp: float) -> None:
        if self._started_at is None:
            self._started_at = timestamp
        if ratio is not None:
            self._samples.append(ratio)

    def elapsed(self, timestamp: float) -> float:
        return 0.0 if self._started_at is None else timestamp - self._started_at

    def done(self, timestamp: float) -> bool:
        return self.elapsed(timestamp) >= self.duration and len(self._samples) >= self.min_samples

    def threshold(self) -> float:
        if len(self._samples) < self.min_samples:
            raise RuntimeError(
                f"need at least {self.min_samples} face samples, got {len(self._samples)}"
            )
        return self.factor * statistics.median(self._samples)
