"""Turns per-frame eye openness into a drowsiness state over time."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    NO_FACE = "no_face"
    AWAKE = "awake"
    EYES_CLOSED = "eyes_closed"  # closed, but not yet for long enough to alert
    DROWSY = "drowsy"


@dataclass(frozen=True)
class Status:
    state: State
    ratio: float | None
    closed_for: float  # seconds the eyes have been continuously closed
    blinks: int

    @property
    def alarm(self) -> bool:
        return self.state is State.DROWSY


class DrowsinessMonitor:
    """Tracks how long the eyes stay closed and raises DROWSY after ``closed_seconds``.

    A frame counts as closed when the ratio drops below ``threshold`` and as
    open again only once it rises above ``threshold * (1 + hysteresis)``. The
    gap between the two stops the state flickering when the ratio hovers near
    the threshold. Closures shorter than ``closed_seconds`` are counted as blinks.

    Timestamps are passed in rather than read from a clock so the logic can be
    driven by video-file timestamps and tested deterministically.
    """

    def __init__(self, threshold: float, closed_seconds: float, hysteresis: float = 0.1) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if closed_seconds < 0:
            raise ValueError("closed_seconds must not be negative")
        if hysteresis < 0:
            raise ValueError("hysteresis must not be negative")
        self.threshold = threshold
        self.closed_seconds = closed_seconds
        self.hysteresis = hysteresis
        self.blinks = 0
        self._closed_since: float | None = None

    @property
    def open_threshold(self) -> float:
        return self.threshold * (1 + self.hysteresis)

    def reset(self) -> None:
        self.blinks = 0
        self._closed_since = None

    def update(self, ratio: float | None, timestamp: float) -> Status:
        """Feed one frame's openness ratio (``None`` if no face) taken at ``timestamp`` seconds."""
        if ratio is None:
            # Without a face we cannot tell whether the eyes are closed, so start over.
            self._closed_since = None
            return Status(State.NO_FACE, None, 0.0, self.blinks)

        if self._closed_since is None:
            if ratio < self.threshold:
                self._closed_since = timestamp
        elif ratio > self.open_threshold:
            if timestamp - self._closed_since < self.closed_seconds:
                self.blinks += 1
            self._closed_since = None

        if self._closed_since is None:
            return Status(State.AWAKE, ratio, 0.0, self.blinks)

        closed_for = max(0.0, timestamp - self._closed_since)
        state = State.DROWSY if closed_for >= self.closed_seconds else State.EYES_CLOSED
        return Status(state, ratio, closed_for, self.blinks)
