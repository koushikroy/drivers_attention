"""Pure geometry helpers. No OpenCV or MediaPipe dependency, so they are easy to test."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

# Below this iris area (in squared pixels) the landmarks are degenerate and a
# ratio would be meaningless or divide by zero.
MIN_IRIS_AREA = 1e-6


def polygon_area(points: ArrayLike) -> float:
    """Return the area of a simple polygon using the shoelace formula.

    ``points`` is an ``(N, 2)`` array of vertices in order around the outline,
    in either direction. Fewer than three points have zero area.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) array of points, got shape {pts.shape}")
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def openness_ratio(eye: ArrayLike, iris: ArrayLike) -> float | None:
    """Return eye area divided by iris area, or ``None`` if the iris is degenerate.

    The iris has a nearly constant physical size and the face landmark model
    keeps estimating it while the lid closes, so dividing by it makes the ratio
    independent of how far the face is from the camera. An open eye scores
    well above 1; a closed eye collapses towards 0.
    """
    iris_area = polygon_area(iris)
    if iris_area < MIN_IRIS_AREA:
        return None
    return polygon_area(eye) / iris_area


def mean_ratio(ratios: Sequence[float | None]) -> float | None:
    """Average the valid ratios, ignoring ``None``. Returns ``None`` if none are valid."""
    valid = [r for r in ratios if r is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)
