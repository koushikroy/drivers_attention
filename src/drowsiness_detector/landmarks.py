"""MediaPipe Face Landmarker indices used by the detector.

The Face Landmarker model returns 478 landmarks: 468 face-mesh points plus
10 iris points (468-477). "Left" and "right" refer to the subject's own eyes,
so the subject's left eye appears on the right side of an unmirrored image.

Each contour is listed in order around its outline, which is required for the
shoelace area formula in :mod:`drowsiness_detector.geometry`.
"""

from __future__ import annotations

NUM_LANDMARKS = 478

# Eye outlines: upper lid from the outer/inner corner, then back along the lower lid.
LEFT_EYE: tuple[int, ...] = (
    362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,
)  # fmt: skip
RIGHT_EYE: tuple[int, ...] = (
    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,
)  # fmt: skip

# Iris outlines: the four boundary points, excluding the centre point (473 / 468).
LEFT_IRIS: tuple[int, ...] = (474, 475, 476, 477)
RIGHT_IRIS: tuple[int, ...] = (469, 470, 471, 472)

LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468
