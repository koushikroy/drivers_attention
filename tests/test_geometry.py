import math

import numpy as np
import pytest

from drowsiness_detector.geometry import mean_ratio, openness_ratio, polygon_area


def square(side: float, origin=(0.0, 0.0)) -> np.ndarray:
    x, y = origin
    return np.array([(x, y), (x + side, y), (x + side, y + side), (x, y + side)])


def circle(radius: float, n: int = 400) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def test_square_area():
    assert polygon_area(square(3)) == pytest.approx(9)


def test_area_ignores_winding_direction_and_position():
    pts = square(2, origin=(100, -50))
    assert polygon_area(pts) == pytest.approx(polygon_area(pts[::-1])) == pytest.approx(4)


def test_triangle_area():
    assert polygon_area([(0, 0), (4, 0), (0, 3)]) == pytest.approx(6)


def test_circle_area_approximation():
    assert polygon_area(circle(10)) == pytest.approx(math.pi * 100, rel=1e-3)


@pytest.mark.parametrize("points", [[], [(0, 0)], [(0, 0), (1, 1)]])
def test_degenerate_polygons_have_zero_area(points):
    assert polygon_area(np.array(points, dtype=float).reshape(-1, 2)) == 0.0


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        polygon_area(np.zeros((4, 3)))


def test_openness_ratio():
    assert openness_ratio(square(4), square(2)) == pytest.approx(4)


def test_openness_ratio_is_scale_invariant():
    eye, iris = square(4), square(2)
    near = openness_ratio(eye * 3, iris * 3)
    far = openness_ratio(eye * 0.5, iris * 0.5)
    assert near == pytest.approx(far) == pytest.approx(4)


def test_openness_ratio_none_for_degenerate_iris():
    flat_iris = np.array([(0, 0), (1, 0), (2, 0), (3, 0)], dtype=float)
    assert openness_ratio(square(4), flat_iris) is None


def test_mean_ratio_skips_missing_values():
    assert mean_ratio([1.0, None, 3.0]) == pytest.approx(2.0)
    assert mean_ratio([None, None]) is None
