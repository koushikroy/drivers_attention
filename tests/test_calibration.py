import pytest

from drowsiness_detector.calibration import Calibrator


def test_threshold_is_factor_times_median():
    cal = Calibrator(duration=1.0, factor=0.5, min_samples=3)
    for i, ratio in enumerate([2.0, 2.2, 0.2, 1.8, 2.0]):  # 0.2 is a blink
        cal.add(ratio, i * 0.25)
    assert cal.done(1.0)
    assert cal.threshold() == pytest.approx(1.0)


def test_not_done_until_duration_and_samples():
    cal = Calibrator(duration=1.0, factor=0.5, min_samples=3)
    cal.add(2.0, 0.0)
    cal.add(None, 0.5)  # no face
    cal.add(2.0, 1.5)
    assert not cal.done(1.5)  # long enough, but only 2 samples
    cal.add(2.0, 1.6)
    assert cal.done(1.6)


def test_threshold_requires_samples():
    cal = Calibrator(duration=1.0, factor=0.5, min_samples=3)
    with pytest.raises(RuntimeError):
        cal.threshold()


@pytest.mark.parametrize("kwargs", [{"duration": 0, "factor": 0.5}, {"duration": 1, "factor": 1}])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        Calibrator(**kwargs)
