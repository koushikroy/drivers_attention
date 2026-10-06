import pytest

from drowsiness_detector.monitor import DrowsinessMonitor, State

OPEN, CLOSED = 2.0, 0.3
FPS = 30


def feed(monitor, ratios, start=0.0):
    """Feed ratios at 30 FPS and return the list of statuses."""
    return [monitor.update(r, start + i / FPS) for i, r in enumerate(ratios)]


@pytest.fixture
def monitor():
    return DrowsinessMonitor(threshold=1.0, closed_seconds=1.0, hysteresis=0.1)


def test_open_eyes_are_awake(monitor):
    statuses = feed(monitor, [OPEN] * 60)
    assert {s.state for s in statuses} == {State.AWAKE}


def test_short_closure_is_a_blink_not_drowsiness(monitor):
    statuses = feed(monitor, [OPEN] * 10 + [CLOSED] * 6 + [OPEN] * 10)
    assert State.DROWSY not in {s.state for s in statuses}
    assert State.EYES_CLOSED in {s.state for s in statuses}
    assert statuses[-1].blinks == 1


def test_sustained_closure_becomes_drowsy_after_window(monitor):
    statuses = feed(monitor, [CLOSED] * 45)  # 1.5 s
    first_drowsy = next(i for i, s in enumerate(statuses) if s.state is State.DROWSY)
    assert first_drowsy == FPS  # exactly 1.0 s after the first closed frame
    assert statuses[-1].alarm
    assert statuses[-1].closed_for == pytest.approx(44 / FPS)


def test_long_closure_is_not_counted_as_blink(monitor):
    statuses = feed(monitor, [CLOSED] * 45 + [OPEN] * 5)
    assert statuses[-1].state is State.AWAKE
    assert statuses[-1].blinks == 0


def test_hysteresis_keeps_eyes_closed_until_clearly_open(monitor):
    # 1.05 is above the closed threshold but below the open threshold (1.1).
    statuses = feed(monitor, [CLOSED] + [1.05] * 40)
    assert statuses[-1].state is State.DROWSY


def test_hovering_near_threshold_from_open_side_stays_awake(monitor):
    statuses = feed(monitor, [OPEN] + [1.05] * 40)
    assert {s.state for s in statuses} == {State.AWAKE}


def test_missing_face_resets_closed_timer(monitor):
    feed(monitor, [CLOSED] * 20)
    assert monitor.update(None, 1.0).state is State.NO_FACE
    statuses = feed(monitor, [CLOSED] * 20, start=1.1)
    assert State.DROWSY not in {s.state for s in statuses}


def test_zero_closed_seconds_alerts_immediately():
    monitor = DrowsinessMonitor(threshold=1.0, closed_seconds=0.0)
    assert monitor.update(CLOSED, 0.0).state is State.DROWSY


def test_reset_clears_blinks(monitor):
    feed(monitor, [CLOSED] * 3 + [OPEN] * 3)
    assert monitor.blinks == 1
    monitor.reset()
    assert monitor.blinks == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0, "closed_seconds": 1},
        {"threshold": 1, "closed_seconds": -1},
        {"threshold": 1, "closed_seconds": 1, "hysteresis": -0.1},
    ],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        DrowsinessMonitor(**kwargs)
