from pathlib import Path

import pytest

from drowsiness_detector.cli import parse_config
from drowsiness_detector.config import DEFAULT_THRESHOLD


def test_defaults_use_webcam_zero():
    config, _ = parse_config([])
    assert config.source == 0
    assert config.is_camera
    assert config.should_mirror
    assert config.threshold == DEFAULT_THRESHOLD
    assert config.display and config.sound


def test_video_file_source_is_not_mirrored_by_default():
    config, _ = parse_config(["--source", "clip.mp4"])
    assert config.source == "clip.mp4"
    assert not config.is_camera
    assert not config.should_mirror


def test_options_are_mapped():
    config, _ = parse_config(
        [
            "--source", "1", "--threshold", "1.2", "--closed-seconds", "2",
            "--calibrate", "--no-display", "--no-sound", "--no-mirror",
            "--output", "out.mp4", "--log-csv", "log.csv",
        ]
    )  # fmt: skip
    assert config.source == 1
    assert config.threshold == 1.2
    assert config.closed_seconds == 2
    assert config.calibrate
    assert not config.display and not config.sound and not config.should_mirror
    assert config.output == Path("out.mp4")
    assert config.log_csv == Path("log.csv")


@pytest.mark.parametrize(
    "argv",
    [
        ["--threshold", "0"],
        ["--closed-seconds", "-1"],
        ["--calibrate", "--calibration-factor", "2"],
    ],
)
def test_invalid_values_exit(argv):
    with pytest.raises(SystemExit):
        parse_config(argv)
