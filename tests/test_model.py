import pytest

from drowsiness_detector import model


def test_explicit_path_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        model.ensure_model(tmp_path / "missing.task")


def test_explicit_path_is_returned(tmp_path):
    path = tmp_path / "face_landmarker.task"
    path.write_bytes(b"x")
    assert model.ensure_model(path) == path


def test_cached_model_is_reused_without_download(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cached = tmp_path / "drowsiness_detector" / model.MODEL_FILENAME
    cached.parent.mkdir()
    cached.write_bytes(b"x")
    monkeypatch.setattr(model, "download_model", lambda *_: pytest.fail("should not download"))
    assert model.ensure_model() == cached


def test_checksum_mismatch_leaves_no_file(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"not the model")
    dest = tmp_path / "cache" / "model.task"
    with pytest.raises(RuntimeError, match="checksum"):
        model.download_model(dest, url=src.as_uri(), sha256="0" * 64)
    assert not dest.exists()
    assert list(dest.parent.iterdir()) == []
