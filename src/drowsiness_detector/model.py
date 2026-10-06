"""Locates the Face Landmarker model file, downloading it on first use."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODEL_FILENAME = "face_landmarker.task"


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "drowsiness_detector"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(dest: Path, url: str = MODEL_URL, sha256: str = MODEL_SHA256) -> Path:
    """Download the model to ``dest``, verifying its checksum before moving it into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading face landmark model to %s", dest)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as resp:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        actual = _sha256(tmp)
        if actual != sha256:
            raise RuntimeError(f"model checksum mismatch: expected {sha256}, got {actual}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def ensure_model(path: str | os.PathLike[str] | None = None) -> Path:
    """Return a usable model path.

    An explicit ``path`` must already exist. Otherwise the model is cached in
    ``~/.cache/drowsiness_detector`` (or ``$XDG_CACHE_HOME``) and downloaded
    the first time it is needed.
    """
    if path is not None:
        explicit = Path(path)
        if not explicit.is_file():
            raise FileNotFoundError(f"model file not found: {explicit}")
        return explicit

    cached = default_cache_dir() / MODEL_FILENAME
    if cached.is_file():
        return cached
    return download_model(cached)
