"""Audible alarm that repeats while the driver is drowsy, without blocking the video loop."""

from __future__ import annotations

import sys
import threading


def _beep() -> None:
    if sys.platform == "win32":
        import winsound

        winsound.Beep(1000, 400)
    else:
        # Terminal bell: no extra audio dependency needed on macOS or Linux.
        sys.stdout.write("\a")
        sys.stdout.flush()


class Alarm:
    """Beeps every ``interval`` seconds on a background thread while active."""

    def __init__(self, enabled: bool = True, interval: float = 1.0) -> None:
        self.enabled = enabled
        self.interval = interval
        self._active = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set(self, active: bool) -> None:
        if not self.enabled:
            return
        if active:
            self._active.set()
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="alarm", daemon=True)
                self._thread.start()
        else:
            self._active.clear()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._active.wait()
            if self._stop.is_set():
                break
            _beep()
            self._stop.wait(self.interval)

    def close(self) -> None:
        self._stop.set()
        self._active.set()  # wake the thread so it can exit
        if self._thread is not None:
            self._thread.join(timeout=2)
