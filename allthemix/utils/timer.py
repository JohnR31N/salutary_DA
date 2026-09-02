from __future__ import annotations

import time


class Timer:
    def __init__(self) -> None:
        """Initialize the object state."""
        self.start_time = 0.0

    def start(self) -> None:
        """Record the current high-resolution clock time."""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Return elapsed seconds since the last start call."""
        end_time = time.perf_counter()

        return end_time - self.start_time  # Elapsed wall-clock seconds.
