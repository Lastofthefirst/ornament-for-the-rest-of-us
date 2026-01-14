"""
Progress reporting utilities for terminal output.

Provides clean, updating progress indicators instead of log spam.
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Iterator, TypeVar

T = TypeVar('T')


@dataclass
class ProgressStats:
    """Statistics for progress tracking."""

    total: int
    current: int = 0
    start_time: float = field(default_factory=time.time)
    successful: int = 0
    failed: int = 0

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def rate(self) -> float:
        """Items per second."""
        if self.elapsed == 0:
            return 0
        return self.current / self.elapsed

    @property
    def eta(self) -> float | None:
        """Estimated time remaining in seconds."""
        if self.rate == 0 or self.current == 0:
            return None
        remaining = self.total - self.current
        return remaining / self.rate

    @property
    def percent(self) -> float:
        """Completion percentage."""
        if self.total == 0:
            return 100.0
        return (self.current / self.total) * 100


def format_time(seconds: float | None) -> str:
    """Format seconds as human-readable time."""
    if seconds is None:
        return "--:--"

    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


class ProgressReporter:
    """Reports progress with updating terminal output.

    Usage:
        with ProgressReporter(total=100, desc="Processing") as progress:
            for item in items:
                process(item)
                progress.update()
    """

    def __init__(
        self,
        total: int,
        desc: str = "Progress",
        unit: str = "items",
        show_rate: bool = True,
    ):
        """Initialize progress reporter.

        Args:
            total: Total number of items to process
            desc: Description prefix for progress line
            unit: Unit name for items (e.g., "pages", "images")
            show_rate: Whether to show processing rate
        """
        self.stats = ProgressStats(total=total)
        self.desc = desc
        self.unit = unit
        self.show_rate = show_rate
        # Check both stderr and stdout for TTY (some terminals only have one)
        self._is_tty = sys.stderr.isatty() or sys.stdout.isatty()
        self._output = sys.stderr if sys.stderr.isatty() else sys.stdout
        self._last_line_len = 0
        self._item_name: str | None = None

    def __enter__(self):
        self.stats.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def update(
        self,
        n: int = 1,
        success: bool = True,
        item_name: str | None = None,
    ) -> None:
        """Update progress.

        Args:
            n: Number of items completed (default 1)
            success: Whether the item was processed successfully
            item_name: Optional name of current item for display
        """
        self.stats.current += n
        if success:
            self.stats.successful += n
        else:
            self.stats.failed += n
        self._item_name = item_name
        self._render()

    def set_item(self, name: str) -> None:
        """Set the current item name without updating count.

        The name will appear on the next update() call.
        """
        self._item_name = name

    def _render(self) -> None:
        """Render the progress line."""
        stats = self.stats

        # Build progress bar
        bar_width = 20
        filled = int(bar_width * stats.percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Build status line
        parts = [
            f"{self.desc}: [{bar}]",
            f"{stats.current}/{stats.total}",
            f"({stats.percent:.0f}%)",
        ]

        # Add timing info
        elapsed_str = format_time(stats.elapsed)
        eta_str = format_time(stats.eta)
        parts.append(f"[{elapsed_str}<{eta_str}]")

        # Add rate if requested
        if self.show_rate and stats.rate > 0:
            if stats.rate >= 1:
                parts.append(f"{stats.rate:.1f} {self.unit}/s")
            else:
                secs_per_item = 1 / stats.rate
                parts.append(f"{secs_per_item:.1f}s/{self.unit[:-1] if self.unit.endswith('s') else self.unit}")

        # Add current item name if set
        if self._item_name:
            # Truncate long names
            name = self._item_name
            if len(name) > 25:
                name = "..." + name[-22:]
            parts.append(f"| {name}")

        line = " ".join(parts)

        if self._is_tty:
            # Use carriage return to overwrite line, pad with spaces to clear old content
            clear = " " * max(0, self._last_line_len - len(line))
            self._output.write(f"\r{line}{clear}")
            self._output.flush()
            self._last_line_len = len(line)
        else:
            # Non-TTY: just print periodic updates (every 10%)
            if stats.current == 1 or stats.current == stats.total or stats.current % max(1, stats.total // 10) == 0:
                self._output.write(line + "\n")
                self._output.flush()

    def finish(self) -> None:
        """Finish progress and print summary."""
        stats = self.stats

        # Move to new line
        if self._is_tty:
            self._output.write("\n")

        # Print summary
        elapsed_str = format_time(stats.elapsed)

        if stats.failed > 0:
            summary = (
                f"✓ {self.desc} complete: {stats.successful}/{stats.total} "
                f"succeeded, {stats.failed} failed ({elapsed_str})"
            )
        else:
            summary = f"✓ {self.desc} complete: {stats.total} {self.unit} ({elapsed_str})"

        self._output.write(summary + "\n")
        self._output.flush()


def progress_iter(
    items: list[T],
    desc: str = "Processing",
    unit: str = "items",
) -> Iterator[tuple[T, ProgressReporter]]:
    """Iterate over items with progress reporting.

    Usage:
        for item, progress in progress_iter(items, "OCR", "pages"):
            result = process(item)
            progress.update(success=result.ok, item_name=item.name)

    Args:
        items: List of items to iterate
        desc: Description for progress display
        unit: Unit name for items

    Yields:
        Tuple of (item, progress_reporter)
    """
    with ProgressReporter(len(items), desc=desc, unit=unit) as progress:
        for item in items:
            yield item, progress
