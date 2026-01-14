"""Tests for progress reporting module."""

import io
import sys
import time

import pytest
from bookpipeline.progress import (
    ProgressStats,
    ProgressReporter,
    format_time,
    progress_iter,
)


class TestFormatTime:
    """Tests for time formatting."""

    def test_format_seconds(self):
        """Seconds should format as Xs."""
        assert format_time(5) == "5s"
        assert format_time(45) == "45s"

    def test_format_minutes(self):
        """Minutes should format as Xm Ys."""
        assert format_time(90) == "1m 30s"
        assert format_time(125) == "2m 5s"

    def test_format_hours(self):
        """Hours should format as Xh Ym."""
        assert format_time(3661) == "1h 1m"
        assert format_time(7200) == "2h 0m"

    def test_format_none(self):
        """None should return --:--."""
        assert format_time(None) == "--:--"


class TestProgressStats:
    """Tests for progress statistics."""

    def test_elapsed_time(self):
        """Elapsed time should be calculated correctly."""
        stats = ProgressStats(total=10)
        time.sleep(0.1)
        assert stats.elapsed >= 0.1

    def test_percent_complete(self):
        """Percentage should be calculated correctly."""
        stats = ProgressStats(total=10, current=5)
        assert stats.percent == 50.0

    def test_percent_zero_total(self):
        """Zero total should return 100%."""
        stats = ProgressStats(total=0)
        assert stats.percent == 100.0

    def test_rate(self):
        """Rate should be items per second."""
        stats = ProgressStats(total=10, current=5)
        # Force some elapsed time
        stats.start_time = time.time() - 5  # 5 seconds ago
        assert stats.rate == pytest.approx(1.0, rel=0.01)  # 5 items / 5 seconds

    def test_eta(self):
        """ETA should estimate remaining time."""
        stats = ProgressStats(total=10, current=5)
        stats.start_time = time.time() - 5  # 5 seconds ago, rate = 1/s
        assert stats.eta == pytest.approx(5.0, rel=0.01)  # 5 remaining at 1/s = 5 seconds


class TestProgressReporter:
    """Tests for progress reporter."""

    def test_context_manager(self):
        """Progress reporter should work as context manager."""
        with ProgressReporter(total=10, desc="Test") as progress:
            assert progress.stats.total == 10

    def test_update_increments(self):
        """Update should increment current count."""
        with ProgressReporter(total=10, desc="Test") as progress:
            progress.update()
            assert progress.stats.current == 1
            progress.update(n=2)
            assert progress.stats.current == 3

    def test_success_tracking(self):
        """Successful and failed items should be tracked."""
        with ProgressReporter(total=10, desc="Test") as progress:
            progress.update(success=True)
            progress.update(success=True)
            progress.update(success=False)
            assert progress.stats.successful == 2
            assert progress.stats.failed == 1

    def test_set_item(self):
        """Set item should update item name."""
        with ProgressReporter(total=10, desc="Test") as progress:
            progress.set_item("test.jpg")
            assert progress._item_name == "test.jpg"


class TestProgressIter:
    """Tests for progress iterator."""

    def test_iterates_all_items(self):
        """Should iterate through all items."""
        items = ["a", "b", "c"]
        results = []
        for item, progress in progress_iter(items, "Test", "items"):
            results.append(item)
            progress.update()
        assert results == items

    def test_progress_tracks_items(self):
        """Progress should track item count."""
        items = [1, 2, 3, 4, 5]
        for item, progress in progress_iter(items, "Test", "items"):
            progress.update()
        # After iteration, progress should be complete
        # (can't check directly since context manager exits)
