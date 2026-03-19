"""Tests for the spaced repetition module."""
import pytest
from datetime import datetime, timedelta

from ml_educator.spaced_repetition import SpacedRepetitionScheduler, ConceptState


class TestSpacedRepetition:
    def test_initial_state(self, tmp_path):
        scheduler = SpacedRepetitionScheduler(str(tmp_path / "state.pkl"))
        state = scheduler.get_state("learning_rate")
        assert state.ease_factor == 2.5
        assert state.interval == 1
        assert state.repetitions == 0

    def test_due_concepts_initially_all_due(self, tmp_path):
        scheduler = SpacedRepetitionScheduler(str(tmp_path / "state.pkl"))
        concepts = ["learning_rate", "batch_size"]
        due = scheduler.get_due_concepts(concepts)
        assert set(due) == set(concepts)

    def test_record_correct_review(self, tmp_path):
        scheduler = SpacedRepetitionScheduler(str(tmp_path / "state.pkl"))
        now = datetime(2025, 1, 1)
        scheduler.record_review("learning_rate", quality=5, now=now)
        state = scheduler.get_state("learning_rate")
        assert state.repetitions == 1
        assert state.interval == 1
        assert state.next_review == now + timedelta(days=1)

    def test_second_correct_review(self, tmp_path):
        scheduler = SpacedRepetitionScheduler(str(tmp_path / "state.pkl"))
        now = datetime(2025, 1, 1)
        scheduler.record_review("learning_rate", quality=5, now=now)
        scheduler.record_review("learning_rate", quality=5, now=now + timedelta(days=1))
        state = scheduler.get_state("learning_rate")
        assert state.repetitions == 2
        assert state.interval == 6

    def test_failed_review_resets(self, tmp_path):
        scheduler = SpacedRepetitionScheduler(str(tmp_path / "state.pkl"))
        now = datetime(2025, 1, 1)
        scheduler.record_review("learning_rate", quality=5, now=now)
        scheduler.record_review("learning_rate", quality=5, now=now + timedelta(days=1))
        # Fail
        scheduler.record_review("learning_rate", quality=1, now=now + timedelta(days=7))
        state = scheduler.get_state("learning_rate")
        assert state.repetitions == 0
        assert state.interval == 1

    def test_not_due_after_review(self, tmp_path):
        scheduler = SpacedRepetitionScheduler(str(tmp_path / "state.pkl"))
        now = datetime(2025, 1, 1)
        scheduler.record_review("learning_rate", quality=5, now=now)
        due = scheduler.get_due_concepts(["learning_rate"], now=now)
        assert "learning_rate" not in due

    def test_due_after_interval(self, tmp_path):
        scheduler = SpacedRepetitionScheduler(str(tmp_path / "state.pkl"))
        now = datetime(2025, 1, 1)
        scheduler.record_review("learning_rate", quality=5, now=now)
        future = now + timedelta(days=2)
        due = scheduler.get_due_concepts(["learning_rate"], now=future)
        assert "learning_rate" in due

    def test_invalid_quality_raises(self, tmp_path):
        scheduler = SpacedRepetitionScheduler(str(tmp_path / "state.pkl"))
        with pytest.raises(ValueError):
            scheduler.record_review("learning_rate", quality=6)
        with pytest.raises(ValueError):
            scheduler.record_review("learning_rate", quality=-1)

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "state.pkl")
        scheduler = SpacedRepetitionScheduler(path)
        now = datetime(2025, 1, 1)
        scheduler.record_review("learning_rate", quality=4, now=now)

        scheduler2 = SpacedRepetitionScheduler(path)
        state = scheduler2.get_state("learning_rate")
        assert state.repetitions == 1

    def test_ease_factor_decreases_on_difficulty(self, tmp_path):
        scheduler = SpacedRepetitionScheduler(str(tmp_path / "state.pkl"))
        now = datetime(2025, 1, 1)
        scheduler.record_review("learning_rate", quality=3, now=now)
        state = scheduler.get_state("learning_rate")
        assert state.ease_factor < 2.5

    def test_ease_factor_minimum(self, tmp_path):
        scheduler = SpacedRepetitionScheduler(str(tmp_path / "state.pkl"))
        now = datetime(2025, 1, 1)
        for i in range(20):
            scheduler.record_review("learning_rate", quality=3, now=now + timedelta(days=i))
        state = scheduler.get_state("learning_rate")
        assert state.ease_factor >= 1.3
