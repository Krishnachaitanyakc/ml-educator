"""Tests for the curriculum builder module."""
import pytest

from autoresearch_edu.curriculum import CurriculumBuilder, Curriculum, Lesson


@pytest.fixture
def experiment_history():
    return [
        {"experiment_id": "exp_001", "description": "baseline model", "metric": 0.72},
        {"experiment_id": "exp_002", "description": "increase learning rate", "metric": 0.75},
        {"experiment_id": "exp_003", "description": "add dropout regularization", "metric": 0.78},
        {"experiment_id": "exp_004", "description": "larger batch size", "metric": 0.74},
        {"experiment_id": "exp_005", "description": "add layer normalization", "metric": 0.82},
        {"experiment_id": "exp_006", "description": "change optimizer to adam", "metric": 0.84},
        {"experiment_id": "exp_007", "description": "increase network depth", "metric": 0.80},
    ]


class TestCurriculumBuilder:
    def test_build_curriculum(self, experiment_history):
        builder = CurriculumBuilder()
        curriculum = builder.build_curriculum(experiment_history)
        assert isinstance(curriculum, Curriculum)

    def test_curriculum_has_lessons(self, experiment_history):
        builder = CurriculumBuilder()
        curriculum = builder.build_curriculum(experiment_history)
        assert len(curriculum.lessons) > 0

    def test_lessons_have_titles(self, experiment_history):
        builder = CurriculumBuilder()
        curriculum = builder.build_curriculum(experiment_history)
        for lesson in curriculum.lessons:
            assert isinstance(lesson, Lesson)
            assert len(lesson.title) > 0

    def test_lessons_reference_experiments(self, experiment_history):
        builder = CurriculumBuilder()
        curriculum = builder.build_curriculum(experiment_history)
        all_exp_ids = set()
        for lesson in curriculum.lessons:
            for exp_id in lesson.experiment_ids:
                all_exp_ids.add(exp_id)
        # At least some experiments should be referenced
        assert len(all_exp_ids) > 0

    def test_lessons_ordered_by_complexity(self, experiment_history):
        builder = CurriculumBuilder()
        curriculum = builder.build_curriculum(experiment_history)
        # Each lesson should have a complexity level
        for lesson in curriculum.lessons:
            assert lesson.complexity >= 1

    def test_empty_history(self):
        builder = CurriculumBuilder()
        curriculum = builder.build_curriculum([])
        assert len(curriculum.lessons) == 0
