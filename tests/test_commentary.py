"""Tests for the commentary generator module."""
import pytest

from ml_educator.annotator import ExperimentAnnotator, AnnotatedExperiment
from ml_educator.commentary import CommentaryGenerator, Commentary


@pytest.fixture
def annotation():
    ann = ExperimentAnnotator()
    return ann.annotate_experiment("increased learning rate from 0.001 to 0.01", 0.03)


class TestCommentaryGenerator:
    def test_generate_commentary_beginner(self, annotation):
        gen = CommentaryGenerator()
        commentary = gen.generate_commentary(
            experiment_desc="increased learning rate",
            annotation=annotation,
            level="beginner",
        )
        assert isinstance(commentary, Commentary)
        assert len(commentary.markdown_text) > 0

    def test_generate_commentary_intermediate(self, annotation):
        gen = CommentaryGenerator()
        commentary = gen.generate_commentary(
            experiment_desc="increased learning rate",
            annotation=annotation,
            level="intermediate",
        )
        assert len(commentary.markdown_text) > 0

    def test_generate_commentary_advanced(self, annotation):
        gen = CommentaryGenerator()
        commentary = gen.generate_commentary(
            experiment_desc="increased learning rate",
            annotation=annotation,
            level="advanced",
        )
        assert len(commentary.markdown_text) > 0

    def test_concepts_used(self, annotation):
        gen = CommentaryGenerator()
        commentary = gen.generate_commentary(
            experiment_desc="increased learning rate",
            annotation=annotation,
            level="beginner",
        )
        assert isinstance(commentary.concepts_used, list)
        assert len(commentary.concepts_used) > 0

    def test_questions_for_reader(self, annotation):
        gen = CommentaryGenerator()
        commentary = gen.generate_commentary(
            experiment_desc="increased learning rate",
            annotation=annotation,
            level="beginner",
        )
        assert isinstance(commentary.questions_for_reader, list)
        assert len(commentary.questions_for_reader) > 0

    def test_invalid_level(self, annotation):
        gen = CommentaryGenerator()
        with pytest.raises(ValueError):
            gen.generate_commentary(
                experiment_desc="test",
                annotation=annotation,
                level="expert",
            )
