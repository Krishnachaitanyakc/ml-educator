"""Tests for the experiment annotator module."""
import pytest

from autoresearch_edu.annotator import ExperimentAnnotator, AnnotatedExperiment


class TestExperimentAnnotator:
    def test_annotate_experiment(self):
        ann = ExperimentAnnotator()
        result = ann.annotate_experiment("increased learning rate", 0.03)
        assert isinstance(result, AnnotatedExperiment)

    def test_concept_references(self):
        ann = ExperimentAnnotator()
        result = ann.annotate_experiment("increased learning rate", 0.03)
        assert len(result.concept_references) > 0
        assert any("learning_rate" in c for c in result.concept_references)

    def test_explanation_text(self):
        ann = ExperimentAnnotator()
        result = ann.annotate_experiment("increased learning rate", 0.03)
        assert len(result.explanation_text) > 0

    def test_skill_level_explanations(self):
        ann = ExperimentAnnotator()
        result = ann.annotate_experiment("added dropout regularization", -0.01)
        assert "beginner" in result.skill_level_explanations
        assert "intermediate" in result.skill_level_explanations
        assert "advanced" in result.skill_level_explanations

    def test_negative_metric_change(self):
        ann = ExperimentAnnotator()
        result = ann.annotate_experiment("larger batch size", -0.05)
        # Should explain why metric decreased
        assert "decrease" in result.explanation_text.lower() or "hurt" in result.explanation_text.lower() or "negative" in result.explanation_text.lower() or "regression" in result.explanation_text.lower()

    def test_annotate_batch_size(self):
        ann = ExperimentAnnotator()
        result = ann.annotate_experiment("changed batch size to 64", 0.02)
        assert any("batch_size" in c for c in result.concept_references)

    def test_annotate_normalization(self):
        ann = ExperimentAnnotator()
        result = ann.annotate_experiment("added layer normalization", 0.05)
        assert any("normalization" in c for c in result.concept_references)
