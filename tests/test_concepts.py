"""Tests for the concept library module."""
import pytest

from ml_educator.concepts import ConceptLibrary, Concept


class TestConceptLibrary:
    def test_get_concept(self):
        lib = ConceptLibrary()
        concept = lib.get("learning_rate")
        assert isinstance(concept, Concept)
        assert concept.name == "learning_rate"

    def test_concept_has_definition(self):
        lib = ConceptLibrary()
        concept = lib.get("learning_rate")
        assert len(concept.definition) > 0

    def test_concept_has_intuition(self):
        lib = ConceptLibrary()
        concept = lib.get("learning_rate")
        assert len(concept.intuition) > 0

    def test_concept_levels(self):
        lib = ConceptLibrary()
        concept = lib.get("learning_rate")
        beginner = concept.get_explanation("beginner")
        intermediate = concept.get_explanation("intermediate")
        advanced = concept.get_explanation("advanced")
        assert len(beginner) > 0
        assert len(intermediate) > 0
        assert len(advanced) > 0

    def test_concept_common_mistakes(self):
        lib = ConceptLibrary()
        concept = lib.get("learning_rate")
        assert isinstance(concept.common_mistakes, list)
        assert len(concept.common_mistakes) > 0

    def test_concept_related_concepts(self):
        lib = ConceptLibrary()
        concept = lib.get("learning_rate")
        assert isinstance(concept.related_concepts, list)
        assert len(concept.related_concepts) > 0

    def test_list_concepts(self):
        lib = ConceptLibrary()
        concepts = lib.list_concepts()
        assert "learning_rate" in concepts
        assert "batch_size" in concepts
        assert "regularization" in concepts
        assert "attention" in concepts
        assert "normalization" in concepts
        assert "optimizer" in concepts
        assert "loss_function" in concepts
        assert "architecture_depth" in concepts

    def test_unknown_concept(self):
        lib = ConceptLibrary()
        with pytest.raises(KeyError):
            lib.get("nonexistent_concept")

    def test_concept_math_optional(self):
        lib = ConceptLibrary()
        concept = lib.get("learning_rate")
        # math can be None or a string
        assert concept.math is None or isinstance(concept.math, str)

    def test_search_concepts(self):
        lib = ConceptLibrary()
        results = lib.search("rate")
        assert len(results) >= 1
        assert any("learning_rate" in r.name for r in results)
