"""Tests for newly added ML concepts."""
import pytest

from autoresearch_edu.concepts import ConceptLibrary, Concept


NEW_CONCEPTS = [
    "transformer", "self_attention", "positional_encoding", "gan",
    "discriminator", "policy_gradient", "reward_shaping",
    "embedding", "tokenization", "fine_tuning",
]


class TestNewConcepts:
    def test_all_new_concepts_exist(self):
        lib = ConceptLibrary()
        for name in NEW_CONCEPTS:
            concept = lib.get(name)
            assert isinstance(concept, Concept)
            assert concept.name == name

    @pytest.mark.parametrize("name", NEW_CONCEPTS)
    def test_concept_has_definition(self, name):
        lib = ConceptLibrary()
        concept = lib.get(name)
        assert len(concept.definition) > 0

    @pytest.mark.parametrize("name", NEW_CONCEPTS)
    def test_concept_has_intuition(self, name):
        lib = ConceptLibrary()
        concept = lib.get(name)
        assert len(concept.intuition) > 0

    @pytest.mark.parametrize("name", NEW_CONCEPTS)
    def test_concept_has_three_levels(self, name):
        lib = ConceptLibrary()
        concept = lib.get(name)
        for level in ("beginner", "intermediate", "advanced"):
            assert len(concept.get_explanation(level)) > 0

    @pytest.mark.parametrize("name", NEW_CONCEPTS)
    def test_concept_has_common_mistakes(self, name):
        lib = ConceptLibrary()
        concept = lib.get(name)
        assert len(concept.common_mistakes) >= 3

    @pytest.mark.parametrize("name", NEW_CONCEPTS)
    def test_concept_has_related_concepts(self, name):
        lib = ConceptLibrary()
        concept = lib.get(name)
        assert len(concept.related_concepts) >= 2

    def test_total_concept_count(self):
        lib = ConceptLibrary()
        assert len(lib.list_concepts()) == 18  # 8 original + 10 new
