"""Tests for YAML concept loading."""
import pytest

from ml_educator.concepts import ConceptLibrary


@pytest.fixture
def yaml_file(tmp_path):
    content = """
- name: test_concept
  definition: A test concept for unit testing
  intuition: This is a test
  math: "y = x + 1"
  common_mistakes:
    - Mistake 1
    - Mistake 2
  related_concepts:
    - learning_rate
  explanations:
    beginner: Simple explanation
    intermediate: Moderate explanation
    advanced: Complex explanation
"""
    p = tmp_path / "test_concepts.yaml"
    p.write_text(content)
    return str(p)


@pytest.fixture
def yaml_multi(tmp_path):
    content = """
- name: concept_a
  definition: Concept A
- name: concept_b
  definition: Concept B
"""
    p = tmp_path / "multi.yaml"
    p.write_text(content)
    return str(p)


class TestYAMLConcepts:
    def test_load_from_yaml(self, yaml_file):
        lib = ConceptLibrary()
        count = lib.load_from_yaml(yaml_file)
        assert count == 1
        concept = lib.get("test_concept")
        assert concept.definition == "A test concept for unit testing"

    def test_yaml_concept_levels(self, yaml_file):
        lib = ConceptLibrary()
        lib.load_from_yaml(yaml_file)
        concept = lib.get("test_concept")
        assert concept.get_explanation("beginner") == "Simple explanation"
        assert concept.get_explanation("advanced") == "Complex explanation"

    def test_yaml_concept_math(self, yaml_file):
        lib = ConceptLibrary()
        lib.load_from_yaml(yaml_file)
        concept = lib.get("test_concept")
        assert concept.math == "y = x + 1"

    def test_load_multiple_from_yaml(self, yaml_multi):
        lib = ConceptLibrary()
        count = lib.load_from_yaml(yaml_multi)
        assert count == 2
        assert "concept_a" in lib.list_concepts()
        assert "concept_b" in lib.list_concepts()

    def test_yaml_concept_in_search(self, yaml_file):
        lib = ConceptLibrary()
        lib.load_from_yaml(yaml_file)
        results = lib.search("test")
        assert any(c.name == "test_concept" for c in results)

    def test_empty_yaml(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        lib = ConceptLibrary()
        count = lib.load_from_yaml(str(p))
        assert count == 0
