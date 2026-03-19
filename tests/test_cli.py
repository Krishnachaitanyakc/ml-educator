"""Tests for the edu CLI."""
import pytest
from click.testing import CliRunner

from ml_educator.cli import cli


@pytest.fixture
def sample_tsv(tmp_path):
    content = (
        "experiment_id\tdescription\tmetric\ttimestamp\n"
        "exp_001\tbaseline model\t0.72\t2025-01-01T10:00:00\n"
        "exp_002\tincrease learning rate\t0.75\t2025-01-02T10:00:00\n"
        "exp_003\tadd dropout regularization\t0.78\t2025-01-03T10:00:00\n"
    )
    p = tmp_path / "results.tsv"
    p.write_text(content)
    return str(p)


@pytest.fixture
def yaml_dir(tmp_path):
    content = """
- name: custom_concept
  definition: A custom concept loaded from YAML
  intuition: Test intuition
  common_mistakes:
    - Mistake 1
  related_concepts:
    - learning_rate
  explanations:
    beginner: Simple
    intermediate: Medium
    advanced: Complex
"""
    d = tmp_path / "concepts"
    d.mkdir()
    (d / "custom.yaml").write_text(content)
    return str(d)


@pytest.fixture
def sample_diff(tmp_path):
    content = """\
--- a/config.py
+++ b/config.py
@@ -1,3 +1,3 @@
-    lr = 0.001
+    lr = 0.01
"""
    p = tmp_path / "changes.diff"
    p.write_text(content)
    return str(p)


class TestCLI:
    def test_concepts_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["concepts"])
        assert result.exit_code == 0
        assert "learning_rate" in result.output

    def test_concepts_with_name(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["concepts", "--name", "learning_rate"])
        assert result.exit_code == 0
        assert "learning_rate" in result.output

    def test_concepts_with_yaml_dir(self, yaml_dir):
        runner = CliRunner()
        result = runner.invoke(cli, ["concepts", "--concepts-dir", yaml_dir])
        assert result.exit_code == 0
        assert "custom_concept" in result.output

    def test_annotate_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "annotate",
            "--description", "increased learning rate",
            "--metric-change", "0.03",
        ])
        assert result.exit_code == 0

    def test_curriculum_command(self, sample_tsv):
        runner = CliRunner()
        result = runner.invoke(cli, ["curriculum", "--results", sample_tsv])
        assert result.exit_code == 0

    def test_quiz_command(self, sample_tsv):
        runner = CliRunner()
        result = runner.invoke(cli, ["quiz", "--results", sample_tsv])
        assert result.exit_code == 0

    def test_quiz_interactive(self, sample_tsv):
        runner = CliRunner()
        # Provide answers for interactive mode
        result = runner.invoke(
            cli,
            ["quiz", "--results", sample_tsv, "--interactive"],
            input="B\nA\nA\nB\nA\nA\nA\n" * 3,
        )
        assert result.exit_code == 0
        assert "Score:" in result.output

    def test_visualize_command(self, tmp_path):
        runner = CliRunner()
        out = str(tmp_path / "test.png")
        result = runner.invoke(cli, ["visualize", "--concept", "learning_rate", "--output", out])
        assert result.exit_code == 0
        assert "saved" in result.output

    def test_visualize_unknown_concept(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "--concept", "nonexistent"])
        assert result.exit_code == 0
        assert "No diagram available" in result.output

    def test_diff_analyze_command(self, sample_diff):
        runner = CliRunner()
        result = runner.invoke(cli, ["diff-analyze", "--diff-file", sample_diff])
        assert result.exit_code == 0
        assert "learning_rate" in result.output.lower() or "lr" in result.output.lower()

    def test_new_concepts_listed(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["concepts"])
        assert result.exit_code == 0
        assert "transformer" in result.output
        assert "embedding" in result.output
        assert "fine_tuning" in result.output
