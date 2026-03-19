"""Tests for the edu CLI."""
import pytest
from click.testing import CliRunner

from autoresearch_edu.cli import cli


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
