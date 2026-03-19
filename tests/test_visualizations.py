"""Tests for the visualizations module."""
import os
import pytest

from ml_educator.visualizations import (
    plot_learning_rate_landscape,
    plot_attention_heatmap,
    plot_optimizer_paths,
    DIAGRAM_GENERATORS,
)


class TestVisualizations:
    def test_learning_rate_diagram(self, tmp_path):
        out = str(tmp_path / "lr.png")
        result = plot_learning_rate_landscape(out)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_attention_heatmap(self, tmp_path):
        out = str(tmp_path / "attn.png")
        result = plot_attention_heatmap(out)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_optimizer_paths(self, tmp_path):
        out = str(tmp_path / "opt.png")
        result = plot_optimizer_paths(out)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_diagram_generators_registry(self):
        assert "learning_rate" in DIAGRAM_GENERATORS
        assert "attention" in DIAGRAM_GENERATORS
        assert "optimizer" in DIAGRAM_GENERATORS

    def test_all_generators_callable(self):
        for name, func in DIAGRAM_GENERATORS.items():
            assert callable(func)
