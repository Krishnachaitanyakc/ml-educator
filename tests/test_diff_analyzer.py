"""Tests for the diff analyzer module."""
import pytest

from autoresearch_edu.diff_analyzer import DiffAnalyzer, DiffDetection


SAMPLE_DIFF = """\
diff --git a/config.py b/config.py
--- a/config.py
+++ b/config.py
@@ -10,7 +10,7 @@
 class Config:
-    lr = 0.001
+    lr = 0.01
-    batch_size = 32
+    batch_size = 64
     epochs = 100
"""

SAMPLE_DIFF_DROPOUT = """\
--- a/model.py
+++ b/model.py
@@ -5,3 +5,3 @@
-    dropout = 0.3
+    dropout = 0.5
"""

SAMPLE_DIFF_OPTIMIZER = """\
--- a/train.py
+++ b/train.py
@@ -1,3 +1,3 @@
-    optimizer = 'sgd'
+    optimizer = 'adam'
"""


class TestDiffAnalyzer:
    def test_detect_lr_change(self):
        analyzer = DiffAnalyzer()
        detections = analyzer.analyze(SAMPLE_DIFF)
        lr_detections = [d for d in detections if d.parameter == "lr"]
        assert len(lr_detections) == 1
        assert lr_detections[0].old_value == "0.001"
        assert lr_detections[0].new_value == "0.01"
        assert lr_detections[0].concept_name == "learning_rate"

    def test_detect_batch_size_change(self):
        analyzer = DiffAnalyzer()
        detections = analyzer.analyze(SAMPLE_DIFF)
        bs = [d for d in detections if d.parameter == "batch_size"]
        assert len(bs) == 1
        assert bs[0].old_value == "32"
        assert bs[0].new_value == "64"
        assert bs[0].concept_name == "batch_size"

    def test_detect_dropout_change(self):
        analyzer = DiffAnalyzer()
        detections = analyzer.analyze(SAMPLE_DIFF_DROPOUT)
        assert len(detections) == 1
        assert detections[0].concept_name == "regularization"

    def test_detect_optimizer_change(self):
        analyzer = DiffAnalyzer()
        detections = analyzer.analyze(SAMPLE_DIFF_OPTIMIZER)
        assert len(detections) == 1
        assert detections[0].concept_name == "optimizer"

    def test_get_concepts_from_diff(self):
        analyzer = DiffAnalyzer()
        concepts = analyzer.get_concepts_from_diff(SAMPLE_DIFF)
        assert "learning_rate" in concepts
        assert "batch_size" in concepts

    def test_no_changes_detected(self):
        analyzer = DiffAnalyzer()
        detections = analyzer.analyze("some random text\nno hyperparams here")
        assert len(detections) == 0

    def test_detection_dataclass(self):
        d = DiffDetection(
            parameter="lr",
            old_value="0.001",
            new_value="0.01",
            concept_name="learning_rate",
        )
        assert d.parameter == "lr"
        assert d.concept_name == "learning_rate"
