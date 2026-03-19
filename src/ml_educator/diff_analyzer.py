"""Analyze git diffs to detect hyperparameter changes and map to concepts."""
import re
from dataclasses import dataclass
from typing import List

from ml_educator.concepts import ConceptLibrary


@dataclass
class DiffDetection:
    """A detected hyperparameter change from a diff."""
    parameter: str
    old_value: str
    new_value: str
    concept_name: str


# Patterns mapping regex to concept names
_DIFF_PATTERNS = [
    (re.compile(r"[-]\s*.*(?:lr|learning.?rate)\s*[=:]\s*([0-9.e\-]+)", re.IGNORECASE), "learning_rate", "lr"),
    (re.compile(r"[+]\s*.*(?:lr|learning.?rate)\s*[=:]\s*([0-9.e\-]+)", re.IGNORECASE), "learning_rate", "lr"),
    (re.compile(r"[-]\s*.*batch.?size\s*[=:]\s*([0-9]+)", re.IGNORECASE), "batch_size", "batch_size"),
    (re.compile(r"[+]\s*.*batch.?size\s*[=:]\s*([0-9]+)", re.IGNORECASE), "batch_size", "batch_size"),
    (re.compile(r"[-]\s*.*dropout\s*[=:]\s*([0-9.]+)", re.IGNORECASE), "regularization", "dropout"),
    (re.compile(r"[+]\s*.*dropout\s*[=:]\s*([0-9.]+)", re.IGNORECASE), "regularization", "dropout"),
    (re.compile(r"[-]\s*.*weight.?decay\s*[=:]\s*([0-9.e\-]+)", re.IGNORECASE), "regularization", "weight_decay"),
    (re.compile(r"[+]\s*.*weight.?decay\s*[=:]\s*([0-9.e\-]+)", re.IGNORECASE), "regularization", "weight_decay"),
    (re.compile(r"[-]\s*.*(?:num.?layers|n.?layers|depth)\s*[=:]\s*([0-9]+)", re.IGNORECASE), "architecture_depth", "depth"),
    (re.compile(r"[+]\s*.*(?:num.?layers|n.?layers|depth)\s*[=:]\s*([0-9]+)", re.IGNORECASE), "architecture_depth", "depth"),
    (re.compile(r"[-]\s*.*(?:optimizer)\s*[=:]\s*['\"]?(\w+)", re.IGNORECASE), "optimizer", "optimizer"),
    (re.compile(r"[+]\s*.*(?:optimizer)\s*[=:]\s*['\"]?(\w+)", re.IGNORECASE), "optimizer", "optimizer"),
    (re.compile(r"[-]\s*.*(?:embed.?dim|d.?model|hidden.?size)\s*[=:]\s*([0-9]+)", re.IGNORECASE), "embedding", "embed_dim"),
    (re.compile(r"[+]\s*.*(?:embed.?dim|d.?model|hidden.?size)\s*[=:]\s*([0-9]+)", re.IGNORECASE), "embedding", "embed_dim"),
]


class DiffAnalyzer:
    """Analyze unified diffs to detect hyperparameter changes."""

    def __init__(self):
        self._library = ConceptLibrary()

    def analyze(self, diff_text: str) -> List[DiffDetection]:
        """Parse a unified diff and detect hyperparameter changes."""
        lines = diff_text.splitlines()
        # Group old (-) and new (+) values by parameter
        old_values = {}
        new_values = {}

        for line in lines:
            for pattern, concept, param in _DIFF_PATTERNS:
                m = pattern.match(line)
                if m:
                    value = m.group(1)
                    if line.startswith("-"):
                        old_values[param] = (value, concept)
                    elif line.startswith("+"):
                        new_values[param] = (value, concept)

        detections = []
        # Find parameters that changed
        all_params = set(old_values.keys()) | set(new_values.keys())
        for param in sorted(all_params):
            old_val, concept = old_values.get(param, ("", ""))
            new_val, concept2 = new_values.get(param, ("", ""))
            concept = concept or concept2
            if old_val != new_val:
                detections.append(DiffDetection(
                    parameter=param,
                    old_value=old_val,
                    new_value=new_val,
                    concept_name=concept,
                ))

        return detections

    def get_concepts_from_diff(self, diff_text: str) -> List[str]:
        """Return unique concept names detected in a diff."""
        detections = self.analyze(diff_text)
        seen = []
        for d in detections:
            if d.concept_name not in seen:
                seen.append(d.concept_name)
        return seen
