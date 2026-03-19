"""Build curricula from experiment history."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

from autoresearch_edu.annotator import ExperimentAnnotator


@dataclass
class Lesson:
    """A lesson in the curriculum."""
    title: str
    concept_names: List[str]
    experiment_ids: List[str]
    experiments: List[Dict[str, Any]]
    complexity: int = 1


@dataclass
class Curriculum:
    """An ordered curriculum of lessons."""
    lessons: List[Lesson]


# Concept complexity levels for ordering
_CONCEPT_COMPLEXITY = {
    "learning_rate": 1,
    "batch_size": 1,
    "loss_function": 2,
    "optimizer": 2,
    "regularization": 2,
    "normalization": 3,
    "architecture_depth": 3,
    "attention": 4,
}

# Lesson topic groups
_TOPIC_GROUPS = {
    "Training Basics: Learning Rate & Batch Size": ["learning_rate", "batch_size"],
    "Optimization: Loss Functions & Optimizers": ["loss_function", "optimizer"],
    "Preventing Overfitting: Regularization": ["regularization"],
    "Stabilizing Training: Normalization": ["normalization"],
    "Architecture Decisions: Depth & Structure": ["architecture_depth"],
    "Advanced: Attention Mechanisms": ["attention"],
}


class CurriculumBuilder:
    """Build curricula from experiment history."""

    def __init__(self):
        self._annotator = ExperimentAnnotator()

    def build_curriculum(self, experiment_history: List[Dict[str, Any]]) -> Curriculum:
        """Build a curriculum from experiment history.

        Groups experiments into lessons based on the concepts they involve,
        ordered by complexity.
        """
        if not experiment_history:
            return Curriculum(lessons=[])

        # Annotate all experiments
        annotated = []
        for exp in experiment_history:
            metric_change = 0.0
            annotation = self._annotator.annotate_experiment(
                exp["description"], metric_change
            )
            annotated.append((exp, annotation))

        # Group experiments by topic
        lessons = []
        for title, topic_concepts in _TOPIC_GROUPS.items():
            matching_exps = []
            for exp, ann in annotated:
                if any(c in ann.concept_references for c in topic_concepts):
                    matching_exps.append(exp)

            if matching_exps:
                complexity = min(
                    _CONCEPT_COMPLEXITY.get(c, 5) for c in topic_concepts
                )
                lessons.append(Lesson(
                    title=title,
                    concept_names=topic_concepts,
                    experiment_ids=[e["experiment_id"] for e in matching_exps],
                    experiments=matching_exps,
                    complexity=complexity,
                ))

        # Sort by complexity
        lessons.sort(key=lambda l: l.complexity)

        return Curriculum(lessons=lessons)
