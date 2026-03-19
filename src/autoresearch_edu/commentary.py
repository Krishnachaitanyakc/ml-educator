"""Generate educational commentary for experiments."""
from dataclasses import dataclass, field
from typing import List

from jinja2 import Environment, BaseLoader

from autoresearch_edu.annotator import AnnotatedExperiment
from autoresearch_edu.concepts import ConceptLibrary


@dataclass
class Commentary:
    """Educational commentary for an experiment."""
    markdown_text: str
    concepts_used: List[str]
    questions_for_reader: List[str]


_VALID_LEVELS = ("beginner", "intermediate", "advanced")

_COMMENTARY_TEMPLATE = """\
## Experiment Commentary: {{ description }}

### What happened?
{{ direction_text }}

### Why does this matter?
{{ level_explanation }}

{% for concept_name, concept_text in concept_explanations %}
### Understanding {{ concept_name }}
{{ concept_text }}

{% endfor %}
### Key Takeaway
{{ takeaway }}
"""

_QUESTION_TEMPLATES = {
    "beginner": [
        "What do you think would happen if we changed {concept} in the opposite direction?",
        "Can you explain in your own words why {concept} matters for model training?",
    ],
    "intermediate": [
        "How might {concept} interact with other hyperparameters in this experiment?",
        "What experiments would you design to isolate the effect of {concept}?",
    ],
    "advanced": [
        "What theoretical framework best explains the role of {concept} in this result?",
        "How would you expect this result to change at different scales (data size, model size)?",
    ],
}


class CommentaryGenerator:
    """Generate educational commentary in journal style."""

    def __init__(self):
        self._env = Environment(loader=BaseLoader())
        self._library = ConceptLibrary()

    def generate_commentary(
        self,
        experiment_desc: str,
        annotation: AnnotatedExperiment,
        level: str = "beginner",
    ) -> Commentary:
        """Generate commentary at a given skill level."""
        if level not in _VALID_LEVELS:
            raise ValueError(
                f"Unknown level: {level}. Available: {', '.join(_VALID_LEVELS)}"
            )

        # Direction text
        if annotation.metric_change > 0:
            direction_text = f"The experiment '{experiment_desc}' improved the metric by {annotation.metric_change:.4f}. This is a positive result worth understanding."
        elif annotation.metric_change < 0:
            direction_text = f"The experiment '{experiment_desc}' decreased the metric by {abs(annotation.metric_change):.4f}. Understanding why is just as valuable as a positive result."
        else:
            direction_text = f"The experiment '{experiment_desc}' had no measurable effect. This can still teach us something about the model."

        # Concept explanations at the right level
        concept_explanations = []
        for concept_name in annotation.concept_references:
            concept = self._library.get(concept_name)
            concept_explanations.append(
                (concept_name, concept.get_explanation(level))
            )

        # Level-specific explanation
        level_explanation = annotation.skill_level_explanations.get(level, "")

        # Takeaway
        if annotation.metric_change > 0:
            takeaway = f"This experiment shows that {experiment_desc} was beneficial. Consider exploring variations of this approach."
        elif annotation.metric_change < 0:
            takeaway = f"This experiment shows that {experiment_desc} was not helpful in this context. This helps narrow down the search space."
        else:
            takeaway = f"While {experiment_desc} did not change the metric, it helps us understand what the model is sensitive to."

        # Render template
        template = self._env.from_string(_COMMENTARY_TEMPLATE)
        markdown_text = template.render(
            description=experiment_desc,
            direction_text=direction_text,
            level_explanation=level_explanation,
            concept_explanations=concept_explanations,
            takeaway=takeaway,
        )

        # Generate questions
        questions = []
        q_templates = _QUESTION_TEMPLATES.get(level, _QUESTION_TEMPLATES["beginner"])
        for concept_name in annotation.concept_references:
            for q_template in q_templates:
                questions.append(q_template.format(concept=concept_name))

        if not questions:
            questions = ["What did you learn from this experiment?"]

        return Commentary(
            markdown_text=markdown_text,
            concepts_used=list(annotation.concept_references),
            questions_for_reader=questions,
        )
