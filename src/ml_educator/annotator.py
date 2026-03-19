"""Annotate experiments with relevant ML concepts."""
from dataclasses import dataclass, field
from typing import Dict, List

from ml_educator.concepts import ConceptLibrary


@dataclass
class AnnotatedExperiment:
    """An experiment annotated with concept references and explanations."""
    description: str
    metric_change: float
    concept_references: List[str]
    explanation_text: str
    skill_level_explanations: Dict[str, str]


# Keywords that map to concepts
_KEYWORD_MAP = {
    "learning_rate": ["learning rate", "lr", "step size"],
    "batch_size": ["batch size", "batch", "mini-batch", "minibatch"],
    "regularization": ["regularization", "dropout", "weight decay", "l1", "l2", "data augmentation"],
    "attention": ["attention", "self-attention", "cross-attention", "multi-head"],
    "normalization": ["normalization", "batch norm", "layer norm", "group norm", "batchnorm", "layernorm"],
    "optimizer": ["optimizer", "adam", "sgd", "adamw", "rmsprop", "momentum"],
    "loss_function": ["loss", "cross-entropy", "mse", "focal loss", "contrastive"],
    "architecture_depth": ["depth", "layer", "deeper", "shallow", "network depth", "add layer", "remove layer"],
    "transformer": ["transformer"],
    "self_attention": ["self-attention", "self attention"],
    "positional_encoding": ["positional encoding", "position embedding", "rotary"],
    "gan": ["gan", "generative adversarial"],
    "discriminator": ["discriminator"],
    "policy_gradient": ["policy gradient", "reinforce", "ppo", "trpo"],
    "reward_shaping": ["reward shaping", "reward design", "shaped reward"],
    "embedding": ["embedding", "word2vec", "glove"],
    "tokenization": ["tokenization", "tokenizer", "bpe", "sentencepiece"],
    "fine_tuning": ["fine-tuning", "fine tuning", "finetune", "finetuning", "lora", "adapter"],
}


class ExperimentAnnotator:
    """Annotate experiments with relevant ML concepts."""

    def __init__(self):
        self.library = ConceptLibrary()

    def annotate_experiment(
        self, description: str, metric_change: float
    ) -> AnnotatedExperiment:
        """Annotate an experiment with concept references and explanations."""
        # Find relevant concepts
        desc_lower = description.lower()
        matched_concepts = []

        for concept_name, keywords in _KEYWORD_MAP.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    if concept_name not in matched_concepts:
                        matched_concepts.append(concept_name)
                    break

        # Generate explanation
        if metric_change > 0:
            direction = "improved"
            direction_detail = f"This change improved the metric by {metric_change:.4f}."
        elif metric_change < 0:
            direction = "regression"
            direction_detail = f"This change caused a regression of {abs(metric_change):.4f} in the metric."
        else:
            direction = "neutral"
            direction_detail = "This change had no measurable effect on the metric."

        # Build explanation text
        explanations = []
        explanations.append(f"Experiment: {description}")
        explanations.append(direction_detail)

        for concept_name in matched_concepts:
            concept = self.library.get(concept_name)
            explanations.append(f"\nRelated concept - {concept.name}: {concept.definition}")

        explanation_text = "\n".join(explanations)

        # Build skill-level explanations
        skill_levels = {}
        for level in ("beginner", "intermediate", "advanced"):
            parts = [f"Experiment: {description}", direction_detail, ""]
            for concept_name in matched_concepts:
                concept = self.library.get(concept_name)
                parts.append(f"**{concept.name}**: {concept.get_explanation(level)}")
            skill_levels[level] = "\n".join(parts)

        return AnnotatedExperiment(
            description=description,
            metric_change=metric_change,
            concept_references=matched_concepts,
            explanation_text=explanation_text,
            skill_level_explanations=skill_levels,
        )
