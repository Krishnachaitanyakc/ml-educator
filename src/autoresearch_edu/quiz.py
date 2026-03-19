"""Generate quizzes from lessons."""
from dataclasses import dataclass, field
from typing import List, Optional

from autoresearch_edu.concepts import ConceptLibrary
from autoresearch_edu.curriculum import Lesson


@dataclass
class Question:
    """A quiz question."""
    text: str
    question_type: str  # "multiple_choice" or "free_response"
    options: List[str] = field(default_factory=list)
    correct_answer: str = ""


@dataclass
class Quiz:
    """A quiz with questions."""
    questions: List[Question]


# Pre-defined question templates per concept
_MC_QUESTIONS = {
    "learning_rate": Question(
        text="What happens when the learning rate is set too high?",
        question_type="multiple_choice",
        options=[
            "Training converges faster and more accurately",
            "Training diverges and loss increases",
            "Training is unaffected",
            "The model becomes more regularized",
        ],
        correct_answer="Training diverges and loss increases",
    ),
    "batch_size": Question(
        text="What is a common effect of using a very large batch size?",
        question_type="multiple_choice",
        options=[
            "Better generalization",
            "More noisy gradient estimates",
            "Sharper minima and potentially worse generalization",
            "Slower training per step",
        ],
        correct_answer="Sharper minima and potentially worse generalization",
    ),
    "regularization": Question(
        text="What is the primary purpose of regularization?",
        question_type="multiple_choice",
        options=[
            "Speed up training",
            "Prevent overfitting",
            "Increase model capacity",
            "Reduce the learning rate",
        ],
        correct_answer="Prevent overfitting",
    ),
    "attention": Question(
        text="What is the computational complexity of standard self-attention with respect to sequence length n?",
        question_type="multiple_choice",
        options=["O(n)", "O(n log n)", "O(n^2)", "O(n^3)"],
        correct_answer="O(n^2)",
    ),
    "normalization": Question(
        text="Why is Layer Normalization preferred over Batch Normalization in transformers?",
        question_type="multiple_choice",
        options=[
            "It is faster to compute",
            "It normalizes across the batch dimension",
            "It is independent of batch size",
            "It requires fewer parameters",
        ],
        correct_answer="It is independent of batch size",
    ),
    "optimizer": Question(
        text="What advantage does Adam have over vanilla SGD?",
        question_type="multiple_choice",
        options=[
            "It always converges to a better solution",
            "It adapts learning rates per parameter",
            "It uses less memory",
            "It does not require a learning rate",
        ],
        correct_answer="It adapts learning rates per parameter",
    ),
    "loss_function": Question(
        text="Which loss function is most appropriate for multi-class classification?",
        question_type="multiple_choice",
        options=[
            "Mean Squared Error",
            "Cross-Entropy Loss",
            "Hinge Loss",
            "Huber Loss",
        ],
        correct_answer="Cross-Entropy Loss",
    ),
    "architecture_depth": Question(
        text="What technique allows training very deep networks by addressing vanishing gradients?",
        question_type="multiple_choice",
        options=[
            "Dropout",
            "Residual connections (skip connections)",
            "Larger learning rate",
            "Batch normalization only",
        ],
        correct_answer="Residual connections (skip connections)",
    ),
}

_FR_QUESTIONS = {
    "learning_rate": Question(
        text="Explain why learning rate schedules (like cosine annealing) are often beneficial for training.",
        question_type="free_response",
        correct_answer="Learning rate schedules allow the optimizer to take larger steps early in training for faster convergence and smaller steps later for fine-tuning to a better minimum.",
    ),
    "batch_size": Question(
        text="How does batch size interact with learning rate, and what is the linear scaling rule?",
        question_type="free_response",
        correct_answer="The linear scaling rule states that when you increase batch size by k, you should also increase learning rate by k to maintain similar training dynamics.",
    ),
    "regularization": Question(
        text="Compare L1 and L2 regularization and when you might prefer one over the other.",
        question_type="free_response",
        correct_answer="L1 regularization encourages sparsity (many weights become exactly zero) and is good for feature selection. L2 regularization encourages small but non-zero weights and is generally better for preventing overfitting.",
    ),
}


class QuizGenerator:
    """Generate quizzes from lessons."""

    def __init__(self):
        self._library = ConceptLibrary()

    def generate_quiz(self, lesson: Lesson) -> Quiz:
        """Generate a quiz from a lesson."""
        questions = []

        for concept_name in lesson.concept_names:
            # Add multiple choice question
            if concept_name in _MC_QUESTIONS:
                questions.append(_MC_QUESTIONS[concept_name])

            # Add free response question
            if concept_name in _FR_QUESTIONS:
                questions.append(_FR_QUESTIONS[concept_name])

        # Add experiment-based questions
        for exp in lesson.experiments:
            questions.append(Question(
                text=f"Given the experiment '{exp['description']}' which achieved a metric of {exp['metric']}, what would you try next to improve results?",
                question_type="free_response",
                correct_answer="Consider varying related hyperparameters or trying complementary techniques.",
            ))

        return Quiz(questions=questions)
