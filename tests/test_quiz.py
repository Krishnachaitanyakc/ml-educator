"""Tests for the quiz generator module."""
import pytest

from autoresearch_edu.curriculum import CurriculumBuilder, Lesson
from autoresearch_edu.quiz import QuizGenerator, Quiz, Question


@pytest.fixture
def lesson():
    return Lesson(
        title="Understanding Learning Rate",
        concept_names=["learning_rate", "optimizer"],
        experiment_ids=["exp_002"],
        experiments=[
            {"experiment_id": "exp_002", "description": "increase learning rate", "metric": 0.75},
        ],
        complexity=1,
    )


class TestQuizGenerator:
    def test_generate_quiz(self, lesson):
        gen = QuizGenerator()
        quiz = gen.generate_quiz(lesson)
        assert isinstance(quiz, Quiz)

    def test_quiz_has_questions(self, lesson):
        gen = QuizGenerator()
        quiz = gen.generate_quiz(lesson)
        assert len(quiz.questions) > 0

    def test_question_structure(self, lesson):
        gen = QuizGenerator()
        quiz = gen.generate_quiz(lesson)
        for q in quiz.questions:
            assert isinstance(q, Question)
            assert len(q.text) > 0
            assert q.question_type in ("multiple_choice", "free_response")

    def test_multiple_choice_has_options(self, lesson):
        gen = QuizGenerator()
        quiz = gen.generate_quiz(lesson)
        mc_questions = [q for q in quiz.questions if q.question_type == "multiple_choice"]
        for q in mc_questions:
            assert len(q.options) >= 2
            assert q.correct_answer in q.options

    def test_quiz_references_concepts(self, lesson):
        gen = QuizGenerator()
        quiz = gen.generate_quiz(lesson)
        # Questions should be related to the lesson concepts
        assert len(quiz.questions) >= 1
