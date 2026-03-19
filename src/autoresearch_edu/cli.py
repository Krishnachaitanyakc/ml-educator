"""Command-line interface for autoresearch-edu."""
import csv

import click

from autoresearch_edu.concepts import ConceptLibrary
from autoresearch_edu.annotator import ExperimentAnnotator
from autoresearch_edu.commentary import CommentaryGenerator
from autoresearch_edu.curriculum import CurriculumBuilder
from autoresearch_edu.quiz import QuizGenerator


@click.group()
def cli():
    """Educational mode for autoresearch -- explains ML reasoning."""
    pass


@cli.command()
@click.option("--name", default=None, help="Specific concept name")
@click.option("--level", default="beginner", help="Explanation level")
@click.option("--concepts-dir", default=None, help="Path to directory of YAML concept files")
def concepts(name, level, concepts_dir):
    """Browse the ML concept library."""
    lib = ConceptLibrary()
    if concepts_dir:
        _load_yaml_concepts(lib, concepts_dir)
    if name:
        concept = lib.get(name)
        click.echo(f"\n{concept.name}")
        click.echo(f"  Definition: {concept.definition}")
        click.echo(f"  Intuition: {concept.intuition}")
        click.echo(f"  Explanation ({level}): {concept.get_explanation(level)}")
        click.echo(f"  Common mistakes: {', '.join(concept.common_mistakes)}")
        click.echo(f"  Related: {', '.join(concept.related_concepts)}")
    else:
        click.echo("Available ML Concepts:")
        for c_name in lib.list_concepts():
            concept = lib.get(c_name)
            click.echo(f"  - {c_name}: {concept.definition[:80]}...")


@cli.command()
@click.option("--description", required=True, help="Experiment description")
@click.option("--metric-change", required=True, type=float, help="Change in metric")
@click.option("--level", default="beginner", help="Explanation level")
def annotate(description, metric_change, level):
    """Annotate a specific experiment with ML concepts."""
    annotator = ExperimentAnnotator()
    annotation = annotator.annotate_experiment(description, metric_change)
    click.echo(f"\nExperiment: {description}")
    click.echo(f"Metric change: {metric_change}")
    click.echo(f"Concepts: {', '.join(annotation.concept_references)}")
    click.echo(f"\n{annotation.skill_level_explanations.get(level, annotation.explanation_text)}")


@cli.command()
@click.option("--results", required=True, help="Path to results.tsv")
@click.option("--level", default="beginner", help="Explanation level")
def commentary(results, level):
    """Generate educational commentary for experiments."""
    experiments = _load_experiments(results)
    annotator = ExperimentAnnotator()
    gen = CommentaryGenerator()

    for i, exp in enumerate(experiments):
        metric_change = 0.0
        if i > 0:
            metric_change = exp["metric"] - experiments[i - 1]["metric"]
        annotation = annotator.annotate_experiment(exp["description"], metric_change)
        comm = gen.generate_commentary(exp["description"], annotation, level)
        click.echo(comm.markdown_text)


@cli.command()
@click.option("--results", required=True, help="Path to results.tsv")
def curriculum(results):
    """Build a curriculum from experiment history."""
    experiments = _load_experiments(results)
    builder = CurriculumBuilder()
    curr = builder.build_curriculum(experiments)

    click.echo("Curriculum")
    click.echo("=" * 40)
    for i, lesson in enumerate(curr.lessons, 1):
        click.echo(f"\nLesson {i}: {lesson.title}")
        click.echo(f"  Concepts: {', '.join(lesson.concept_names)}")
        click.echo(f"  Complexity: {lesson.complexity}")
        click.echo(f"  Experiments: {', '.join(lesson.experiment_ids)}")


@cli.command()
@click.option("--results", required=True, help="Path to results.tsv")
@click.option("--interactive", is_flag=True, help="Run quiz interactively")
def quiz(results, interactive):
    """Generate a quiz from experiment history."""
    experiments = _load_experiments(results)
    builder = CurriculumBuilder()
    curr = builder.build_curriculum(experiments)
    gen = QuizGenerator()

    if interactive:
        _run_interactive_quiz(curr, gen)
    else:
        for lesson in curr.lessons:
            q = gen.generate_quiz(lesson)
            click.echo(f"\nQuiz: {lesson.title}")
            click.echo("-" * 40)
            for i, question in enumerate(q.questions, 1):
                click.echo(f"\n  Q{i}. {question.text}")
                if question.question_type == "multiple_choice":
                    for j, opt in enumerate(question.options):
                        letter = chr(65 + j)
                        click.echo(f"      {letter}) {opt}")


@cli.command()
@click.option("--concept", required=True, help="Concept name to visualize")
@click.option("--output", default=None, help="Output file path")
def visualize(concept, output):
    """Generate a diagram for an ML concept."""
    from autoresearch_edu.visualizations import DIAGRAM_GENERATORS
    if concept not in DIAGRAM_GENERATORS:
        available = ", ".join(DIAGRAM_GENERATORS.keys())
        click.echo(f"No diagram available for '{concept}'. Available: {available}")
        return
    out_path = output or f"{concept}.png"
    DIAGRAM_GENERATORS[concept](out_path)
    click.echo(f"Diagram saved to {out_path}")


@cli.command()
@click.option("--state-file", default=".spaced_reps.pkl", help="Path to state file")
def review(state_file):
    """Review due concepts using spaced repetition."""
    from autoresearch_edu.spaced_repetition import SpacedRepetitionScheduler
    from autoresearch_edu.quiz import QuizGenerator, _MC_QUESTIONS
    from autoresearch_edu.curriculum import Lesson

    lib = ConceptLibrary()
    scheduler = SpacedRepetitionScheduler(state_file)
    due = scheduler.get_due_concepts(lib.list_concepts())

    if not due:
        click.echo("No concepts due for review!")
        return

    click.echo(f"{len(due)} concept(s) due for review.\n")
    gen = QuizGenerator()

    for concept_name in due:
        if concept_name not in _MC_QUESTIONS:
            continue
        question = _MC_QUESTIONS[concept_name]
        click.echo(f"Concept: {concept_name}")
        click.echo(f"  Q: {question.text}")
        for j, opt in enumerate(question.options):
            click.echo(f"    {chr(65 + j)}) {opt}")

        answer = click.prompt("Your answer (A/B/C/D)")
        answer_idx = ord(answer.upper()) - 65
        if 0 <= answer_idx < len(question.options):
            chosen = question.options[answer_idx]
            if chosen == question.correct_answer:
                click.echo("Correct!\n")
                scheduler.record_review(concept_name, 5)
            else:
                click.echo(f"Incorrect. The answer is: {question.correct_answer}\n")
                scheduler.record_review(concept_name, 1)
        else:
            click.echo(f"Invalid choice. The answer is: {question.correct_answer}\n")
            scheduler.record_review(concept_name, 1)


@cli.command()
def assess():
    """Assess your ML knowledge level to personalize your curriculum."""
    from autoresearch_edu.quiz import _MC_QUESTIONS

    topics = {
        "basics": ["learning_rate", "batch_size", "loss_function"],
        "intermediate": ["regularization", "optimizer", "normalization"],
        "advanced": ["attention", "transformer", "self_attention"],
    }

    scores = {}
    total_correct = 0
    total_questions = 0

    for topic, concept_names in topics.items():
        correct = 0
        asked = 0
        for cname in concept_names:
            if cname not in _MC_QUESTIONS:
                continue
            q = _MC_QUESTIONS[cname]
            click.echo(f"\n[{topic.upper()}] {q.text}")
            for j, opt in enumerate(q.options):
                click.echo(f"  {chr(65 + j)}) {opt}")
            answer = click.prompt("Your answer (A/B/C/D)")
            answer_idx = ord(answer.upper()) - 65
            asked += 1
            if 0 <= answer_idx < len(q.options) and q.options[answer_idx] == q.correct_answer:
                click.echo("Correct!")
                correct += 1
            else:
                click.echo(f"Incorrect. Answer: {q.correct_answer}")
        if asked > 0:
            scores[topic] = correct / asked
            total_correct += correct
            total_questions += asked

    click.echo("\n" + "=" * 40)
    click.echo("Assessment Results")
    click.echo("=" * 40)
    for topic, score in scores.items():
        pct = int(score * 100)
        click.echo(f"  {topic}: {pct}%")

    if total_questions > 0:
        overall = total_correct / total_questions
        if overall >= 0.8:
            level = "advanced"
        elif overall >= 0.5:
            level = "intermediate"
        else:
            level = "beginner"
        click.echo(f"\nRecommended level: {level}")
        click.echo(f"Overall: {int(overall * 100)}% ({total_correct}/{total_questions})")


@cli.command(name="diff-analyze")
@click.option("--diff-file", required=True, help="Path to a unified diff file")
@click.option("--level", default="beginner", help="Explanation level")
def diff_analyze(diff_file, level):
    """Detect hyperparameter changes in a git diff and explain related concepts."""
    from autoresearch_edu.diff_analyzer import DiffAnalyzer

    with open(diff_file, "r") as f:
        diff_text = f.read()

    analyzer = DiffAnalyzer()
    detections = analyzer.analyze(diff_text)

    if not detections:
        click.echo("No hyperparameter changes detected in the diff.")
        return

    lib = ConceptLibrary()
    click.echo(f"Detected {len(detections)} hyperparameter change(s):\n")
    for d in detections:
        click.echo(f"  {d.parameter}: {d.old_value or '(new)'} -> {d.new_value or '(removed)'}")
        concept = lib.get(d.concept_name)
        click.echo(f"  Related concept: {concept.name}")
        click.echo(f"  {concept.get_explanation(level)}\n")


@cli.command()
@click.option("--description", required=True, help="Experiment description")
@click.option("--level", default="beginner", help="Explanation level")
def explain(description, level):
    """Get an LLM-powered explanation of an experiment."""
    from autoresearch_edu.llm_explain import LLMExplainer
    explainer = LLMExplainer()
    result = explainer.explain(description, level)
    click.echo(result)


def _run_interactive_quiz(curr, gen):
    """Run quiz in interactive mode with scoring."""
    total_correct = 0
    total_asked = 0

    for lesson in curr.lessons:
        q = gen.generate_quiz(lesson)
        click.echo(f"\nQuiz: {lesson.title}")
        click.echo("-" * 40)
        for i, question in enumerate(q.questions, 1):
            click.echo(f"\n  Q{i}. {question.text}")
            if question.question_type == "multiple_choice":
                for j, opt in enumerate(question.options):
                    letter = chr(65 + j)
                    click.echo(f"      {letter}) {opt}")
                answer = click.prompt("  Your answer")
                total_asked += 1
                answer_idx = ord(answer.upper()) - 65
                if 0 <= answer_idx < len(question.options):
                    chosen = question.options[answer_idx]
                    if chosen == question.correct_answer:
                        click.echo("  Correct!")
                        total_correct += 1
                    else:
                        click.echo(f"  Incorrect. The answer is: {question.correct_answer}")
                else:
                    click.echo(f"  Invalid choice. The answer is: {question.correct_answer}")
            else:
                answer = click.prompt("  Your answer")
                click.echo(f"  Reference answer: {question.correct_answer}")
                total_asked += 1

    if total_asked > 0:
        pct = int(total_correct / total_asked * 100)
        click.echo(f"\nScore: {total_correct}/{total_asked} ({pct}%)")


def _load_yaml_concepts(lib, concepts_dir):
    """Load YAML concept files from a directory."""
    import glob
    import os
    pattern = os.path.join(concepts_dir, "*.yaml")
    for path in glob.glob(pattern):
        lib.load_from_yaml(path)
    pattern_yml = os.path.join(concepts_dir, "*.yml")
    for path in glob.glob(pattern_yml):
        lib.load_from_yaml(path)


def _load_experiments(path):
    """Load experiments from a TSV file."""
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [
            {
                "experiment_id": row["experiment_id"],
                "description": row["description"],
                "metric": float(row["metric"]),
            }
            for row in reader
        ]
