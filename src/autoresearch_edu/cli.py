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
def concepts(name, level):
    """Browse the ML concept library."""
    lib = ConceptLibrary()
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
def quiz(results):
    """Generate a quiz from experiment history."""
    experiments = _load_experiments(results)
    builder = CurriculumBuilder()
    curr = builder.build_curriculum(experiments)
    gen = QuizGenerator()

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
