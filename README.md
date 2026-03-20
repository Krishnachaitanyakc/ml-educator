# ml-educator

**Learn ML by doing experiments.**

ml-educator turns your experiment history into a personalized learning experience. It annotates each experiment with relevant ML concepts, generates educational commentary at your skill level, builds structured curricula, and quizzes you on what you have learned -- all driven by real experiments you have run.

## Quick Start

```bash
pip install -e .

# Browse the built-in concept library
ml-educator concepts

# Get a beginner-friendly annotation of an experiment
ml-educator annotate --description "increased learning rate" --metric-change 0.03
```

## Features

- **Concept library** -- curated ML concepts with definitions, intuitions, and common mistakes
- **Three difficulty levels** -- beginner (intuitive, no math), intermediate, and advanced (full mathematical treatment)
- **Experiment annotation** -- automatically links experiments to relevant ML concepts
- **Educational commentary** -- generates Markdown explanations of why experiments succeeded or failed
- **Curriculum builder** -- organizes experiment history into structured lessons
- **Interactive quizzes** -- multiple-choice and open-ended questions generated from your experiments
- **Spaced repetition** -- review due concepts on an optimal schedule
- **Knowledge assessment** -- take a diagnostic quiz to find your recommended difficulty level
- **Diff analysis** -- detect hyperparameter changes in git diffs and explain related concepts
- **LLM explanations** -- use Claude for deeper, context-aware explanations
- **Concept visualizations** -- generate diagrams for key ML concepts
- **YAML concept packs** -- extend the library with custom concept files

## Usage

### CLI

```bash
# Explain a concept at a specific level
ml-educator concepts --name learning_rate --level intermediate

# Generate commentary for a full experiment run
ml-educator commentary --results results.tsv --level beginner

# Build a curriculum from experiment history
ml-educator curriculum --results results.tsv

# Take an interactive quiz
ml-educator quiz --results results.tsv --interactive

# Review concepts via spaced repetition
ml-educator review

# Assess your ML knowledge level
ml-educator assess

# Analyze hyperparameter changes in a diff
ml-educator diff-analyze --diff-file changes.diff --level beginner

# Visualize a concept
ml-educator visualize --concept learning_rate --output lr.png
```

### Python API

```python
from ml_educator.concepts import ConceptLibrary
from ml_educator.annotator import ExperimentAnnotator
from ml_educator.curriculum import CurriculumBuilder

library = ConceptLibrary()
concept = library.get("learning_rate")
print(concept.get_explanation("beginner"))

annotator = ExperimentAnnotator()
annotation = annotator.annotate_experiment("increased learning rate", 0.03)
print(annotation.concept_references)

builder = CurriculumBuilder()
curriculum = builder.build_curriculum(experiments)
for lesson in curriculum.lessons:
    print(f"{lesson.title} -- {', '.join(lesson.concept_names)}")
```

## License

MIT
