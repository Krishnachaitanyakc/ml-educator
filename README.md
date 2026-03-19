# ml-educator

Educational mode where the autoresearch agent explains its ML reasoning. Provides a rich ML concept library, annotates experiments with relevant concepts, generates educational commentary, builds curricula from experiment history, and generates quizzes.

## Installation

```bash
pip install -e .
```

## Usage

### CLI

```bash
# Browse concept library
ml-educator concepts
ml-educator concepts --name learning_rate --level beginner

# Annotate a specific experiment
ml-educator annotate --description "increased learning rate" --metric-change 0.03

# Generate commentary
ml-educator commentary --results results.tsv --level beginner

# Build curriculum from experiment history
ml-educator curriculum --results results.tsv

# Generate quiz
ml-educator quiz --results results.tsv
```

### Python API

```python
from ml_educator.concepts import ConceptLibrary
from ml_educator.annotator import ExperimentAnnotator

library = ConceptLibrary()
concept = library.get("learning_rate")
print(concept.get_explanation("beginner"))

annotator = ExperimentAnnotator()
annotation = annotator.annotate_experiment("increased learning rate", 0.03)
```

## Difficulty Levels

- **beginner** -- Intuitive explanations, no math
- **intermediate** -- Some technical detail
- **advanced** -- Full mathematical treatment

## Dependencies

- click
- jinja2
