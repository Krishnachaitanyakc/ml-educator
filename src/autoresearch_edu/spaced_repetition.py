"""Spaced repetition using SM-2 algorithm."""
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


@dataclass
class ConceptState:
    """Spaced repetition state for a single concept."""
    ease_factor: float = 2.5
    interval: int = 1  # days
    repetitions: int = 0
    next_review: datetime = field(default_factory=datetime.now)


class SpacedRepetitionScheduler:
    """SM-2 based spaced repetition scheduler."""

    def __init__(self, state_path: str = ".spaced_reps.pkl"):
        self._state_path = Path(state_path)
        self._states: Dict[str, ConceptState] = {}
        self._load()

    def _load(self):
        if self._state_path.exists():
            with open(self._state_path, "rb") as f:
                self._states = pickle.load(f)

    def save(self):
        with open(self._state_path, "wb") as f:
            pickle.dump(self._states, f)

    def get_state(self, concept_name: str) -> ConceptState:
        if concept_name not in self._states:
            self._states[concept_name] = ConceptState()
        return self._states[concept_name]

    def get_due_concepts(self, concept_names: List[str], now: datetime | None = None) -> List[str]:
        """Return concepts that are due for review."""
        if now is None:
            now = datetime.now()
        due = []
        for name in concept_names:
            if name not in self._states:
                # Never reviewed = always due
                due.append(name)
            else:
                state = self._states[name]
                if now >= state.next_review:
                    due.append(name)
        return due

    def record_review(self, concept_name: str, quality: int, now: datetime | None = None):
        """Record a review with quality 0-5 (SM-2 scale).

        0-1: complete blackout / wrong
        2: wrong but upon seeing correct answer it felt familiar
        3: correct with serious difficulty
        4: correct after hesitation
        5: perfect response
        """
        if quality < 0 or quality > 5:
            raise ValueError("Quality must be between 0 and 5")
        if now is None:
            now = datetime.now()

        state = self.get_state(concept_name)

        if quality < 3:
            # Reset on failure
            state.repetitions = 0
            state.interval = 1
        else:
            if state.repetitions == 0:
                state.interval = 1
            elif state.repetitions == 1:
                state.interval = 6
            else:
                state.interval = round(state.interval * state.ease_factor)
            state.repetitions += 1

        # Update ease factor
        state.ease_factor = max(
            1.3,
            state.ease_factor + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
        )
        state.next_review = now + timedelta(days=state.interval)
        self.save()
