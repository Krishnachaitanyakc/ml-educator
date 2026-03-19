"""LLM-powered explanations using Anthropic Claude."""
import os

import anthropic

from autoresearch_edu.concepts import ConceptLibrary


class LLMExplainer:
    """Generate tailored explanations using Claude."""

    def __init__(self, api_key: str | None = None):
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._library = ConceptLibrary()

    def explain(self, experiment_description: str, level: str = "beginner") -> str:
        """Get a tailored explanation of an experiment using Claude."""
        concept_context = self._build_context(level)
        prompt = (
            f"You are an ML educator. The user is at the '{level}' level.\n\n"
            f"Available concept knowledge:\n{concept_context}\n\n"
            f"Explain this experiment in a way appropriate for the user's level:\n"
            f"{experiment_description}\n\n"
            f"Focus on: what the experiment does, why it might work or fail, "
            f"and what concepts are relevant. Keep it concise (2-3 paragraphs)."
        )
        message = self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _build_context(self, level: str) -> str:
        parts = []
        for name in self._library.list_concepts():
            concept = self._library.get(name)
            explanation = concept.get_explanation(level)
            parts.append(f"- {name}: {explanation}")
        return "\n".join(parts)
