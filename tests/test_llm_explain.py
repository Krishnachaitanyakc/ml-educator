"""Tests for LLM explanations module."""
import pytest
from unittest.mock import patch, MagicMock

from autoresearch_edu.llm_explain import LLMExplainer


class TestLLMExplainer:
    @patch("autoresearch_edu.llm_explain.anthropic")
    def test_explain_calls_api(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test explanation")]
        mock_client.messages.create.return_value = mock_response

        explainer = LLMExplainer(api_key="test-key")
        result = explainer.explain("increased learning rate", "beginner")

        assert result == "Test explanation"
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["max_tokens"] == 1024

    @patch("autoresearch_edu.llm_explain.anthropic")
    def test_explain_includes_level_in_prompt(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Advanced explanation")]
        mock_client.messages.create.return_value = mock_response

        explainer = LLMExplainer(api_key="test-key")
        explainer.explain("test experiment", "advanced")

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]
        assert "advanced" in prompt

    @patch("autoresearch_edu.llm_explain.anthropic")
    def test_explain_includes_concepts_context(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Explanation")]
        mock_client.messages.create.return_value = mock_response

        explainer = LLMExplainer(api_key="test-key")
        explainer.explain("test", "beginner")

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]
        assert "learning_rate" in prompt
        assert "batch_size" in prompt
