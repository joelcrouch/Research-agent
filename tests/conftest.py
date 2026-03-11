"""
tests/conftest.py

Shared pytest fixtures available to all test modules.
"""

from unittest.mock import MagicMock, patch

import pytest

from config.settings import Settings


@pytest.fixture(autouse=True)
def mock_settings():
    """Patch get_settings() so no real API keys are needed in tests."""
    mock = MagicMock(spec=Settings)
    mock.anthropic_api_key = "test-anthropic-key"
    mock.langsmith_api_key = "test-langsmith-key"
    mock.log_level = "DEBUG"
    mock.default_top_k = 10
    mock.default_date_range_years = 5
    mock.langsmith_project = "research-agent-test"

    with patch("config.settings.get_settings", return_value=mock):
        yield mock
