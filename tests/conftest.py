"""
tests/conftest.py

Shared pytest fixtures available to all test modules.
"""

from unittest.mock import patch
import pytest
import os
from config.settings import Settings

@pytest.fixture(autouse=True)
def mock_settings():
    """
    Force a specific configuration for all tests.
    """
    # Overwrite environment variables to ensure Pydantic picks up the right values
    os.environ["USE_LOCAL_LLM"] = "true"
    os.environ["LOCAL_LLM_MODEL"] = "qwen2.5:1.5b"
    
    test_settings = Settings(
        anthropic_api_key="test-anthropic-key",
        langsmith_api_key="test-langsmith-key",
        log_level="DEBUG",
        langsmith_project="research-agent-test",
        use_local_llm=True,
        local_llm_model="qwen2.5:1.5b"
    )

    with patch("config.settings.get_settings", return_value=test_settings):
        yield test_settings
