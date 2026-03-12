from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from config.settings import settings


def get_llm() -> BaseChatModel:
    s = settings()
    if s.use_local_llm:
        from langchain_ollama import ChatOllama

        return ChatOllama(model=s.local_llm_model, base_url=s.local_llm_base_url)
    else:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(  # type: ignore[call-arg]
            model_name="claude-3-5-sonnet-20240620",
            api_key=SecretStr(s.anthropic_api_key),
        )
