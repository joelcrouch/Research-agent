from langchain_core.language_models import BaseChatModel
from config.settings import settings

def get_llm() -> BaseChatModel:
    s=settings()
    if s.use_local_llm:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=s.local_llm_model, 
            base_url=s.local_llm_base_url
        )
    else: 
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=s.anthropic_api_key,
        )

