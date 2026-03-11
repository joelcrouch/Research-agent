




    
    
from functools import lru_cache
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required for cloud LLMs — not needed if use_local_llm=true
    anthropic_api_key: str = Field(default="")
    langsmith_api_key: str = Field(default="")

    # Optional
    log_level: str = Field("INFO")
    default_top_k: int = Field(10, ge=1, le=50)
    default_date_range_years: int = Field(5, ge=1, le=30)
    langsmith_project: str = Field("research-agent")

    # Local LLM via Ollama
    use_local_llm: bool = Field(False)
    local_llm_model: str = Field("llama3:latest")
    local_llm_base_url: str = Field("http://localhost:11434")

    @model_validator(mode="after")
    def validate_keys(self) -> "Settings":
        if not self.use_local_llm:
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set. Check your .env file or set USE_LOCAL_LLM=true.")
            if not self.langsmith_api_key:
                raise ValueError("LANGSMITH_API_KEY is not set. Check your .env file or set USE_LOCAL_LLM=true.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings