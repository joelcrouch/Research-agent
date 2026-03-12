from functools import lru_cache
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env file explicitly
load_dotenv()

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
    local_llm_model: str = Field("qwen2.5:1.5b")
    local_llm_base_url: str = Field("http://localhost:11434")

    @model_validator(mode="after")
    def validate_keys(self) -> "Settings":
        if not self.use_local_llm:
            if not self.anthropic_api_key:
                pass
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings
