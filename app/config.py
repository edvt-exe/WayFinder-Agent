"""
Centralized settings. Reads from environment variables / .env so the
Anthropic API key never lives in source control.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-5"
    max_tokens: int = 8000
    request_timeout_seconds: int = 60

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

settings = Settings()
