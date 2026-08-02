"""Central configuration, loaded from environment variables / .env."""
from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- core ---
    app_name: str = "Hone AI"
    environment: str = "development"

    # --- database / redis ---
    database_url: str = "postgresql+asyncpg://hone:hone@db:5432/hone"
    sync_database_url: str = "postgresql+psycopg2://hone:hone@db:5432/hone"
    redis_url: str = "redis://redis:6379/0"

    @field_validator("database_url")
    @classmethod
    def _asyncpg_scheme(cls, v: str) -> str:
        # Managed Postgres providers (Render, etc.) hand out plain
        # postgres:// / postgresql:// URLs; SQLAlchemy's async engine needs
        # the +asyncpg driver in the scheme.
        if v.startswith("postgres://"):
            return "postgresql+asyncpg://" + v[len("postgres://") :]
        if v.startswith("postgresql://"):
            return "postgresql+asyncpg://" + v[len("postgresql://") :]
        return v

    # --- auth ---
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 30
    cookie_secure: bool = False          # True behind HTTPS
    cookie_domain: str | None = None

    # --- google oauth (optional) ---
    google_client_id: str | None = None
    google_client_secret: str | None = None

    # --- llm provider ---
    # one of: "mock" | "anthropic" | "openai" | "gemini"
    llm_provider: str = "mock"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    llm_model: str = "claude-3-5-sonnet-latest"
    embed_dim: int = 1536

    # --- rate limiting (per user or IP) ---
    rate_limit_optimize: int = 20        # requests
    rate_limit_window_seconds: int = 60  # per window

    # --- cors ---
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # --- observability (optional) ---
    sentry_dsn: str | None = None

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
