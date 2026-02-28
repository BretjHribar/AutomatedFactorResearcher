"""
Global configuration using pydantic-settings.
Reads from environment variables and .env file.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Platform-wide configuration. Resolves from env vars / .env file."""

    # Data vendor API keys (optional)
    fmp_api_key: str = ""
    eodhd_api_key: str = ""

    # Database
    database_url: str = "sqlite:///local.db"

    # Data directory
    data_dir: str = "./data"

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Simulation defaults
    default_booksize: float = 20_000_000.0
    default_delay: int = 1
    default_neutralization: str = "subindustry"
    default_universe: str = "TOP3000"
    default_decay: int = 0
    default_max_instrument_weight: float = 0.0
    default_duration_years: int = 5
    trading_days_per_year: int = 252

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get (or create) the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
