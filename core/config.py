"""Application configuration via Pydantic Settings."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _HAS_PYDANTIC_SETTINGS = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC_SETTINGS = False


if _HAS_PYDANTIC_SETTINGS:

    class Settings(BaseSettings):
        """Environment-driven settings.

        Override via environment variables or a .env file.
        """
        model_config = SettingsConfigDict(env_prefix="VOLMCP_", env_file=".env", extra="ignore")

        # Data providers
        default_provider: str = "yfinance"
        alpha_vantage_api_key: str | None = None
        polygon_api_key: str | None = None
        kite_api_key: str | None = None
        kite_access_token: str | None = None

        # Cache
        price_cache_ttl_seconds: int = 3600
        fit_cache_ttl_seconds: int = 3600

        # Model fitting defaults
        default_distribution: str = "t"
        default_maxiter: int = 1000

        # Reporting
        report_output_dir: str = "/tmp/volmcp_reports"

        # Storage
        session_backend: str = "memory"  # "memory" | "sqlite" | "duckdb"
        session_db_path: str = "/tmp/volmcp_sessions.db"

    @lru_cache
    def get_settings() -> Settings:
        return Settings()

else:  # Fallback when pydantic_settings not installed

    class Settings:  # type: ignore[no-redef]
        def __init__(self) -> None:
            import os
            self.default_provider = os.getenv("VOLMCP_DEFAULT_PROVIDER", "yfinance")
            self.alpha_vantage_api_key = os.getenv("VOLMCP_ALPHA_VANTAGE_API_KEY")
            self.polygon_api_key = os.getenv("VOLMCP_POLYGON_API_KEY")
            self.kite_api_key = os.getenv("VOLMCP_KITE_API_KEY")
            self.kite_access_token = os.getenv("VOLMCP_KITE_ACCESS_TOKEN")
            self.price_cache_ttl_seconds = int(os.getenv("VOLMCP_PRICE_CACHE_TTL", "3600"))
            self.fit_cache_ttl_seconds = int(os.getenv("VOLMCP_FIT_CACHE_TTL", "3600"))
            self.default_distribution = os.getenv("VOLMCP_DEFAULT_DISTRIBUTION", "t")
            self.default_maxiter = int(os.getenv("VOLMCP_DEFAULT_MAXITER", "1000"))
            self.report_output_dir = os.getenv("VOLMCP_REPORT_DIR", "/tmp/volmcp_reports")
            self.session_backend = os.getenv("VOLMCP_SESSION_BACKEND", "memory")
            self.session_db_path = os.getenv("VOLMCP_SESSION_DB", "/tmp/volmcp_sessions.db")

    @lru_cache
    def get_settings() -> Settings:
        return Settings()
