from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings. Prefix is MONITOR_."""

    model_config = SettingsConfigDict(env_prefix="MONITOR_", env_file=".env", extra="ignore")

    store_dir: Path = Path("var/monitor")
    psi_watch: float = 0.1
    psi_alert: float = 0.2
    accuracy_floor: float = 0.75
    error_rate_alert: float = 0.08
    window: int = 200
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
