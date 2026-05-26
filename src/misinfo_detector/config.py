from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="misinformation-checker", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    model_dir: Path = Field(default=Path("artifacts/checkpoints"), alias="MODEL_DIR")
    config_path: Path = Field(default=Path("configs/train_config.yaml"), alias="CONFIG_PATH")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_training_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else get_settings().config_path
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return config or {}
