import logging
from pathlib import Path
from typing import Optional

import torch
import wandb
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    data_dir: str = "data/coco"
    ann_file: str = "data/coco/annotations/instances_val2017.json"
    img_dir: str = "data/coco/val2017"
    # Relative to cwd (run from project/), or absolute: JSON from scripts/build_eval_split.py
    eval_split_path: Optional[str] = None
    api_key: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_level: int = logging.INFO

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("eval_split_path", mode="before")
    @classmethod
    def empty_eval_split_as_none(cls, v):
        if v is None or v == "":
            return None
        return v

    def resolved_eval_split_path(self) -> Path | None:
        if not self.eval_split_path:
            return None
        p = Path(self.eval_split_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p.resolve()

settings = Settings()

logging.basicConfig(level=settings.log_level)
logger.info(f'Using device: {settings.device}')
logger.info(f"Using dataset: {settings.data_dir}")

if settings.api_key:
    wandb.login(key=settings.api_key)
    logger.info("Logging on Weights & Biases")