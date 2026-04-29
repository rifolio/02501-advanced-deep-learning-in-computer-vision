import logging
from pathlib import Path
from typing import Optional

import torch
import wandb
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    data_dir: str = "data/coco"
    ann_file: str = "data/coco/annotations/instances_val2017.json"
    img_dir: str = "data/coco/val2017"
    # Relative to cwd (run from project/), or absolute: JSON from scripts/build_eval_split.py
    eval_split_path: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("EVAL_SPLIT_PATH", "eval_split_path"),
    )
    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("API_KEY", "api_key"),
    )
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    #log_level: int = logging.INFO
    #####
    log_level: int | str = Field(
        default=logging.INFO,
        validation_alias=AliasChoices("LOG_LEVEL", "log_level"),
    )
    
    # 2. Add the VLM full text flag as a boolean
    vlm_log_full_text: bool = Field(
        default=False,
        validation_alias=AliasChoices("VLM_LOG_FULL_TEXT", "vlm_log_full_text"),)
    ####
    experiment_mode: str = Field(
        default="zero_shot",
        validation_alias=AliasChoices("EXPERIMENT_MODE", "experiment_mode"),
    )
    model_name: str = Field(
        default="qwen",
        validation_alias=AliasChoices("MODEL_NAME", "model_name"),
    )
    k_shot: int = Field(
        default=1,
        validation_alias=AliasChoices("K_SHOT", "k_shot"),
    )
    few_shot_seed: int = Field(
        default=42,
        validation_alias=AliasChoices("FEW_SHOT_SEED", "few_shot_seed"),
    )
    prompt_strategy: str = Field(
        default="side_by_side",
        validation_alias=AliasChoices("PROMPT_STRATEGY", "prompt_strategy"),
    )
    log_viz_artifact: bool = Field(
        default=True,
        validation_alias=AliasChoices("LOG_VIZ_ARTIFACT", "log_viz_artifact"),
    )
    viz_max_images: int = Field(
        default=50,
        validation_alias=AliasChoices("VIZ_MAX_IMAGES", "viz_max_images"),
    )
    viz_preview_count: int = Field(
        default=12,
        validation_alias=AliasChoices("VIZ_PREVIEW_COUNT", "viz_preview_count"),
    )
    viz_target_category: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("VIZ_TARGET_CATEGORY", "viz_target_category"),
    )
    viz_output_dir: str = Field(
        default="viz",
        validation_alias=AliasChoices("VIZ_OUTPUT_DIR", "viz_output_dir"),
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    @field_validator("eval_split_path", mode="before")
    @classmethod
    def empty_eval_split_as_none(cls, v):
        if v is None or v == "":
            return None
        return v
    
    ########
    #validator to convert "DEBUG" from your .env into the logging module's integer
    @field_validator("log_level", mode="before")
    @classmethod
    def parse_log_level(cls, v):
        if isinstance(v, str) and not v.isdigit():
            return getattr(logging, v.upper(), logging.INFO)
        return int(v)
    ########

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
logger.info(
    (
        "Runtime selection: model_name=%s experiment_mode=%s k_shot=%s prompt_strategy=%s "
        "log_viz_artifact=%s viz_max_images=%s viz_target_category=%s"
    ),
    settings.model_name,
    settings.experiment_mode,
    settings.k_shot,
    settings.prompt_strategy,
    settings.log_viz_artifact,
    settings.viz_max_images,
    settings.viz_target_category,
)
if settings.eval_split_path:
    logger.info("Runtime selection: eval_split_path=%s", settings.eval_split_path)

if settings.api_key:
    wandb.login(key=settings.api_key)
    logger.info("Logging on Weights & Biases")