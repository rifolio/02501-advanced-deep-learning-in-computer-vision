import logging
import torch
import wandb
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    data_dir: str = 'data/coco'
    ann_file: str = 'data/coco/annotations/instances_val2017.json'
    img_dir: str = 'data/coco/val2017'
    api_key: str
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_level: int = logging.INFO

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

settings = Settings()

logging.basicConfig(level=settings.log_level)
logger.info(f'Using device: {settings.device}')
logger.info(f"Using dataset: {settings.data_dir}")

if settings.api_key:
    wandb.login(key=settings.api_key)
    logger.info("Logging on Weights & Biases")