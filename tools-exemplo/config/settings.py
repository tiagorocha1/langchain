import yaml
import logging
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Configurações da aplicação, carregadas de arquivo YAML ou .env.
    Campos extras serão simplesmente ignorados.
    """
    model_name: str = "gemma3:1b"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    cache_dir: str = "cache_dir"

    # ignora qualquer variável adicional em config.yaml/.env
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

def load_config(path: str = "config.yaml") -> Settings:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("Arquivo de config não encontrado: %s. Usando defaults.", path)
        data = {}
    return Settings(**data)
