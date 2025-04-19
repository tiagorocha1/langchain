import os
import json
import hashlib
import logging
from typing import Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class CacheStrategy(ABC):
    """Interface para estratégias de cache."""

    @abstractmethod
    def lookup(self, key: str, llm_string: str) -> Optional[str]:
        """Busca um valor em cache pelo identificador 'key'."""
        ...

    @abstractmethod
    def update(self, key: str, value: str, llm_string: str) -> None:
        """Atualiza o cache armazenando 'value' sob o identificador 'key'."""
        ...

class DiskCache(CacheStrategy):
    """Cache simples em disco usando arquivos JSON."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def _get_path(self, key: str) -> str:
        hash_name = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.directory, f"{hash_name}.json")

    def lookup(self, key: str, llm_string: str = "") -> Optional[str]:
        path = self._get_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("response")
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Erro ao ler cache %s: %s", path, e)
            return None

    def update(self, key: str, value: str, llm_string: str = "") -> None:
        path = self._get_path(key)
        temp_path = f"{path}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump({"response": value}, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, path)
        except IOError as e:
            logger.error("Erro ao gravar cache %s: %s", temp_path, e)
