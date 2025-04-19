# llm_service/llm_service.py

import logging
from typing import Optional

from langchain_ollama.chat_models import ChatOllama
from langchain.globals import set_llm_cache

from cache.disk_cache import CacheStrategy

logger = logging.getLogger(__name__)


class LLMService:
    """
    Serviço para invocação de LLM com cache.
    """

    def __init__(self, llm: ChatOllama, cache: CacheStrategy):
        """
        :param llm: instância de ChatOllama ou compatível
        :param cache: estratégia de cache que implementa CacheStrategy
        """
        self.llm = llm
        self.cache = cache
        # registra o cache na biblioteca do langchain
        set_llm_cache(cache)

    def get_response(self, prompt: str) -> str:
        """
        Retorna a resposta do LLM, buscando no cache antes de invocar a API.

        :param prompt: Texto do prompt a ser enviado ao modelo.
        :return: Resposta gerada pelo modelo.
        """
        # Tenta obter do cache
        cached = self.cache.lookup(prompt)
        if cached:
            logger.info("Cache hit para prompt: %s", prompt)
            return cached

        # Caso não exista, invoca a API
        message = self.llm.invoke(prompt)
        content = getattr(message, "content", str(message))

        # Atualiza o cache e retorna
        self.cache.update(prompt, content)
        return content
