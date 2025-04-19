from langchain_ollama.chat_models import ChatOllama

#Cache para LLMs
from langchain_community.cache import SQLiteCache, InMemoryCache
from langchain.globals import set_llm_cache

# P
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from pydantic import ValidationError
import hashlib
import os
import yaml
import time
import logging
import json

# --- configurações e LLM -------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


llm = ChatOllama(
    model=config.get("model_name", "gemma3:1b"),
    temperature=config.get("temperature", 0.0),
    max_tokens=config.get("max_tokens"),
)

# --- cache personalizado -----------------------------
class SimpleDiskCache:
    def __init__(self, cache_dir: str = "cache_dir"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, key: str) -> str:
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.json")

    def lookup(self, key: str, llm_string: str):
        cache_path = self._get_cache_path(key)
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("response")
        except (json.JSONDecodeError, IOError):
            # arquivo vazio ou corrompido: ignora
            return None

    def update(self, key: str, value: str, llm_string: str):
        cache_path = self._get_cache_path(key)
        tmp_path = cache_path + ".tmp"
        # escreve em .tmp e depois renomeia
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"response": value}, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, cache_path)


# registra nosso cache customizado
cache = SimpleDiskCache()
set_llm_cache(cache)

def invoke_with_cache(llm, prompt: str, cache: SimpleDiskCache) -> str:
    cached = cache.lookup(prompt, "")
    if cached:
        print("Usando cache:")
        return cached

    # faz a chamada ao modelo e extrai só o texto
    message = llm.invoke(prompt)
    response_text = message.content if hasattr(message, "content") else str(message)
    cache.update(prompt, response_text, "")
    return response_text


# --- exemplo de uso -----------------------------
prompt = 'Me diga em poucas palavras quem foi Neil Degrasse Tyson.'

# primeira chamada (API)
start = time.perf_counter()
response1 = invoke_with_cache(llm, prompt, cache)
elapsed_ms = (time.perf_counter() - start) * 1000
print("Primeira resposta (API chamada):", response1.replace("\n", " "))
print(f"⏱ Tempo: {elapsed_ms:.2f} ms\n")

# segunda chamada (cache)
start = time.perf_counter()
response2 = invoke_with_cache(llm, prompt, cache)
elapsed_ms = (time.perf_counter() - start) * 1000
print("Segunda resposta (usando cache):", response2.replace("\n", " "))
print(f"⏱ Tempo: {elapsed_ms:.2f} ms")