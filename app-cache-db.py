from langchain_ollama.chat_models import ChatOllama
from langchain_community.cache import SQLiteCache, InMemoryCache
from langchain.globals import set_llm_cache
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

# Opcional: ajuste do host/porta se precisar
# os.environ["OLLAMA_HOST"] = config["ollama_host"]
# os.environ["OLLAMA_PORT"] = str(config["ollama_port"])

llm = ChatOllama(
    model=config.get("model_name", "gemma3:1b"),
    temperature=config.get("temperature", 0.0),
    max_tokens=config.get("max_tokens"),
)

# --- cache em memória ------------------------------
#set_llm_cache(InMemoryCache())

# --- cache persistente -----------------------------
cache_path = "gemma3-cache.db"
cache = SQLiteCache(database_path=cache_path)
set_llm_cache(cache)

# --- chamada com retry on bad-cache ---
messages = [
    ("system", "Você é um assistente prestativo especializado em física."),
    ("user", "Explique a Lei de Ohm."),
]

start = time.perf_counter()
try:
    response = llm.invoke(messages)
except ValidationError as e:
    logging.warning(
        "Cache antigo em formato Legacy detectado: "
        "limpando cache e tentando de novo."
    )
    cache.clear()              # limpa o DB apenas uma vez
    response = llm.invoke(messages)
end = time.perf_counter()

print(response.content)
print(f"\n⏱ Tempo de resposta: {end - start:.2f} s")
