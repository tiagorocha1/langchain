from langchain_ollama.chat_models import ChatOllama  # :contentReference[oaicite:0]{index=0}
import os
import yaml

# Carrega configurações
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Se necessário, aponta para o servidor Ollama local
os.environ["OLLAMA_HOST"] = config.get("ollama_host", "127.0.0.1")
os.environ["OLLAMA_PORT"] = str(config.get("ollama_port", 11434))

# Instancia o modelo DeepSeek rodando no Ollama
llm = ChatOllama(
    model=config.get("model_name", "deepseek-r1"),
    temperature=config.get("temperature", 0),
    max_tokens=config.get("max_tokens", None),
)

# Exemplo de chamada
messages = [
    ("system", "Você é um assistente útil."),
    ("user", "Explique a Lei de Ohm."),
]
response = llm.invoke(messages)
print(response.content)


