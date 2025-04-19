from langchain_ollama.chat_models import ChatOllama
from config import Settings

def create_llm(settings: Settings) -> ChatOllama:
    """
    Cria e configura o cliente LLM a partir das Settings.
    """
    return ChatOllama(
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
