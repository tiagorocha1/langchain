import time
from typing import Protocol, Union
from langchain_core.prompt_values import ChatPromptValue

class LLMServiceProtocol(Protocol):
    def get_response(self, prompt: str) -> str: ...

def run_prompt(service: LLMServiceProtocol, prompt: Union[str, ChatPromptValue]) -> None:
    """
    Executa o prompt duas vezes (API e cache), exibe resposta e tempo.
    Suporta prompts do tipo string ou ChatPromptValue.
    """
    # Converte ChatPromptValue para string, se necessário
    if isinstance(prompt, ChatPromptValue):
        prompt = prompt.to_string()

    for label in ("API chamada", "cache"):
        start = time.perf_counter()
        resp = service.get_response(prompt)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{label}: {resp.replace(chr(10), ' ')}")
        print(f"⏱ Tempo: {elapsed:.2f} ms\n")
