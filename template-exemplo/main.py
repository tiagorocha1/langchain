from config import load_config
from cache import DiskCache
from llm_factory import create_llm
from llm_service import LLMService
from analysis.financial_analysis import execute_financial_analysis_completion
from langchain.prompts import PromptTemplate

def main():
    # 1. Carrega configurações
    settings = load_config()

    # 2. Cria o LLM e o cache
    llm = create_llm(settings)
    cache = DiskCache(settings.cache_dir)
    service = LLMService(llm, cache)

    # 3. Executa a análise financeira
    execute_financial_analysis_completion(service)


    




if __name__ == "__main__":
    main()
