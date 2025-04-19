from config import load_config
from cache import DiskCache
from llm_factory import create_llm
from llm_service import LLMService
from analysis.financial_analysis import execute_financial_analysis_completion
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

def main():
    
    settings = load_config()
   
    llm = create_llm(settings)
    cache = DiskCache(settings.cache_dir)
    service = LLMService(llm, cache)
    
    # Executa Código Python
    python_repl = PythonREPL()
    result = python_repl.run("print(5 * 5)")
    print(result)

    # Executa Busca no Web
    ddg_search = DuckDuckGoSearchRun()
    query = "Qual é a capital da Guiana?"
    search_results = ddg_search.run(query)
    print("Resultados da busca:", search_results)

    # Consulta Wikipedia
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    query = "Guiana"
    wikipedia_results = wikipedia.run(query)
    print("Resultados da Wikipedia:", wikipedia_results)



if __name__ == "__main__":
        main()
