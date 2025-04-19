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

from langchain_experimental.agents.agent_toolkits import create_python_agent

def main():
    
    settings = load_config()
   
    llm = create_llm(settings)
    cache = DiskCache(settings.cache_dir)
    service = LLMService(llm, cache)
    
    ddg_search = DuckDuckGoSearchRun()

    agent_executor = create_python_agent(
        llm=llm,
        tool=ddg_search,  
        verbose=True
    )

    prompt_template = PromptTemplate(
        input_variables=["query"],
        template="""
        Pesquise na web sobre {query} e forneça um resumo abrangente sobre o assunto no idioma português.
        """
    )

    query = "Carl Sagan"
    prompt = prompt_template.format(query=query)

    print(prompt) 

    response = agent_executor.invoke(prompt)

    print("Entrada do agente:", response['input'])

    print("Saída do agente:", response['output'])

if __name__ == "__main__":
        main()
