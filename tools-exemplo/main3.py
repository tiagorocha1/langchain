from config import load_config
from cache import DiskCache
from llm_factory import create_llm
from llm_service import LLMService
from analysis.financial_analysis import execute_financial_analysis_completion
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_experimental.agents.agent_toolkits import create_python_agent

from langchain import hub
from langchain.agents import Tool, AgentExecutor, initialize_agent, create_react_agent

def validate_python_code(code):
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError as e:
        print(f"Erro de sintaxe no código gerado: {e}")
        return False

def main():
    
    settings = load_config()
   
    llm = create_llm(settings)
    cache = DiskCache(settings.cache_dir)
    service = LLMService(llm, cache)

    prompt = '''
    Como assistente financeiro pessoal, ajude a responder as seguintes perguntas com ajuda da internet.
    Responda em português e use a ferramenta correta para cada pergunta.
    Perguntas: {q}
    '''
    prompt_template = PromptTemplate.from_template(prompt)

    react_instructions  = hub.pull('hwchase17/react')

    # Ferramenta 1 Python REPL
    python_repl = PythonREPLTool()
    python_repl_tool = Tool(
        name='Python REPL',
        func=python_repl.run,
        description='''Qualquer tipo de cálculo deve usar esta ferramenta. Você não deve realizar
                        o cálculo diretamente. Você deve inserir código Python.'''
    )

    # Ferramenta  2 busca DuckDuckGo 
    search = DuckDuckGoSearchRun()
    duckduckgo_tool = Tool(
        name='Busca DuckDuckGo',
        func=search.run,
        description='''Útil para encontrar informações e dicas de economia e opções de investimento.
                    Você sempre deve pesquisar na internet as melhores dicas usando esta ferramenta, não
                    responda diretamente. Sua resposta deve informar que há elementos pesquisados na internet'''
    )
    
    tools = [python_repl_tool, duckduckgo_tool]
    agent = create_react_agent(llm, tools, react_instructions)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )

    question = """
    Eu ganho R$4000 por mês mas o total de minhas despesas é de R$3800 mais 500 de aluguel.
    Como posso ajustar meu orçamento para economizar dinheiro?
    """
    output = agent_executor.invoke({
        'input': prompt_template.format(q=question)
    })

    print(output['input'])

    print(output['output'])

if __name__ == "__main__":
        main()
