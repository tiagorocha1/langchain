from config import load_config
from cache import DiskCache
from llm_factory import create_llm
from llm_service import LLMService
from analysis.financial_analysis import execute_financial_analysis_completion
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser

def main():
    # 1. Carrega configurações
    settings = load_config()

    # 2. Cria o LLM e o cache
    llm = create_llm(settings)
    #cache = DiskCache(settings.cache_dir)
    #service = LLMService(llm, cache)

    # 3. Chain de execução com PromptTemplate
    #prompt_template = PromptTemplate.from_template("Descreva as tendências tecnológicas em {ano}.")
    #runnable_sequence = prompt_template | llm 
    #output = runnable_sequence.invoke({"ano": "2024"})
    #print("Output:\n", output)

    # ela inicial (classificação)
    chain = (
        PromptTemplate.from_template(
            """
            Classifique a pergunta do usuário em uma das seguintes categorias:
            - Assuntos Financeiros
            - Suporte Técnico
            - Atualização de Cadastro
            - Outras Informações

            Pergunta: {query}
            Classificação:
            """
        )
        | llm
        | StrOutputParser() 
    )    

    #print(chain)

    # elos específicos
    financial_chain = PromptTemplate.from_template(
        """
        Você é um especialista financeiro.
        Sempre responda às perguntas começando com "Bem-vindo ao Suporte Financeiro".
        Responda à pergunta do usuário:
        Pergunta: {query}
        Resposta:
        """
    ) | llm
    tech_support_chain = PromptTemplate.from_template(
        """
        Você é um especialista em suporte técnico.
        Sempre responda às perguntas começando com "Bem-vindo ao Suporte Técnico".
        Ajude o usuário com seu problema técnico.
        Pergunta: {query}
        Resposta:
        """
    ) | llm
    update_registration_chain = PromptTemplate.from_template(
        """
        Você é um representante de atendimento ao cliente.
        Sempre responda às perguntas começando com "Bem-vindo ao Suporte de Cadastro".
        Guie o usuário na atualização de suas informações de cadastro.
        Pergunta: {query}
        Resposta:
        """
    ) | llm
    other_info_chain = PromptTemplate.from_template(
        """
        Você é um assistente de informações gerais.
        Sempre responda às perguntas começando com "Bem-vindo ao Suporte Geral".
        Forneça informações ao usuário sobre sua pergunta.
        Pergunta: {query}
        Resposta:
        """
    ) | llm


    # Função de roteamento
    def route(info):
        topic = info["topic"].lower()
        if "financeiro" in topic:
            return financial_chain
        elif "técnico" in topic:
            return tech_support_chain
        elif "atualização" in topic or "cadastro" in topic:
            return update_registration_chain
        else:
            return other_info_chain

    # Exemplos 1 suporte técnico
    classification = chain.invoke({"query": "Como faço para redefinir minha senha?"})
    print(classification)

    #chama a função rote, passando o topico
    response_chain = route({"topic": classification})
    #print(response_chain)

    print("------------")
    # Exemplo 2 (Assuntos Financeiros)
    classification = chain.invoke({"query": "Como posso pagar uma fatura atrasada?"})
    response_chain = route({"topic": classification})
    response = response_chain.invoke({"query": "Como posso pagar uma fatura atrasada?"})
    print(response.content)

    print("------------")
    # Exemplo 3 (Atualização de Cadastro)
    classification = chain.invoke({"query": "Preciso alterar meu endereço de e-mail."})
    response_chain = route({"topic": classification})
    response = response_chain.invoke({"query": "Preciso alterar meu endereço de e-mail."})
    print(response.content)

    print("------------")
    # Exemplo 4 (Outras Informações)
    classification = chain.invoke({"query": "Qual é a missão da empresa?"})
    response_chain = route({"topic": classification})
    response = response_chain.invoke({"query": "Qual é a missão da empresa?"})
    print(response.content)


        
if __name__ == "__main__":
    main()
