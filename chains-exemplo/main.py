from config import load_config
from cache import DiskCache
from llm_factory import create_llm
from llm_service import LLMService
from analysis.financial_analysis import execute_financial_analysis_completion
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader

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


    # Carregar os documentos
    loader = TextLoader('base_conhecimento_britadeira.txt')
    documents = loader.load()
    # Carregar histórico de conversas 
    historico_conversas = """Cliente: Minha britadeira não liga. Chatbot: Você já verificou 
                            se a bateria está carregada e conectada corretamente?"""
    # Pergunta do cliente
    pergunta = "Minha britadeira não liga. Eu já veriquei e a bateria está carregada e conectada corretamente"

    inputs = {
        "context": "\n".join(doc.page_content for doc in documents),
        "question": pergunta,
        "historico": historico_conversas
    }

    # 1) Prompt para resposta a partir de uma base de conhecimento
    prompt_base_conhecimento = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    Você é um assistente especializado. Use apenas o contexto fornecido para responder.
    Não introduza informações externas, nem repita procedimentos já realizados.
    Contexto:
    {context}

    Pergunta:
    {question}

    Resposta:"""
    )

    # 2) Prompt para resposta a partir do histórico de conversas
    prompt_historico_conversas = PromptTemplate(
        input_variables=["historico", "question"],
        template="""
    Você é um assistente que considera apenas o histórico abaixo.
    Não insira dados novos nem repita procedimentos já executados.
    Histórico de conversas:
    {historico}

    Pergunta atual:
    {question}

    Resposta:"""
    )

    # 3) Prompt para combinação de respostas
    prompt_final = PromptTemplate(
        input_variables=["resposta_base_conhecimento", "resposta_historico_conversas"],
        template="""
    Você recebeu duas respostas:
    1) Da base de conhecimento:
    {resposta_base_conhecimento}

    2) Do histórico de conversas:
    {resposta_historico_conversas}

    Combine-as de forma coesa, extraia o essencial de cada uma e gere uma única resposta final.
    Evite repetir instruções de processos já concluídos.

    Resposta combinada:"""
    )

    # Definir as cadeias  
    chain_base_conhecimento = prompt_base_conhecimento | llm
    chain_historico_conversas = prompt_historico_conversas | llm
    chain_final = prompt_final | llm

    # Passando dados e executando
    resultado_base_conhecimento = chain_base_conhecimento.invoke({"context": inputs["context"], "question": inputs["question"]})
    resultado_historico_conversas = chain_historico_conversas.invoke({"historico": inputs["historico"], "question": inputs["question"]})
    resultado_final = chain_final.invoke({"resposta_base_conhecimento": resultado_base_conhecimento, 
                                        "resposta_historico_conversas": resultado_historico_conversas})

    #print("Resultado Base de Conhecimento:\n", resultado_base_conhecimento)
    #print("----")
    #print("Resultado Histórico de Conversas:\n", resultado_historico_conversas)  

    print(resultado_final.content)       
    #print(resultado_final)                    


if __name__ == "__main__":
    main()
