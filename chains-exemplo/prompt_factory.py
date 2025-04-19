from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate




def analise_financeira(
    empresa: str,
    periodo: str,
    idioma: str,
    analises: list[str],
) -> str:
    """
    Monta um relatório financeiro segundo um template padrão.
    """
    template = """
        Você é um analista financeiro.
        Escreva um relatório financeiro detalhado para a empresa "{empresa}" para o período {periodo}.

        O relatório deve ser escrito em {idioma} e incluir as seguintes análises:
        {analises}

        Certifique-se de fornecer insights e conclusões para cada seção.
    """
    prompt_template = PromptTemplate.from_template(template=template)
    analises_str = "\n".join(analises)
    return prompt_template.format(
        empresa=empresa,
        periodo=periodo,
        idioma=idioma,
        analises=analises_str,
    )

def create_chat_analise_financeira(
    empresa: str,
    periodo: str,
    idioma: str,
    analises: list[str],
) -> ChatPromptTemplate:
    """
    Cria um template de chat para gerar um relatório financeiro detalhado.
    """
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="Você é um analista financeiro especializado em relatórios empresariais."),
            HumanMessagePromptTemplate.from_template(
                'Por favor, gere um relatório financeiro detalhado para a empresa "{empresa}" no período {periodo}, escrito em {idioma}.'
            ),
            AIMessage(content="Claro, vou começar analisando os dados disponíveis e estruturando o relatório."),
            HumanMessage(content=f"Certifique-se de incluir as seguintes análises:\n{', '.join(analises)}"),
            AIMessage(content="Entendido. Aqui está o relatório completo:")
        ]
    )
    return chat_template.format_prompt(
        empresa=empresa,
        periodo=periodo,
        idioma=idioma,
    )
