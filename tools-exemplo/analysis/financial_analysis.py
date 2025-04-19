from prompt_factory import analise_financeira, create_chat_analise_financeira
from prompt_runner import run_prompt

def execute_financial_analysis_completion(service):
    # 3. Parâmetros para o prompt
    empresa = "Facebook"
    periodo = "2025"
    idioma = "português"
    analises = [
        "Análise de receita",
        "Análise de despesas",
        "Análise de lucro",
    ]

    # 4. Cria e executa o prompt template  (Completion)
    prompt = analise_financeira(empresa, periodo, idioma, analises)
    run_prompt(service, prompt)


def execute_financial_analysis_chat(service):
    # 3. Parâmetros para o prompt
    empresa = "Facebook"
    periodo = "2025"
    idioma = "português"
    analises = [
        "Análise de receita",
        "Análise de despesas",
        "Análise de lucro",
    ]

    # Cria e executa o chat template (Chat)
    prompt = create_chat_analise_financeira(empresa, periodo, idioma, analises)
    run_prompt(service, prompt)