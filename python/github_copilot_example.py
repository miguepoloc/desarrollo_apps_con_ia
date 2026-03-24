"""
Ejemplo de uso de GitHub Copilot a través de la API de modelos de GitHub.

GitHub ofrece acceso a modelos de IA (incluyendo GPT-4o) mediante tokens de GitHub.
Esta API es compatible con el cliente de OpenAI.

Modelos disponibles:
  - gpt-4o
  - gpt-4o-mini
  - o1-mini
  - o1-preview

Documentación: https://docs.github.com/en/github-models
Obtén tu token en: https://github.com/settings/tokens
"""

from openai import OpenAI
from utils import get_api_key

GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"


def chat_with_github_models(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Envía un mensaje a través de la API de modelos de GitHub.

    Args:
        prompt: El mensaje del usuario
        model: El modelo a utilizar (por defecto gpt-4o-mini)

    Returns:
        La respuesta del modelo como texto
    """
    client = OpenAI(
        api_key=get_api_key("GITHUB_TOKEN"),
        base_url=GITHUB_MODELS_BASE_URL,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Eres un asistente de programación experto."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


def code_review_with_copilot(code: str, language: str = "python") -> str:
    """
    Solicita una revisión de código usando GitHub Copilot.

    Args:
        code: El código a revisar
        language: El lenguaje de programación

    Returns:
        Sugerencias y revisión del código
    """
    client = OpenAI(
        api_key=get_api_key("GITHUB_TOKEN"),
        base_url=GITHUB_MODELS_BASE_URL,
    )

    prompt = f"Revisa el siguiente código {language} e identifica mejoras, errores o malas prácticas:\n\n```{language}\n{code}\n```"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Eres GitHub Copilot, un experto revisor de código. "
                "Proporciona retroalimentación constructiva y detallada.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Ejemplo 1: Pregunta de programación
    print("=== Ejemplo 1: Pregunta de programación ===")
    respuesta = chat_with_github_models(
        "¿Cómo implemento un API REST con FastAPI en Python? Dame un ejemplo básico."
    )
    print(f"Respuesta: {respuesta}\n")

    # Ejemplo 2: Revisión de código
    print("=== Ejemplo 2: Revisión de código ===")
    codigo = """
def suma_lista(lista):
    total = 0
    for i in range(len(lista)):
        total = total + lista[i]
    return total
"""
    revision = code_review_with_copilot(codigo)
    print(f"Revisión: {revision}\n")
