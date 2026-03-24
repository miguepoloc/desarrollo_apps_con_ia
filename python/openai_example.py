"""
Ejemplo de uso de la API de OpenAI (ChatGPT).

Modelos disponibles:
  - gpt-4o
  - gpt-4-turbo
  - gpt-3.5-turbo

Documentación: https://platform.openai.com/docs
Obtén tu clave en: https://platform.openai.com/api-keys
"""

from openai import OpenAI
from utils import get_api_key


def chat_with_gpt(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Envía un mensaje al modelo de OpenAI y devuelve la respuesta.

    Args:
        prompt: El mensaje del usuario
        model: El modelo a utilizar (por defecto gpt-4o-mini)

    Returns:
        La respuesta del modelo como texto
    """
    client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Eres un asistente útil y amigable."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


def chat_with_history(messages: list[dict], model: str = "gpt-4o-mini") -> str:
    """
    Envía una conversación completa al modelo de OpenAI.

    Args:
        messages: Lista de mensajes con formato [{"role": "...", "content": "..."}]
        model: El modelo a utilizar

    Returns:
        La respuesta del modelo como texto
    """
    client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Ejemplo 1: Pregunta simple
    print("=== Ejemplo 1: Pregunta simple ===")
    respuesta = chat_with_gpt("¿Cuál es la capital de Colombia?")
    print(f"Respuesta: {respuesta}\n")

    # Ejemplo 2: Conversación con historial
    print("=== Ejemplo 2: Conversación con historial ===")
    historial = [
        {"role": "system", "content": "Eres un experto en programación Python."},
        {"role": "user", "content": "¿Qué es una lista por comprensión en Python?"},
    ]
    respuesta = chat_with_history(historial)
    print(f"Respuesta: {respuesta}\n")
