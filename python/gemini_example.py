"""
Ejemplo de uso de la API de Google Gemini.

Modelos disponibles:
  - gemini-1.5-pro
  - gemini-1.5-flash
  - gemini-2.0-flash

Documentación: https://ai.google.dev/gemini-api/docs
Obtén tu clave en: https://aistudio.google.com/app/apikey
"""

import google.generativeai as genai
from utils import get_api_key


def chat_with_gemini(prompt: str, model: str = "gemini-1.5-flash") -> str:
    """
    Envía un mensaje al modelo Gemini y devuelve la respuesta.

    Args:
        prompt: El mensaje del usuario
        model: El modelo a utilizar (por defecto gemini-1.5-flash)

    Returns:
        La respuesta del modelo como texto
    """
    genai.configure(api_key=get_api_key("GEMINI_API_KEY"))

    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(prompt)

    return response.text


def chat_with_history(history: list[dict], model: str = "gemini-1.5-flash") -> str:
    """
    Envía una conversación con historial al modelo Gemini.

    Args:
        history: Lista de mensajes con formato [{"role": "user/model", "parts": ["..."]}]
        model: El modelo a utilizar

    Returns:
        La respuesta del modelo como texto
    """
    genai.configure(api_key=get_api_key("GEMINI_API_KEY"))

    gemini_model = genai.GenerativeModel(model)
    chat = gemini_model.start_chat(history=history[:-1])

    last_message = history[-1]["parts"][0]
    response = chat.send_message(last_message)

    return response.text


if __name__ == "__main__":
    # Ejemplo 1: Pregunta simple
    print("=== Ejemplo 1: Pregunta simple ===")
    respuesta = chat_with_gemini("Explica qué es la inteligencia artificial en 3 líneas.")
    print(f"Respuesta: {respuesta}\n")

    # Ejemplo 2: Conversación con historial
    print("=== Ejemplo 2: Conversación con historial ===")
    historial = [
        {"role": "user", "parts": ["¿Qué es Python?"]},
        {"role": "model", "parts": ["Python es un lenguaje de programación de alto nivel."]},
        {"role": "user", "parts": ["¿Para qué se usa en IA?"]},
    ]
    respuesta = chat_with_history(historial)
    print(f"Respuesta: {respuesta}\n")
