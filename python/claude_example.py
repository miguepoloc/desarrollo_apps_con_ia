"""
Ejemplo de uso de la API de Anthropic (Claude).

Modelos disponibles:
  - claude-3-5-sonnet-20241022
  - claude-3-5-haiku-20241022
  - claude-3-opus-20240229

Documentación: https://docs.anthropic.com
Obtén tu clave en: https://console.anthropic.com/settings/keys
"""

import anthropic
from utils import get_api_key


def chat_with_claude(prompt: str, model: str = "claude-3-5-haiku-20241022") -> str:
    """
    Envía un mensaje al modelo Claude y devuelve la respuesta.

    Args:
        prompt: El mensaje del usuario
        model: El modelo a utilizar (por defecto claude-3-5-haiku)

    Returns:
        La respuesta del modelo como texto
    """
    client = anthropic.Anthropic(api_key=get_api_key("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    return message.content[0].text


def chat_with_system(prompt: str, system: str, model: str = "claude-3-5-haiku-20241022") -> str:
    """
    Envía un mensaje con un prompt de sistema personalizado.

    Args:
        prompt: El mensaje del usuario
        system: El prompt de sistema que define el comportamiento
        model: El modelo a utilizar

    Returns:
        La respuesta del modelo como texto
    """
    client = anthropic.Anthropic(api_key=get_api_key("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    return message.content[0].text


if __name__ == "__main__":
    # Ejemplo 1: Pregunta simple
    print("=== Ejemplo 1: Pregunta simple ===")
    respuesta = chat_with_claude("¿Cuáles son las ventajas de usar Claude para desarrollo de software?")
    print(f"Respuesta: {respuesta}\n")

    # Ejemplo 2: Con prompt de sistema
    print("=== Ejemplo 2: Con prompt de sistema personalizado ===")
    respuesta = chat_with_system(
        prompt="Escribe una función en Python para calcular el factorial de un número.",
        system="Eres un experto en Python. Responde siempre con código limpio y comentado.",
    )
    print(f"Respuesta: {respuesta}\n")
