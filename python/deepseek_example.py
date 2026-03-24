"""
Ejemplo de uso de la API de DeepSeek.

DeepSeek es compatible con la API de OpenAI, por lo que usamos el cliente openai
apuntando al endpoint de DeepSeek.

Modelos disponibles:
  - deepseek-chat    (DeepSeek-V3)
  - deepseek-reasoner (DeepSeek-R1)

Documentación: https://platform.deepseek.com/api-docs
Obtén tu clave en: https://platform.deepseek.com/api_keys
"""

from openai import OpenAI
from utils import get_api_key

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def chat_with_deepseek(prompt: str, model: str = "deepseek-chat") -> str:
    """
    Envía un mensaje al modelo DeepSeek y devuelve la respuesta.

    Args:
        prompt: El mensaje del usuario
        model: El modelo a utilizar (por defecto deepseek-chat / DeepSeek-V3)

    Returns:
        La respuesta del modelo como texto
    """
    client = OpenAI(
        api_key=get_api_key("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Eres un asistente útil y amigable."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


def reason_with_deepseek(prompt: str) -> tuple[str, str]:
    """
    Usa el modelo razonador DeepSeek-R1 que muestra su cadena de pensamiento.

    Args:
        prompt: El problema o pregunta a razonar

    Returns:
        Tupla con (cadena_de_pensamiento, respuesta_final)
    """
    client = OpenAI(
        api_key=get_api_key("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
    )

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    reasoning = response.choices[0].message.reasoning_content or ""
    answer = response.choices[0].message.content or ""

    return reasoning, answer


if __name__ == "__main__":
    # Ejemplo 1: Pregunta simple con DeepSeek-V3
    print("=== Ejemplo 1: Chat con DeepSeek-V3 ===")
    respuesta = chat_with_deepseek("¿Qué diferencia hay entre DeepSeek-V3 y DeepSeek-R1?")
    print(f"Respuesta: {respuesta}\n")

    # Ejemplo 2: Razonamiento con DeepSeek-R1
    print("=== Ejemplo 2: Razonamiento con DeepSeek-R1 ===")
    pensamiento, respuesta = reason_with_deepseek(
        "¿Cuántos días hay entre el 1 de enero y el 1 de julio de 2025?"
    )
    if pensamiento:
        print(f"Cadena de pensamiento:\n{pensamiento}\n")
    print(f"Respuesta final: {respuesta}\n")
