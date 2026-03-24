"""
Ejemplo de uso de la API de Google Gemini.

Modelos disponibles:
  - gemini-1.5-pro
  - gemini-1.5-flash
  - gemini-2.0-flash

Documentación: https://ai.google.dev/gemini-api/docs
Obtén tu clave en: https://aistudio.google.com/app/apikey
"""

from google import genai
from google.genai import types
from utils import get_api_key


DEFAULT_FLASH_CANDIDATES = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-1.5-flash",
]


def _get_client() -> genai.Client:
    """Crea un cliente Gemini con la clave del entorno."""
    return genai.Client(api_key=get_api_key("GEMINI_API_KEY"))


def _history_item_to_content(item: dict) -> types.Content:
    """Convierte mensajes del historial al formato de google.genai."""
    role = item.get("role", "user")
    raw_parts = item.get("parts", [])

    parts: list[types.Part] = []
    for part in raw_parts:
        if isinstance(part, str):
            parts.append(types.Part.from_text(text=part))
        elif isinstance(part, dict) and "text" in part:
            parts.append(types.Part.from_text(text=str(part["text"])))

    if not parts:
        parts.append(types.Part.from_text(text=""))

    return types.Content(role=role, parts=parts)


def _pick_available_model(client: genai.Client, preferred: str) -> str:
    """Devuelve un modelo válido para generate_content en la cuenta actual."""
    available = []
    for item in client.models.list():
        name = getattr(item, "name", "")
        if "generateContent" in getattr(item, "supported_actions", []):
            available.append(name)

    preferred_full = preferred if preferred.startswith("models/") else f"models/{preferred}"
    if preferred_full in available:
        return preferred_full

    for candidate in DEFAULT_FLASH_CANDIDATES:
        if candidate in available:
            return candidate

    if available:
        return available[0]

    raise RuntimeError("No hay modelos de Gemini disponibles para generate_content.")


def chat_with_gemini(prompt: str, model: str = "gemini-1.5-flash") -> str:
    """
    Envía un mensaje al modelo Gemini y devuelve la respuesta.

    Args:
        prompt: El mensaje del usuario
        model: El modelo a utilizar (por defecto gemini-1.5-flash)

    Returns:
        La respuesta del modelo como texto
    """
    client = _get_client()
    selected_model = _pick_available_model(client, model)
    response = client.models.generate_content(
        model=selected_model,
        contents=prompt,
    )

    return response.text or ""


def chat_with_history(history: list[dict], model: str = "gemini-1.5-flash") -> str:
    """
    Envía una conversación con historial al modelo Gemini.

    Args:
        history: Lista de mensajes con formato [{"role": "user/model", "parts": ["..."]}]
        model: El modelo a utilizar

    Returns:
        La respuesta del modelo como texto
    """
    if not history:
        return ""

    client = _get_client()
    selected_model = _pick_available_model(client, model)
    contents = [_history_item_to_content(item) for item in history]
    response = client.models.generate_content(
        model=selected_model,
        contents=contents,
    )

    return response.text or ""


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
