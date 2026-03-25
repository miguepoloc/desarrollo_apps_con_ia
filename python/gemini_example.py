"""
Ejemplo de uso de la API de Google Gemini.

Modelos disponibles:
  - gemini-1.5-pro
  - gemini-1.5-flash
  - gemini-2.0-flash
  - gemini-2.5-flash (recomendado para clase — gratuito)

Documentación: https://ai.google.dev/gemini-api/docs
Obtén tu clave en: https://aistudio.google.com/app/apikey
"""

import csv
import io
import json
import time
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


# ---------------------------------------------------------------------------
# TAREA 1 — Funciones con system_instruction
# ---------------------------------------------------------------------------

def chat_with_system(
    system_instruction: str,
    user_message: str,
    model: str = "gemini-2.5-flash",
) -> str:
    """
    Envía un mensaje al modelo Gemini con una instrucción de sistema (rol/reglas).

    Args:
        system_instruction: Rol y reglas que definen el comportamiento del modelo.
        user_message: Mensaje puntual del usuario.
        model: Nombre del modelo a utilizar.

    Returns:
        Texto de la respuesta del modelo.
    """
    client = _get_client()
    selected_model = _pick_available_model(client, model)

    # GenerateContentConfig permite pasar system_instruction de forma nativa
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
    )

    response = client.models.generate_content(
        model=selected_model,
        contents=user_message,
        config=config,
    )
    return response.text or ""


def chat_with_system_and_history(
    system_instruction: str,
    history: list[dict],
    model: str = "gemini-2.5-flash",
) -> str:
    """
    Envía una conversación completa (system + historial) al modelo Gemini.

    El historial debe incluir todos los turnos anteriores. El último elemento
    debe ser el mensaje más reciente del usuario.

    Args:
        system_instruction: Rol y reglas del modelo.
        history: Lista de mensajes con formato [{"role": "user/model", "parts": [...]}].
        model: Nombre del modelo a utilizar.

    Returns:
        Texto de la respuesta del modelo al último turno del historial.
    """
    if not history:
        return ""

    client = _get_client()
    selected_model = _pick_available_model(client, model)

    # Convertir historial al formato nativo de google.genai
    contents = [_history_item_to_content(item) for item in history]

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
    )

    response = client.models.generate_content(
        model=selected_model,
        contents=contents,
        config=config,
    )
    return response.text or ""


# ---------------------------------------------------------------------------
# Bloque principal — ejemplos base y experimentos didácticos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    MODELO = "gemini-2.5-flash"  # Modelo principal de clase (gratuito)

    # --- Ejemplo 1: Pregunta simple ---
    print("=== Ejemplo 1: Pregunta simple ===")
    respuesta = chat_with_gemini("Explica qué es la inteligencia artificial en 3 líneas.")
    print(f"Respuesta: {respuesta}\n")

    # --- Ejemplo 2: Conversación con historial ---
    print("=== Ejemplo 2: Conversación con historial ===")
    historial = [
        {"role": "user", "parts": ["¿Qué es Python?"]},
        {"role": "model", "parts": ["Python es un lenguaje de programación de alto nivel."]},
        {"role": "user", "parts": ["¿Para qué se usa en IA?"]},
    ]
    respuesta = chat_with_history(historial)
    print(f"Respuesta: {respuesta}\n")

    # -----------------------------------------------------------------------
    # EXPERIMENTO A — Mismo prompt, 3 system prompts distintos
    # Objetivo: mostrar cómo el system instruction cambia tono, formato y
    # profundidad de la respuesta para el mismo mensaje del usuario.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENTO A — Mismo prompt, 3 system prompts distintos")
    print("=" * 60)

    PREGUNTA_A = "¿Qué es machine learning?"

    SYSTEMS_A = {
        "A (vago)": "Eres un asistente útil.",
        "B (técnico)": (
            "Eres un profesor universitario de ingeniería de sistemas. "
            "Responde de forma técnica y precisa en máximo 3 párrafos."
        ),
        "C (JSON)": (
            "Eres un asistente técnico. "
            "RESPONDE ÚNICAMENTE con un objeto JSON válido con esta estructura exacta, "
            "sin texto adicional ni markdown: "
            '{"concepto": str, "definicion": str, "ejemplo": str, '
            '"nivel": "basico|intermedio|avanzado"}'
        ),
    }

    inicio_a = time.time()
    respuestas_a: dict[str, str] = {}

    for nombre_system, system_text in SYSTEMS_A.items():
        print(f"\n--- System {nombre_system} ---")
        try:
            resp = chat_with_system(
                system_instruction=system_text,
                user_message=PREGUNTA_A,
                model=MODELO,
            )
            respuestas_a[nombre_system] = resp
            print(resp)
        except Exception as e:
            respuestas_a[nombre_system] = ""
            print(f"[ERROR] No se pudo obtener respuesta: {e}")

    # — Resumen del experimento A —
    print("\n--- Análisis de respuestas ---")
    longitudes: dict[str, int] = {}
    for nombre, resp in respuestas_a.items():
        palabras = len(resp.split())
        longitudes[nombre] = palabras
        print(f"  System {nombre}: {palabras} palabras")

    # Validar si la respuesta C es JSON válido
    resp_c_texto = respuestas_a.get("C (JSON)", "")
    try:
        texto_limpio = (
            resp_c_texto.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        json.loads(texto_limpio)
        print("  ✅ Respuesta C es JSON válido")
    except (json.JSONDecodeError, ValueError):
        print("  ❌ Respuesta C NO es JSON válido")

    if longitudes:
        mas_larga = max(longitudes, key=lambda k: longitudes[k])
        mas_corta = min(longitudes, key=lambda k: longitudes[k])
        print(f"  Más larga : System {mas_larga} ({longitudes[mas_larga]} palabras)")
        print(f"  Más corta : System {mas_corta} ({longitudes[mas_corta]} palabras)")

    tiempo_a = time.time() - inicio_a
    print("=" * 60)
    print(f"EXPERIMENTO A completado en {tiempo_a:.2f}s")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # EXPERIMENTO B — Mismo system, 3 formatos de salida distintos
    # Objetivo: mostrar cómo el formato pedido en el mensaje del usuario
    # (texto libre / JSON / CSV) cambia la estructura de la respuesta.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENTO B — Mismo system, 3 formatos de salida")
    print("=" * 60)

    SYSTEM_B = "Eres un experto en inteligencia artificial."
    PREGUNTA_BASE_B = "Explica las 3 aplicaciones más importantes de la IA"

    FORMATOS_B = {
        "1 — Texto libre": PREGUNTA_BASE_B,
        "2 — JSON": (
            f"{PREGUNTA_BASE_B}. "
            "Responde SOLO con JSON: "
            '[{"aplicacion": str, "descripcion": str, "industria": str}]'
        ),
        "3 — CSV": (
            f"{PREGUNTA_BASE_B}. "
            "Responde SOLO con CSV, primera fila son headers: "
            "aplicacion,descripcion,industria"
        ),
    }

    inicio_b = time.time()

    for nombre_formato, mensaje_usuario in FORMATOS_B.items():
        print(f"\n--- Formato {nombre_formato} ---")
        try:
            resp = chat_with_system(
                system_instruction=SYSTEM_B,
                user_message=mensaje_usuario,
                model=MODELO,
            )
            print(resp)

            # Validaciones adicionales según el formato
            if nombre_formato.startswith("2"):
                try:
                    texto_limpio = (
                        resp.strip()
                        .removeprefix("```json")
                        .removeprefix("```")
                        .removesuffix("```")
                        .strip()
                    )
                    datos = json.loads(texto_limpio)
                    print(f"  ✅ JSON válido — {len(datos)} elemento(s) en la lista")
                except (json.JSONDecodeError, ValueError):
                    print("  ❌ La respuesta no es JSON parseable")

            elif nombre_formato.startswith("3"):
                try:
                    reader = csv.reader(io.StringIO(resp.strip()))
                    filas = list(reader)
                    print("  ✅ CSV parseado — filas:")
                    for fila in filas:
                        print(f"    {fila}")
                except Exception as csv_err:
                    print(f"  ❌ No se pudo parsear el CSV: {csv_err}")

        except Exception as e:
            print(f"[ERROR] No se pudo obtener respuesta: {e}")

    tiempo_b = time.time() - inicio_b
    print("=" * 60)
    print(f"EXPERIMENTO B completado en {tiempo_b:.2f}s")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # EXPERIMENTO C — Conversación con memoria manual vs sin memoria
    # Objetivo: demostrar que sin historial el modelo "olvida" los turnos
    # anteriores, y con historial los "recuerda".
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENTO C — Memoria: sin historial vs con historial")
    print("=" * 60)

    MSG_1 = "Mi nombre es Carlos y soy de Barranquilla"
    MSG_2 = "¿Cómo me llamo y de dónde soy?"

    inicio_c = time.time()

    # — Sin historial: dos llamadas independientes —
    print("\nSIN HISTORIAL (llamadas independientes):")
    try:
        _ = chat_with_gemini(MSG_1, model=MODELO)  # sesión 1 — se descarta
        resp_sin = chat_with_gemini(MSG_2, model=MODELO)  # sesión 2 — contexto nuevo
        print(f"Respuesta: {resp_sin}")
    except Exception as e:
        print(f"[ERROR] {e}")

    # — Con historial: todos los turnos acumulados en una sola petición —
    print("\nCON HISTORIAL (historial acumulado):")
    try:
        historial_c = [
            {"role": "user",  "parts": [{"text": MSG_1}]},
            {"role": "model", "parts": [{"text": "¡Hola Carlos! Encantado de conocerte."}]},
            {"role": "user",  "parts": [{"text": MSG_2}]},
        ]
        resp_con = chat_with_history(historial_c, model=MODELO)
        print(f"Respuesta: {resp_con}")
    except Exception as e:
        print(f"[ERROR] {e}")

    print(
        "\n¿Por qué son diferentes?\n"
        "  — Sin historial: cada llamada a chat_with_gemini() abre una sesión nueva.\n"
        "    El modelo no tiene contexto de lo que se dijo antes.\n"
        "  — Con historial: chat_with_history() envía todos los turnos en una sola\n"
        "    petición. El modelo puede 'leer' lo que se dijo y responder con contexto.\n"
        "  Esta es la base del manejo de memoria en chatbots conversacionales."
    )

    tiempo_c = time.time() - inicio_c
    print("=" * 60)
    print(f"EXPERIMENTO C completado en {tiempo_c:.2f}s")
    print("=" * 60)
