"""
Utilidades compartidas para los ejemplos de IA.
Carga las variables de entorno desde el archivo .env
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Buscar .env en el directorio raíz del proyecto
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # python-dotenv no instalado, usar variables del sistema


def get_api_key(key_name: str) -> str:
    """
    Obtiene una clave de API desde las variables de entorno.

    Args:
        key_name: Nombre de la variable de entorno (ej: 'OPENAI_API_KEY')

    Returns:
        El valor de la clave de API

    Raises:
        ValueError: Si la clave no está configurada
    """
    value = os.getenv(key_name)
    if not value:
        raise ValueError(
            f"La variable de entorno '{key_name}' no está configurada.\n"
            f"Copia '.env.example' como '.env' y añade tu clave."
        )
    return value
