# Desarrollo de Apps con IA

Proyecto educativo para aprender a crear aplicaciones de **backend** y **frontend** utilizando las principales APIs de Inteligencia Artificial disponibles en el mercado.

## 🤖 Proveedores de IA soportados

| Proveedor | Librería | Modelos destacados |
|-----------|----------|-------------------|
| **OpenAI / ChatGPT** | `openai` | GPT-4o, GPT-4-turbo, GPT-3.5-turbo |
| **Google Gemini** | `google-genai` | Gemini 1.5 Pro, Gemini 1.5 Flash |
| **Anthropic Claude** | `anthropic` | Claude 3.5 Sonnet, Claude 3 Opus |
| **DeepSeek** | `openai` (compatible) | DeepSeek-V3, DeepSeek-R1 |
| **GitHub Copilot** | `openai` (Azure) | GPT-4o via Azure OpenAI |

## 📁 Estructura del proyecto

```
desarrollo_apps_con_ia/
│
├── notebooks/                  # Jupyter Notebooks interactivos
│   ├── 01_openai_chatgpt.ipynb        # Ejemplos con OpenAI / ChatGPT
│   ├── 02_gemini.ipynb                # Ejemplos con Google Gemini
│   ├── 03_claude_anthropic.ipynb      # Ejemplos con Anthropic Claude
│   ├── 04_deepseek.ipynb              # Ejemplos con DeepSeek
│   └── 05_github_copilot.ipynb        # Ejemplos con GitHub Copilot (Azure OpenAI)
│
├── python/                     # Scripts de Python
│   ├── utils.py                       # Utilidades compartidas
│   ├── openai_example.py              # Ejemplo con OpenAI
│   ├── gemini_example.py              # Ejemplo con Google Gemini
│   ├── claude_example.py              # Ejemplo con Anthropic Claude
│   ├── deepseek_example.py            # Ejemplo con DeepSeek
│   └── github_copilot_example.py      # Ejemplo con GitHub Copilot
│
├── .env.example               # Plantilla de variables de entorno
├── requirements.txt           # Dependencias de Python
└── README.md                  # Este archivo
```

## 🚀 Configuración inicial

### 1. Clonar el repositorio

```bash
git clone https://github.com/miguepoloc/desarrollo_apps_con_ia.git
cd desarrollo_apps_con_ia
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tus claves de API
```

### 5. Ejecutar Jupyter Notebook

```bash
jupyter notebook
```

## 🔑 Obtener claves de API

- **OpenAI**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Google Gemini**: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- **Anthropic Claude**: [https://console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
- **DeepSeek**: [https://platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)
- **GitHub Copilot**: Requiere suscripción activa a GitHub Copilot

## 📚 Recursos de aprendizaje

- [Documentación OpenAI](https://platform.openai.com/docs)
- [Documentación Google Gemini](https://ai.google.dev/gemini-api/docs)
- [Documentación Anthropic Claude](https://docs.anthropic.com)
- [Documentación DeepSeek](https://platform.deepseek.com/api-docs)
- [GitHub Copilot API](https://docs.github.com/en/copilot/using-github-copilot/using-github-copilot-in-the-command-line)

## 📄 Licencia

Este proyecto es de uso educativo libre.
