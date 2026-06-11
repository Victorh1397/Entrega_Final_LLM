# Asistente de Turismo de Tenerife — RAG + LLM

Sistema de preguntas y respuestas sobre turismo en Tenerife basado en
Retrieval-Augmented Generation (RAG). El proyecto incluye:

1. Un **notebook de exploración** (`notebook/RAG_LLM.ipynb`) donde se desarrolla
   toda la pipeline: carga del PDF, chunking, creación de la base vectorial
   ChromaDB, definición de herramientas (tool calling) y bucle de generación con
   OpenAI.
2. Un **paquete Python reutilizable** (`tenerife_rag/`) que refactoriza la lógica
   del notebook en módulos importables.
3. Una **aplicación web** (`app.py`) construida con Streamlit que expone un chat
   interactivo reutilizando el paquete anterior y la base vectorial ya persistida.

## Estructura del proyecto

```
Entrega_Final_LLM/
├── .env                    # Variables de entorno (no versionado)
├── .gitignore              # Reglas de exclusión de Git
├── analisis_final.md       # Análisis del proyecto (diseño, decisiones, resultados)
├── README.md               # Este archivo
├── requirements.txt        # Dependencias del proyecto
├── app.py                  # Aplicación web Streamlit (interfaz de chat)
├── tenerife_rag/           # Paquete Motor RAG (módulos reutilizables)
│   ├── __init__.py         # Marca el directorio como paquete Python
│   ├── config.py           # Carga de configuración y cliente OpenAI
│   ├── vector_store.py     # Conexión a ChromaDB y búsqueda documental
│   ├── tools.py            # Herramientas: search_tenerife_info, get_weather
│   └── tool_loop.py        # Bucle de tool calling (run_llm_with_tools)
├── data/                   # Documentos fuente
│   └── TENERIFE.pdf        # Guía turística de Tenerife (fuente de datos)
├── notebook/               # Notebook de desarrollo y exploración
│   └── RAG_LLM.ipynb       # Pipeline RAG completa (ingesta + consulta)
└── tenerife_db/            # Base vectorial ChromaDB persistida
    └── ...                 # Archivos generados por ChromaDB
```

## Descripción de componentes

### Notebook (`notebook/RAG_LLM.ipynb`)

Contiene el desarrollo completo de la pipeline RAG:

- Carga y extracción de texto del PDF (`TENERIFE.pdf`) con PyPDF.
- Particionado del texto en chunks.
- Creación de embeddings con el modelo `text-embedding-3-small` de OpenAI.
- Almacenamiento en una colección ChromaDB persistida en `tenerife_db/`.
- Definición de herramientas (schemas JSON) para tool calling.
- Bucle de generación con el modelo `gpt-4.1-mini` y tool calling.

### Paquete `tenerife_rag/`

Refactorización del notebook en módulos independientes:

| Módulo | Responsabilidad |
|--------|----------------|
| `config.py` | Carga `.env`, valida `OPENAI_API_KEY`, resuelve modelos con defaults, construye el cliente OpenAI |
| `vector_store.py` | Conecta a la colección ChromaDB persistida y expone la función de búsqueda |
| `tools.py` | Define las herramientas `search_tenerife_info` (búsqueda RAG) y `get_weather` (clima simulado) |
| `tool_loop.py` | Implementa el bucle de tool calling (`run_llm_with_tools`) con trazabilidad y validación |

### Aplicación web (`app.py`)

Interfaz de chat con Streamlit que:

- Reutiliza la base vectorial ya existente (no re-ingesta el PDF).
- Mantiene un historial de conversación (máximo 10 turnos).
- Valida la entrada del usuario (no vacía, máximo 1000 caracteres).
- Sanitiza las respuestas eliminando logs técnicos y preservando la atribución de fuentes.
- Gestiona errores de configuración, timeout y límite de rondas de herramientas.

## Requisitos previos

- Python 3.10 o superior
- La base vectorial `tenerife_db/` presente en la raíz del proyecto (se genera
  ejecutando el notebook)
- Una clave de API de OpenAI válida

## Instalación y configuración

### 1. Crear y activar el entorno virtual

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```dotenv
# Requerida
OPENAI_API_KEY=tu_clave_de_openai

# Opcionales (se muestran los valores por defecto)
OPENAI_GENERATION_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

| Variable | Requerida | Default | Descripción |
|----------|-----------|---------|-------------|
| `OPENAI_API_KEY` | Sí | — | Clave de API de OpenAI |
| `OPENAI_GENERATION_MODEL` | No | `gpt-4.1-mini` | Modelo de generación de texto |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Modelo de embeddings |

## Uso

### Ejecutar el notebook

Abre `notebook/RAG_LLM.ipynb` en Jupyter y ejecútalo para:

- Generar la base vectorial en `tenerife_db/` (si no existe).
- Explorar la pipeline RAG de forma interactiva.

### Ejecutar la aplicación web

```bash
streamlit run app.py
```

La interfaz se abrirá en el navegador en `http://localhost:8501`.

## Tecnologías utilizadas

- **OpenAI API** — Generación de texto (GPT-4.1-mini) y embeddings
- **ChromaDB** — Base de datos vectorial para búsqueda semántica
- **Streamlit** — Interfaz web de chat
- **PyPDF** — Extracción de texto del PDF
- **python-dotenv** — Carga de variables de entorno
- **jsonschema** — Validación de argumentos de herramientas

## Memoria técnica

El archivo `analisis_final.md` en la raíz del proyecto contiene la memoria técnica completa del proyecto: diseño de la solución, decisiones técnicas justificadas, resultados obtenidos, limitaciones identificadas y propuestas de mejora futura.
