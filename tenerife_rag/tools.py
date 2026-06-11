"""Herramientas: definición de ``search_tenerife_info`` y ``get_weather``.

Este módulo importa del notebook ``RAG_LLM.ipynb`` las herramientas que el
modelo puede invocar, junto con sus esquemas JSON:

- :data:`SEARCH_TENERIFE_SCHEMA` (≡ ``rag_tool_schema`` del notebook): mismo dict de
  esquema (``query`` string y ``k`` integer en rango 1..8).
- ``search_tenerife_info``: mismo cuerpo y firma, conservando el clamp
  ``safe_k = max(1, min(k, 8))`` (rango 1..8) y devolviendo ``{query, results, context}``.
- :func:`format_search_results`: mismo cuerpo, produciendo bloques
  ``Fuente: <source> (chunk N, score S)`` seguidos del texto.
- :data:`WEATHER_TOOL_SCHEMA` (≡ ``weather_tool_schema`` del notebook) y
  :func:`get_weather`: mismo cuerpo, firma y esquema.

El notebook usaba una ``collection`` a nivel de módulo a través de ``search_internal_docs``. Aquí, :func:`make_search_tool`
enlaza vía closure la ``collection`` conectada (construyendo ``search_internal_docs`` con
:func:`tenerife_rag.vector_store.make_search_internal_docs`) y devuelve el
:class:`~tenerife_rag.tool_loop.ToolSpec` correspondiente. El cuerpo portado de
``search_tenerife_info`` no se toca.

``ToolSpec`` NO se redefine aquí: se importa de ``tenerife_rag.tool_loop`` (tal como el notebook lo declara en la sección del tool-loop).

"""

from __future__ import annotations

import logging
from typing import Any

from .tool_loop import ToolSpec
from .vector_store import make_search_internal_docs


# Configuramos el logger (traido del notebook)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ClimaTenerife")


def format_search_results(results: list[dict[str, Any]]) -> str:
    """Convierte resultados de búsqueda en un bloque de contexto legible para el modelo."""
    blocks = []
    for result in results:
        blocks.append(
            "\n".join(
                [
                    f"Fuente: {result['source']} (chunk {result['chunk_index']}, score {result['score']:.3f})",
                    result["text"],
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


# traido del notebook (mismo dict de esquema; k entre 1 y 8).
# rag_tool_schema del notebook.
SEARCH_TENERIFE_SCHEMA = {
    "type": "function",
    "name": "search_tenerife_info",
    "description": (
        "Busca fragmentos relevantes en la documentación sobre Tenerife. "
        "Úsala para preguntas sobre turismo en Tenerife: lugares a visitar, actividades a realizar, gastronomía "
        "No la uses para preguntas generales que no dependan de documentación sobre Tenerife."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Pregunta o búsqueda semántica que representa la necesidad informativa del usuario.",
            },
            "k": {
                "type": "integer",
                "description": "Número de fragmentos a recuperar. Usa 3 o 4 salvo que necesites más cobertura.",
                "minimum": 1,
                "maximum": 8,
            },
        },
        "required": ["query", "k"],
        "additionalProperties": False,
    },
}


def make_search_tool(collection) -> ToolSpec:
    """Enlaza la ``collection`` conectada y devuelve el ``ToolSpec`` de ``search_tenerife_info``.

    El cuerpo de ``search_tenerife_info`` se importa del notebook (incluido el clamp
    ``safe_k = max(1, min(k, 8))``) y devuelve ``{'query', 'results', 'context'}``. La única
    adaptación es que ``search_internal_docs`` se obtiene enlazando la ``collection`` vía
    closure (en lugar del global de módulo del notebook).
    """

    search_internal_docs = make_search_internal_docs(collection)

    def search_tenerife_info(query: str, k: int) -> dict[str, Any]:
        """Busca información en la documentación en .pdf sobre Tenerife."""
        safe_k = max(1, min(k, 8))
        results = search_internal_docs(query, k=safe_k)
        return {
            "query": query,
            "results": results,
            "context": format_search_results(results),
        }

    return ToolSpec(schema=SEARCH_TENERIFE_SCHEMA, function=search_tenerife_info)


# Realizamos una función de Python simulada que nos dé el clima (con manejo de errores)
def get_weather(location: str, date: str = "hoy") -> dict:
    """Obtiene el clima simulado de una ubicación en Tenerife."""
    logger.info(f"[INTENTO] Buscando clima para la ubicación: {location}, fecha: {date}")

    try:
        # Simulamos una llamada a una API que podría fallar
        location_lower = location.lower()
        if "teide" in location_lower:
            resultado = {"temperatura_celsius": 6, "condicion": "Despejado pero muy frío", "viento": "Alto"}
        elif "playa" in location_lower or "sur" in location_lower:
            resultado = {"temperatura_celsius": 24, "condicion": "Soleado ideal para baño", "viento": "Leve"}
        elif "laguna" in location_lower:
            resultado = {"temperatura_celsius": 18, "condicion": "Nublado y húmedo", "viento": "Moderado"}
        else:
            resultado = {"temperatura_celsius": 21, "condicion": "Parcialmente nublado", "viento": "Moderado"}

        logger.info(f"[ÉXITO] Clima recuperado para la locación: {location}")
        return resultado

    except Exception as e:
        logger.error(f"[ERROR] Falló la recuperación del clima para la ubicación {location}: {str(e)}")
        # Devolvemos un JSON de error controlado
        return {"error": "El servicio meteorológico no está disponible en este momento."}


# Definimos el esquema JSON
# Traido del weather_tool_schema del notebook.
WEATHER_TOOL_SCHEMA = {
    "type": "function",
    "name": "get_weather",
    "description": "Obtiene el pronóstico del clima actual para una ubicación específica en Tenerife.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Nombre del municipio, playa o punto de interés en Tenerife (ej. 'El Teide', 'Playa de las Américas').",
            },
            "date": {
                "type": "string",
                "description": "Fecha para el pronóstico. Por defecto es 'hoy'.",
            }
        },
        "required": ["location", "date"],
        "additionalProperties": False,
    },
}
