"""Base_Vectorial: conexión a ChromaDB persistida y búsqueda documental.

Este módulo porta la lógica de búsqueda del notebook ``RAG_LLM.ipynb`` y añade la
conexión a la colección ``guia_tenerife`` ya persistida en ``tenerife_db/``.

Reutilización:

- :func:`search_internal_docs` se importa del notebook (mismo cuerpo y
  firma ``search_internal_docs(query, k=4) -> list[dict]``), devolviendo ``list[dict]``
  con las claves exactas ``score``, ``source``, ``chunk_index`` y ``text``. Los
  resultados llegan pre-ordenados por distancia de ChromaDB (``n_results=k``); No se
  añade ninguna función de ordenación propia. La única adaptación es enlazar la
  ``collection`` vía closure en lugar del global de módulo del notebook.

Nuevo:

- :func:`connect_collection` reutiliza el bloque de setup del notebook
  (``PersistentClient`` + ``OpenAIEmbeddingFunction(model_name=EMBEDDING_MODEL)`` +
  ``get_or_create_collection(name="guia_tenerife", ...)``). EXCLUYE intencionadamente
  toda la ingesta del PDF (``cargar_documento_tenerife``, ``chunk_text`` y
  ``collection.add(...)``), porque la app reutiliza ``tenerife_db/`` sin re-ingestar.
  Añade un control de timeout de 10 s y el lanzamiento de
  :class:`VectorStoreUnavailableError` ante inexistencia, colección no disponible,
  error de red, error de permisos o expiración del tiempo máximo.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable

import chromadb
from chromadb.utils import embedding_functions

from .config import AppConfig


class VectorStoreUnavailableError(Exception):
    """Indisponibilidad de la Base_Vectorial.

    Cubre inexistencia, colección no disponible, error de red, error de permisos o
    expiración del tiempo máximo de conexión.
    """


def make_search_internal_docs(collection) -> Callable[..., list[dict]]:
    """Enlaza ``collection`` vía closure y devuelve la función ``search_internal_docs``.

    El cuerpo y la firma de la función interna se portan VERBATIM del notebook; la única
    diferencia es que ``collection`` se captura por closure en lugar de ser un global de
    módulo.
    """

    def search_internal_docs(query: str, k: int = 4) -> list[dict]:
        """Busca en ChromaDB y devuelve el formato exacto que se requiere"""

        results = collection.query(
            query_texts=[query],
            n_results=k
        )

        formatted_results = []

        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "score": results["distances"][0][i] if "distances" in results else 0.0,
                    "source": results["metadatas"][0][i]["source"],
                    "chunk_index": results["metadatas"][0][i]["chunk_index"],
                    "text": results["documents"][0][i],
                })

        return formatted_results

    return search_internal_docs


def connect_collection(config: AppConfig):
    """Conecta a la colección ``guia_tenerife`` persistida en ``tenerife_db/``.

    Reutiliza el bloque de setup del notebook (PersistentClient +
    OpenAIEmbeddingFunction con el modelo de embeddings + get_or_create_collection). La
    embedding function sigue siendo requerida para consultar la colección, por lo que se
    reutiliza tal cual. No hace re-ingesta del documento.

    Aplica un timeout de ``config.db_connect_timeout_seconds`` (10 s por defecto) y lanza
    :class:`VectorStoreUnavailableError` si la base de datos no existe, la colección no
    está disponible, hay un error de red o permisos, o se excede el tiempo máximo.
    """

    def _connect():
        # Reutilización del notebook (sin la ingesta)
        funcion_embeddings = embedding_functions.OpenAIEmbeddingFunction(
            api_key=config.openai_api_key,
            model_name=config.embedding_model,
        )

        chroma_client = chromadb.PersistentClient(path=config.chroma_path)

        collection = chroma_client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=funcion_embeddings,
        )

        return collection

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_connect)
            return future.result(timeout=config.db_connect_timeout_seconds)
    except FuturesTimeoutError as exc:
        raise VectorStoreUnavailableError(
            "No se pudo conectar con la base de datos: se excedió el tiempo máximo de "
            f"{config.db_connect_timeout_seconds:g} segundos."
        ) from exc
    except VectorStoreUnavailableError:
        raise
    except Exception as exc:  # Red, permisos, inexistencia, colección no disponible
        raise VectorStoreUnavailableError(
            "La base de datos no está disponible. No se pudo conectar con la colección "
            f"'{config.collection_name}' en '{config.chroma_path}'."
        ) from exc
