"""Modulo_Config: carga de configuración del entorno y construcción del Cliente_LLM.

Este módulo porta la lógica de entorno del notebook ``RAG_LLM.ipynb`` (``load_dotenv``,
resolución de ``OPENAI_GENERATION_MODEL``/``OPENAI_EMBEDDING_MODEL`` con sus defaults
``gpt-4.1-mini``/``text-embedding-3-small`` y los ``LLM_PARAMS``
``{"temperature": 0.2, "max_output_tokens": 800}``) hacia funciones reutilizables.

ÚNICA adaptación respecto al notebook: el fallback interactivo ``getpass`` se reemplaza
por el lanzamiento de :class:`MissingApiKeyError`, ya que no hay terminal interactiva en
una app Streamlit (Req 7.5). El valor de ``OPENAI_API_KEY`` nunca se incluye en mensajes
de error ni en la representación de la configuración.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Defaults de modelos del notebook.
DEFAULT_GENERATION_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class MissingApiKeyError(Exception):
    """Se lanza cuando ``OPENAI_API_KEY`` falta, está vacía o contiene solo espacios.

    El valor de la clave nunca se incluye en el mensaje; solo se refiere a ella por el
    nombre de su variable de entorno.
    """

    def __init__(self, variable_name: str = "OPENAI_API_KEY") -> None:
        self.variable_name = variable_name
        super().__init__(
            f"La variable de entorno requerida '{variable_name}' no está definida, "
            "está vacía o contiene solo espacios en blanco."
        )


class MissingEnvVarError(Exception):
    """Se lanza cuando falta una variable de entorno requerida.

    Lleva el nombre de la variable ausente, nunca su valor.
    """

    def __init__(self, variable_name: str) -> None:
        self.variable_name = variable_name
        super().__init__(
            f"Falta la variable de entorno requerida: '{variable_name}'."
        )


@dataclass(frozen=True)
class AppConfig:
    """Configuración de la aplicación cargada del entorno.

    El valor de ``openai_api_key`` se excluye de la representación textual para evitar
    fugas del secreto.
    """

    openai_api_key: str
    generation_model: str
    embedding_model: str
    temperature: float = 0.2
    max_output_tokens: int = 800
    request_timeout_seconds: float = 60.0
    max_tool_rounds: int = 5
    chroma_path: str = "tenerife_db"
    collection_name: str = "guia_tenerife"
    db_connect_timeout_seconds: float = 10.0

    def __repr__(self) -> str:  #  evita filtrar el valor de la clave
        return (
            "AppConfig("
            "openai_api_key=<redacted>, "
            f"generation_model={self.generation_model!r}, "
            f"embedding_model={self.embedding_model!r}, "
            f"temperature={self.temperature!r}, "
            f"max_output_tokens={self.max_output_tokens!r}, "
            f"request_timeout_seconds={self.request_timeout_seconds!r}, "
            f"max_tool_rounds={self.max_tool_rounds!r}, "
            f"chroma_path={self.chroma_path!r}, "
            f"collection_name={self.collection_name!r}, "
            f"db_connect_timeout_seconds={self.db_connect_timeout_seconds!r})"
        )


def resolve_model_name(raw: str | None, default: str) -> str:
    """Función pura: devuelve ``raw.strip()`` si tiene contenido, si no ``default``.

    Reproduce el comportamiento del notebook (``os.getenv(var, default)``) pero además
    trata como ausente cualquier valor vacío o compuesto solo por espacios en blanco.
    """

    if raw is not None:
        stripped = raw.strip()
        if stripped:
            return stripped
    return default


def load_config() -> AppConfig:
    """Carga ``.env`` y variables de entorno, aplica defaults y validación.

    - Si ``OPENAI_API_KEY`` falta, está vacía o contiene solo espacios en blanco, lanza
      :class:`MissingApiKeyError` (en lugar del fallback ``getpass`` del notebook).
    - Resuelve los modelos de generación y embeddings con sus defaults cuando la variable
      está ausente, vacía o es solo espacios.

    El valor de la clave nunca se filtra en mensajes ni en la representación devuelta.
    """

    # Portado del notebook: localizar y cargar el .env.
    env_path = Path(".env")
    load_dotenv(dotenv_path=env_path)
    # Cargar también cualquier .env presente en el directorio de trabajo / entorno.
    load_dotenv()

    # Adaptación: el notebook usa getpass como fallback interactivo, aquí no
    # hay terminal, así que validamos y lanzamos MissingApiKeyError.
    raw_api_key = os.getenv("OPENAI_API_KEY")
    if raw_api_key is None or not raw_api_key.strip():
        raise MissingApiKeyError("OPENAI_API_KEY")

    generation_model = resolve_model_name(
        os.getenv("OPENAI_GENERATION_MODEL"), DEFAULT_GENERATION_MODEL
    )
    embedding_model = resolve_model_name(
        os.getenv("OPENAI_EMBEDDING_MODEL"), DEFAULT_EMBEDDING_MODEL
    )

    return AppConfig(
        openai_api_key=raw_api_key,
        generation_model=generation_model,
        embedding_model=embedding_model,
    )


def build_openai_client(config: AppConfig) -> OpenAI:
    """Construye el Cliente_LLM de OpenAI a partir de la configuración.

    Equivale a ``client = OpenAI()`` del notebook, pero pasando la clave y el timeout
    de petición de forma explícita desde la configuración cargada.
    """

    return OpenAI(
        api_key=config.openai_api_key,
        timeout=config.request_timeout_seconds,
    )


def generation_params(config: AppConfig) -> dict:
    """Devuelve los ``LLM_PARAMS`` del notebook: ``temperature`` y ``max_output_tokens``.

    Se construye a partir de la configuración para aplicarse en cada llamada de
    generación del Cliente_LLM (Req 7.4).
    """

    return {
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
    }
