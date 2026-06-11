"""
app.py — Aplicacion_Web (Tenerife Chat Web UI)

Capa de presentación Streamlit con funciones puras de soporte aisladas
para testeo (validación, historial, saneamiento de respuesta).
"""

import logging
import re
from dataclasses import dataclass

import openai
import streamlit as st

from tenerife_rag.config import (
    MissingApiKeyError,
    MissingEnvVarError,
    build_openai_client,
    generation_params,
    load_config,
)
from tenerife_rag.tool_loop import (
    INSTRUCCIONES_GUIA_TENERIFE,
    ToolRoundLimitError,
    ToolSpec,
    run_llm_with_tools,
)
from tenerife_rag.tools import (
    WEATHER_TOOL_SCHEMA,
    get_weather,
    make_search_tool,
)
from tenerife_rag.vector_store import (
    VectorStoreUnavailableError,
    connect_collection,
)



# Funciones (sin depedencias de Streamlit)



@dataclass
class Turn:
    """Un turno de conversación: mensaje del usuario y respuesta del asistente."""

    user: str  # mensaje del usuario (hasta 1000 caracteres, no vacío)
    assistant: str  # Respuesta_Usuario saneada


def validate_user_input(text: str, max_len: int = 1000) -> tuple[bool, str | None]:
    """Valida la entrada del usuario.

    Returns:
        (True, None) si el texto recortado tiene entre 1 y max_len caracteres.
        (False, motivo) si es vacío, solo espacios, o supera max_len.
    """
    stripped = text.strip()
    if not stripped:
        return (False, "La pregunta no puede estar vacía")
    if len(stripped) > max_len:
        return (False, f"La pregunta no puede superar {max_len} caracteres")
    return (True, None)


def truncate_history(turns: list["Turn"], max_turns: int = 10) -> list["Turn"]:
    """Conserva los `max_turns` turnos más recientes, descartando los antiguos."""
    if len(turns) <= max_turns:
        return turns
    return turns[-max_turns:]


def history_to_llm_messages(turns: list["Turn"]) -> list[dict]:
    """Aplana turnos a mensajes role/content para el Bucle_Tool_Calling."""
    messages = []
    for turn in turns:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})
    return messages



# Saneamiento de respuesta y atribución de fuente 


# Patrones de bloques técnicos a eliminar para mejorar la experiencia de usuario

# "RESUMEN DE HERRAMIENTAS UTILIZADAS" y todo lo que le sigue
_RESUMEN_HERRAMIENTAS_RE = re.compile(
    r"(?:^|\n)\s*-*\s*RESUMEN DE HERRAMIENTAS UTILIZADAS\s*-*\s*.*",
    re.DOTALL | re.IGNORECASE,
)

# Listados de herramientas invocadas con argumentos
#    "- search_tenerife_info(query=..., k=...)" o "- get_weather(location=..., date=...)"
_TOOL_LISTING_RE = re.compile(
    r"^\s*[-*]\s*\w+\(.*?=.*?\)\s*$",
    re.MULTILINE,
)

# Logs de llamadas a API y de ejecución
#    Ej: "Llamada a API: ..."
_LOG_LINE_RE = re.compile(
    r"^\s*(Llamada a API|Log de ejecución|API call|Execution log)\s*:.*$",
    re.MULTILINE | re.IGNORECASE,
)


def sanitize_response(text: str) -> str:
    """Elimina fragmentos de Registro_Herramientas / logs técnicos del texto,
    preservando la Atribucion_Fuente ('Fuente: ...')

    Remueve:
    El bloque "RESUMEN DE HERRAMIENTAS UTILIZADAS" y todo lo que esté después.

    Cualquier listado de herramientas invocadas con sus argumentos
    (por ejemplo, "- search_tenerife_info(query=..., k=...)").

    Registros de llamadas a la API y registros de ejecución
    (por ejemplo, "Llamada a API:", "Log de ejecución:").

    Lo que se PRESERVA:
    Líneas de atribución del tipo "Fuente: ..." (por ejemplo, "Fuente: TENERIFE.pdf")

    Todo el texto en lenguaje natural que no sean registros técnicos.
    
    """
    if not text:
        return text

    # Paso 1: Eliminar el bloque "RESUMEN DE HERRAMIENTAS UTILIZADAS" y todo lo que sigue
    result = _RESUMEN_HERRAMIENTAS_RE.sub("", text)

    # Paso 2: Eliminar líneas de listado de herramientas (pero NO líneas con "Fuente:")
    # Procesamos línea por línea para preservar las líneas "Fuente:" de forma segura
    lines = result.split("\n")
    cleaned_lines = []
    for line in lines:
        # Verificar si es una línea de listado de herramientas
        if _TOOL_LISTING_RE.match(line):
            # Asegurarse de que no sea una línea "Fuente:" antes de eliminarla
            if "Fuente:" not in line:
                continue
        # Verificar si es una línea de log
        if _LOG_LINE_RE.match(line):
            # Asegurarse de que no sea una línea "Fuente:" antes de eliminarla
            if "Fuente:" not in line:
                continue
        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)

    # Limpiar líneas en blanco excesivas (más de 2 consecutivas)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def has_source_attribution(text: str) -> bool:
    """True si el texto contiene 'Fuente:' (Req 6.1, 6.2)."""
    return "Fuente:" in text



# ---------------------------------------------------------------------------
# Conexión cacheada e inicialización de sesión (Req 2.4, 2.6, 7.5, 9.6)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


@st.cache_resource
def get_collection():
    """Conecta a la Base_Vectorial una sola vez por sesión y reutiliza la conexión.

    Usa ``@st.cache_resource`` para que la conexión se inicialice una vez y se
    reutilice en todas las consultas posteriores de la sesión (Req 2.6).

    Returns:
        La colección ChromaDB conectada, o None si la conexión falla.
        En caso de fallo, el error se reporta vía session_state en
        ``init_session_state``.
    """
    try:
        config = load_config()
        collection = connect_collection(config)
        return collection
    except (MissingApiKeyError, MissingEnvVarError, VectorStoreUnavailableError):
        # Re-raise para que init_session_state pueda capturar y clasificar
        raise
    except Exception as exc:
        # Error inesperado durante la conexión
        raise VectorStoreUnavailableError(
            "Error inesperado al conectar con la base de datos."
        ) from exc


def init_session_state() -> None:
    """Inicializa el estado de sesión de Streamlit y gestiona errores de arranque.

    Claves inicializadas en ``st.session_state``:
    - ``history`` (list[Turn]): historial de conversación, máx 10 turnos.
    - ``db_available`` (bool): True si la Base_Vectorial está disponible.
    - ``config_error`` (str | None): mensaje de error de configuración (sin secretos).

    Comportamiento ante errores (Req 2.4, 2.5, 7.5, 9.6):
    - Si ``OPENAI_API_KEY`` falta: muestra error y bloquea llamadas al LLM.
    - Si falta otra variable requerida: muestra error con nombre de la variable.
    - Si la Base_Vectorial no está disponible: muestra advertencia y deshabilita
      la búsqueda, pero conserva la sesión activa.
    """
    # Inicializar claves por defecto (solo si no existen aún)
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "db_available" not in st.session_state:
        st.session_state["db_available"] = False
    if "config_error" not in st.session_state:
        st.session_state["config_error"] = None

    # Intentar conectar a la colección (cacheada: solo se ejecuta una vez)
    try:
        collection = get_collection()
        st.session_state["db_available"] = True
    except MissingApiKeyError as exc:
        # Req 7.5: mostrar mensaje sin valor de la clave, solo nombre de variable
        error_msg = (
            f"⚠️ Error de configuración: la variable de entorno "
            f"'{exc.variable_name}' no está definida o está vacía. "
            f"Las llamadas al modelo de lenguaje están deshabilitadas."
        )
        st.session_state["config_error"] = error_msg
        st.session_state["db_available"] = False
        logger.error("Falta la variable de entorno: %s", exc.variable_name)
    except MissingEnvVarError as exc:
        # Req 9.6: indicar cuál variable falta
        error_msg = (
            f"⚠️ Error de configuración: falta la variable de entorno "
            f"requerida '{exc.variable_name}'."
        )
        st.session_state["config_error"] = error_msg
        st.session_state["db_available"] = False
        logger.error("Falta la variable de entorno: %s", exc.variable_name)
    except VectorStoreUnavailableError as exc:
        # Req 2.4, 2.5: base de datos no disponible, deshabilitar búsqueda
        st.session_state["db_available"] = False
        st.session_state["config_error"] = None  # No es error de config
        logger.warning("Base de datos no disponible: %s", exc)
    except Exception as exc:
        # Error inesperado
        st.session_state["db_available"] = False
        st.session_state["config_error"] = (
            "⚠️ Error inesperado durante la inicialización."
        )
        logger.exception("Error inesperado en init_session_state: %s", exc)

    # Mostrar mensajes de error/advertencia en la UI
    if st.session_state["config_error"]:
        st.error(st.session_state["config_error"])

    if not st.session_state["db_available"] and not st.session_state["config_error"]:
        st.warning(
            "⚠️ La base de datos no está disponible. "
            "La búsqueda documental está deshabilitada.\n\n"
            "**Acción requerida:** Asegúrese de que el directorio `tenerife_db/` "
            "existe y contiene la colección `guia_tenerife`. "
            "Puede generarla ejecutando el notebook `notebook/RAG_LLM.ipynb`."
        )


# ---------------------------------------------------------------------------
# Manejo de mensajes del usuario (Req 3.2, 3.6, 4.2, 4.3, 5.6, 6.2, 6.3,
#                                  8.1, 8.2, 8.5, 8.6)
# ---------------------------------------------------------------------------


def handle_user_message(text: str) -> None:
    """Procesa una pregunta del usuario: valida, invoca al LLM y actualiza el historial.

    Flujo:
    1. Valida la entrada con ``validate_user_input``.
    2. Si la configuración tiene errores o la BD no está disponible, rechaza.
    3. Construye el historial para el LLM (máx 10 turnos).
    4. Invoca ``run_llm_with_tools`` con spinner.
    5. Sanitiza la respuesta y verifica atribución de fuente.
    6. Añade el turno al historial y trunca.

    Errores (conservan el historial intacto):
    - ``openai.APITimeoutError``: timeout de 60 s (Req 8.2).
    - ``ToolRoundLimitError``: límite de rondas (Req 8.5).
    - ``openai.APIError`` / ``Exception``: fallo del LLM (Req 8.1).
    - Respuesta vacía tras sanear (Req 5.6).
    - Atribución ausente en respuesta basada en docs (Req 6.2).
    """
    # 1. Validar entrada
    stripped = text.strip()
    valid, reason = validate_user_input(text)
    if not valid:
        st.warning(reason)
        return

    # 2. Verificar que el sistema está disponible para llamar al LLM
    if st.session_state.get("config_error"):
        st.error(
            "No se puede procesar la pregunta: hay un error de configuración. "
            "Revise los mensajes de error anteriores."
        )
        return

    if not st.session_state.get("db_available"):
        st.error(
            "No se puede procesar la pregunta: la base de datos no está disponible."
        )
        return

    # 3. Construir historial para el LLM (máx 10 turnos) — Req 4.2
    llm_history = None
    try:
        truncated = truncate_history(st.session_state["history"])
        llm_history = history_to_llm_messages(truncated)
    except Exception as exc:
        # Req 4.2: si falla la construcción del historial, proceder sin él
        logger.error("Fallo al construir historial para el LLM: %s", exc)
        llm_history = None

    # 4. Invocar run_llm_with_tools con spinner (Req 3.6)
    with st.spinner("Generando respuesta..."):
        try:
            config = load_config()
            client = build_openai_client(config)

            # Construir herramientas
            collection = get_collection()
            search_tool = make_search_tool(collection)
            weather_tool = ToolSpec(schema=WEATHER_TOOL_SCHEMA, function=get_weather)

            result = run_llm_with_tools(
                client,
                stripped,
                tools=[search_tool, weather_tool],
                history=llm_history,
                instructions=INSTRUCCIONES_GUIA_TENERIFE,
                model=config.generation_model,
                max_tool_rounds=config.max_tool_rounds,
                **generation_params(config),
            )

        except openai.APITimeoutError:
            # Req 8.2: timeout de 60 s
            st.error(
                "La operación ha excedido el tiempo de espera. Inténtalo de nuevo."
            )
            return

        except ToolRoundLimitError:
            # Req 8.5: límite de rondas alcanzado
            st.error(
                "No se pudo completar la respuesta. El sistema alcanzó el límite "
                "de procesamiento."
            )
            return

        except (openai.APIError, Exception):
            # Req 8.1: fallo del LLM (genérico)
            st.error(
                "Ha ocurrido un error al procesar tu pregunta. Inténtalo de nuevo."
            )
            return

    # 5. Obtener texto de respuesta y sanear (Req 5.1, 5.5)
    response_text = result.output_text
    sanitized = sanitize_response(response_text)

    # Req 5.6: respuesta vacía tras sanear
    if not sanitized:
        st.error("No se pudo generar una respuesta.")
        return

    # 6. Verificar atribución de fuente (Req 6.2, 6.3, 8.6)
    search_was_used = "search_tenerife_info" in result.tool_names
    if search_was_used and not has_source_attribution(sanitized):
        # Req 6.2: la respuesta se basa en docs pero no tiene atribución.
        # Excepción Req 6.3 / 8.6: si la respuesta indica que no posee información,
        # es válido omitir la atribución.
        no_info_indicators = [
            "no poseo información",
            "no dispongo de información",
            "no tengo información",
            "no se encontró información",
            "no cuento con información",
        ]
        response_lower = sanitized.lower()
        lacks_info = any(
            indicator in response_lower for indicator in no_info_indicators
        )
        if not lacks_info:
            # Req 6.2: fallo por falta de atribución
            st.error("No se pudo determinar la fuente de la información.")
            return

    # 7. Añadir turno al historial y truncar (Req 4.3)
    new_turn = Turn(user=stripped, assistant=sanitized)
    st.session_state["history"] = truncate_history(
        st.session_state["history"] + [new_turn]
    )


# ---------------------------------------------------------------------------
# Renderizado del chat (Req 3.4, 3.5)
# ---------------------------------------------------------------------------


def render_chat_history() -> None:
    """Renderiza el Historial_Conversacion en orden cronológico ascendente.

    Itera por ``st.session_state["history"]`` (ya almacenado del más antiguo al
    más reciente) y muestra cada turno con ``st.chat_message``.

    Si la renderización falla (Req 3.5), muestra un mensaje de error y permite
    que el usuario siga haciendo preguntas (no bloquea la interfaz).
    """
    try:
        for turn in st.session_state.get("history", []):
            with st.chat_message("user"):
                st.markdown(turn.user)
            with st.chat_message("assistant"):
                st.markdown(turn.assistant)
    except Exception as exc:
        logger.error("Error al renderizar el historial: %s", exc)
        st.error("El historial no pudo mostrarse.")


# ---------------------------------------------------------------------------
# Reinicio de conversación (Req 4.5, 4.6)
# ---------------------------------------------------------------------------


def reset_conversation() -> None:
    """Reinicia el Historial_Conversacion a un estado vacío (cero turnos).

    Debe completarse en <=1 segundo (trivial para vaciar una lista).
    Si falla, muestra un mensaje de error y conserva el estado previo (Req 4.6).
    """
    try:
        st.session_state["history"] = []
    except Exception as exc:
        logger.error("Error al reiniciar la conversación: %s", exc)
        st.error("La conversación no pudo reiniciarse.")


# ---------------------------------------------------------------------------
# Función principal de la Aplicacion_Web (Req 3.7, 5.4, 9.2)
# ---------------------------------------------------------------------------


def main() -> None:
    """Punto de entrada principal de la aplicación Streamlit.

    Configura logging (Req 5.4), la página, inicializa la sesión, muestra el
    título y mensaje de bienvenida, renderiza el historial, y procesa la
    entrada del usuario.
    """
    # Configurar logging: logs van a consola, NUNCA a la UI de Streamlit (Req 5.4)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Configurar la página de Streamlit
    st.set_page_config(page_title="Asistente de Turismo - Tenerife", page_icon="🏝️")

    # Inicializar estado de sesión (conexión BD, config, historial)
    init_session_state()

    # Título de la aplicación (Req 3.7)
    st.title("🏝️ Asistente de Turismo de Tenerife")

    # Mensaje de bienvenida del asistente (siempre visible como primer mensaje)
    with st.chat_message("assistant"):
        st.markdown(
            "¡Hola! Soy tu asistente virtual de turismo para Tenerife. "
            "Puedo ayudarte con información sobre lugares de interés, playas, "
            "rutas, gastronomía, transporte y mucho más. ¿En qué puedo ayudarte?"
        )

    # Renderizar historial de conversación
    render_chat_history()

    # Sidebar: botón "Nueva conversación" (Req 4.5)
    with st.sidebar:
        if st.button("Nueva conversación"):
            reset_conversation()
            st.rerun()

    # Campo de entrada del usuario (debe estar a nivel top-level del script)
    user_input = st.chat_input("Escribe tu pregunta sobre Tenerife...")
    if user_input:
        # Req 3.3: rechazar preguntas vacías sin invocar al Motor_RAG
        stripped = user_input.strip()
        if stripped:
            handle_user_message(user_input)
            st.rerun()


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

main()
