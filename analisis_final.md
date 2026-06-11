# Análisis Final del Proyecto: Asistente Turístico de Tenerife (RAG + LLM)

## 1. Diseño de la solución

El objetivo del proyecto es construir un asistente conversacional especializado en turismo para la isla de Tenerife, capaz de responder preguntas basándose exclusivamente en una guía turística proporcionada como fuente de datos (TENERIFE.pdf).

La arquitectura sigue el patrón **Retrieval-Augmented Generation (RAG)**, que combina búsqueda semántica sobre una base documental con generación de texto mediante un LLM. El flujo de datos es el siguiente:

1. **Ingesta y chunking**: el PDF se extrae con PyPDF, se limpia y se divide en fragmentos (chunks) de 1000 caracteres con un solapamiento de 200 caracteres para preservar el contexto entre segmentos.
2. **Embedding y almacenamiento**: cada chunk se transforma en un vector mediante el modelo `text-embedding-3-small` de OpenAI y se almacena en una colección de ChromaDB persistida en disco (`tenerife_db/`).
3. **Búsqueda semántica**: ante una consulta del usuario, se genera el embedding de la pregunta y se recuperan los k chunks más relevantes por distancia coseno.
4. **Generación con tool calling**: el LLM (`gpt-4.1-mini`) recibe las instrucciones del sistema, el historial de conversación y los chunks recuperados como contexto, y genera una respuesta atribuida a la fuente.
5. **Herramienta auxiliar de clima**: como complemento, se define una función simulada (`get_weather`) que el modelo puede invocar cuando el usuario pregunta por condiciones meteorológicas en Tenerife.

La interfaz web (Streamlit) reutiliza la base vectorial ya persistida sin necesidad de re-ingestar el PDF en cada ejecución, lo que reduce significativamente el tiempo de arranque y los costes de API.

## 2. Decisiones técnicas

### Modelos de OpenAI

Se emplean dos modelos distintos, cada uno optimizado para su tarea:

- **Generación**: `gpt-4.1-mini` — un modelo ligero pero con buena capacidad de razonamiento y seguimiento de instrucciones. Posee un balance adecuado entre coste y calidad.
- **Embeddings**: `text-embedding-3-small` — modelo de embeddings más económico y rápido que su variante `large`. Para un corpus reducido (20 chunks de una sola guía) ofrece una capacidad de discriminación semántica adecuada sin necesidad de la mayor dimensionalidad del modelo grande.

### Parámetros de generación

- **temperature = 0.2**: un valor bajo reduce la aleatoriedad en la selección de tokens, haciendo que el modelo se adhiera más estrictamente al contexto recuperado. Esto es fundamental en un sistema RAG donde se quiere minimizar la invención de información (alucinaciones). Al configurar la temperatura explícitamente, no se modifica `top_p` para evitar interacciones impredecibles entre ambos parámetros de muestreo.
- **max_output_tokens = 800**: limita la extensión de las respuestas para mantenerlas concisas y relevantes, evitando que el modelo genere texto excesivamente largo que pueda diluir la precisión de la información.

### Tool calling vs. sistema de agentes

Se optó por tool calling nativo de la API de OpenAI en lugar de un framework de agentes (como LangChain Agents o AutoGen) por las siguientes razones:

1. **Simplicidad del dominio**: el asistente solo necesita dos capacidades externas: buscar en la base documental y consultar el clima. No existe un grafo de decisiones complejo ni tareas que requieran planificación multi-paso autónoma.
2. **Control determinista**: con tool calling, el desarrollador define exactamente qué funciones están disponibles y cómo se validan sus argumentos (JSON Schema con `strict: true`). Esto otorga un control preciso sobre el comportamiento del sistema sin delegar la orquestación a un agente que podría tomar decisiones no deseadas.
3. **Menor latencia y coste**: un bucle de tool calling simple (máximo 5 rondas) evita la sobrecarga de frameworks de agentes que añaden capas de abstracción, prompts adicionales de reflexión y múltiples llamadas al LLM para planificar.
4. **Transparencia y depurabilidad**: el registro de ejecución (`ToolExecution`) captura cada llamada, sus argumentos, resultado y tiempo, facilitando la depuración sin la opacidad de un sistema multi-agente.

Un sistema multi-agente sería más apropiado si el proyecto creciese en complejidad: por ejemplo, si se requirieran múltiples fuentes documentales con estrategias de búsqueda diferentes, integración con APIs reales de reservas, o razonamiento multi-paso donde un agente planificador delegue sub-tareas a agentes especializados.

### ChromaDB como base vectorial

Se eligió ChromaDB por las siguientes ventajas para este caso de uso:

- **Persistencia local en disco**: los datos se guardan en un directorio (`tenerife_db/`) sin necesidad de un servidor separado. Esto simplifica el despliegue y elimina dependencias de infraestructura.
- **Almacenamiento integral**: ChromaDB almacena conjuntamente los vectores de embeddings, los documentos originales (texto de los chunks) y los metadatos (fuente, índice del chunk). Esto permite recuperar tanto la representación vectorial para la búsqueda como el texto completo para pasarlo al LLM, sin necesidad de un almacén secundario.
- **Integración nativa con OpenAI**: mediante `OpenAIEmbeddingFunction`, ChromaDB genera automáticamente los embeddings en la ingesta y en las consultas, simplificando el código.
- **Adecuado para volúmenes pequeños**: con solo 20 chunks, no se necesita un sistema vectorial distribuido (como Pinecone o Weaviate). ChromaDB opera eficientemente en memoria para colecciones de este tamaño.

### Atribución de fuentes

Cada respuesta generada a partir de la búsqueda documental incluye una línea de atribución con el formato `Fuente: TENERIFE.pdf (chunk N, score S)`. Esto cumple dos funciones:

1. **Trazabilidad**: el usuario puede verificar de dónde proviene la información.
2. **Confianza**: al mostrar explícitamente la fuente, se refuerza la percepción de que el sistema no está inventando respuestas.

El prompt del sistema instruye al modelo para que incluya la fuente cuando responde basándose en los documentos recuperados, y para que indique explícitamente cuándo no dispone de información suficiente.

### Interfaz web con Streamlit y multiturno

Como extensión del notebook (que solo contemplaba 3 preguntas predefinidas), se desarrolló una interfaz web con Streamlit que permite conversaciones de hasta 10 turnos. La gestión del historial funciona así:

- **`st.session_state`**: Streamlit proporciona un diccionario de estado de sesión que persiste entre re-ejecuciones del script (cada interacción del usuario provoca un rerun del script completo). El historial de conversación se almacena en `st.session_state["history"]` como una lista de objetos `Turn(user, assistant)`.
- **Truncamiento a 10 turnos**: antes de cada llamada al LLM, se conservan solo los 10 turnos más recientes. Esto evita exceder la ventana de contexto del modelo y controla el coste por llamada, ya que todo el historial se envía como parte del input.
- **Conversión a mensajes**: los turnos se aplanan a una lista de diccionarios `{"role": "user"/"assistant", "content": ...}` que se anteponen al nuevo mensaje del usuario en la llamada al LLM. Así, el modelo tiene contexto de la conversación previa y puede resolver referencias indirectas (por ejemplo "¿Y qué más hay ahí para comer?" refiriéndose a una ubicación mencionada antes).
- **Sanitización de la respuesta**: antes de mostrar la respuesta al usuario, se eliminan bloques técnicos (registros de herramientas, logs de ejecución) preservando las líneas de atribución de fuente.

## 3. Resultados

El sistema logra los objetivos propuestos:

- **Respuestas fundamentadas**: el modelo responde basándose en la guía turística y cuando la información no está disponible, lo indica explícitamente sin inventar datos.
- **Tool calling funcional**: el modelo invoca correctamente `search_tenerife_info` para consultas turísticas y `get_weather` para preguntas sobre el clima, demostrando la capacidad de seleccionar la herramienta apropiada según el contexto.
- **Conversación multiturno coherente**: el historial permite que el modelo resuelva referencias contextuales entre turnos (por ejemplo, preguntar "¿qué comidas hay ahí?" después de hablar del Teide).
- **Atribución consistente**: las respuestas incluyen la referencia a los chunks fuente, proporcionando trazabilidad al usuario.
- **Validación robusta de argumentos**: el sistema valida los argumentos de las herramientas contra su JSON Schema antes de ejecutarlas, capturando errores sin interrumpir la sesión.

## 4. Limitaciones

- El sistema depende de un único documento PDF. Esto limita la cobertura temática ya que no hay información sobre precios de entradas, horarios actualizados o eventos temporales.
-  La herramienta `get_weather` devuelve datos estáticos basados en reglas simples. No se conecta a una API meteorológica real, por lo que las respuestas sobre el clima no reflejan condiciones actuales.
- No se implementó un benchmark formal (como RAGAS o evaluación con juez LLM) para medir la calidad de las respuestas en términos de faithfulness, relevancia o recall de los chunks.
- La estrategia de chunking divide por longitud fija sin considerar la estructura semántica del documento (secciones, párrafos). Esto puede cortar información relevante entre dos chunks, aunque el solapamiento de 200 caracteres mitiga parcialmente este problema.
- A utilizar la API de OpenAI tanto para embeddings como para generación, el sistema requiere conexión a internet y está sujeto a latencia de red y posibles errores de disponibilidad.
- La aplicación Streamlit no implementa control de acceso, por lo que cualquier usuario con acceso a la URL puede interactuar con ella y generar costes de API.

## 5. Mejoras futuras

### Optimización de la recuperación

- **Experimentar con valores más bajos de k**: actualmente se recuperan entre 3 y 4 chunks por consulta (con un máximo de 8). Evaluar si reducir k a 2 mejora la precisión al eliminar chunks menos relevantes que podrían confundir al modelo, especialmente dado que el corpus es pequeño y cada chunk tiene alta probabilidad de ser parcialmente relevante.
- **Chunking semántico**: reemplazar el chunking por caracteres con una estrategia basada en la estructura del documento (dividir por secciones temáticas) o usar recursive text splitting con separadores jerárquicos (párrafos > oraciones) para preservar la coherencia semántica de cada fragmento.

### Ajuste de parámetros de generación

- **Variación de temperatura**: evaluar valores como 0.4 o 0.5 para lograr un tono más conversacional y natural, comparando contra la precisión factual. Un valor ligeramente más alto podría hacer las respuestas menos rígidas sin comprometer significativamente la fidelidad a la fuente.
- **Ajuste de max_output_tokens**: para preguntas que requieren respuestas detalladas (p.ej., itinerarios de varios días), un límite de 800 tokens puede resultar insuficiente. Se podría implementar un ajuste dinámico basado en la complejidad de la consulta.

### Herramientas adicionales

- **Conversión de divisas**: una herramienta que consulte tasas de cambio en tiempo real sería útil para turistas internacionales que quieran estimar costes en su moneda local.
- **Reservas y disponibilidad**: integración con APIs de servicios turísticos (teleférico del Teide, excursiones) para consultar disponibilidad y precios actualizados.
- **Transporte**: conexión con APIs de transporte público para informar sobre rutas y horarios reales.

### Multimodalidad

- **Entrada de audio**: permitir que el usuario haga preguntas por voz, mejorando la accesibilidad y la experiencia en contextos móviles durante el viaje.
- **Salida de audio**: usar el API de text-to-speech de OpenAI para leer las respuestas en voz alta, creando una experiencia de audioguía turística.
- **Entrada de imágenes**: permitir al usuario enviar fotos de un lugar para identificarlo y proporcionar información contextual como por ejemplo fotografiar un monumento y recibir su descripción e historia.

### Mejoras de infraestructura

- **Evaluación automatizada**: implementar un pipeline de evaluación con métricas como faithfulness (fidelidad al contexto recuperado), answer relevancy (relevancia de la respuesta a la pregunta) y context recall, usando frameworks como RAGAS o evaluación con un LLM juez.
- **Conexión a API meteorológica real**: sustituir la función simulada por una integración con OpenWeatherMap o AEMET para ofrecer datos climáticos reales.
- **Caché de respuestas**: para preguntas frecuentes, almacenar respuestas previas y servirlas directamente sin consumir tokens adicionales.
