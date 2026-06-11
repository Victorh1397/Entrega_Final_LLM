"""Motor_RAG: paquete de módulos reutilizables para el asistente de turismo de Tenerife.

Este paquete refactoriza la lógica RAG + LLM del notebook `RAG_LLM.ipynb` en
módulos Python importables:

- ``config``: carga de configuración del entorno y cliente LLM (Modulo_Config).
- ``vector_store``: conexión a la Base_Vectorial persistida y búsqueda documental.
- ``tools``: definición de las herramientas ``search_tenerife_info`` y ``get_weather``.
- ``tool_loop``: bucle de tool calling ``run_llm_with_tools``.
"""
