# test_full_pipeline_llm_comparison.py
# -*- coding: utf-8 -*-

import pandas as pd
import json
import os
from dotenv import load_dotenv
import time

# Importa tu función principal del pipeline y la variable global que queremos modificar
try:
    import app as main_app_module # Para acceder y modificar CURRENT_LLM_PROVIDER
    from app import (
        run_full_dashboard_pipeline_st,
        get_faiss_index_instance, # Para obtener el índice
        get_sentence_transformer_model, # Para el modelo de embedding
        EMBEDDING_MODEL, # Para el modelo de embedding
        # Asegúrate de que las API keys se carguen al importar app o cárgalas aquí
    )
    load_dotenv() # Carga .env para que app.py pueda acceder a las keys

    # Configurar Gemini y Groq (esto ya debería ocurrir al importar app.py si lo tienes ahí)
    # Si no, necesitarías replicar la inicialización de main_app_module.gemini_model y main_app_module.groq_client aquí.
    # Es crucial que main_app_module.gemini_model y main_app_module.groq_client (o como llames al cliente Llama3)
    # estén listos ANTES de llamar a run_full_dashboard_pipeline_st.
    # El inicio de tu app.py ya maneja esto, así que debería estar bien.

except ModuleNotFoundError:
    print("Asegúrate de que app.py está accesible.")
    exit()
except Exception as e:
    print(f"Error importación/configuración inicial: {e}")
    exit()

# --- Casos de Prueba (Solo la consulta de usuario) ---
# Usaremos el sistema de búsqueda de app.py para encontrar el dataset
QUERIES_FOR_LLM_TEST = [
    "Centros educativos en Valencia",
    "Muestra la disponibilidad de Valenbisi",
    "Muestra el estado del tráfico en tiempo real",
    # Puedes añadir 1-2 más si quieres
]

# LLMs a probar
LLM_PROVIDERS_TO_TEST = ["gemini", "llama3"] # Deben coincidir con los de make_llm_call

# Directorio para guardar resultados
RESULTS_DIR = "full_pipeline_llm_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Mocking de Streamlit UI para capturar salidas ---
# Necesitamos "capturar" lo que run_full_dashboard_pipeline_st normalmente mostraría en Streamlit.
# Esto es la parte más complicada de este enfoque.
# Opciones:
# 1. Modificar run_full_dashboard_pipeline_st para que DEVUELVA los resultados (JSON de viz, texto de insights)
#    en lugar de (o además de) mostrarlos con st.plotly_chart / st.markdown. Esta es la más limpia.
# 2. Usar un "mock" muy elaborado de Streamlit que intercepte las llamadas a st.plotly_chart, etc. (difícil).

# ASUMIREMOS LA OPCIÓN 1: Modificar run_full_dashboard_pipeline_st

# --- Modificación Sugerida para run_full_dashboard_pipeline_st en app.py ---
# Al final de la función, en lugar de solo st.success("--- ✅ Pipeline Completado ---"):
#
#   pipeline_output = {
#       "user_query": user_query,
#       "selected_dataset_title": dataset_title if 'dataset_title' in locals() else "N/A",
#       "dataset_id": dataset_id if 'dataset_id' in locals() else "N/A",
#       "viz_configs_suggested": viz_configs_suggested, # La lista de dicts del planner
#       "insights_text": insights_text,
#       "llm_provider_used": CURRENT_LLM_PROVIDER # Para saber qué LLM generó esto
#   }
#   st.success("--- ✅ Pipeline Completado ---")
#   st.balloons()
#   return pipeline_output # ¡DEVOLVER LOS RESULTADOS!
#
# Si no devuelve nada, o no encuentras el dataset, devuelve un dict con errores.


# --- Script Principal ---
if __name__ == "__main__":
    print("Iniciando Evaluación del Pipeline Completo con Comparación de LLMs...")

    # Cargar recursos comunes una vez (índice y modelo de embedding)
    # Asegúrate de que EMBEDDING_MODEL en app.py sea el que quieres usar para la búsqueda (MiniLM)
    print(f"Usando modelo de embedding para búsqueda: {EMBEDDING_MODEL}")
    faiss_index = get_faiss_index_instance()
    sentence_model = get_sentence_transformer_model(EMBEDDING_MODEL)

    if not faiss_index.is_ready():
        print("Error: Índice FAISS no está listo. Ejecuta la construcción del índice primero.")
        exit()
    
    # Verificar que los clientes LLM estén listos en app.py
    if not main_app_module.gemini_model:
        print("Advertencia: El modelo Gemini no parece estar inicializado en app.py.")
    if LLM_PROVIDERS_TO_TEST.count("llama3") > 0 and not main_app_module.groq_client : # o como hayas llamado a tu cliente Llama3
        print("Advertencia: El cliente Groq/Llama3 no parece estar inicializado en app.py.")


    all_pipeline_results = []

    for query in QUERIES_FOR_LLM_TEST:
        print(f"\n\n--- PROCESANDO CONSULTA: \"{query}\" ---")
        for llm_provider in LLM_PROVIDERS_TO_TEST:
            print(f"  --- Usando LLM Provider: {llm_provider} ---")
            
            # Cambiar el proveedor de LLM en app.py
            main_app_module.CURRENT_LLM_PROVIDER = llm_provider
            
            # Ejecutar el pipeline completo
            # Necesitamos "silenciar" los st.spinner, st.balloons, etc., o que no den error.
            # La forma más fácil es modificar run_full_dashboard_pipeline_st para que
            # tenga un flag `silent_mode=True` que deshabilite las llamadas a st.* que no sean st.error/warning
            # o que simplemente las ignore si el contexto de Streamlit no está.
            # Por ahora, asumimos que se ejecutarán pero los warnings de "missing ScriptRunContext" son ignorables.
            
            # Limpiar st.session_state si tu app lo usa extensivamente para evitar
            # contaminación entre ejecuciones (esto es avanzado y puede no ser necesario)
            # if hasattr(st, 'session_state'):
            #     for key in list(st.session_state.keys()):
            #         del st.session_state[key]

            try:
                # Aquí es donde necesitas que run_full_dashboard_pipeline_st DEVUELVA algo
                output = run_full_dashboard_pipeline_st(query, faiss_index, sentence_model)
                if output: # Si devuelve algo
                    all_pipeline_results.append(output)
                else: # Si no devuelve nada o devuelve None en error
                    all_pipeline_results.append({
                        "user_query": query,
                        "llm_provider_used": llm_provider,
                        "error": "Pipeline no devolvió resultados o encontró un error no capturado."
                    })

            except Exception as e:
                print(f"    ERROR ejecutando el pipeline para {query} con {llm_provider}: {e}")
                all_pipeline_results.append({
                    "user_query": query,
                    "llm_provider_used": llm_provider,
                    "error": str(e)
                })
            
            # Pequeña pausa para asegurar que las API no se saturen (opcional)
            time.sleep(2) 

    # --- Guardar Resultados ---
    if all_pipeline_results:
        output_json_path = os.path.join(RESULTS_DIR, "results_full_pipeline_llm_comparison.json")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_pipeline_results, f, ensure_ascii=False, indent=4)
        print(f"\nResultados del pipeline completo guardados en: {output_json_path}")
        
        try:
            df_pipeline_results = pd.DataFrame(all_pipeline_results)
            csv_path = os.path.join(RESULTS_DIR, "results_full_pipeline_llm_comparison.csv")
            df_pipeline_results.to_csv(csv_path, index=False)
            print(f"Resultados también guardados en: {csv_path}")
        except Exception as e:
            print(f"No se pudo guardar como CSV directamente: {e}")

    print("\nEvaluación del pipeline completo con LLMs completada.")