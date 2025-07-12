# test_semantic_search.py (Versión para 3 modelos)
# -*- coding: utf-8 -*-

import time
import os
import pandas as pd
from dotenv import load_dotenv

try:
    from app import (
        APIQueryAgent,
        FAISSIndex,
        get_sentence_transformer_model,
        validate_dataset_relevance,
        GOOGLE_LLM_MODEL, API_KEY_GEMINI
    )
    load_dotenv()
    
    # Configuración de Gemini para validación (más robusta)
    gemini_model_for_test = None
    if API_KEY_GEMINI:
        try:
            import google.generativeai as genai
            from app import safety_settings 
            genai.configure(api_key=API_KEY_GEMINI)
            gemini_model_for_test = genai.GenerativeModel(
                model_name=GOOGLE_LLM_MODEL,
                safety_settings=safety_settings
            )
            # Monkey patch la variable global en el módulo app si validate_dataset_relevance la usa
            import app as main_app_module
            main_app_module.gemini_model = gemini_model_for_test
            print("Gemini configurado para validación en este script.")
        except Exception as e:
            print(f"Error configurando Gemini: {e}. La validación LLM podría fallar.")
    else:
        print("ADVERTENCIA: API_KEY_GEMINI no encontrada. La validación LLM se omitirá.")

except ModuleNotFoundError:
    print("Asegúrate de que app.py está accesible.")
    exit()
except Exception as e:
    print(f"Error importación/configuración inicial: {e}")
    exit()

# --- Configuración del Experimento ---
QUERIES_TO_TEST = [
    "Información sobre aparcamientos para bicicletas en Valencia",
    "Muéstrame datos de calidad del aire",
    "Bocas de metro en Valencia",
    "Muestra viviendas de protección civil",
    "¿Cuántos contenedores de reciclaje hay por barrio?Enseña los centros educativos en València",
    "Muestra los Presupuestos de gastos por artículo en 2023",
    "Obras ejecutadas por el ayuntamiento durante todos estos años",
    "Análisis de las fallas plantadas en los últimos años",
    "Mapa de las farmacias de la ciudad"
]
TOP_K_FAISS = 3

# Definición de los modelos y sus archivos de índice
MODELS_TO_EVALUATE = [
    {
        "name": 'paraphrase-MiniLM-L6-v2', # Tu nuevo modelo principal
        "index_path": 'faiss_opendata_valencia.idx',
        "metadata_path": 'faiss_metadata.pkl'
    },
    {
        "name": 'all-mpnet-base-v2',
        "index_path": "faiss_compare_all-mpnet-base-v2.idx", # Nombre de generate_custom_index.py
        "metadata_path": "faiss_compare_metadata_all-mpnet-base-v2.pkl"
    },
    {
        "name": 'multi-qa-mpnet-base-dot-v1',
        "index_path": "faiss_compare_multi-qa-mpnet-base-dot-v1.idx", # Nombre de generate_custom_index.py
        "metadata_path": "faiss_compare_metadata_multi-qa-mpnet-base-dot-v1.pkl"
    }
]

# (La función evaluate_search_for_model es la misma que te proporcioné antes,
#  asegurándote de que use el gemini_model_for_test para la validación si es necesario,
#  o que el monkey patch funcione.)
# ... (Pega aquí la función evaluate_search_for_model de la respuesta anterior) ...
def evaluate_search_for_model(
    queries: list, 
    embedding_model_name: str, 
    faiss_index_path: str, 
    faiss_metadata_path: str
    ):
    # ... (Copia la función completa de la respuesta anterior donde te la di) ...
    # ... (Asegúrate que la lógica de validación LLM dentro de esta función
    #      funciona con el gemini_model_for_test que configuramos arriba) ...
    print(f"\n--- Evaluando con Modelo de Embedding: {embedding_model_name} ---")
    print(f"Usando Índice FAISS: {faiss_index_path}, Metadatos: {faiss_metadata_path}")
    results_summary = []
    try:
        sentence_model_instance = get_sentence_transformer_model(embedding_model_name)
        faiss_index_instance = FAISSIndex(index_path=faiss_index_path, metadata_path=faiss_metadata_path)

        if not faiss_index_instance.is_ready():
            print(f"Error: Índice FAISS en {faiss_index_path} no está listo.")
            # Intenta dar más info si los archivos no existen
            if not os.path.exists(faiss_index_path): print(f"  Archivo de índice NO ENCONTRADO: {faiss_index_path}")
            if not os.path.exists(faiss_metadata_path): print(f"  Archivo de metadatos NO ENCONTRADO: {faiss_metadata_path}")
            return []
        api_agent = APIQueryAgent(faiss_index=faiss_index_instance, sentence_model=sentence_model_instance)
    except Exception as e:
        print(f"Error en la configuración para {embedding_model_name}: {e}")
        return []

    for query_idx, user_query in enumerate(queries):
        print(f"\nProcesando Consulta ({query_idx+1}/{len(queries)}): \"{user_query}\"")
        query_results = {
            "query": user_query,
            "model_embedding": embedding_model_name,
            "faiss_candidates_titles": [], # Lista de títulos
            "faiss_candidates_scores": [], # Lista de scores
            "selected_dataset_title": "N/A",
            "selected_dataset_similarity": 0.0,
            "llm_validation_passed": False,
            "error": None
        }
        start_time = time.time()
        try:
            search_results_faiss = api_agent.search_dataset(user_query, top_k=TOP_K_FAISS)
            
            if search_results_faiss:
                for res in search_results_faiss:
                    query_results["faiss_candidates_titles"].append(res["metadata"]["title"])
                    query_results["faiss_candidates_scores"].append(round(res['similarity'], 4))
                print(f"  FAISS encontró {len(search_results_faiss)} candidato(s):")
                for i in range(len(search_results_faiss)):
                    print(f"    {i+1}. Título: {query_results['faiss_candidates_titles'][i][:70]}... (Sim: {query_results['faiss_candidates_scores'][i]})")
            else:
                print("  FAISS no encontró candidatos.")
                query_results["error"] = "FAISS no encontró candidatos"
                results_summary.append(query_results) # Añadir incluso si no hay candidatos
                continue # Ir a la siguiente consulta

            selected_dataset_info = None
            for result_idx, result in enumerate(search_results_faiss): # Iterar sobre los resultados de FAISS
                dataset_title = result["metadata"]["title"]
                similarity_score = result['similarity']

                if similarity_score < api_agent.SIMILARITY_THRESHOLD:
                    print(f"    Descartando '{dataset_title[:70]}...' por baja similitud ({similarity_score:.4f}).")
                    continue
                
                print(f"    Validando con LLM: '{dataset_title[:70]}...'")
                if not gemini_model_for_test: # Usar la instancia configurada en el script
                    print("    Advertencia: Modelo LLM para validación no disponible globalmente. Asumiendo relevancia.")
                    is_relevant_by_llm = True
                else:
                    is_relevant_by_llm = validate_dataset_relevance( # Esta función usa app.gemini_model
                        user_query, 
                        dataset_title, 
                        result["metadata"].get("description", "")
                    ) 
                
                if is_relevant_by_llm:
                    print(f"    LLM validó '{dataset_title[:70]}...' como relevante.")
                    selected_dataset_info = result
                    query_results["selected_dataset_title"] = dataset_title
                    query_results["selected_dataset_similarity"] = round(similarity_score, 4)
                    query_results["llm_validation_passed"] = True
                    break 
                else:
                    print(f"    LLM descartó '{dataset_title[:70]}...' como no relevante.")
            
            if not selected_dataset_info:
                print("  Ningún dataset pasó la validación LLM o el umbral de similitud.")
                if not query_results["error"]: query_results["error"] = "Ningún dataset pasó validación/umbral"


        except Exception as e:
            print(f"  Error durante la búsqueda o validación: {e}")
            query_results["error"] = str(e)
        
        end_time = time.time()
        print(f"  Tiempo para esta consulta: {end_time - start_time:.2f} segundos.")
        results_summary.append(query_results)
        
    return results_summary

# --- Script Principal ---
if __name__ == "__main__":
    all_results_data = []
    print("Iniciando Evaluación del Módulo de Búsqueda Semántica para Múltiples Modelos...")

    for model_config in MODELS_TO_EVALUATE:
        model_name = model_config["name"]
        index_p = model_config["index_path"]
        metadata_p = model_config["metadata_path"]
        
        model_results = evaluate_search_for_model(
            QUERIES_TO_TEST,
            model_name,
            index_p,
            metadata_p
        )
        if model_results: # Solo añadir si la función no devolvió lista vacía por error de config
            all_results_data.extend(model_results)

    if not all_results_data:
        print("No se generaron resultados. Verifique la configuración y los archivos de índice.")
    else:
        # --- Presentación de Resultados ---
        print("\n\n--- RESUMEN DE RESULTADOS COMPLETOS ---")
        df_all_results = pd.DataFrame(all_results_data)
        
        # Mejorar la visualización de candidatos FAISS
        df_all_results['faiss_top1_title'] = df_all_results['faiss_candidates_titles'].apply(lambda x: x[0] if isinstance(x, list) and len(x)>0 else 'N/A')
        df_all_results['faiss_top1_score'] = df_all_results['faiss_candidates_scores'].apply(lambda x: x[0] if isinstance(x, list) and len(x)>0 else 0.0)

        # Seleccionar columnas para el resumen
        summary_columns = [
            "query", "model_embedding", 
            "faiss_top1_title", "faiss_top1_score",
            "selected_dataset_title", "selected_dataset_similarity", 
            "llm_validation_passed", "error"
        ]
        df_summary_display = df_all_results[summary_columns]

        print(df_summary_display.to_string()) # Imprimir todo el DataFrame sin truncar
        
        # Guardar en un CSV
        output_csv_path = "results_semantic_search_comparison.csv"
        df_all_results.to_csv(output_csv_path, index=False)
        print(f"\nResultados completos guardados en: {output_csv_path}")

    print("\nEvaluación completada.")