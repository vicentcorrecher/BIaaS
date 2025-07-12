# generate_custom_index.py (Versión para 3 modelos)
# -*- coding: utf-8 -*-

import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv

# Importa las clases y funciones necesarias de tu app.py
try:
    from app import (
        APIQueryAgent,
        FAISSIndex,
        get_sentence_transformer_model,
        BASE_URL
    )
    # Mock Streamlit para ejecución standalone
    class MockStreamlit:
        def progress(self, value): return self 
        def empty(self): return self
        def text(self, value): print(value) 
        def success(self, value): print(f"SUCCESS: {value}")
        def error(self, value): print(f"ERROR: {value}")
        def warning(self, value): print(f"WARNING: {value}")
        def info(self, value): print(f"INFO: {value}")
        def balloons(self): pass
    st = MockStreamlit()

except ModuleNotFoundError:
    print("Asegúrate de que app.py está accesible y las importaciones son correctas.")
    exit()
except ImportError as e:
    if 'streamlit' in str(e).lower():
        print("Streamlit no disponible. Usando mock.")
        class MockStreamlit:
            def progress(self, value): return self; # Permite encadenar .text(value)
            def empty(self): return self;
            def text(self, value): print(value);
            def success(self, value): print(f"SUCCESS: {value}");
            def error(self, value): print(f"ERROR: {value}");
            def warning(self, value): print(f"WARNING: {value}");
            def info(self, value): print(f"INFO: {value}");
            def balloons(self): pass;
        st = MockStreamlit()
    else:
        raise e

def build_index_for_model(
    target_model_name: str, 
    output_index_path: str, 
    output_metadata_path: str
    ):
    print(f"\n--- Construyendo Índice FAISS para modelo: {target_model_name} ---")
    print(f"Guardando en: {output_index_path}, {output_metadata_path}")
    
    sentence_model_instance = get_sentence_transformer_model(target_model_name)
    
    # APIQueryAgent necesita una instancia de FAISSIndex, aunque sea temporal y no se use para búsqueda aquí.
    # Asumimos que FAISSIndex en app.py ahora toma rutas en __init__ y no falla si los archivos no existen.
    temp_faiss_index = FAISSIndex(index_path="dummy.idx", metadata_path="dummy.pkl") 
    agent_for_building = APIQueryAgent(temp_faiss_index, sentence_model_instance)

    # El st mock manejará las llamadas a st.progress/st.text en get_all_datasets_metadata
    datasets_metadata = agent_for_building.get_all_datasets_metadata()
    if not datasets_metadata:
        st.error("No se pudieron obtener metadatos. Abortando.")
        return

    st.text(f"Vectorizando {len(datasets_metadata)} datasets usando {target_model_name}...")
    embeddings = []
    valid_metadata = []
    
    total_datasets = len(datasets_metadata)
    for i, meta in enumerate(datasets_metadata):
        text_to_embed = f"{meta['title']} {meta['description']}"
        embedding = agent_for_building.get_embedding(text_to_embed)
        embeddings.append(embedding)
        valid_metadata.append(meta)
        if (i + 1) % 50 == 0 or (i + 1) == total_datasets: # Actualiza menos frecuentemente para consola
            progress_percent = (i + 1) / total_datasets
            st.text(f"Vectorizados {i+1}/{total_datasets} ({progress_percent*100:.0f}%)...")

    if not embeddings:
        st.error("No se generaron embeddings. Abortando.")
        return

    embeddings_np = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings_np)

    embedding_dimension = sentence_model_instance.get_sentence_embedding_dimension()
    st.text(f"Creando índice FAISS (IndexFlatIP) con dimensión {embedding_dimension}...")
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings_np)
    st.success(f"Índice creado con {index.ntotal} vectores.")

    st.text(f"Guardando índice en {output_index_path}...")
    faiss.write_index(index, output_index_path)
    st.text(f"Guardando metadatos en {output_metadata_path}...")
    with open(output_metadata_path, 'wb') as f:
        pickle.dump(valid_metadata, f)
    st.success(f"--- Índice para {target_model_name} Completado ---")

if __name__ == "__main__":
    load_dotenv()

    # MODELO 1: Tu nuevo modelo principal (MiniLM)
    MODEL_1_NAME = 'paraphrase-MiniLM-L6-v2'
    MODEL_1_INDEX_PATH = 'faiss_opendata_valencia.idx' # Será el índice principal de tu app
    MODEL_1_METADATA_PATH = 'faiss_metadata.pkl'     # Será el metadato principal de tu app

    # MODELO 2: Tu anterior modelo principal (all-mpnet)
    MODEL_2_NAME = 'all-mpnet-base-v2'
    MODEL_2_INDEX_PATH = f"faiss_compare_{MODEL_2_NAME.replace('/', '_')}.idx"
    MODEL_2_METADATA_PATH = f"faiss_compare_metadata_{MODEL_2_NAME.replace('/', '_')}.pkl"

    # MODELO 3: Nuevo modelo para comparación (multi-qa-mpnet)
    MODEL_3_NAME = 'multi-qa-mpnet-base-dot-v1'
    MODEL_3_INDEX_PATH = f"faiss_compare_{MODEL_3_NAME.replace('/', '_')}.idx"
    MODEL_3_METADATA_PATH = f"faiss_compare_metadata_{MODEL_3_NAME.replace('/', '_')}.pkl"

    # Generar los índices (descomenta según necesites)
    print(f"Generando índice para el modelo: {MODEL_1_NAME}")
    build_index_for_model(MODEL_1_NAME, MODEL_1_INDEX_PATH, MODEL_1_METADATA_PATH)
    
    print(f"\nGenerando índice para el modelo: {MODEL_2_NAME}")
    build_index_for_model(MODEL_2_NAME, MODEL_2_INDEX_PATH, MODEL_2_METADATA_PATH)

    print(f"\nGenerando índice para el modelo: {MODEL_3_NAME}")
    build_index_for_model(MODEL_3_NAME, MODEL_3_INDEX_PATH, MODEL_3_METADATA_PATH)

    print("\nProceso de generación de todos los índices completado.")